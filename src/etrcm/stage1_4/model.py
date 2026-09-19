"""Predictive ET-RCM; only the current event and (H,F,M) enter a transition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as Fnn

from etrcm.stage1_1.model import LearnedState, RMSNorm, _normalize
from etrcm.stage1_3.events import ContinuousEvent
from etrcm.stage1_3.model import ContinuousETRCM, Stage13Config


@dataclass(frozen=True)
class Stage14Config(Stage13Config):
    read_epsilon: float = 1e-6

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "Stage14Config":
        model = mapping["model"]
        fields = cls.__dataclass_fields__
        return cls(**{name: model[name] for name in fields if name in model})


class PredictiveETRCM(ContinuousETRCM):
    """R0/R1/R2 and matched baselines without an expression training target.

    The superclass supplies the unchanged external delta write, state shape,
    event encoder, recurrent core, and readout-conserving transfer. This class
    changes the read interface and adds future-event heads only.
    """

    MODES = {
        "B0_no_memory": ("none", "gated"),
        "B1_gru": ("none", "gru"),
        "B2_single_memory": ("single_persistent", "gated"),
        "B3_joint": ("joint", "gated"),
        "B4_shared": ("arbitration", "gated"),
        "B5_separate": ("arbitration", "gated"),
        "B6_gamma_zero": ("arbitration", "gated"),
        "B7_random_query": ("arbitration", "gated"),
    }

    def __init__(self, config: Stage14Config, mode: str = "B5_separate"):
        if mode not in self.MODES:
            raise ValueError(mode)
        read_mode, core_kind = self.MODES[mode]
        super().__init__(config, read_mode=read_mode, core_kind=core_kind)
        self.mode = mode
        self.q_fast_projection = nn.Linear(config.hidden_dim, config.key_dim, bias=False)
        self.q_slow_projection = nn.Linear(config.hidden_dim, config.key_dim, bias=False)
        self.fast_read_norm = RMSNorm(config.value_dim)
        self.slow_read_norm = RMSNorm(config.value_dim)
        self.two_way_gate = nn.Linear(2 * config.hidden_dim, 2)
        self.future_heads = nn.ModuleDict(
            {str(h): nn.Linear(config.hidden_dim, config.symbol_count) for h in (1, 2, 4, 8)}
        )
        random_direction = _normalize(torch.randn(config.key_dim))
        self.register_buffer("random_q_M", random_direction, persistent=True)
        self.register_buffer(
            "random_null_matrix",
            _normalize(torch.randn(config.hidden_dim, config.hidden_dim), dim=0) * 0.1,
            persistent=True,
        )

    def predict_logits(self, state: LearnedState) -> dict[int, torch.Tensor]:
        pooled = self._pool(state.H)
        return {int(h): head(pooled) for h, head in self.future_heads.items()}

    def _queries(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self._pool(H)
        if self.mode in {"B5_separate", "B6_gamma_zero", "B7_random_query"}:
            q_F = _normalize(self.q_fast_projection(pooled))
            q_M = _normalize(self.q_slow_projection(pooled))
            if self.mode == "B7_random_query":
                q_M = self.random_q_M.to(H).expand_as(q_M)
            return q_F, q_M
        shared = self.query(H)
        return shared, shared

    def _read_stage14(
        self, F: torch.Tensor, M: torch.Tensor, q_F: torch.Tensor,
        q_M: torch.Tensor, H: torch.Tensor, encoded: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        r_F = torch.einsum("bvk,bk->bv", F, q_F)
        r_M = torch.einsum("bvk,bk->bv", M, q_M)
        eps = self.config.read_epsilon
        # Independent RMS normalizers are only part of the R2 architecture.
        n_F = self.fast_read_norm(r_F) if self.mode.startswith(("B5", "B6", "B7")) else r_F
        n_M = self.slow_read_norm(r_M) if self.mode.startswith(("B5", "B6", "B7")) else r_M
        if self.mode.startswith(("B5", "B6", "B7")):
            gates = torch.softmax(self.two_way_gate(torch.cat([self._pool(H), encoded], -1)), -1)
            read = gates[:, :1] * n_F + gates[:, 1:] * n_M
        elif self.mode == "B4_shared":
            fast_gate = torch.sigmoid(self.arbitration_head(torch.cat([self._pool(H), encoded], -1)))
            gates = torch.cat([fast_gate, 1 - fast_gate], -1)
            read = gates[:, :1] * r_F + gates[:, 1:] * r_M
        elif self.mode == "B3_joint":
            gates = torch.ones(H.shape[0], 2, device=H.device, dtype=H.dtype)
            read = r_F + r_M
        elif self.mode == "B2_single_memory":
            gates = torch.stack([torch.zeros_like(r_M[:, 0]), torch.ones_like(r_M[:, 0])], -1)
            read = r_M
        else:
            gates = torch.zeros(H.shape[0], 2, device=H.device, dtype=H.dtype)
            read = torch.zeros_like(r_F)
        return {
            "read": read, "q_F": q_F, "q_M": q_M, "r_F": r_F, "r_M": r_M,
            "normalized_r_F": n_F, "normalized_r_M": n_M, "gates": gates,
            "r_F_norm": r_F.norm(dim=-1), "r_M_norm": r_M.norm(dim=-1),
            "normalized_r_F_norm": n_F.norm(dim=-1), "normalized_r_M_norm": n_M.norm(dim=-1),
            "effective_F_norm": (gates[:, :1] * n_F).norm(dim=-1),
            "effective_M_norm": (gates[:, 1:] * n_M).norm(dim=-1),
            "read_epsilon": torch.full((H.shape[0],), eps, device=H.device, dtype=H.dtype),
        }

    def step(
        self, state: LearnedState, event: ContinuousEvent | None,
        *, freeze_null_H: bool = False, random_null_H: bool = False,
        random_M_query: bool = False,
        block_transfer_key: torch.Tensor | None = None,
    ) -> tuple[LearnedState, dict[str, torch.Tensor]]:
        state.validate()
        batch = state.H.shape[0]
        encoded = torch.zeros(batch, self.config.hidden_dim, device=state.H.device, dtype=state.H.dtype)
        external_update = torch.zeros_like(state.F)
        F, M = state.F, state.M
        external_write_flag = torch.zeros(batch, dtype=torch.bool, device=state.H.device)
        external_event = 0
        if event is not None:
            event.validate(batch)
            encoded = self.event_encoder(event).to(state.H.dtype)
            # The validated event schema makes SELF_OUTPUT/NULL writes impossible.
            if bool(event.write_mask.any()):
                F, M, external_update = self._external_write(F, M, event)
            external_write_flag = event.write_mask
            external_event = int((~event.self_output_mask).any())
        event_slots = (
            self.event_to_slots(encoded).view(batch, self.config.latent_slots, self.config.hidden_dim)
            if event is not None else torch.zeros_like(state.H)
        )
        H_pre = state.H + event_slots
        q_F, q_M = self._queries(H_pre)
        if random_M_query:
            q_M = self.random_q_M.to(H_pre).expand_as(q_M)
        reads = self._read_stage14(F, M, q_F, q_M, H_pre, encoded)
        read = reads["read"]
        if event is None and freeze_null_H:
            H_new = state.H
        elif event is None and random_null_H:
            # This matrix is random, fixed, and never optimized by world loss.
            H_new = state.H + torch.tanh(self.norm(state.H) @ self.random_null_matrix)
        elif self.core_kind == "gru":
            H_new = self.gru(torch.cat([encoded, read], -1), state.H.reshape(batch, -1)).view_as(state.H)
        else:
            core_input = torch.cat([
                self.norm(H_pre), read[:, None, :].expand(-1, self.config.latent_slots, -1),
                encoded[:, None, :].expand(-1, self.config.latent_slots, -1),
            ], -1)
            H_new = H_pre + torch.sigmoid(self.core_gate(core_input)) * self.core_out(Fnn.silu(self.core_in(core_input)))
        access = torch.sigmoid(self.access_head(self._pool(H_new))).squeeze(-1)
        total_before = F + M
        if self.mode == "B6_gamma_zero":
            transfer = torch.zeros_like(F)
        else:
            F_raw, M_raw = F, M
            F, M, transfer = self._consolidate(F, M, q_F, access)
            if block_transfer_key is not None:
                if block_transfer_key.shape != q_F.shape:
                    raise ValueError("block_transfer_key must be [batch,key_dim]")
                block_key = _normalize(block_transfer_key)
                blocked_value = torch.einsum("bvk,bk->bv", transfer, block_key)
                blocked = torch.einsum("bv,bk->bvk", blocked_value, block_key)
                transfer = transfer - blocked
                F, M = F_raw - transfer, M_raw + transfer
        conservation_error = (F + M - total_before).norm(dim=(-2, -1))
        F = self.config.rho_fast * F
        M = self.config.rho_slow * M
        next_state = LearnedState(H_new, F, M, state.tau + 1, state.external_time + external_event)
        outputs = {
            **reads, "access": access, "transfer": transfer,
            "external_update": external_update, "external_write_flag": external_write_flag,
            "conservation_error": conservation_error,
            "H_delta_norm": (H_new - state.H).norm(dim=(-2, -1)),
        }
        return next_state, outputs
