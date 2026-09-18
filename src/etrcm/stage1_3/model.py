"""Continuous ET-RCM with read arbitration and non-halting expression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as Fnn

from etrcm.stage1_1.model import LearnedState, RMSNorm, _normalize
from etrcm.stage1_3.events import ContinuousEvent, Stage13EventKind


@dataclass(frozen=True)
class Stage13Config:
    hidden_dim: int = 64
    latent_slots: int = 2
    symbol_count: int = 32
    key_dim: int = 16
    value_dim: int = 16
    event_type_dim: int = 8
    gamma: float = 0.12
    rho_fast: float = 0.97
    rho_slow: float = 0.9995
    eta_external: float = 0.6
    expression_content_count: int = 32

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Stage13Config":
        model = config["model"]
        return cls(**{field: model[field] for field in cls.__dataclass_fields__})


class ContinuousEventEncoder(nn.Module):
    def __init__(self, config: Stage13Config, symbols: nn.Embedding):
        super().__init__()
        self.symbols = symbols
        self.kind = nn.Embedding(len(Stage13EventKind), config.event_type_dim)
        width = config.event_type_dim + 3 * config.value_dim + 4
        self.network = nn.Sequential(
            nn.Linear(width, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def _symbol(self, ids: torch.Tensor) -> torch.Tensor:
        present = ids.ge(0).unsqueeze(-1)
        value = self.symbols(ids.clamp_min(0))
        return torch.where(present, value, torch.zeros_like(value))

    def forward(self, event: ContinuousEvent) -> torch.Tensor:
        event.validate()
        return self.network(
            torch.cat(
                [
                    self.kind(event.kind),
                    self._symbol(event.key_id),
                    self._symbol(event.value_id),
                    self._symbol(event.aux_id),
                    event.scalars.to(self.symbols.weight.dtype),
                ],
                dim=-1,
            )
        )


class ContinuousETRCM(nn.Module):
    """One shared recurrent transition; expression is an action, never halting."""

    READ_MODES = {"joint", "m_only", "f_only", "arbitration", "none", "single_persistent"}

    def __init__(
        self,
        config: Stage13Config,
        *,
        read_mode: str = "arbitration",
        core_kind: str = "gated",
        self_output_writes: bool = False,
    ):
        super().__init__()
        if read_mode not in self.READ_MODES:
            raise ValueError(read_mode)
        if core_kind not in {"gated", "gru"}:
            raise ValueError(core_kind)
        self.config = config
        self.read_mode = read_mode
        self.core_kind = core_kind
        self.self_output_writes = self_output_writes
        self.has_memory = read_mode not in {"none"}

        self.symbol_embedding = nn.Embedding(config.symbol_count, config.value_dim)
        nn.init.normal_(self.symbol_embedding.weight, std=0.25)
        self.event_encoder = ContinuousEventEncoder(config, self.symbol_embedding)
        self.initial_H = nn.Parameter(torch.randn(config.latent_slots, config.hidden_dim) * 0.02)
        self.event_to_slots = nn.Linear(config.hidden_dim, config.latent_slots * config.hidden_dim)
        self.norm = RMSNorm(config.hidden_dim)
        self.query_projection = nn.Linear(config.hidden_dim, config.key_dim, bias=False)
        recurrent_width = config.hidden_dim + config.value_dim + config.hidden_dim
        self.core_in = nn.Linear(recurrent_width, 2 * config.hidden_dim)
        self.core_out = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        self.core_gate = nn.Linear(recurrent_width, config.hidden_dim)
        flat_hidden = config.latent_slots * config.hidden_dim
        self.gru = nn.GRUCell(config.hidden_dim + config.value_dim, flat_hidden)
        self.arbitration_head = nn.Linear(2 * config.hidden_dim, 1)
        self.access_head = nn.Linear(config.hidden_dim, 1)
        self.expression_head = nn.Linear(config.hidden_dim, 1)
        self.content_head = nn.Linear(config.hidden_dim, config.expression_content_count)
        self.self_output_to_slots = nn.Linear(
            config.value_dim, config.latent_slots * config.hidden_dim, bias=False
        )
        self.self_output_gate = nn.Linear(config.hidden_dim, config.hidden_dim)

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> LearnedState:
        device = device or self.initial_H.device
        dtype = dtype or self.initial_H.dtype
        H = self.initial_H.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).clone()
        memory = torch.zeros(
            batch_size, self.config.value_dim, self.config.key_dim, device=device, dtype=dtype
        )
        return LearnedState(H, memory, memory.clone())

    def reset_active(self, state: LearnedState) -> LearnedState:
        H = self.initial_H.to(state.H).unsqueeze(0).expand(state.H.shape[0], -1, -1).clone()
        return LearnedState(H, state.F, state.M, state.tau, state.external_time)

    def _pool(self, H: torch.Tensor) -> torch.Tensor:
        return self.norm(H).mean(1)

    def target_key(self, ids: torch.Tensor) -> torch.Tensor:
        return _normalize(self.symbol_embedding(ids), dim=-1)

    def value_vector(self, ids: torch.Tensor) -> torch.Tensor:
        return _normalize(self.symbol_embedding(ids), dim=-1)

    def query(self, H: torch.Tensor) -> torch.Tensor:
        return _normalize(self.query_projection(self._pool(H)), dim=-1)

    def _external_write(
        self, F: torch.Tensor, M: torch.Tensor, event: ContinuousEvent
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.has_memory:
            return F, M, torch.zeros_like(F)
        key = self.target_key(event.key_id.clamp_min(0))
        value = self.value_vector(event.value_id.clamp_min(0))
        if self.read_mode == "single_persistent":
            current = torch.einsum("bvk,bk->bv", M, key)
            update = self.config.eta_external * torch.einsum("bv,bk->bvk", value - current, key)
            update = torch.where(event.write_mask[:, None, None], update, torch.zeros_like(update))
            return F, M + update, update
        current = torch.einsum("bvk,bk->bv", F + M, key)
        update = self.config.eta_external * torch.einsum("bv,bk->bvk", value - current, key)
        update = torch.where(event.write_mask[:, None, None], update, torch.zeros_like(update))
        return F + update, M, update

    def _read(
        self, F: torch.Tensor, M: torch.Tensor, q: torch.Tensor, H: torch.Tensor, encoded: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r_fast = torch.einsum("bvk,bk->bv", F, q)
        r_slow = torch.einsum("bvk,bk->bv", M, q)
        if self.read_mode == "joint":
            gate = torch.full((H.shape[0],), 0.5, device=H.device, dtype=H.dtype)
            read = r_fast + r_slow
        elif self.read_mode in {"m_only", "single_persistent"}:
            gate = torch.zeros(H.shape[0], device=H.device, dtype=H.dtype)
            read = r_slow
        elif self.read_mode == "f_only":
            gate = torch.ones(H.shape[0], device=H.device, dtype=H.dtype)
            read = r_fast
        elif self.read_mode == "arbitration":
            gate = torch.sigmoid(self.arbitration_head(torch.cat([self._pool(H), encoded], -1))).squeeze(-1)
            read = gate[:, None] * r_fast + (1.0 - gate[:, None]) * r_slow
        else:
            gate = torch.zeros(H.shape[0], device=H.device, dtype=H.dtype)
            read = torch.zeros_like(r_fast)
        return read, r_fast, r_slow, gate

    def _consolidate(
        self, F: torch.Tensor, M: torch.Tensor, q: torch.Tensor, access: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.has_memory or self.read_mode == "single_persistent":
            return F, M, torch.zeros_like(F)
        fast_read = torch.einsum("bvk,bk->bv", F, q)
        delta = self.config.gamma * access[:, None, None] * torch.einsum("bv,bk->bvk", fast_read, q)
        return F - delta, M + delta, delta

    def apply_self_output_feedback(
        self, state: LearnedState, content_id: torch.Tensor, mask: torch.Tensor | None = None
    ) -> LearnedState:
        """Feed expression back into H only; F/M and clocks are untouched."""
        if mask is None:
            mask = torch.ones(content_id.shape[0], dtype=torch.bool, device=content_id.device)
        content = self.symbol_embedding(content_id.clamp_min(0))
        proposal = self.self_output_to_slots(content).view(
            content.shape[0], self.config.latent_slots, self.config.hidden_dim
        )
        gate = torch.sigmoid(self.self_output_gate(self.norm(state.H)))
        H = state.H + mask[:, None, None] * gate * torch.tanh(proposal)
        return LearnedState(H, state.F, state.M, state.tau, state.external_time)

    def expression(self, state: LearnedState) -> dict[str, torch.Tensor]:
        pooled = self._pool(state.H)
        logit = self.expression_head(pooled).squeeze(-1)
        return {
            "expression_logit": logit,
            "expression_score": torch.sigmoid(logit),
            "content_logits": self.content_head(pooled),
        }

    def step(
        self,
        state: LearnedState,
        event: ContinuousEvent | None,
        *,
        threshold: float | None = None,
        feedback_emission: bool = True,
    ) -> tuple[LearnedState, dict[str, torch.Tensor | bool]]:
        state.validate()
        batch = state.H.shape[0]
        F, M = state.F, state.M
        encoded = torch.zeros(batch, self.config.hidden_dim, device=state.H.device, dtype=state.H.dtype)
        external_update = torch.zeros_like(F)
        external_write_flag = torch.zeros(batch, dtype=torch.bool, device=state.H.device)
        self_input_flag = torch.zeros(batch, dtype=torch.bool, device=state.H.device)
        external_event = False
        if event is not None:
            event.validate(batch)
            encoded = self.event_encoder(event).to(state.H.dtype)
            self_input_flag = event.self_output_mask
            external_event = bool((~event.self_output_mask).any())
            F, M, external_update = self._external_write(F, M, event)
            external_write_flag = event.write_mask.clone()

        event_slots = self.event_to_slots(encoded).view(batch, self.config.latent_slots, self.config.hidden_dim)
        if event is None:
            event_slots = torch.zeros_like(event_slots)
        H_pre = state.H + event_slots
        q = self.query(H_pre)
        read, r_fast, r_slow, arbitration_gate = self._read(F, M, q, H_pre, encoded)
        if self.core_kind == "gru":
            flat = state.H.reshape(batch, -1)
            H_new = self.gru(torch.cat([encoded, read], -1), flat).view_as(state.H)
        else:
            core_input = torch.cat(
                [
                    self.norm(H_pre),
                    read[:, None, :].expand(-1, self.config.latent_slots, -1),
                    encoded[:, None, :].expand(-1, self.config.latent_slots, -1),
                ],
                -1,
            )
            proposal = self.core_out(Fnn.silu(self.core_in(core_input)))
            H_new = H_pre + torch.sigmoid(self.core_gate(core_input)) * proposal
        access = torch.sigmoid(self.access_head(self._pool(H_new))).squeeze(-1)
        total_before = F + M
        F, M, transfer = self._consolidate(F, M, q, access)
        conservation_error = torch.linalg.vector_norm(F + M - total_before, dim=(-2, -1))
        F = self.config.rho_fast * F
        M = self.config.rho_slow * M
        new_state = LearnedState(
            H_new, F, M, state.tau + 1, state.external_time + int(external_event)
        )
        outputs: dict[str, torch.Tensor | bool] = self.expression(new_state)
        content_id = outputs["content_logits"].argmax(-1)  # type: ignore[union-attr]
        score = outputs["expression_score"]  # type: ignore[assignment]
        emitted = torch.zeros(batch, dtype=torch.bool, device=state.H.device)
        if threshold is not None:
            emitted = score.ge(threshold)  # type: ignore[union-attr]
        self_memory_update = torch.zeros_like(F)
        if bool(emitted.any()) and feedback_emission:
            new_state = self.apply_self_output_feedback(new_state, content_id, emitted)
            if self.self_output_writes:
                value = self.value_vector(content_id)
                self_memory_update = 0.25 * torch.einsum("bv,bk->bvk", value, q)
                self_memory_update = torch.where(
                    emitted[:, None, None], self_memory_update, torch.zeros_like(self_memory_update)
                )
                new_state = LearnedState(
                    new_state.H,
                    new_state.F,
                    new_state.M + self_memory_update,
                    new_state.tau,
                    new_state.external_time,
                )
        outputs.update(
            {
                "query": q,
                "read": read,
                "r_fast": r_fast,
                "r_slow": r_slow,
                "arbitration_gate": arbitration_gate,
                "access": access,
                "transfer": transfer,
                "external_update": external_update,
                "external_write_flag": external_write_flag,
                "self_output_input_flag": self_input_flag,
                "self_output_memory_update": self_memory_update,
                "conservation_error": conservation_error,
                "emitted": emitted,
                "emitted_content_id": content_id,
            }
        )
        return new_state, outputs

    def lesion(self, state: LearnedState, component: str) -> LearnedState:
        if component == "fast":
            return LearnedState(state.H, torch.zeros_like(state.F), state.M, state.tau, state.external_time)
        if component == "slow":
            return LearnedState(state.H, state.F, torch.zeros_like(state.M), state.tau, state.external_time)
        if component == "both":
            return LearnedState(state.H, torch.zeros_like(state.F), torch.zeros_like(state.M), state.tau, state.external_time)
        if component == "none":
            return state
        raise ValueError(component)

    def persistent_state_bytes(self) -> int:
        memory = 0 if not self.has_memory else 2 * self.config.value_dim * self.config.key_dim
        return 4 * (self.config.latent_slots * self.config.hidden_dim + memory)

    def trainable_parameters(self) -> int:
        return sum(item.numel() for item in self.parameters() if item.requires_grad)
