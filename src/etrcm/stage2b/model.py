"""Read-old-memory, update-H, then contextual external write (E1/E2)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from etrcm.stage1_1.model import LearnedState, _normalize
from etrcm.stage1_3.events import ContinuousEvent
from etrcm.stage1_4.model import Stage14Config
from etrcm.stage2a.language import LanguageETRCM


class ContextualLanguageETRCM(LanguageETRCM):
    """Only the external write encoding and its necessary ordering differ.

    E1: normalized contextual H -> K,V. E2: contextual H plus current token.
    E0_late is a diagnostic raw-token control with the same read-before-write
    ordering, separate from the untouched Stage 2A E0 main baseline.
    """

    ARMS = {"E1", "E2", "E0_late"}

    def __init__(self, config: Stage14Config, arm: str = "E1"):
        if arm not in self.ARMS:
            raise ValueError(arm)
        super().__init__(config, mode="B5_separate")
        self.arm = arm
        width = config.hidden_dim + (config.value_dim if arm == "E2" else 0)
        if arm != "E0_late":
            self.context_key = nn.Linear(width, config.key_dim, bias=False)
            self.context_value = nn.Linear(width, config.value_dim, bias=False)

    def contextual_kv(self, H: torch.Tensor, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.arm == "E0_late":
            return self.target_key(token), self.value_vector(token)
        context = self._pool(H)
        if self.arm == "E2":
            context = torch.cat([context, self.symbol_embedding(token)], dim=-1)
        key = _normalize(self.context_key(context))
        value = self.context_value(context)
        return key, value

    def step(self, state: LearnedState, event: ContinuousEvent | None,
             **unused_kwargs) -> tuple[LearnedState, dict[str, torch.Tensor]]:
        if unused_kwargs:
            raise TypeError(f"unsupported contextual step arguments: {sorted(unused_kwargs)}")
        state.validate()
        batch = state.H.shape[0]
        zeros = torch.zeros(batch, self.config.hidden_dim, device=state.H.device, dtype=state.H.dtype)
        encoded = zeros
        write_mask = torch.zeros(batch, dtype=torch.bool, device=state.H.device)
        external_event = 0
        if event is not None:
            event.validate(batch)
            encoded = self.event_encoder(event).to(state.H.dtype)
            write_mask = event.write_mask
            external_event = int((~event.self_output_mask).any())

        # A: queries see the current token and old F/M, never the new write.
        slots = (self.event_to_slots(encoded).view_as(state.H)
                 if event is not None else torch.zeros_like(state.H))
        H_pre = state.H + slots
        q_F, q_M = self._queries(H_pre)
        reads = self._read_stage14(state.F, state.M, q_F, q_M, H_pre, encoded)
        read = reads["read"]

        # B: the frozen gated-residual integration operator.
        core_input = torch.cat([
            self.norm(H_pre), read[:, None, :].expand(-1, self.config.latent_slots, -1),
            encoded[:, None, :].expand(-1, self.config.latent_slots, -1),
        ], -1)
        H_new = H_pre + torch.sigmoid(self.core_gate(core_input)) * self.core_out(F.silu(self.core_in(core_input)))

        # C/D: only genuine external evidence can produce a delta write.
        key = torch.zeros(batch, self.config.key_dim, device=state.H.device, dtype=state.H.dtype)
        value = torch.zeros(batch, self.config.value_dim, device=state.H.device, dtype=state.H.dtype)
        update = torch.zeros_like(state.F)
        F_mem, M_mem = state.F, state.M
        if event is not None and bool(write_mask.any()):
            key, value = self.contextual_kv(H_new, event.value_id.clamp_min(0))
            prior = torch.einsum("bvk,bk->bv", state.F + state.M, key)
            update = self.config.eta_external * torch.einsum("bv,bk->bvk", value - prior, key)
            update = torch.where(write_mask[:, None, None], update, torch.zeros_like(update))
            F_mem = F_mem + update

        # Existing use-driven, readout-conserving F->M transfer and decay.
        access = torch.sigmoid(self.access_head(self._pool(H_new))).squeeze(-1)
        total_before = F_mem + M_mem
        F_mem, M_mem, transfer = self._consolidate(F_mem, M_mem, q_F, access)
        conservation_error = (F_mem + M_mem - total_before).norm(dim=(-2, -1))
        next_state = LearnedState(H_new, self.config.rho_fast * F_mem,
                                  self.config.rho_slow * M_mem,
                                  state.tau + 1, state.external_time + external_event)
        diagnostics = {
            **reads, "context_key": key, "context_value": value,
            "external_update": update, "external_write_flag": write_mask,
            "transfer": transfer, "access": access,
            "conservation_error": conservation_error,
            "H_delta_norm": (H_new - state.H).norm(dim=(-2, -1)),
            "read_pre_write_F": reads["r_F"], "read_pre_write_M": reads["r_M"],
        }
        return next_state, diagnostics
