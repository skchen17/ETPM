"""Stage 1.5 intervention interface; standard trained transition is unchanged."""

from __future__ import annotations

import torch
from torch.nn import functional as Fnn

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_3.events import ContinuousEvent
from etrcm.stage1_4.model import PredictiveETRCM


class AnatomicalETRCM(PredictiveETRCM):
    """Adds read clamps without changing parameters or the memory law."""

    def step_with_read(
        self,
        state: LearnedState,
        event: ContinuousEvent | None,
        *,
        fast_read_override: torch.Tensor | None = None,
        slow_read_override: torch.Tensor | None = None,
        fused_read_override: torch.Tensor | None = None,
    ) -> tuple[LearnedState, dict[str, torch.Tensor]]:
        if all(item is None for item in (fast_read_override, slow_read_override, fused_read_override)):
            return self.step(state, event)
        if self.mode not in {"B5_separate", "B6_gamma_zero", "B7_random_query"}:
            raise ValueError("read clamps require the separate-query interface")
        state.validate()
        batch = state.H.shape[0]
        encoded = torch.zeros(batch, self.config.hidden_dim, device=state.H.device, dtype=state.H.dtype)
        F, M = state.F, state.M
        external_update = torch.zeros_like(F)
        external_write_flag = torch.zeros(batch, dtype=torch.bool, device=state.H.device)
        external_event = 0
        if event is not None:
            event.validate(batch)
            encoded = self.event_encoder(event).to(state.H.dtype)
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
        memory_r_F = torch.einsum("bvk,bk->bv", F, q_F)
        memory_r_M = torch.einsum("bvk,bk->bv", M, q_M)

        def checked(item: torch.Tensor | None, fallback: torch.Tensor) -> torch.Tensor:
            if item is None:
                return fallback
            if item.shape != fallback.shape or item.device != fallback.device:
                raise ValueError("read override shape/device mismatch")
            return item.to(dtype=fallback.dtype)

        r_F = checked(fast_read_override, memory_r_F)
        r_M = checked(slow_read_override, memory_r_M)
        n_F = self.fast_read_norm(r_F)
        n_M = self.slow_read_norm(r_M)
        gates = torch.softmax(self.two_way_gate(torch.cat([self._pool(H_pre), encoded], -1)), -1)
        read = gates[:, :1] * n_F + gates[:, 1:] * n_M
        read = checked(fused_read_override, read)
        if self.core_kind == "gru":
            H_new = self.gru(torch.cat([encoded, read], -1), state.H.reshape(batch, -1)).view_as(state.H)
        else:
            core_input = torch.cat([
                self.norm(H_pre), read[:, None, :].expand(-1, self.config.latent_slots, -1),
                encoded[:, None, :].expand(-1, self.config.latent_slots, -1),
            ], -1)
            H_new = H_pre + torch.sigmoid(self.core_gate(core_input)) * self.core_out(
                Fnn.silu(self.core_in(core_input))
            )
        access = torch.sigmoid(self.access_head(self._pool(H_new))).squeeze(-1)
        total_before = F + M
        if self.mode == "B6_gamma_zero":
            transfer = torch.zeros_like(F)
        else:
            F, M, transfer = self._consolidate(F, M, q_F, access)
        conservation_error = (F + M - total_before).norm(dim=(-2, -1))
        F = self.config.rho_fast * F
        M = self.config.rho_slow * M
        next_state = LearnedState(H_new, F, M, state.tau + 1, state.external_time + external_event)
        output = {
            "read": read, "q_F": q_F, "q_M": q_M,
            "r_F": r_F, "r_M": r_M, "memory_r_F": memory_r_F, "memory_r_M": memory_r_M,
            "normalized_r_F": n_F, "normalized_r_M": n_M, "gates": gates,
            "r_F_norm": r_F.norm(dim=-1), "r_M_norm": r_M.norm(dim=-1),
            "effective_F_norm": (gates[:, :1] * n_F).norm(dim=-1),
            "effective_M_norm": (gates[:, 1:] * n_M).norm(dim=-1),
            "access": access, "transfer": transfer, "external_update": external_update,
            "external_write_flag": external_write_flag,
            "conservation_error": conservation_error,
            "H_delta_norm": (H_new - state.H).norm(dim=(-2, -1)),
            "fast_read_restored": torch.full((batch,), fast_read_override is not None, device=state.H.device),
            "slow_read_restored": torch.full((batch,), slow_read_override is not None, device=state.H.device),
        }
        return next_state, output
