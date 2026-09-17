"""Stage-1.2 ET-RCM extension with controlled query and memory interventions."""

from __future__ import annotations

import torch
from torch.nn import functional as Fnn

from etrcm.stage1_1.events import StructuredEvent
from etrcm.stage1_1.model import LearnedETRCM, LearnedState


class Stage12ETRCM(LearnedETRCM):
    """Preserves Stage-1.1 laws while exposing preregistered interventions."""

    def proposed_query(
        self, state: LearnedState, event: StructuredEvent | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = state.H.shape[0]
        if event is None:
            encoded = torch.zeros(
                batch, self.config.hidden_dim, device=state.H.device, dtype=state.H.dtype
            )
        else:
            event.validate(batch)
            encoded = self.event_encoder(event).to(dtype=state.H.dtype)
        event_slots = self.event_to_slots(encoded).view(
            batch, self.config.latent_slots, self.config.hidden_dim
        )
        H_pre = state.H + event_slots
        return self._query(H_pre), H_pre, encoded

    def step_with_forced_query(
        self,
        state: LearnedState,
        event: StructuredEvent | None,
        forced_query: torch.Tensor,
    ):
        """Run exactly one normal transition except for replacing q.

        This method is evaluation-only. It does not add evidence and preserves
        the same external-write, consolidation, decay and learned-core paths.
        """

        state.validate()
        batch = state.H.shape[0]
        if forced_query.shape != (batch, self.config.key_dim):
            raise ValueError("forced query must be [batch,key_dim]")
        F, M = state.F, state.M
        external_update = torch.zeros_like(F)
        if event is None:
            encoded = torch.zeros(
                batch, self.config.hidden_dim, device=state.H.device, dtype=state.H.dtype
            )
            written = False
        else:
            event.validate(batch)
            encoded = self.event_encoder(event).to(dtype=state.H.dtype)
            F, M, external_update = self._external_write(F, M, event)
            written = bool(event.write_mask.any())
        H_pre = state.H + self.event_to_slots(encoded).view(
            batch, self.config.latent_slots, self.config.hidden_dim
        )
        q = forced_query.to(H_pre)
        read = torch.einsum("bvk,bk->bv", F + M, q)
        core_input = torch.cat(
            (
                self.norm(H_pre),
                read[:, None, :].expand(-1, self.config.latent_slots, -1),
                encoded[:, None, :].expand(-1, self.config.latent_slots, -1),
            ),
            dim=-1,
        )
        proposal = self.core_out(Fnn.silu(self.core_in(core_input)))
        gate = torch.sigmoid(self.core_gate(core_input))
        H_new = H_pre if self.freeze_core else H_pre + gate * proposal
        access = torch.sigmoid(self.access_head(self._pool(H_new))).squeeze(-1)
        total_before = F + M
        F, M, transfer = self._consolidate(F, M, q, access)
        conservation_error = torch.linalg.vector_norm((F + M) - total_before, dim=(-2, -1))
        rho_fast = self.config.rho_slow if self.equal_timescales else self.config.rho_fast
        F, M = rho_fast * F, self.config.rho_slow * M
        new_state = LearnedState(
            H_new, F, M, state.tau + 1, state.external_time + int(written)
        )
        output = self.predict(new_state)
        output.update(
            {
                "query": q,
                "read": read,
                "access": access,
                "transfer": transfer,
                "external_update": external_update,
                "conservation_error": conservation_error,
                "event_written": written,
                "h_change": torch.linalg.vector_norm((H_new - H_pre).flatten(1), dim=-1),
            }
        )
        return new_state, output

    def lesion(self, state: LearnedState, component: str) -> LearnedState:
        if component not in {"none", "fast", "slow", "both"}:
            raise ValueError(component)
        F = torch.zeros_like(state.F) if component in {"fast", "both"} else state.F
        M = torch.zeros_like(state.M) if component in {"slow", "both"} else state.M
        return LearnedState(state.H, F, M, state.tau, state.external_time)

