"""Trained Stage-1.1 baselines with explicit state/capacity accounting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from etrcm.stage1_1.events import StructuredEvent
from etrcm.stage1_1.model import (
    LearnedETRCM,
    LearnedEventEncoder,
    LearnedModelConfig,
    LearnedState,
    RMSNorm,
    _normalize,
)


class NoMemoryRecurrentMLP(LearnedETRCM):
    """B0: learned recurrent H, but memory writes and reads are disabled."""

    def __init__(self, config: LearnedModelConfig):
        super().__init__(config, consolidation="none")

    def _external_write(self, F, M, event):  # type: ignore[override]
        return F, M, torch.zeros_like(F)

    def persistent_state_bytes(self, batch_item: bool = True) -> int:
        return self.config.latent_slots * self.config.hidden_dim * 4


@dataclass
class GRUState:
    H: torch.Tensor
    F: torch.Tensor
    M: torch.Tensor
    tau: int = 0
    external_time: int = 0

    def validate(self) -> None:
        if self.H.ndim != 3 or self.H.shape[1] != 1:
            raise ValueError("GRU H must be [batch,1,hidden]")

    def detach(self) -> "GRUState":
        return GRUState(
            self.H.detach(), self.F.detach(), self.M.detach(), self.tau, self.external_time
        )

    def clone(self) -> "GRUState":
        return GRUState(
            self.H.clone(), self.F.clone(), self.M.clone(), self.tau, self.external_time
        )


class TrainedGRU(nn.Module):
    """B1: ordinary GRU with a learned current-event encoder and no matrix memory."""

    def __init__(self, config: LearnedModelConfig, hidden_dim: int | None = None):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim or (config.hidden_dim * config.latent_slots)
        self.symbol_embedding = nn.Embedding(config.symbol_count, config.value_dim)
        nn.init.normal_(self.symbol_embedding.weight, std=0.3)
        self.event_encoder = LearnedEventEncoder(config, self.symbol_embedding)
        self.cell = nn.GRUCell(config.hidden_dim, self.hidden_dim)
        self.initial_H = nn.Parameter(torch.zeros(self.hidden_dim))
        self.norm = RMSNorm(self.hidden_dim)
        self.symbol_head = nn.Linear(self.hidden_dim, config.symbol_count)
        self.binary_head = nn.Linear(self.hidden_dim, 1)
        self.query_projection = nn.Linear(self.hidden_dim, config.key_dim, bias=False)

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> GRUState:
        parameter = self.initial_H
        device, dtype = device or parameter.device, dtype or parameter.dtype
        H = self.initial_H.to(device=device, dtype=dtype).expand(batch_size, -1).clone()
        empty = torch.zeros(batch_size, 0, 0, device=device, dtype=dtype)
        return GRUState(H[:, None, :], empty, empty.clone())

    def reset_active(self, state: GRUState) -> GRUState:
        H = self.initial_H.to(state.H).expand(state.H.shape[0], -1).clone()
        return GRUState(H[:, None, :], state.F, state.M, state.tau, state.external_time)

    def target_key(self, ids: torch.Tensor) -> torch.Tensor:
        return _normalize(self.symbol_embedding(ids), dim=-1)

    def step(self, state: GRUState, event: StructuredEvent | None):
        state.validate()
        batch = state.H.shape[0]
        if event is None:
            encoded = torch.zeros(batch, self.config.hidden_dim, device=state.H.device)
            written = False
        else:
            event.validate(batch)
            encoded = self.event_encoder(event)
            written = bool(event.write_mask.any())
        hidden = self.cell(encoded, state.H[:, 0, :])
        new_state = GRUState(
            hidden[:, None, :],
            state.F,
            state.M,
            state.tau + 1,
            state.external_time + int(written),
        )
        pooled = self.norm(hidden)
        outputs = {
            "symbol_logits": self.symbol_head(pooled),
            "binary_logit": self.binary_head(pooled).squeeze(-1),
            "query": _normalize(self.query_projection(pooled), dim=-1),
            "read": torch.zeros(batch, self.config.value_dim, device=hidden.device),
            "access": torch.zeros(batch, device=hidden.device),
            "transfer": torch.zeros(batch, 0, 0, device=hidden.device),
            "external_update": torch.zeros(batch, 0, 0, device=hidden.device),
            "conservation_error": torch.zeros(batch, device=hidden.device),
            "event_written": written,
        }
        return new_state, outputs

    def predict(self, state: GRUState):
        pooled = self.norm(state.H[:, 0, :])
        return {
            "symbol_logits": self.symbol_head(pooled),
            "binary_logit": self.binary_head(pooled).squeeze(-1),
        }

    def memory_lesion(self, state: GRUState) -> GRUState:
        return state

    def persistent_state_bytes(self, batch_item: bool = True) -> int:
        return self.hidden_dim * 4

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class TwoHeadSinglePersistent(LearnedETRCM):
    """B2: two slow matrices, matching the F+M float count of ET-RCM."""

    def __init__(self, config: LearnedModelConfig):
        super().__init__(config, consolidation="none")
        self.second_key = nn.Linear(config.value_dim, config.key_dim, bias=False)
        self.second_query = nn.Linear(config.hidden_dim, config.key_dim, bias=False)

    def _key2(self, ids: torch.Tensor) -> torch.Tensor:
        return _normalize(self.second_key(self.symbol_vectors(ids)), dim=-1)

    def step(self, state: LearnedState, event: StructuredEvent | None):  # noqa: C901
        state.validate()
        batch = state.H.shape[0]
        A1, A2 = state.F, state.M
        external_update = torch.zeros_like(A1)
        if event is None:
            encoded = torch.zeros(
                batch, self.config.hidden_dim, device=state.H.device, dtype=state.H.dtype
            )
            written = False
        else:
            event.validate(batch)
            encoded = self.event_encoder(event).to(dtype=state.H.dtype)
            key1 = self.target_key(event.key_id.clamp_min(0))
            key2 = self._key2(event.key_id.clamp_min(0))
            value = self.symbol_vectors(event.value_id.clamp_min(0))
            read1 = torch.einsum("bvk,bk->bv", A1, key1)
            read2 = torch.einsum("bvk,bk->bv", A2, key2)
            update1 = self.config.eta_external * torch.einsum(
                "bv,bk->bvk", value - read1, key1
            )
            update2 = self.config.eta_external * torch.einsum(
                "bv,bk->bvk", value - read2, key2
            )
            mask = event.write_mask[:, None, None]
            update1 = torch.where(mask, update1, torch.zeros_like(update1))
            update2 = torch.where(mask, update2, torch.zeros_like(update2))
            A1, A2 = A1 + update1, A2 + update2
            external_update = 0.5 * (update1 + update2)
            written = bool(event.write_mask.any())

        H_pre = state.H + self.event_to_slots(encoded).view(
            batch, self.config.latent_slots, self.config.hidden_dim
        )
        pooled = self._pool(H_pre)
        q1 = _normalize(self.query_projection(pooled), dim=-1)
        q2 = _normalize(self.second_query(pooled), dim=-1)
        read = 0.5 * (
            torch.einsum("bvk,bk->bv", A1, q1)
            + torch.einsum("bvk,bk->bv", A2, q2)
        )
        core_input = torch.cat(
            (
                self.norm(H_pre),
                read[:, None, :].expand(-1, self.config.latent_slots, -1),
                encoded[:, None, :].expand(-1, self.config.latent_slots, -1),
            ),
            dim=-1,
        )
        proposal = self.core_out(torch.nn.functional.silu(self.core_in(core_input)))
        gate = torch.sigmoid(self.core_gate(core_input))
        H_new = H_pre + gate * proposal
        A1, A2 = self.config.rho_slow * A1, self.config.rho_slow * A2
        new_state = LearnedState(
            H_new, A1, A2, state.tau + 1, state.external_time + int(written)
        )
        outputs = self.predict(new_state)
        outputs.update(
            {
                "query": q1,
                "query2": q2,
                "read": read,
                "access": torch.ones(batch, device=state.H.device),
                "transfer": torch.zeros_like(A1),
                "external_update": external_update,
                "conservation_error": torch.zeros(batch, device=state.H.device),
                "event_written": written,
            }
        )
        return new_state, outputs


def build_baseline(name: str, config: LearnedModelConfig) -> nn.Module:
    if name == "B0_no_memory_mlp":
        return NoMemoryRecurrentMLP(config)
    if name == "B1_gru":
        return TrainedGRU(config)
    if name == "B2_single_persistent":
        return TwoHeadSinglePersistent(config)
    if name == "B3_uniform":
        return LearnedETRCM(config, consolidation="uniform")
    if name == "B5_no_idle":
        return LearnedETRCM(config, idle_updates=False)
    if name == "B6_full":
        return LearnedETRCM(config)
    raise KeyError(name)


TRAINED_BASELINES = (
    "B0_no_memory_mlp",
    "B1_gru",
    "B2_single_persistent",
    "B3_uniform",
    "B5_no_idle",
    "B6_full",
)
