"""Batched learned ET-RCM with a fixed readout-conserving memory law."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as Fnn

from etrcm.stage1_1.events import StructuredEvent


def _normalize(value: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=dim, keepdim=True).clamp_min(eps)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(value.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return value * scale * self.weight


@dataclass(frozen=True)
class LearnedModelConfig:
    hidden_dim: int = 128
    latent_slots: int = 4
    symbol_count: int = 64
    key_dim: int = 32
    value_dim: int = 32
    event_type_dim: int = 16
    gamma: float = 0.12
    rho_fast: float = 0.97
    rho_slow: float = 0.9995
    eta_external: float = 0.6

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "LearnedModelConfig":
        model = config["model"]
        return cls(
            hidden_dim=int(model["hidden_dim"]),
            latent_slots=int(model["latent_slots"]),
            symbol_count=int(model["symbol_count"]),
            key_dim=int(model["key_dim"]),
            value_dim=int(model["value_dim"]),
            event_type_dim=int(model["event_type_dim"]),
            gamma=float(model["gamma"]),
            rho_fast=float(model["rho_fast"]),
            rho_slow=float(model["rho_slow"]),
            eta_external=float(model["eta_external"]),
        )


@dataclass
class LearnedState:
    H: torch.Tensor
    F: torch.Tensor
    M: torch.Tensor
    tau: int = 0
    external_time: int = 0

    def validate(self) -> None:
        if self.H.ndim != 3:
            raise ValueError("H must be [batch,slots,hidden]")
        if self.F.ndim != 3 or self.M.ndim != 3 or self.F.shape != self.M.shape:
            raise ValueError("F/M must be equal [batch,value,key] tensors")
        if self.H.shape[0] != self.F.shape[0]:
            raise ValueError("state batch mismatch")

    def clone(self) -> "LearnedState":
        return LearnedState(
            self.H.clone(), self.F.clone(), self.M.clone(), self.tau, self.external_time
        )

    def detach(self) -> "LearnedState":
        return LearnedState(
            self.H.detach(), self.F.detach(), self.M.detach(), self.tau, self.external_time
        )


class LearnedEventEncoder(nn.Module):
    def __init__(self, config: LearnedModelConfig, symbol_embedding: nn.Embedding):
        super().__init__()
        self.config = config
        self.symbol_embedding = symbol_embedding
        self.kind_embedding = nn.Embedding(6, config.event_type_dim)
        width = config.event_type_dim + 3 * config.value_dim + 4
        self.network = nn.Sequential(
            nn.Linear(width, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

    def _symbol(self, ids: torch.Tensor) -> torch.Tensor:
        mask = ids.ge(0).unsqueeze(-1)
        embedded = self.symbol_embedding(ids.clamp_min(0))
        return torch.where(mask, embedded, torch.zeros_like(embedded))

    def forward(self, event: StructuredEvent) -> torch.Tensor:
        event.validate()
        pieces = (
            self.kind_embedding(event.kind),
            self._symbol(event.key_id),
            self._symbol(event.value_id),
            self._symbol(event.aux_id),
            event.scalars.to(dtype=self.symbol_embedding.weight.dtype),
        )
        return self.network(torch.cat(pieces, dim=-1))


class LearnedETRCM(nn.Module):
    """Full learned model plus fixed-law ablations selected at construction."""

    def __init__(
        self,
        config: LearnedModelConfig,
        *,
        consolidation: str = "query",
        idle_updates: bool = True,
        random_query: bool = False,
        freeze_core: bool = False,
        equal_timescales: bool = False,
    ):
        super().__init__()
        if consolidation not in {"query", "uniform", "none", "nonconserving"}:
            raise ValueError(consolidation)
        self.config = config
        self.consolidation = consolidation
        self.idle_updates = idle_updates
        self.random_query = random_query
        self.freeze_core = freeze_core
        self.equal_timescales = equal_timescales

        self.symbol_embedding = nn.Embedding(config.symbol_count, config.value_dim)
        nn.init.normal_(self.symbol_embedding.weight, std=0.3)
        self.event_encoder = LearnedEventEncoder(config, self.symbol_embedding)
        self.initial_H = nn.Parameter(
            torch.randn(config.latent_slots, config.hidden_dim) * 0.02
        )
        self.event_to_slots = nn.Linear(
            config.hidden_dim, config.latent_slots * config.hidden_dim
        )
        self.norm = RMSNorm(config.hidden_dim)
        self.query_projection = nn.Linear(config.hidden_dim, config.key_dim, bias=False)
        self.core_in = nn.Linear(
            config.hidden_dim + config.value_dim + config.hidden_dim,
            2 * config.hidden_dim,
        )
        self.core_out = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        self.core_gate = nn.Linear(
            config.hidden_dim + config.value_dim + config.hidden_dim,
            config.hidden_dim,
        )
        self.access_head = nn.Linear(config.hidden_dim, 1)
        self.symbol_head = nn.Linear(config.hidden_dim, config.symbol_count)
        self.binary_head = nn.Linear(config.hidden_dim, 1)
        random = torch.randn(config.key_dim)
        self.register_buffer("random_query_direction", _normalize(random), persistent=True)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> LearnedState:
        parameter = self.initial_H
        device = device or parameter.device
        dtype = dtype or parameter.dtype
        H = self.initial_H.to(device=device, dtype=dtype).unsqueeze(0).expand(
            batch_size, -1, -1
        ).clone()
        memory = torch.zeros(
            batch_size,
            self.config.value_dim,
            self.config.key_dim,
            device=device,
            dtype=dtype,
        )
        return LearnedState(H, memory, memory.clone())

    def reset_active(self, state: LearnedState) -> LearnedState:
        H = self.initial_H.to(state.H).unsqueeze(0).expand(state.H.shape[0], -1, -1)
        return LearnedState(H.clone(), state.F, state.M, state.tau, state.external_time)

    def symbol_vectors(self, ids: torch.Tensor) -> torch.Tensor:
        return _normalize(self.symbol_embedding(ids), dim=-1)

    def target_key(self, ids: torch.Tensor) -> torch.Tensor:
        value = self.symbol_vectors(ids)
        if value.shape[-1] != self.config.key_dim:
            raise ValueError("Stage-1.1 v1 requires key_dim=value_dim")
        return value

    def _pool(self, H: torch.Tensor) -> torch.Tensor:
        return self.norm(H).mean(dim=1)

    def _query(self, H: torch.Tensor) -> torch.Tensor:
        if self.random_query:
            return self.random_query_direction.to(H).expand(H.shape[0], -1)
        return _normalize(self.query_projection(self._pool(H)), dim=-1)

    def _external_write(
        self, F: torch.Tensor, M: torch.Tensor, event: StructuredEvent
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = self.target_key(event.key_id.clamp_min(0))
        value = self.symbol_vectors(event.value_id.clamp_min(0))
        total_read = torch.einsum("bvk,bk->bv", F + M, key)
        update = self.config.eta_external * torch.einsum(
            "bv,bk->bvk", value - total_read, key
        )
        mask = event.write_mask[:, None, None]
        update = torch.where(mask, update, torch.zeros_like(update))
        return F + update, M, update

    def _consolidate(
        self, F: torch.Tensor, M: torch.Tensor, q: torch.Tensor, access: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.consolidation == "none":
            delta = torch.zeros_like(F)
            return F, M, delta
        if self.consolidation == "uniform":
            delta = (self.config.gamma / self.config.key_dim) * access[:, None, None] * F
        else:
            fast_read = torch.einsum("bvk,bk->bv", F, q)
            delta = self.config.gamma * access[:, None, None] * torch.einsum(
                "bv,bk->bvk", fast_read, q
            )
        if self.consolidation == "nonconserving":
            return F, M + delta, delta
        return F - delta, M + delta, delta

    def step(
        self, state: LearnedState, event: StructuredEvent | None
    ) -> tuple[LearnedState, dict[str, torch.Tensor | float | bool]]:
        state.validate()
        batch = state.H.shape[0]
        if event is None and not self.idle_updates:
            zeros_q = self._query(state.H)
            logits = self.predict(state)
            return state, {
                **logits,
                "query": zeros_q,
                "read": torch.zeros(batch, self.config.value_dim, device=state.H.device),
                "access": torch.zeros(batch, device=state.H.device),
                "transfer": torch.zeros_like(state.F),
                "external_update": torch.zeros_like(state.F),
                "conservation_error": torch.zeros(batch, device=state.H.device),
                "event_written": False,
            }

        F, M = state.F, state.M
        external_update = torch.zeros_like(F)
        if event is None:
            encoded = torch.zeros(
                batch, self.config.hidden_dim, device=state.H.device, dtype=state.H.dtype
            )
            event_written = False
        else:
            event.validate(batch)
            encoded = self.event_encoder(event).to(dtype=state.H.dtype)
            F, M, external_update = self._external_write(F, M, event)
            event_written = bool(event.write_mask.any())

        event_slots = self.event_to_slots(encoded).view(
            batch, self.config.latent_slots, self.config.hidden_dim
        )
        H_pre = state.H + event_slots
        q = self._query(H_pre)
        read = torch.einsum("bvk,bk->bv", F + M, q)
        core_input = torch.cat(
            (
                self.norm(H_pre),
                read[:, None, :].expand(-1, self.config.latent_slots, -1),
                encoded[:, None, :].expand(-1, self.config.latent_slots, -1),
            ),
            dim=-1,
        )
        if self.freeze_core:
            H_new = H_pre
        else:
            proposal = self.core_out(Fnn.silu(self.core_in(core_input)))
            gate = torch.sigmoid(self.core_gate(core_input))
            H_new = H_pre + gate * proposal
        access = torch.sigmoid(self.access_head(self._pool(H_new))).squeeze(-1)

        total_before = F + M
        F, M, transfer = self._consolidate(F, M, q, access)
        conservation_error = torch.linalg.vector_norm(
            (F + M) - total_before, dim=(-2, -1)
        )
        rho_fast = self.config.rho_slow if self.equal_timescales else self.config.rho_fast
        F = rho_fast * F
        M = self.config.rho_slow * M
        new_state = LearnedState(
            H_new,
            F,
            M,
            state.tau + 1,
            state.external_time + int(event_written),
        )
        outputs = self.predict(new_state)
        outputs.update(
            {
                "query": q,
                "read": read,
                "access": access,
                "transfer": transfer,
                "external_update": external_update,
                "conservation_error": conservation_error,
                "event_written": event_written,
            }
        )
        return new_state, outputs

    def predict(self, state: LearnedState) -> dict[str, torch.Tensor]:
        pooled = self._pool(state.H)
        return {
            "symbol_logits": self.symbol_head(pooled),
            "binary_logit": self.binary_head(pooled).squeeze(-1),
        }

    def memory_lesion(self, state: LearnedState) -> LearnedState:
        return LearnedState(
            state.H,
            torch.zeros_like(state.F),
            torch.zeros_like(state.M),
            state.tau,
            state.external_time,
        )

    def persistent_state_bytes(self, batch_item: bool = True) -> int:
        elements = self.config.latent_slots * self.config.hidden_dim
        elements += 2 * self.config.value_dim * self.config.key_dim
        return elements * 4

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

