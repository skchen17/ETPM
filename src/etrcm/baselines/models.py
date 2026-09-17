"""Minimal, explicit baselines; no hidden memory-engineering heuristics."""

from __future__ import annotations

import torch
from torch import nn

from etrcm.memory import (
    decay_memory,
    external_delta_write,
    nonconserving_replay,
    normalize,
    uniform_consolidation,
)
from etrcm.model import ETRCM


class NoMemoryMLP(nn.Module):
    """B0: output depends only on the current event."""

    def __init__(self, event_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(event_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, event: torch.Tensor) -> torch.Tensor:
        return self.network(event)


class GRUBaseline(nn.Module):
    """B1: ordinary recurrent state with no explicit fast/slow split."""

    def __init__(self, event_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.cell = nn.GRUCell(event_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, output_dim)

    def step(
        self, hidden: torch.Tensor, event: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.cell(event, hidden)
        return hidden, self.readout(hidden)


class SinglePersistentMatrix:
    """B2: one DeltaNet-like associative matrix A."""

    def __init__(self, value_dim: int, key_dim: int, eta: float, rho: float):
        self.A = torch.zeros(value_dim, key_dim, dtype=torch.float64)
        self.eta = eta
        self.rho = rho

    def write(self, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        key = normalize(key)
        update = self.eta * torch.outer(value - self.A @ key, key)
        self.A = self.A + update
        return update

    def tick(self) -> None:
        self.A = self.rho * self.A

    def read(self, query: torch.Tensor) -> torch.Tensor:
        return self.A @ normalize(query)


class FastSlowUniformTransfer:
    """B3: conserving F->M transfer independent of the query."""

    def __init__(self, value_dim: int, key_dim: int):
        self.F = torch.zeros(value_dim, key_dim, dtype=torch.float64)
        self.M = torch.zeros_like(self.F)

    def write(self, key: torch.Tensor, value: torch.Tensor, eta: float) -> None:
        self.F, self.M, _ = external_delta_write(self.F, self.M, key, value, eta)

    def tick(self, gamma: float, rho_fast: float, rho_slow: float) -> torch.Tensor:
        self.F, self.M, delta = uniform_consolidation(self.F, self.M, gamma)
        self.F, self.M = decay_memory(self.F, self.M, rho_fast, rho_slow)
        return delta


class NonConservingReplay:
    """B4: internal access adds a new copy to M (negative control)."""

    def __init__(self, value_dim: int, key_dim: int):
        self.F = torch.zeros(value_dim, key_dim, dtype=torch.float64)
        self.M = torch.zeros_like(self.F)

    def write(self, key: torch.Tensor, value: torch.Tensor, eta: float) -> None:
        self.F, self.M, _ = external_delta_write(self.F, self.M, key, value, eta)

    def tick(self, query: torch.Tensor, gamma: float) -> torch.Tensor:
        self.F, self.M, delta = nonconserving_replay(
            self.F, self.M, query, gamma
        )
        return delta


class NoIdleETRCM(ETRCM):
    """B5: NULL_EVENT leaves state unchanged."""

    def step(self, state, event):  # type: ignore[override]
        if event is None or bool(torch.count_nonzero(event) == 0):
            return state, {"event_written": False, "idle_skipped": True}
        return super().step(state, event)


BASELINE_REGISTRY = {
    "B0_no_memory_mlp": NoMemoryMLP,
    "B1_gru": GRUBaseline,
    "B2_single_persistent_matrix": SinglePersistentMatrix,
    "B3_uniform_transfer": FastSlowUniformTransfer,
    "B4_nonconserving_replay": NonConservingReplay,
    "B5_no_idle": NoIdleETRCM,
    "B6_full_etrcm": ETRCM,
}

