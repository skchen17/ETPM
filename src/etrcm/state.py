"""Explicit state containers for internal-time dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ETState:
    """Stage-1 state S=(H,F,M) with separate internal/external clocks."""

    H: torch.Tensor
    F: torch.Tensor
    M: torch.Tensor
    tau: int = 0
    external_time: int = 0

    @classmethod
    def zeros(
        cls,
        *,
        latent_slots: int,
        hidden_dim: int,
        value_dim: int,
        key_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "ETState":
        return cls(
            H=torch.zeros(latent_slots, hidden_dim, device=device, dtype=dtype),
            F=torch.zeros(value_dim, key_dim, device=device, dtype=dtype),
            M=torch.zeros(value_dim, key_dim, device=device, dtype=dtype),
        )

    def validate(self) -> None:
        if self.H.ndim != 2:
            raise ValueError(f"H must be rank-2, got {tuple(self.H.shape)}")
        if self.F.ndim != 2 or self.M.ndim != 2:
            raise ValueError("F and M must be rank-2 associative matrices")
        if self.F.shape != self.M.shape:
            raise ValueError(f"F/M shapes differ: {self.F.shape} vs {self.M.shape}")
        if not (self.H.device == self.F.device == self.M.device):
            raise ValueError("H, F, and M must share a device")
        if not (self.H.dtype == self.F.dtype == self.M.dtype):
            raise ValueError("H, F, and M must share a dtype")
        if self.tau < 0 or self.external_time < 0:
            raise ValueError("state clocks must be non-negative")

    def clone(self) -> "ETState":
        return ETState(
            H=self.H.clone(),
            F=self.F.clone(),
            M=self.M.clone(),
            tau=self.tau,
            external_time=self.external_time,
        )

    def detach(self) -> "ETState":
        return ETState(
            H=self.H.detach(),
            F=self.F.detach(),
            M=self.M.detach(),
            tau=self.tau,
            external_time=self.external_time,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "H": self.H,
            "F": self.F,
            "M": self.M,
            "tau": self.tau,
            "external_time": self.external_time,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ETState":
        state = cls(
            H=value["H"],
            F=value["F"],
            M=value["M"],
            tau=int(value.get("tau", 0)),
            external_time=int(value.get("external_time", 0)),
        )
        state.validate()
        return state

