"""Unified ET-RCM step API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from etrcm.dynamics import MinimalLatentDynamics
from etrcm.memory import (
    decay_memory,
    external_delta_write,
    memory_read,
    normalize,
    readout_conserving_consolidation,
)
from etrcm.state import ETState


@dataclass(frozen=True)
class ETRCMConfig:
    hidden_dim: int = 64
    latent_slots: int = 1
    event_dim: int = 64
    key_dim: int = 32
    value_dim: int = 32
    gamma: float = 0.1
    rho_fast: float = 0.95
    rho_slow: float = 0.9995
    eta_external: float = 0.5

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "ETRCMConfig":
        model = config.get("model", {})
        memory = config.get("memory", {})
        return cls(
            hidden_dim=int(model.get("hidden_dim", 64)),
            latent_slots=int(model.get("latent_slots", 1)),
            event_dim=int(model.get("event_dim", model.get("hidden_dim", 64))),
            key_dim=int(memory.get("key_dim", 32)),
            value_dim=int(memory.get("value_dim", 32)),
            gamma=float(memory.get("gamma", 0.1)),
            rho_fast=float(memory.get("rho_fast", 0.95)),
            rho_slow=float(memory.get("rho_slow", 0.9995)),
            eta_external=float(memory.get("eta_external", 0.5)),
        )


def is_null_event(event: torch.Tensor | None) -> bool:
    return event is None or event.numel() == 0 or bool(torch.count_nonzero(event) == 0)


class ETRCM(nn.Module):
    def __init__(self, config: ETRCMConfig | None = None):
        super().__init__()
        self.config = config or ETRCMConfig()
        cfg = self.config
        self.key_encoder = nn.Linear(cfg.event_dim, cfg.key_dim, bias=False)
        self.value_encoder = nn.Linear(cfg.event_dim, cfg.value_dim, bias=False)
        self.dynamics = MinimalLatentDynamics(
            cfg.hidden_dim, cfg.key_dim, cfg.value_dim, cfg.event_dim
        )

    def initial_state(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ETState:
        parameter = next(self.parameters())
        return ETState.zeros(
            latent_slots=self.config.latent_slots,
            hidden_dim=self.config.hidden_dim,
            value_dim=self.config.value_dim,
            key_dim=self.config.key_dim,
            device=device or parameter.device,
            dtype=dtype or parameter.dtype,
        )

    def step(
        self,
        state: ETState | dict[str, Any],
        event: torch.Tensor | None,
    ) -> tuple[ETState, dict[str, Any]]:
        state_was_dict = isinstance(state, dict)
        current = ETState.from_dict(state) if state_was_dict else state
        current.validate()
        cfg = self.config
        F, M = current.F, current.M
        event_written = not is_null_event(event)
        external_update = torch.zeros_like(F)
        encoded_event: torch.Tensor | None = None

        if event_written:
            if event is None or event.shape != (cfg.event_dim,):
                raise ValueError(f"event must have shape ({cfg.event_dim},)")
            encoded_event = event.to(device=current.H.device, dtype=current.H.dtype)
            key = normalize(self.key_encoder(encoded_event))
            value = self.value_encoder(encoded_event)
            F, M, external_update = external_delta_write(
                F, M, key, value, cfg.eta_external
            )

        query = self.dynamics.query(current.H)
        read = memory_read(F, M, query)
        H_next = self.dynamics(current.H, read, encoded_event)

        total_before = F + M
        F, M, delta = readout_conserving_consolidation(
            F, M, query, cfg.gamma
        )
        conservation_error = torch.linalg.vector_norm((F + M) - total_before)
        F, M = decay_memory(F, M, cfg.rho_fast, cfg.rho_slow)

        new_state = ETState(
            H=H_next,
            F=F,
            M=M,
            tau=current.tau + 1,
            external_time=current.external_time + int(event_written),
        )
        output: dict[str, Any] = {
            "read": read,
            "query": query,
            "consolidation_delta": delta,
            "external_update": external_update,
            "event_written": event_written,
            "conservation_error": conservation_error,
        }
        return (new_state.as_dict() if state_was_dict else new_state), output

