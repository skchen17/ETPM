"""Minimal active-latent dynamics; intentionally not a Transformer."""

from __future__ import annotations

import torch
from torch import nn

from etrcm.memory import normalize


class MinimalLatentDynamics(nn.Module):
    def __init__(self, hidden_dim: int, key_dim: int, value_dim: int, event_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.event_dim = event_dim
        self.query_projection = nn.Linear(hidden_dim, key_dim, bias=False)
        joint_dim = hidden_dim + value_dim + event_dim
        self.proposal = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gate = nn.Linear(joint_dim, hidden_dim)

    def query(self, H: torch.Tensor) -> torch.Tensor:
        return normalize(self.query_projection(H.mean(dim=0)))

    def forward(
        self, H: torch.Tensor, read: torch.Tensor, event: torch.Tensor | None
    ) -> torch.Tensor:
        if event is None:
            event = torch.zeros(
                self.event_dim, device=H.device, dtype=H.dtype
            )
        joint = torch.cat((H.mean(dim=0), read, event), dim=0)
        update = torch.sigmoid(self.gate(joint)) * self.proposal(joint)
        return H + update.unsqueeze(0).expand_as(H)

