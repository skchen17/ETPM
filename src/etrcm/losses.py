"""Small Stage-1 losses used by trainable toy policies."""

from __future__ import annotations

import torch


def retrieval_mse(read: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((read - target) ** 2)


def calibration_brier(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((probability - target) ** 2)


def slow_state_cost(M: torch.Tensor) -> torch.Tensor:
    return torch.mean(M**2)

