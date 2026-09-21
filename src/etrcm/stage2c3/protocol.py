"""Frozen Stage 2C.3 targets, controls, and arm schedules."""

from __future__ import annotations

import math

import torch

CHECKPOINTS = (0, 25, 50, 100, 200, 300, 500, 1000)
MARGINAL = (0.5, 1 / 6, 1 / 6, 1 / 6)
MARGINAL_CE = -sum(p * math.log(p) for p in MARGINAL)


def target_distribution(latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Simulator-only consequence target; neither tensor is a model input."""
    correct = latent.eq(action)
    target = torch.full((*correct.shape, 4), 1 / 3,
                        dtype=torch.float32, device=latent.device)
    target[..., 0] = 0
    return torch.where(correct[..., None],
                       torch.nn.functional.one_hot(torch.zeros_like(latent), 4).float(),
                       target)


def arm_phase(arm: str, step: int) -> str:
    if arm in {"A0", "A1"}:
        return "joint"
    if arm not in {"A2", "A3"}:
        raise ValueError(arm)
    if step < 500:
        return "frozen_evaluator"
    return "frozen_evaluator" if arm == "A2" else "gradual_unfreeze"


def evaluator_lr_ratio(arm: str, step: int) -> float:
    return 0.1 if arm_phase(arm, step) == "gradual_unfreeze" else 0.0


def gate_count_pass(rows: list[bool], minimum: int = 6) -> bool:
    return len(rows) == 8 and sum(rows) >= minimum
