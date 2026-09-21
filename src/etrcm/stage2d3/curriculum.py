"""Temporary paired-action observational scaffold for conditional binding.

The simulator uses its hidden state only to sample two legal consequences.
Neither latent z, a correct-action label, nor any memory target crosses the
model boundary.  The auxiliary term is exactly zero after the warmup window.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from etrcm.stage2c.world import ACTION, consequence, context_token
from etrcm.stage2d.model import Stage2DModel, context_state
from etrcm.stage2d.world import NoisyExperience


@dataclass(frozen=True)
class PairedWarmup:
    steps: int
    arm: str = "C1"

    def __post_init__(self):
        if self.steps not in {50, 100, 200, 300}:
            raise ValueError(self.steps)
        if self.arm not in {"C1", "C2", "C3"}:
            raise ValueError(self.arm)

    def auxiliary_weight(self, step: int, *, evaluation: bool = False) -> float:
        return 0.0 if evaluation or step >= self.steps else 1.0


def paired_observational_targets(rows: list[NoisyExperience], seed: int,
                                 index: int) -> torch.Tensor:
    """Sample two legal branch outcomes; returned artifact contains no z."""
    targets = []
    for row_index, row in enumerate(rows):
        branch = []
        for action in (0, 1):
            rng = random.Random(seed * 1000003 + index * 7919 + row_index * 101 + action * 17)
            branch.append(consequence(rng, row.latent, action) - 4)
        targets.append(branch)
    return torch.tensor(targets, dtype=torch.long)


def paired_action_loss(model: Stage2DModel, state, rows: list[NoisyExperience],
                       targets: torch.Tensor, arm: str) -> torch.Tensor:
    """Evaluate matched action branches from the same endogenous state."""
    context = context_state(model, state.clone(), rows)
    targets = targets.to(context.H.device)
    losses = []
    for action in (0, 1):
        branch_action = action
        target_action = action
        if arm == "C2":
            target_action = 1 - action  # shuffled action assignment
        elif arm == "C3":
            branch_action = 0
            target_action = 0          # duplicated single-action compute
        ids = torch.full((context.H.shape[0],), ACTION[branch_action], dtype=torch.long,
                         device=context.H.device)
        branch, _ = model.step(context.clone(), context_token(ids))
        actions = torch.full((context.H.shape[0],), branch_action, dtype=torch.long,
                             device=context.H.device)
        losses.append(F.cross_entropy(model.logits(branch, actions), targets[:, target_action]))
    return .5 * (losses[0] + losses[1])
