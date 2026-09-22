"""Conditional Stage 2D.7 evaluator-compatibility training arms.

All arms inherit the exact frozen F/M/H transition law and observed-only
objective. The add-on modules are zero-effect at initialization.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from etrcm.stage2d.model import Stage2DModel


class LowRankResidual(nn.Module):
    def __init__(self, width=32, rank=4):
        super().__init__()
        self.left = nn.Linear(rank, width, bias=False)
        self.right = nn.Linear(width, rank, bias=False)
        nn.init.zeros_(self.left.weight)
        nn.init.normal_(self.right.weight, mean=0., std=0.01)

    def forward(self, x):
        return x + self.left(self.right(x))


class DenseResidual(nn.Module):
    """1056-parameter generic downstream control matching W_H plus bias."""

    def __init__(self, width=32):
        super().__init__()
        self.map = nn.Linear(width, width)
        nn.init.zeros_(self.map.weight)
        nn.init.zeros_(self.map.bias)

    def forward(self, x):
        return x + self.map(x)


class CompatibilityModel(Stage2DModel):
    def __init__(self, arm: str, *, rank: int = 4, gamma=.50,
                 rho_fast=.97, rho_slow=.9995):
        super().__init__(variant="full", gamma=gamma, rho_fast=rho_fast,
                         rho_slow=rho_slow)
        if arm not in {"C0", "C1", "C2", "C3", "C4", "C5"}:
            raise ValueError(arm)
        self.compat_arm = arm
        self.rank = rank
        self.coordinate_adapter = LowRankResidual(32, rank) if arm == "C3" else None
        self.generic_adapter = (DenseResidual(32) if rank == 0 else LowRankResidual(32, rank)) if arm == "C4" else None

    def logits(self, state, action):
        h = self.core._pool(state.H)
        if self.coordinate_adapter is not None:
            h = self.coordinate_adapter(h)
        head = self.action_head
        fusion = torch.cat([h, head.action_embedding(action)], dim=-1)
        pre = head.head[0](fusion)
        post = head.head[1](pre)
        if self.generic_adapter is not None:
            post = self.generic_adapter(post)
        return head.head[2](post)


def configure_training(model: CompatibilityModel, arm: str, *, lr_ratio=.1):
    """Return optimizer groups and protected parameter snapshot.

    Gradient hooks and zero weight decay prevent AdamW changing a frozen
    column inside the shared first-layer tensor.
    """
    for parameter in model.action_head.parameters():
        parameter.requires_grad_(False)
    first = model.action_head.head[0]
    upstream = list(model.core.parameters())
    groups = [{"params": upstream, "lr": .001, "weight_decay": .01}]
    if arm in {"C1", "C2", "C5"}:
        first.weight.requires_grad_(True)
        mask = torch.zeros_like(first.weight)
        if arm in {"C1", "C2"}:
            mask[:, :32] = 1.
            first.bias.requires_grad_(True)
            groups.append({"params": [first.weight, first.bias],
                           "lr": .001 * lr_ratio, "weight_decay": 0.})
        else:
            mask[:, 32:] = 1.
            groups.append({"params": [first.weight],
                           "lr": .001 * lr_ratio, "weight_decay": 0.})
        first.weight.register_hook(lambda grad: grad * mask)
    if arm == "C3":
        groups.append({"params": list(model.coordinate_adapter.parameters()),
                       "lr": .001, "weight_decay": .01})
    if arm == "C4":
        groups.append({"params": list(model.generic_adapter.parameters()),
                       "lr": .001, "weight_decay": .01})
    snapshot = {name: value.detach().clone()
                for name, value in model.action_head.state_dict().items()}
    return groups, snapshot


def protected_exact(model: CompatibilityModel, arm: str, snapshot: dict,
                    *, after_freeze=False):
    current = model.action_head.state_dict()
    for name, original in snapshot.items():
        value = current[name]
        if name == "head.0.weight":
            if arm in {"C1", "C2"} and not after_freeze:
                if not torch.equal(value[:, 32:], original[:, 32:]):
                    return False
                continue
            if arm == "C5":
                if not torch.equal(value[:, :32], original[:, :32]):
                    return False
                continue
        if name == "head.0.bias" and arm in {"C1", "C2"} and not after_freeze:
            continue
        if not torch.equal(value, original):
            return False
    return True
