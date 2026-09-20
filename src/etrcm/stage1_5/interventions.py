"""Finite, exact component and read interventions for Stage 1.5."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import functional as Fnn

from etrcm.stage1_1.model import LearnedState


CHANNELS = ("H", "F", "M", "HF", "HM", "FM", "HFM")


def swap_components(reference: LearnedState, donor: LearnedState, channels: str) -> LearnedState:
    """Take only named channels from donor, preserving reference clocks."""
    if channels not in CHANNELS:
        raise ValueError(f"unregistered swap {channels}")
    reference.validate()
    donor.validate()
    if reference.H.shape != donor.H.shape or reference.F.shape != donor.F.shape:
        raise ValueError("incompatible swap shapes")
    return LearnedState(
        donor.H.clone() if "H" in channels else reference.H.clone(),
        donor.F.clone() if "F" in channels else reference.F.clone(),
        donor.M.clone() if "M" in channels else reference.M.clone(),
        reference.tau, reference.external_time,
    )


def cross_entropy_per_item(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return Fnn.cross_entropy(logits, target, reduction="none")


def retrieval_advantage(
    no_read_logits: torch.Tensor, injected_logits: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Finite realized CE benefit, never an infinitesimal proxy."""
    return cross_entropy_per_item(no_read_logits, target) - cross_entropy_per_item(
        injected_logits, target
    )


def finite_read_intervention(
    rollout: Callable[[torch.Tensor | None], torch.Tensor],
    target: torch.Tensor, candidates: torch.Tensor,
) -> torch.Tensor:
    """Evaluate actual future loss after each historical candidate read.

    `rollout` accepts one read or None and returns future logits from the
    identical cloned pre-intervention state. `candidates` is [n,batch,value].
    """
    if candidates.ndim != 3:
        raise ValueError("candidate bank must be [item,batch,value]")
    baseline = rollout(None)
    return torch.stack(
        [retrieval_advantage(baseline, rollout(read), target) for read in candidates], 0
    )
