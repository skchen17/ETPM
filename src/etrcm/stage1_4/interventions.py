"""Auditable peripheral, workspace, direction-lesion and safety interventions."""

from __future__ import annotations

import torch
from torch.nn import functional as Fnn

from etrcm.stage1_1.model import LearnedState, _normalize
from etrcm.stage1_3.events import ContinuousEvent
from .model import PredictiveETRCM


def peripheral_swap(
    reference: LearnedState, donor: LearnedState, *, channels: str = "FM"
) -> LearnedState:
    if reference.H.shape != donor.H.shape or channels not in {"F", "M", "FM"}:
        raise ValueError("incompatible same-H peripheral swap")
    return LearnedState(
        reference.H.clone(),
        donor.F.clone() if "F" in channels else reference.F.clone(),
        donor.M.clone() if "M" in channels else reference.M.clone(),
        reference.tau, reference.external_time,
    )


def restore_workspace(reference: LearnedState, manipulated: LearnedState) -> LearnedState:
    """Replace only H; peripheral tensors and clocks are unchanged."""
    if reference.H.shape != manipulated.H.shape:
        raise ValueError("workspace shapes differ")
    return LearnedState(reference.H.clone(), manipulated.F.clone(), manipulated.M.clone(),
                        manipulated.tau, manipulated.external_time)


def lesion_direction(state: LearnedState, key: torch.Tensor, *, channel: str = "M") -> LearnedState:
    """Remove one normalized rank-one key direction; leave other channel exact."""
    if channel not in {"F", "M"}:
        raise ValueError(channel)
    q = _normalize(key)
    source = state.F if channel == "F" else state.M
    projection = torch.einsum("bvk,bk->bv", source, q)
    removed = torch.einsum("bv,bk->bvk", projection, q)
    if channel == "F":
        return LearnedState(state.H.clone(), source - removed, state.M.clone(), state.tau, state.external_time)
    return LearnedState(state.H.clone(), state.F.clone(), source - removed, state.tau, state.external_time)


def prediction_js(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    p = first.softmax(-1).clamp_min(1e-9)
    q = second.softmax(-1).clamp_min(1e-9)
    midpoint = 0.5 * (p + q)
    return 0.5 * ((p * (p.log() - midpoint.log())).sum(-1)
                  + (q * (q.log() - midpoint.log())).sum(-1))


def causal_usage_loss(
    full_logits: torch.Tensor, lesion_logits: torch.Tensor, future_target: torch.Tensor
) -> torch.Tensor:
    """Positive means deleting the direction harmed future prediction."""
    return Fnn.cross_entropy(lesion_logits, future_target, reduction="none") - Fnn.cross_entropy(
        full_logits, future_target, reduction="none"
    )


def pathological_self_output_write(
    model: PredictiveETRCM, state: LearnedState, content_id: torch.Tensor,
    *, eta_self: float = 0.6,
) -> tuple[LearnedState, torch.Tensor]:
    """Deliberately unsafe positive control; never used in the primary model."""
    key = model.target_key(content_id.remainder(model.config.symbol_count))
    value = model.value_vector(content_id.remainder(model.config.symbol_count))
    current = torch.einsum("bvk,bk->bv", state.F + state.M, key)
    update = eta_self * torch.einsum("bv,bk->bvk", value - current, key)
    return LearnedState(state.H, state.F + update, state.M, state.tau, state.external_time), update


def output_is_not_evidence(
    model: PredictiveETRCM, state: LearnedState, event: ContinuousEvent
) -> tuple[LearnedState, dict[str, torch.Tensor]]:
    """Normal self-output takes the validated non-writing transition."""
    if not bool(event.self_output_mask.all()):
        raise ValueError("expected SELF_OUTPUT event")
    return model.step(state, event)
