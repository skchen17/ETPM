"""Stage-1.2 current-event-only task generators."""

from __future__ import annotations

import torch

from etrcm.stage1_1.events import EventKind, StructuredEvent


def goal_event(batch_size: int, operation: torch.Tensor, device: torch.device) -> StructuredEvent:
    """One goal cue with no key IDs, values, answer, or retrieval schedule."""

    missing = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    scalars = torch.zeros(batch_size, 4, device=device)
    scalars.scatter_(1, operation.long().clamp(0, 3)[:, None], 1.0)
    return StructuredEvent.create(
        kind=EventKind.TASK_CUE,
        key_id=missing,
        value_id=missing,
        aux_id=missing,
        write=False,
        scalars=scalars,
    )


def autonomous_labels(values: torch.Tensor, operation: torch.Tensor) -> torch.Tensor:
    """Balanced compositional binary tasks requiring two or three stored values."""

    a, b, c = [values[:, index].remainder(2).bool() for index in range(3)]
    labels = torch.empty_like(operation, dtype=torch.bool)
    labels[operation.eq(0)] = torch.logical_xor(a, b)[operation.eq(0)]
    labels[operation.eq(1)] = a.eq(b)[operation.eq(1)]
    selected = torch.where(c, a, b)
    labels[operation.eq(2)] = selected[operation.eq(2)]
    return labels.float()


def fixed_autonomous_keys(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.arange(3, device=device).unsqueeze(0).expand(batch_size, -1)


def path_key_at(sources: torch.Tensor, destinations: torch.Tensor, distance: torch.Tensor, tick: int) -> torch.Tensor:
    """Training-only auxiliary query target; never passed as an inference event."""

    index = torch.minimum(
        torch.full_like(distance, tick), distance.sub(1).clamp_min(0)
    )
    row = torch.arange(sources.shape[0], device=sources.device)
    return sources[row, index]

