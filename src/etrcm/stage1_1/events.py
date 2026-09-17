"""Current-event-only input contract for the Stage-1.1 models."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum

import torch


class EventKind(IntEnum):
    FACT = 0
    QUERY = 1
    GRAPH_QUERY = 2
    INTERRUPTION = 3
    UNKNOWABLE = 4
    TASK_CUE = 5


@dataclass(frozen=True)
class StructuredEvent:
    """One current external event; deliberately contains no history container."""

    kind: torch.Tensor
    key_id: torch.Tensor
    value_id: torch.Tensor
    aux_id: torch.Tensor
    write_mask: torch.Tensor
    scalars: torch.Tensor

    ALLOWED_FIELDS = frozenset(
        {"kind", "key_id", "value_id", "aux_id", "write_mask", "scalars"}
    )

    def validate(self, batch_size: int | None = None) -> None:
        actual = {field.name for field in fields(self)}
        if actual != self.ALLOWED_FIELDS:
            raise ValueError(f"event schema changed: {sorted(actual)}")
        size = int(self.kind.shape[0])
        if batch_size is not None and size != batch_size:
            raise ValueError(f"event batch {size} != state batch {batch_size}")
        for name in ("key_id", "value_id", "aux_id", "write_mask"):
            tensor = getattr(self, name)
            if tensor.shape != (size,):
                raise ValueError(f"{name} must have shape ({size},)")
        if self.scalars.shape != (size, 4):
            raise ValueError(f"scalars must have shape ({size},4)")
        if self.write_mask.dtype != torch.bool:
            raise ValueError("write_mask must be bool")

    @classmethod
    def create(
        cls,
        *,
        kind: int | EventKind,
        key_id: torch.Tensor,
        value_id: torch.Tensor | None = None,
        aux_id: torch.Tensor | None = None,
        write: bool | torch.Tensor = False,
        scalars: torch.Tensor | None = None,
    ) -> "StructuredEvent":
        batch = int(key_id.shape[0])
        device = key_id.device
        if value_id is None:
            value_id = torch.full((batch,), -1, device=device, dtype=torch.long)
        if aux_id is None:
            aux_id = torch.full((batch,), -1, device=device, dtype=torch.long)
        if isinstance(write, bool):
            write_mask = torch.full((batch,), write, device=device, dtype=torch.bool)
        else:
            write_mask = write.to(device=device, dtype=torch.bool)
        if scalars is None:
            scalars = torch.zeros(batch, 4, device=device)
        event = cls(
            kind=torch.full((batch,), int(kind), device=device, dtype=torch.long),
            key_id=key_id.to(dtype=torch.long),
            value_id=value_id.to(dtype=torch.long),
            aux_id=aux_id.to(dtype=torch.long),
            write_mask=write_mask,
            scalars=scalars,
        )
        event.validate(batch)
        return event

