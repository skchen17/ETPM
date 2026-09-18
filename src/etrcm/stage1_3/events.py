"""Current-event-only contract with strict external/self-output separation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum

import torch


class Stage13EventKind(IntEnum):
    EVIDENCE = 0
    CONTEXT = 1
    NOISE = 2
    SELF_OUTPUT = 3


@dataclass(frozen=True)
class ContinuousEvent:
    """One event only; no history, answer label, solved flag, or halt field."""

    kind: torch.Tensor
    key_id: torch.Tensor
    value_id: torch.Tensor
    aux_id: torch.Tensor
    write_mask: torch.Tensor
    self_output_mask: torch.Tensor
    scalars: torch.Tensor

    ALLOWED_FIELDS = frozenset(
        {"kind", "key_id", "value_id", "aux_id", "write_mask", "self_output_mask", "scalars"}
    )

    def validate(self, batch_size: int | None = None) -> None:
        if {item.name for item in fields(self)} != self.ALLOWED_FIELDS:
            raise ValueError("Stage-1.3 event schema changed")
        batch = int(self.kind.shape[0])
        if batch_size is not None and batch != batch_size:
            raise ValueError("event/state batch mismatch")
        for name in ("key_id", "value_id", "aux_id", "write_mask", "self_output_mask"):
            if getattr(self, name).shape != (batch,):
                raise ValueError(f"{name} must be [batch]")
        if self.scalars.shape != (batch, 4):
            raise ValueError("scalars must be [batch,4]")
        if self.write_mask.dtype != torch.bool or self.self_output_mask.dtype != torch.bool:
            raise ValueError("masks must be bool")
        if bool((self.write_mask & self.self_output_mask).any()):
            raise ValueError("SELF_OUTPUT can never request an external write")
        legal_write = self.kind.eq(int(Stage13EventKind.EVIDENCE)) | self.kind.eq(
            int(Stage13EventKind.NOISE)
        )
        if bool((self.write_mask & ~legal_write).any()):
            raise ValueError("only external evidence/noise can write")
        if bool((self.self_output_mask & ~self.kind.eq(int(Stage13EventKind.SELF_OUTPUT))).any()):
            raise ValueError("self_output_mask requires SELF_OUTPUT kind")

    @classmethod
    def create(
        cls,
        *,
        kind: int | Stage13EventKind | torch.Tensor,
        key_id: torch.Tensor,
        value_id: torch.Tensor | None = None,
        aux_id: torch.Tensor | None = None,
        write: bool | torch.Tensor = False,
        self_output: bool | torch.Tensor = False,
        scalars: torch.Tensor | None = None,
    ) -> "ContinuousEvent":
        batch, device = int(key_id.shape[0]), key_id.device
        if value_id is None:
            value_id = torch.full((batch,), -1, dtype=torch.long, device=device)
        if aux_id is None:
            aux_id = torch.full((batch,), -1, dtype=torch.long, device=device)
        if isinstance(kind, torch.Tensor):
            kinds = kind.to(device=device, dtype=torch.long)
        else:
            kinds = torch.full((batch,), int(kind), dtype=torch.long, device=device)
        write_mask = (
            torch.full((batch,), write, dtype=torch.bool, device=device)
            if isinstance(write, bool)
            else write.to(device=device, dtype=torch.bool)
        )
        self_mask = (
            torch.full((batch,), self_output, dtype=torch.bool, device=device)
            if isinstance(self_output, bool)
            else self_output.to(device=device, dtype=torch.bool)
        )
        if scalars is None:
            scalars = torch.zeros(batch, 4, device=device)
        result = cls(
            kinds,
            key_id.to(dtype=torch.long),
            value_id.to(dtype=torch.long),
            aux_id.to(dtype=torch.long),
            write_mask,
            self_mask,
            scalars,
        )
        result.validate(batch)
        return result


def evidence_event(keys: torch.Tensor, values: torch.Tensor, support: float = 1.0) -> ContinuousEvent:
    scalars = torch.zeros(keys.shape[0], 4, device=keys.device)
    scalars[:, 0] = support
    return ContinuousEvent.create(
        kind=Stage13EventKind.EVIDENCE,
        key_id=keys,
        value_id=values,
        write=True,
        scalars=scalars,
    )


def noise_event(keys: torch.Tensor, values: torch.Tensor) -> ContinuousEvent:
    return ContinuousEvent.create(
        kind=Stage13EventKind.NOISE, key_id=keys, value_id=values, write=True
    )


def context_event(keys: torch.Tensor, aux_id: torch.Tensor | None = None) -> ContinuousEvent:
    return ContinuousEvent.create(
        kind=Stage13EventKind.CONTEXT, key_id=keys, aux_id=aux_id, write=False
    )


def self_output_event(content_ids: torch.Tensor) -> ContinuousEvent:
    empty = torch.full_like(content_ids, -1)
    return ContinuousEvent.create(
        kind=Stage13EventKind.SELF_OUTPUT,
        key_id=empty,
        value_id=content_ids,
        write=False,
        self_output=True,
    )
