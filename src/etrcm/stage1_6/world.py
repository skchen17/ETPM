"""Compositional memory-necessary world with a strict past/future boundary."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from etrcm.stage1_3.events import ContinuousEvent, Stage13EventKind


@dataclass(frozen=True)
class ComposedWorld:
    events: tuple[ContinuousEvent, ...]
    past_value: torch.Tensor
    key: torch.Tensor
    offset: torch.Tensor
    target: torch.Tensor
    early_exposures: int = 4
    distractors: int = 8

    @property
    def bridge_index(self) -> int:
        return self.early_exposures + self.distractors


def generate_world(*, batch: int, seed: int, device: str | torch.device = "cpu",
                   early_exposures: int = 4, distractors: int = 8) -> ComposedWorld:
    if batch < 2 or early_exposures < 1 or distractors < 1:
        raise ValueError("invalid world dimensions")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    key = torch.randint(0, 8, (batch,), generator=generator).to(device)
    past_value = (8 + torch.randint(0, 8, (batch,), generator=generator)).to(device)
    offset = (16 + torch.randint(0, 8, (batch,), generator=generator)).to(device)
    target = ((past_value - 8) + (offset - 16)).remainder(8)
    events: list[ContinuousEvent] = []
    for _ in range(early_exposures):
        events.append(ContinuousEvent.create(kind=Stage13EventKind.EVIDENCE,
                                             key_id=key, value_id=past_value, write=True))
    blank = torch.full_like(key, -1)
    for _ in range(distractors):
        events.append(ContinuousEvent.create(kind=Stage13EventKind.CONTEXT,
                                             key_id=blank, write=False))
    events.append(ContinuousEvent.create(kind=Stage13EventKind.CONTEXT,
                                         key_id=key, aux_id=offset, write=False))
    return ComposedWorld(tuple(events), past_value, key, offset, target,
                         early_exposures, distractors)


def audit_no_future_leak(world: ComposedWorld) -> bool:
    """Past bank is invariant to later offset/label when key and A are held fixed."""
    early = world.events[:world.early_exposures]
    return all(bool(e.write_mask.all()) and torch.equal(e.key_id, world.key)
               and torch.equal(e.value_id, world.past_value)
               and bool(e.aux_id.eq(-1).all()) for e in early)
