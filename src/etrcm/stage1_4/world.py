"""Frozen causal toy worlds. Future labels are separate from current events."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from etrcm.stage1_3.events import ContinuousEvent, Stage13EventKind


FAMILIES = ("latent_transition", "long_gap_relation", "latent_regime", "distractor_heavy")


@dataclass(frozen=True)
class WorldBatch:
    family: str
    events: tuple[ContinuousEvent, ...]
    targets: torch.Tensor  # [batch, external time], never supplied to the model
    target_key: torch.Tensor  # audit metadata; never passed to model.step
    exposure_count: torch.Tensor
    bridge_index: int
    metadata: dict[str, torch.Tensor]

    def future(self, external_index: int, horizon: int) -> torch.Tensor:
        index = external_index + horizon
        if index >= self.targets.shape[1]:
            raise IndexError("future target unavailable")
        return self.targets[:, index]


def _sample(generator: torch.Generator, count: int, batch: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, count, (batch,), generator=generator).to(device)


def _event(
    keys: torch.Tensor, values: torch.Tensor, *, write: bool = True,
    kind: Stage13EventKind = Stage13EventKind.EVIDENCE,
) -> ContinuousEvent:
    return ContinuousEvent.create(kind=kind, key_id=keys, value_id=values, write=write)


def generate_world(
    family: str, *, batch: int, length: int, seed: int,
    device: torch.device | str = "cpu", gap: int | None = None,
) -> WorldBatch:
    """Generate an independent batch with no query/answer field in its events."""
    if family not in FAMILIES:
        raise ValueError(family)
    if length < 10:
        raise ValueError("length must allow four horizons")
    device = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if family == "long_gap_relation" and gap is not None:
        length = gap + 6  # four early slots, gap noise, bridge, target
    keys = torch.zeros(batch, length, dtype=torch.long, device=device)
    values = torch.zeros_like(keys)
    tokens = torch.zeros_like(keys)
    write = torch.ones(batch, length, dtype=torch.bool, device=device)
    target_key = torch.full((batch,), -1, dtype=torch.long, device=device)
    exposures = torch.zeros(batch, dtype=torch.long, device=device)
    bridge = length - 2
    metadata: dict[str, torch.Tensor] = {}

    if family == "latent_transition":
        latent = _sample(generator, 8, batch, device)
        for t in range(length):
            if t:
                step = (torch.rand(batch, generator=generator).to(device) < 0.82).long()
                latent = (latent + step + 1).remainder(8)
            observation = latent.clone()
            noisy = torch.rand(batch, generator=generator).to(device) < 0.20
            observation = torch.where(noisy, _sample(generator, 8, batch, device), observation)
            keys[:, t], values[:, t], tokens[:, t] = latent, observation, observation
        target_key = latent
        exposures[:] = 1
        bridge = max(0, length - 9)
    elif family == "long_gap_relation":
        a = _sample(generator, 8, batch, device) + 8
        b = _sample(generator, 8, batch, device)
        c = _sample(generator, 8, batch, device)
        target_key = b
        exposure_choices = torch.tensor([1, 2, 4], device=device)
        exposures = exposure_choices[_sample(generator, 3, batch, device)]
        for t in range(length):
            keys[:, t] = _sample(generator, 16, batch, device)
            values[:, t] = _sample(generator, 24, batch, device)
            tokens[:, t] = 0  # uninformative distractor observations
        for t in range(4):
            mask = exposures > t
            keys[:, t] = torch.where(mask, b, keys[:, t])
            values[:, t] = torch.where(mask, a, values[:, t])
            tokens[:, t] = torch.where(mask, a, tokens[:, t])
        # C→B is a genuine bridge observation. It does not overwrite B→A.
        keys[:, bridge], values[:, bridge], tokens[:, bridge] = c, b, b
        keys[:, -1], values[:, -1], tokens[:, -1] = c, a, a
        write[:, -1] = False  # answer event is never seen when it is predicted
        metadata = {"A": a, "B": b, "C": c}
    elif family == "latent_regime":
        regime = _sample(generator, 4, batch, device)
        target_key = regime
        exposures[:] = 1
        for t in range(length):
            keys[:, t] = _sample(generator, 16, batch, device)
            token = 8 + 2 * regime + _sample(generator, 2, batch, device)
            values[:, t] = token
            tokens[:, t] = token
            write[:, t] = t == 0
        keys[:, 0], values[:, 0], tokens[:, 0] = regime, regime, regime
    else:
        cue = _sample(generator, 4, batch, device)
        target_key = cue
        exposures[:] = 1
        for t in range(length):
            keys[:, t] = _sample(generator, 16, batch, device)
            values[:, t] = _sample(generator, 24, batch, device)
            tokens[:, t] = 0
        keys[:, 0], values[:, 0], tokens[:, 0] = cue, cue + 16, cue + 16
        values[:, -1], tokens[:, -1] = cue + 16, cue + 16
        write[:, -1] = False
        bridge = length - 2
        metadata = {"cue": cue}

    events = tuple(
        ContinuousEvent.create(
            kind=Stage13EventKind.EVIDENCE,
            key_id=keys[:, t], value_id=values[:, t], write=write[:, t]
        )
        for t in range(length)
    )
    return WorldBatch(family, events, tokens, target_key, exposures, bridge, metadata)
