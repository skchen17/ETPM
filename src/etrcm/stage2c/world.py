"""Balanced predictive-consequence world with latent, unobserved rule z."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from etrcm.stage1_3.events import context_event, evidence_event, noise_event


ABSTRACT = 1
ACTION = (2, 3)
OUTCOME = (4, 5, 6, 7)
COLORS = (8, 9, 10, 11, 12, 13)
SHAPES = (14, 15, 16, 17, 18, 19)
NUISANCE = (20, 21, 22, 23)
SYMBOL_COUNT = 24


@dataclass(frozen=True)
class Experience:
    color: int
    shape: int
    nuisance: int
    action: int
    outcome: int


def surface(rng: random.Random, split: str) -> tuple[int, int, int]:
    """Primary OOD holds out combinations, never a single trained token."""
    if split not in {"train", "seen", "novel", "hard_ood"}:
        raise ValueError(split)
    if split == "hard_ood":
        color, shape = rng.choice(COLORS[4:]), rng.choice(SHAPES[4:])
    else:
        parity = 1 if split == "novel" else 0
        color, shape = rng.choice([(c, s) for i, c in enumerate(COLORS[:4])
                                   for j, s in enumerate(SHAPES[:4]) if (i + j) % 2 == parity])
    return color, shape, rng.choice(NUISANCE)


def consequence(rng: random.Random, latent: int, action: int, *, noise: bool = False) -> int:
    if latent not in (0, 1) or action not in (0, 1):
        raise ValueError((latent, action))
    # Development amendment: non-reliable behavior has a three-symbol,
    # higher-entropy future distinct from the reliable action's consequence.
    # Neither a reward nor the correct action is ever emitted to the model.
    if noise:
        return rng.choice(OUTCOME)
    return OUTCOME[0] if action == latent else rng.choice(OUTCOME[1:])


def experience(rng: random.Random, latent: int, *, split: str = "train",
               action: int | None = None, noise: bool = False) -> Experience:
    c, s, n = surface(rng, split)
    a = rng.randrange(2) if action is None else action
    return Experience(c, s, n, a, consequence(rng, latent, a, noise=noise))


def paired_histories(seed: int, n: int, *, split: str = "train", noise: bool = False,
                     start_index: int = 0) -> tuple[list[Experience], list[Experience]]:
    """Same surfaces/actions/clock for A/B; only observed consequences differ."""
    surfaces = random.Random(seed * 997 + 71)
    outcomes_a = random.Random(seed * 997 + 73)
    outcomes_b = random.Random(seed * 997 + 79)
    # Every power-of-two prefix from N=2 upward is action balanced, while
    # within-block order is random. This avoids an exposure-curve confound.
    actions = []
    total = start_index + n
    while len(actions) < total:
        block_size = min(max(2, len(actions)), total-len(actions))
        block = [0, 1] * (block_size // 2)
        if block_size % 2:
            block.append(surfaces.randrange(2))
        surfaces.shuffle(block)
        actions.extend(block)
    shared = [(surface(surfaces, split), actions[i]) for i in range(start_index + n)]
    histories = ([], [])
    for features, action in shared[start_index:]:
        c, s, nuisance = features
        histories[0].append(Experience(c, s, nuisance, action,
                                       consequence(outcomes_a, 0, action, noise=noise)))
        histories[1].append(Experience(c, s, nuisance, action,
                                       consequence(outcomes_b, 1, action, noise=noise)))
    return histories


def tensor_ids(values: list[int], device: torch.device | str) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long, device=device)


def context_token(ids: torch.Tensor):
    return context_event(ids)


def outcome_token(ids: torch.Tensor, *, write: bool = True):
    # The unchanged memory law receives raw observed outcome identity in both
    # key and value; no entity, importance or correct-action key is supplied.
    event = evidence_event(ids, ids)
    if not write:
        from dataclasses import replace
        event = replace(event, write_mask=torch.zeros_like(event.write_mask))
    return event


def unrelated_token(ids: torch.Tensor):
    return noise_event(ids, ids)
