"""Equal-budget future-world training, with no query/importance supervision."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as Fnn

from .model import PredictiveETRCM
from .world import FAMILIES, WorldBatch, generate_world


HORIZONS = (1, 2, 4, 8)
HORIZON_WEIGHTS = (0.4, 0.3, 0.2, 0.1)


@dataclass
class TrainingResult:
    logs: list[dict[str, float | int | str]]
    validation_loss: float
    validation_by_family: dict[str, float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prefix_index(world: WorldBatch, step: int) -> int:
    if world.family in {"long_gap_relation", "distractor_heavy"}:
        return world.bridge_index
    return min(world.targets.shape[1] - 9, 2 + (step % 6))


def run_prefix(
    model: PredictiveETRCM, world: WorldBatch, *, prefix: int, null_ticks: int,
) -> tuple[object, list[dict[str, torch.Tensor]]]:
    state = model.initial_state(world.targets.shape[0], device=world.targets.device)
    diagnostic: list[dict[str, torch.Tensor]] = []
    for index in range(prefix + 1):
        state, output = model.step(state, world.events[index])
        diagnostic.append(output)
    for _ in range(null_ticks):
        state, output = model.step(state, None)
        diagnostic.append(output)
    return state, diagnostic


def world_loss(
    model: PredictiveETRCM, world: WorldBatch, state: object, *, prefix: int,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    logits = model.predict_logits(state)
    weighted = torch.zeros((), device=world.targets.device)
    weight_sum = 0.0
    parts: dict[int, torch.Tensor] = {}
    for horizon, weight in zip(HORIZONS, HORIZON_WEIGHTS):
        if prefix + horizon >= world.targets.shape[1]:
            continue
        target = world.future(prefix, horizon)
        one = Fnn.cross_entropy(logits[horizon], target)
        weighted = weighted + weight * one
        weight_sum += weight
        parts[horizon] = one
    if not parts:
        raise ValueError("no shifted future target")
    return weighted / weight_sum, parts


def validate(
    model: PredictiveETRCM, *, seed: int, batch_size: int, length: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    with torch.no_grad():
        for family_id, family in enumerate(FAMILIES):
            world = generate_world(
                family, batch=batch_size, length=length, seed=seed + 500_000 + family_id,
                device=device,
            )
            prefix = prefix_index(world, 5)
            state, _ = run_prefix(model, world, prefix=prefix, null_ticks=2)
            loss, _ = world_loss(model, world, state, prefix=prefix)
            losses[family] = float(loss)
    return losses


def train_model(
    model: PredictiveETRCM, *, steps: int, batch_size: int, learning_rate: float,
    seed: int, device: torch.device, length: int = 16, weight_decay: float = 0.0001,
    gradient_clip: float = 1.0,
) -> TrainingResult:
    seed_everything(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logs: list[dict[str, float | int | str]] = []
    null_choices = (0, 1, 2, 4)
    for step in range(steps):
        family = FAMILIES[step % len(FAMILIES)]
        world = generate_world(
            family, batch=batch_size, length=length,
            seed=seed * 1_000_000 + step * 17 + 41, device=device,
        )
        prefix = prefix_index(world, step)
        null_ticks = null_choices[(step // len(FAMILIES)) % len(null_choices)]
        optimizer.zero_grad(set_to_none=True)
        state, diagnostics = run_prefix(model, world, prefix=prefix, null_ticks=null_ticks)
        loss, parts = world_loss(model, world, state, prefix=prefix)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at training step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        if step % 10 == 0 or step == steps - 1:
            logs.append({
                "step": step, "family": family, "loss": float(loss.detach()),
                "null_ticks": null_ticks, "external_steps": prefix + 1,
                "transitions": len(diagnostics),
                **{f"h{h}_loss": float(part.detach()) for h, part in parts.items()},
            })
    by_family = validate(model, seed=seed, batch_size=batch_size, length=length, device=device)
    return TrainingResult(logs, float(np.mean(list(by_family.values()))), by_family)
