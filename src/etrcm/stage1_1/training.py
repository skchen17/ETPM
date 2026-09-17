"""Full-BPTT training utilities for frozen Stage-1.1 synthetic tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as Fnn

from etrcm.stage1_1.data import (
    fact_event,
    graph_edge_event,
    make_generator,
    query_event,
    sample_facts,
    sample_graphs,
    unknowable_event,
)


@dataclass(frozen=True)
class TrainSpec:
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    access_regularization: float
    symbol_count: int


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _grad_norm(model: nn.Module) -> float:
    squared = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().pow(2).sum()
    return float(torch.sqrt(squared))


def _query_with_ticks(model, state, keys, ticks: int, target=None):
    event = query_event(keys, target=target)
    state, output = model.step(state, event)
    trajectory = [output]
    for _ in range(ticks):
        state, output = model.step(state, None)
        trajectory.append(output)
    return state, output, trajectory


def train_memory_model(
    model: nn.Module,
    spec: TrainSpec,
    *,
    seed: int,
    device: torch.device,
    log_every: int = 50,
) -> list[dict[str, Any]]:
    """Train autonomous query/reuse under delayed interference with full BPTT."""

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    rng = make_generator(seed)
    records: list[dict[str, Any]] = []
    for step in range(1, spec.steps + 1):
        facts = sample_facts(
            spec.batch_size, 8, spec.symbol_count, rng, device
        )
        state = model.initial_state(spec.batch_size, device=device)
        fact_access = []
        for index in range(8):
            state, output = model.step(
                state, fact_event(facts.keys[:, index], facts.values[:, index])
            )
            fact_access.append(output["access"].mean())

        target_index = torch.randint(0, 8, (1,), generator=rng).item()
        keys = facts.keys[:, target_index]
        targets = facts.values[:, target_index]
        reuse_count = 1 + int(torch.randint(0, 4, (1,), generator=rng).item())
        losses = []
        query_access = []
        for _ in range(reuse_count):
            state = model.reset_active(state)
            state, output, trajectory = _query_with_ticks(model, state, keys, 1)
            losses.append(Fnn.cross_entropy(output["symbol_logits"], targets))
            query_access.extend(item["access"].mean() for item in trajectory)

        distractors = sample_facts(
            spec.batch_size, 16, spec.symbol_count, rng, device
        )
        for index in range(16):
            state, output = model.step(
                state,
                fact_event(distractors.keys[:, index], distractors.values[:, index]),
            )
            fact_access.append(output["access"].mean())

        # Active-state scrub is a preregistered memory-dependence constraint:
        # the answer must be reconstructed from F/M and the current query cue.
        state = model.reset_active(state)
        state, output, trajectory = _query_with_ticks(model, state, keys, 2)
        final_loss = Fnn.cross_entropy(output["symbol_logits"], targets)
        losses.append(2.0 * final_loss)
        query_access.extend(item["access"].mean() for item in trajectory)
        access_penalty = torch.stack(fact_access).mean() if fact_access else 0.0
        loss = torch.stack(losses).mean() + spec.access_regularization * access_penalty

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        preclip = _grad_norm(model)
        clipped = float(torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip))
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == spec.steps:
            accuracy = float((output["symbol_logits"].argmax(-1) == targets).float().mean())
            records.append(
                {
                    "phase": "train_memory",
                    "step": step,
                    "loss": float(loss.detach()),
                    "accuracy": accuracy,
                    "grad_norm_preclip": preclip,
                    "grad_norm_reported": clipped,
                    "learning_rate": spec.learning_rate,
                    "reuse_count": reuse_count,
                    "fact_access": float(torch.stack(fact_access).mean().detach()),
                    "query_access": float(torch.stack(query_access).mean().detach()),
                    "parameters": _parameter_count(model),
                }
            )
    return records


@torch.no_grad()
def evaluate_memory_validation(
    model: nn.Module,
    *,
    seed: int,
    batch_size: int,
    symbol_count: int,
    device: torch.device,
) -> float:
    model.eval()
    rng = make_generator(seed)
    facts = sample_facts(batch_size, 8, symbol_count, rng, device)
    state = model.initial_state(batch_size, device=device)
    for index in range(8):
        state, _ = model.step(
            state, fact_event(facts.keys[:, index], facts.values[:, index])
        )
    target_index = 3
    keys, targets = facts.keys[:, target_index], facts.values[:, target_index]
    for _ in range(3):
        state = model.reset_active(state)
        state, _, _ = _query_with_ticks(model, state, keys, 1)
    distractors = sample_facts(batch_size, 32, symbol_count, rng, device)
    for index in range(32):
        state, _ = model.step(
            state, fact_event(distractors.keys[:, index], distractors.values[:, index])
        )
    state = model.reset_active(state)
    state, output, _ = _query_with_ticks(model, state, keys, 2)
    return float((output["symbol_logits"].argmax(-1) == targets).float().mean())


def train_graph_model(
    model: nn.Module,
    spec: TrainSpec,
    *,
    seed: int,
    device: torch.device,
    log_every: int = 50,
) -> list[dict[str, Any]]:
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    rng = make_generator(seed)
    records: list[dict[str, Any]] = []
    for step in range(1, spec.steps + 1):
        distance = 1 + int(torch.randint(0, 6, (1,), generator=rng).item())
        graph = sample_graphs(
            spec.batch_size,
            distance,
            distance,
            spec.symbol_count,
            rng,
            device,
        )
        state = model.initial_state(spec.batch_size, device=device)
        fact_access = []
        for edge in range(distance):
            state, output = model.step(state, graph_edge_event(graph, edge))
            fact_access.append(output["access"].mean())
        state = model.reset_active(state)
        state, output, trajectory = _query_with_ticks(
            model, state, graph.start, distance, target=graph.target
        )
        loss = Fnn.binary_cross_entropy_with_logits(output["binary_logit"], graph.label)
        loss = loss + spec.access_regularization * torch.stack(fact_access).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        preclip = _grad_norm(model)
        clipped = float(torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip))
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == spec.steps:
            probability = torch.sigmoid(output["binary_logit"])
            accuracy = float(((probability >= 0.5) == graph.label.bool()).float().mean())
            records.append(
                {
                    "phase": "train_graph",
                    "step": step,
                    "loss": float(loss.detach()),
                    "accuracy": accuracy,
                    "grad_norm_preclip": preclip,
                    "grad_norm_reported": clipped,
                    "learning_rate": spec.learning_rate,
                    "distance": distance,
                    "query_access": float(
                        torch.stack([item["access"].mean() for item in trajectory]).mean()
                    ),
                    "parameters": _parameter_count(model),
                }
            )
    return records


def train_unknowable_model(
    model: nn.Module,
    spec: TrainSpec,
    *,
    seed: int,
    device: torch.device,
    log_every: int = 50,
) -> list[dict[str, Any]]:
    """Joint learned-head training on knowable and information-free random labels."""

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
    )
    rng = make_generator(seed)
    records: list[dict[str, Any]] = []
    half = spec.batch_size // 2
    for step in range(1, spec.steps + 1):
        # Knowable half: one external key/value, label is value parity.
        facts = sample_facts(half, 1, spec.symbol_count, rng, device)
        know_state = model.initial_state(half, device=device)
        know_state, _ = model.step(
            know_state, fact_event(facts.keys[:, 0], facts.values[:, 0])
        )
        know_state = model.reset_active(know_state)
        ticks = int(torch.randint(0, 5, (1,), generator=rng).item())
        know_state, know_output, _ = _query_with_ticks(
            model, know_state, facts.keys[:, 0], ticks
        )
        know_target = facts.values[:, 0].remainder(2).float()
        know_loss = Fnn.binary_cross_entropy_with_logits(
            know_output["binary_logit"], know_target
        )

        # Unknowable half: target is freshly random and absent from the event.
        unknown_keys = torch.randint(
            0, spec.symbol_count, (half,), generator=rng
        ).to(device)
        unknown_target = torch.randint(0, 2, (half,), generator=rng).float().to(device)
        unknown_state = model.initial_state(half, device=device)
        unknown_state, unknown_output = model.step(
            unknown_state, unknowable_event(unknown_keys)
        )
        for _ in range(ticks):
            unknown_state, unknown_output = model.step(unknown_state, None)
        unknown_loss = Fnn.binary_cross_entropy_with_logits(
            unknown_output["binary_logit"], unknown_target
        )
        loss = know_loss + unknown_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        preclip = _grad_norm(model)
        clipped = float(torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip))
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == spec.steps:
            know_accuracy = float(
                ((torch.sigmoid(know_output["binary_logit"]) >= 0.5) == know_target.bool())
                .float()
                .mean()
            )
            unknown_accuracy = float(
                ((torch.sigmoid(unknown_output["binary_logit"]) >= 0.5) == unknown_target.bool())
                .float()
                .mean()
            )
            records.append(
                {
                    "phase": "train_unknowable",
                    "step": step,
                    "loss": float(loss.detach()),
                    "knowable_accuracy": know_accuracy,
                    "unknowable_accuracy": unknown_accuracy,
                    "grad_norm_preclip": preclip,
                    "grad_norm_reported": clipped,
                    "learning_rate": spec.learning_rate,
                    "ticks": ticks,
                    "parameters": _parameter_count(model),
                }
            )
    return records

