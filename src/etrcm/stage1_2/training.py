"""End-to-end learned training for Stage-1.2 tasks."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as Fnn

from etrcm.stage1_1.data import (
    fact_event,
    graph_edge_event,
    make_generator,
    query_event,
    sample_facts,
    sample_graphs,
)
from etrcm.stage1_1.training import TrainSpec
from etrcm.stage1_2.data import (
    autonomous_labels,
    fixed_autonomous_keys,
    goal_event,
    path_key_at,
)


def _grad_norm(model) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().pow(2).sum()
    return float(total.sqrt())


def _query_loss(model, query: torch.Tensor, key_ids: torch.Tensor) -> torch.Tensor:
    target = model.target_key(key_ids)
    return (1.0 - (query * target).sum(-1)).mean()


def _optimize(model, optimizer, loss, gradient_clip: float) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    preclip = _grad_norm(model)
    reported = float(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip))
    optimizer.step()
    return preclip, reported


def train_autonomous_model(
    model,
    spec: TrainSpec,
    *,
    seed: int,
    device: torch.device,
    log_every: int = 50,
) -> list[dict[str, Any]]:
    """Train task-goal-only multi-memory retrieval; inference gets no key schedule."""

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    rng = make_generator(seed)
    rows = []
    for step in range(1, spec.steps + 1):
        batch = spec.batch_size
        keys = fixed_autonomous_keys(batch, device)
        values = torch.randint(0, spec.symbol_count, (batch, 3), generator=rng).to(device)
        operation = torch.randint(0, 3, (batch,), generator=rng).to(device)
        labels = autonomous_labels(values, operation)
        state = model.initial_state(batch, device=device)
        for index in range(3):
            state, _ = model.step(state, fact_event(keys[:, index], values[:, index]))
        state = model.reset_active(state)
        state, output = model.step(state, goal_event(batch, operation, device))
        q_losses = [_query_loss(model, output["query"], keys[:, 0])]
        state, output = model.step(state, None)
        q_losses.append(_query_loss(model, output["query"], keys[:, 1]))
        state, output = model.step(state, None)
        third_target = torch.where(operation.eq(2), keys[:, 2], keys[:, 0])
        q_losses.append(_query_loss(model, output["query"], third_target))
        task_loss = Fnn.binary_cross_entropy_with_logits(output["binary_logit"], labels)
        loss = task_loss + 0.20 * torch.stack(q_losses).mean()
        preclip, reported = _optimize(model, optimizer, loss, spec.gradient_clip)
        if step == 1 or step % log_every == 0 or step == spec.steps:
            probability = torch.sigmoid(output["binary_logit"])
            rows.append(
                {
                    "phase": "train_autonomous",
                    "step": step,
                    "loss": float(loss.detach()),
                    "task_loss": float(task_loss.detach()),
                    "query_aux_loss": float(torch.stack(q_losses).mean().detach()),
                    "accuracy": float((probability.ge(0.5) == labels.bool()).float().mean()),
                    "grad_norm_preclip": preclip,
                    "grad_norm_reported": reported,
                }
            )
    return rows


def train_sequential_model(
    model,
    spec: TrainSpec,
    *,
    seed: int,
    device: torch.device,
    train_max_length: int,
    log_every: int = 50,
) -> list[dict[str, Any]]:
    """Train one-query-per-tick graph traversal after graph-to-H scrubbing."""

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    rng = make_generator(seed)
    rows = []
    for step in range(1, spec.steps + 1):
        length = 1 + int(torch.randint(0, train_max_length, (1,), generator=rng))
        graph = sample_graphs(spec.batch_size, length, length, spec.symbol_count, rng, device)
        state = model.initial_state(spec.batch_size, device=device)
        for edge in range(length):
            state, _ = model.step(state, graph_edge_event(graph, edge))
        # Critical bottleneck: erase active write history before the goal.
        state = model.reset_active(state)
        state, output = model.step(state, query_event(graph.start, target=graph.target))
        q_losses = [_query_loss(model, output["query"], path_key_at(graph.sources, graph.destinations, graph.distance, 0))]
        for tick in range(1, train_max_length):
            state, output = model.step(state, None)
            q_losses.append(_query_loss(model, output["query"], path_key_at(graph.sources, graph.destinations, graph.distance, tick)))
        task_loss = Fnn.binary_cross_entropy_with_logits(output["binary_logit"], graph.label)
        loss = task_loss + 0.25 * torch.stack(q_losses).mean()
        preclip, reported = _optimize(model, optimizer, loss, spec.gradient_clip)
        if step == 1 or step % log_every == 0 or step == spec.steps:
            probability = torch.sigmoid(output["binary_logit"])
            rows.append(
                {
                    "phase": "train_sequential",
                    "step": step,
                    "length": length,
                    "loss": float(loss.detach()),
                    "task_loss": float(task_loss.detach()),
                    "query_aux_loss": float(torch.stack(q_losses).mean().detach()),
                    "accuracy": float((probability.ge(0.5) == graph.label.bool()).float().mean()),
                    "grad_norm_preclip": preclip,
                    "grad_norm_reported": reported,
                }
            )
    return rows


def train_no_evidence_model(
    model,
    spec: TrainSpec,
    *,
    seed: int,
    device: torch.device,
    max_ticks: int = 8,
    log_every: int = 50,
) -> list[dict[str, Any]]:
    """Train identical-format knowable/unknowable episodes without a condition cue."""

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    rng = make_generator(seed)
    rows = []
    half = spec.batch_size // 2
    for step in range(1, spec.steps + 1):
        query_keys = torch.randint(0, spec.symbol_count, (spec.batch_size,), generator=rng).to(device)
        values = torch.randint(0, spec.symbol_count, (spec.batch_size,), generator=rng).to(device)
        write_keys = query_keys.clone()
        offsets = 1 + torch.randint(0, spec.symbol_count - 1, (half,), generator=rng).to(device)
        write_keys[half:] = (query_keys[half:] + offsets).remainder(spec.symbol_count)
        targets = values.remainder(2).float()
        targets[half:] = torch.randint(0, 2, (half,), generator=rng).float().to(device)
        state = model.initial_state(spec.batch_size, device=device)
        state, _ = model.step(state, fact_event(write_keys, values))
        state = model.reset_active(state)
        state, output = model.step(state, query_event(query_keys))
        ticks = int(torch.randint(0, max_ticks + 1, (1,), generator=rng))
        for _ in range(ticks):
            state, output = model.step(state, None)
        loss = Fnn.binary_cross_entropy_with_logits(output["binary_logit"], targets)
        preclip, reported = _optimize(model, optimizer, loss, spec.gradient_clip)
        if step == 1 or step % log_every == 0 or step == spec.steps:
            probability = torch.sigmoid(output["binary_logit"])
            predicted = probability.ge(0.5)
            rows.append(
                {
                    "phase": "train_no_evidence",
                    "step": step,
                    "ticks": ticks,
                    "loss": float(loss.detach()),
                    "knowable_accuracy": float((predicted[:half] == targets[:half].bool()).float().mean()),
                    "unknowable_accuracy": float((predicted[half:] == targets[half:].bool()).float().mean()),
                    "grad_norm_preclip": preclip,
                    "grad_norm_reported": reported,
                }
            )
    return rows

