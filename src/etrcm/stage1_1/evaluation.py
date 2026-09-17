"""Formal Stage-1.1 evaluations; every result row is machine-readable."""

from __future__ import annotations

import json
import math
from typing import Any

import torch
from torch.nn import functional as Fnn

from etrcm.stage1_1.baselines import TrainedGRU, TwoHeadSinglePersistent
from etrcm.stage1_1.data import (
    fact_event,
    graph_edge_event,
    interruption_event,
    make_generator,
    query_event,
    sample_facts,
    sample_graphs,
    unknowable_event,
)
from etrcm.stage1_1.model import LearnedETRCM, LearnedState, _normalize


def _safe_cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    numerator = (left * right).sum(-1)
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(
        right, dim=-1
    )
    return numerator / denominator.clamp_min(1e-8)


def _entropy(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-7, 1.0 - 1e-7)
    return -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log())


def _state_norm(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return torch.zeros(tensor.shape[0], device=tensor.device)
    return torch.linalg.vector_norm(tensor.flatten(1), dim=-1)


def _query_json(query: torch.Tensor) -> str:
    return json.dumps([round(float(value), 6) for value in query.detach().cpu().tolist()])


def _base_rows(
    *,
    experiment: str,
    model_name: str,
    seed: int,
    episode_offset: int,
    state,
    output: dict[str, Any],
    target: torch.Tensor | None,
    internal_tick: int,
    event_index: int,
    phase: str,
    state_bytes: int,
    parameter_count: int,
    compute_budget: int,
    notes: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    batch = state.H.shape[0]
    h_norm = _state_norm(state.H)
    f_norm = _state_norm(state.F)
    m_norm = _state_norm(state.M)
    query = output.get("query")
    read = output.get("read")
    transfer = output.get("transfer")
    symbol_logits = output.get("symbol_logits")
    rows: list[dict[str, Any]] = []
    for index in range(batch):
        confidence = None
        entropy = None
        accuracy = None
        loss = None
        if target is not None and symbol_logits is not None:
            probabilities = torch.softmax(symbol_logits[index], dim=-1)
            prediction = int(probabilities.argmax())
            target_value = int(target[index])
            confidence = float(probabilities.max())
            entropy = float(-(probabilities * probabilities.clamp_min(1e-8).log()).sum())
            accuracy = float(prediction == target_value)
            loss = float(Fnn.cross_entropy(symbol_logits[index : index + 1], target[index : index + 1]))
        row = {
            "experiment": experiment,
            "model": model_name,
            "seed": seed,
            "episode": episode_offset + index,
            "event_index": event_index,
            "internal_tick": internal_tick,
            "phase": phase,
            "h_norm": float(h_norm[index]),
            "f_norm": float(f_norm[index]),
            "m_norm": float(m_norm[index]),
            "query": _query_json(query[index]) if query is not None else None,
            "query_target_alignment": None,
            "memory_read_norm": float(torch.linalg.vector_norm(read[index])) if read is not None else 0.0,
            "transfer_norm": float(torch.linalg.vector_norm(transfer[index])) if transfer is not None else 0.0,
            "task_loss": loss,
            "accuracy": accuracy,
            "confidence": confidence,
            "entropy": entropy,
            "ece": None,
            "brier": None,
            "state_bytes": state_bytes,
            "parameter_count": parameter_count,
            "compute_budget": compute_budget,
            "notes": notes,
        }
        if extra:
            for key, value in extra.items():
                if isinstance(value, torch.Tensor):
                    item = value[index]
                    row[key] = float(item) if item.numel() == 1 else json.dumps(item.detach().cpu().tolist())
                else:
                    row[key] = value
        rows.append(row)
    return rows


def _binary_rows(
    *,
    experiment: str,
    model_name: str,
    seed: int,
    episode_offset: int,
    state,
    output: dict[str, Any],
    target: torch.Tensor,
    ticks: int,
    event_index: int,
    state_bytes: int,
    parameter_count: int,
    compute_budget: int,
    notes: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    probability = torch.sigmoid(output["binary_logit"])
    prediction = probability.ge(0.5)
    confidence = torch.maximum(probability, 1.0 - probability)
    accuracy = prediction.eq(target.bool()).float()
    loss = Fnn.binary_cross_entropy_with_logits(
        output["binary_logit"], target.float(), reduction="none"
    )
    brier = (probability - target.float()).pow(2)
    rows = _base_rows(
        experiment=experiment,
        model_name=model_name,
        seed=seed,
        episode_offset=episode_offset,
        state=state,
        output=output,
        target=None,
        internal_tick=ticks,
        event_index=event_index,
        phase="final",
        state_bytes=state_bytes,
        parameter_count=parameter_count,
        compute_budget=compute_budget,
        notes=notes,
        extra=extra,
    )
    for index, row in enumerate(rows):
        row.update(
            {
                "task_loss": float(loss[index]),
                "accuracy": float(accuracy[index]),
                "confidence": float(confidence[index]),
                "entropy": float(_entropy(probability[index])),
                "brier": float(brier[index]),
                "positive_probability": float(probability[index]),
            }
        )
    mean_confidence = float(confidence.mean())
    mean_accuracy = float(accuracy.mean())
    for row in rows:
        row["ece"] = abs(mean_confidence - mean_accuracy)
    return rows


def _query_with_ticks(model, state, keys, ticks: int, target=None):
    state, output = model.step(state, query_event(keys, target=target))
    trajectory = [output]
    for _ in range(ticks):
        state, output = model.step(state, None)
        trajectory.append(output)
    return state, output, trajectory


def _write_facts(model, state, facts):
    accesses = []
    for index in range(facts.keys.shape[1]):
        state, output = model.step(
            state, fact_event(facts.keys[:, index], facts.values[:, index])
        )
        accesses.append(output["access"])
    return state, accesses


def _slow_read(model, state, keys: torch.Tensor) -> torch.Tensor:
    if isinstance(model, TrainedGRU) or state.F.numel() == 0:
        return torch.zeros(keys.shape[0], model.config.value_dim, device=keys.device)
    if isinstance(model, TwoHeadSinglePersistent):
        key1 = model.target_key(keys)
        key2 = model._key2(keys)
        return 0.5 * (
            torch.einsum("bvk,bk->bv", state.F, key1)
            + torch.einsum("bvk,bk->bv", state.M, key2)
        )
    key = model.target_key(keys)
    return torch.einsum("bvk,bk->bv", state.M, key)


@torch.no_grad()
def evaluate_learned_reuse(
    model,
    *,
    model_name: str,
    seed: int,
    reuse_counts: list[int],
    episodes: int,
    interference: int,
    symbol_count: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rng = make_generator(seed + 10_000)
    facts = sample_facts(episodes, 8, symbol_count, rng, device)
    base_state = model.initial_state(episodes, device=device)
    base_state, _ = _write_facts(model, base_state, facts)
    target_keys, targets = facts.keys[:, 0], facts.values[:, 0]
    target_key_vectors = model.target_key(target_keys)
    target_values = _normalize(model.symbol_embedding(targets), dim=-1)
    distractors = sample_facts(episodes, interference, symbol_count, rng, device)
    records: list[dict[str, Any]] = []
    for reuse in reuse_counts:
        state = base_state.clone()
        alignments = []
        cumulative_access = torch.zeros(episodes, device=device)
        cumulative_transfer = torch.zeros(episodes, device=device)
        for _ in range(reuse):
            state = model.reset_active(state)
            state, _, trajectory = _query_with_ticks(model, state, target_keys, 1)
            for output in trajectory:
                alignments.append(_safe_cosine(output["query"], target_key_vectors))
                cumulative_access += output["access"]
                cumulative_transfer += _state_norm(output["transfer"])
        for index in range(interference):
            state, _ = model.step(
                state,
                fact_event(distractors.keys[:, index], distractors.values[:, index]),
            )
        slow_read = _slow_read(model, state, target_keys)
        retention = _safe_cosine(slow_read, target_values)
        state = model.reset_active(state)
        state, output, _ = _query_with_ticks(model, state, target_keys, 2)
        if alignments:
            alignment = torch.stack(alignments).mean(0)
        else:
            alignment = _safe_cosine(output["query"], target_key_vectors)
        rows = _base_rows(
            experiment="A_learned_reuse",
            model_name=model_name,
            seed=seed,
            episode_offset=reuse * episodes,
            state=state,
            output=output,
            target=targets,
            internal_tick=reuse,
            event_index=8 + interference + reuse,
            phase="final",
            state_bytes=model.persistent_state_bytes(),
            parameter_count=model.trainable_parameters(),
            compute_budget=8 + interference + 2 * reuse + 3,
            notes="all target facts have one external exposure; final H scrubbed before query",
            extra={
                "reuse_count": reuse,
                "query_target_alignment": alignment,
                "cumulative_access": cumulative_access,
                "cumulative_transfer": cumulative_transfer,
                "slow_retention": retention,
            },
        )
        records.extend(rows)
    return records


@torch.no_grad()
def evaluate_capacity_pressure(
    model,
    *,
    model_name: str,
    seed: int,
    distractor_counts: list[int],
    episodes: int,
    symbol_count: int,
    device: torch.device,
    lesion: bool = False,
) -> list[dict[str, Any]]:
    model.eval()
    rng = make_generator(seed + 20_000)
    useful = sample_facts(episodes, 8, symbol_count, rng, device)
    base_state = model.initial_state(episodes, device=device)
    base_state, _ = _write_facts(model, base_state, useful)
    for index in range(8):
        base_state = model.reset_active(base_state)
        base_state, _, _ = _query_with_ticks(
            model, base_state, useful.keys[:, index], 1
        )
    records: list[dict[str, Any]] = []
    max_distractors = max(distractor_counts)
    distractors = sample_facts(
        episodes, max_distractors, symbol_count, rng, device
    )
    state = base_state.clone()
    for index in range(max_distractors + 1):
        if index in distractor_counts:
            eval_state = state.clone()
            target_keys, targets = useful.keys[:, 0], useful.values[:, 0]
            target_values = _normalize(model.symbol_embedding(targets), dim=-1)
            slow_read = _slow_read(model, eval_state, target_keys)
            retention = _safe_cosine(slow_read, target_values)
            if lesion:
                eval_state = model.memory_lesion(eval_state)
            eval_state = model.reset_active(eval_state)
            eval_state, output, _ = _query_with_ticks(
                model, eval_state, target_keys, 2
            )
            accuracy = (output["symbol_logits"].argmax(-1) == targets).float()
            efficiency = accuracy / float(model.persistent_state_bytes())
            records.extend(
                _base_rows(
                    experiment="B_selective_persistence",
                    model_name=model_name + ("_memory_lesion" if lesion else ""),
                    seed=seed,
                    episode_offset=index * episodes,
                    state=eval_state,
                    output=output,
                    target=targets,
                    internal_tick=2,
                    event_index=8 + 16 + index,
                    phase="final",
                    state_bytes=model.persistent_state_bytes(),
                    parameter_count=model.trainable_parameters(),
                    compute_budget=8 + 16 + index + 3,
                    notes="8 useful facts each used twice; no useful/importance input bit",
                    extra={
                        "distractor_count": index,
                        "slow_retention": retention,
                        "selective_persistence_efficiency": efficiency,
                        "memory_lesion": lesion,
                    },
                )
            )
        if index == max_distractors:
            break
        state, _ = model.step(
            state,
            fact_event(distractors.keys[:, index], distractors.values[:, index]),
        )
    return records


@torch.no_grad()
def evaluate_frequency_utility(
    model,
    *,
    model_name: str,
    seed: int,
    useless_frequencies: list[int],
    useful_frequencies: list[int],
    episodes: int,
    symbol_count: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rng = make_generator(seed + 30_000)
    facts = sample_facts(episodes, 2, symbol_count, rng, device)
    useful_key, useful_value = facts.keys[:, 0], facts.values[:, 0]
    useless_key, useless_value = facts.keys[:, 1], facts.values[:, 1]
    records: list[dict[str, Any]] = []
    condition_index = 0
    for useful_frequency in useful_frequencies:
        for useless_frequency in useless_frequencies:
            state = model.initial_state(episodes, device=device)
            for _ in range(useful_frequency):
                state, _ = model.step(state, fact_event(useful_key, useful_value))
            for _ in range(useless_frequency):
                state, _ = model.step(state, fact_event(useless_key, useless_value))
            access = torch.zeros(episodes, device=device)
            for _ in range(8):
                state = model.reset_active(state)
                state, _, trajectory = _query_with_ticks(model, state, useful_key, 1)
                access += sum(item["access"] for item in trajectory)
            distractors = sample_facts(episodes, 128, symbol_count, rng, device)
            state, _ = _write_facts(model, state, distractors)
            useful_read = _slow_read(model, state, useful_key)
            useless_read = _slow_read(model, state, useless_key)
            useful_target = _normalize(model.symbol_embedding(useful_value), dim=-1)
            useless_target = _normalize(model.symbol_embedding(useless_value), dim=-1)
            useful_retention = _safe_cosine(useful_read, useful_target)
            useless_retention = _safe_cosine(useless_read, useless_target)
            state = model.reset_active(state)
            state, output, _ = _query_with_ticks(model, state, useful_key, 2)
            records.extend(
                _base_rows(
                    experiment="C_frequency_vs_utility",
                    model_name=model_name,
                    seed=seed,
                    episode_offset=condition_index * episodes,
                    state=state,
                    output=output,
                    target=useful_value,
                    internal_tick=8,
                    event_index=useful_frequency + useless_frequency + 16 + 128,
                    phase="final",
                    state_bytes=model.persistent_state_bytes(),
                    parameter_count=model.trainable_parameters(),
                    compute_budget=useful_frequency + useless_frequency + 16 + 128 + 3,
                    notes="no frequency/utility marker; utility expressed only by downstream queries",
                    extra={
                        "useful_frequency": useful_frequency,
                        "useless_frequency": useless_frequency,
                        "useful_retention": useful_retention,
                        "useless_retention": useless_retention,
                        "cumulative_access": access,
                    },
                )
            )
            condition_index += 1
    return records


@torch.no_grad()
def evaluate_idle_reasoning(
    model,
    *,
    model_name: str,
    seed: int,
    tick_counts: list[int],
    episodes: int,
    symbol_count: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """D: learned graph inference with a learned head and autonomous NULL ticks."""
    model.eval()
    rng = make_generator(seed + 40_000)
    graph = sample_graphs(episodes, 4, 6, symbol_count, rng, device)
    base = model.initial_state(episodes, device=device)
    for edge in range(6):
        base, _ = model.step(base, graph_edge_event(graph, edge))
    base = model.reset_active(base)
    base, _ = model.step(base, query_event(graph.start, target=graph.target))
    records: list[dict[str, Any]] = []
    previous_state = base
    for ticks in range(max(tick_counts) + 1):
        if ticks > 0:
            previous_state, output = model.step(previous_state, None)
        else:
            output = model.predict(previous_state)
            output.update(
                {
                    "query": model.target_key(graph.start),
                    "read": torch.zeros(episodes, model.config.value_dim, device=device),
                    "access": torch.zeros(episodes, device=device),
                    "transfer": torch.zeros_like(previous_state.F),
                }
            )
        if ticks in tick_counts:
            records.extend(
                _binary_rows(
                    experiment="D_idle_reasoning",
                    model_name=model_name,
                    seed=seed,
                    episode_offset=ticks * episodes,
                    state=previous_state,
                    output=output,
                    target=graph.label,
                    ticks=ticks,
                    event_index=7,
                    state_bytes=model.persistent_state_bytes(),
                    parameter_count=model.trainable_parameters(),
                    compute_budget=7 + ticks,
                    notes="all graph edges precede query; NULL ticks contain no label or history",
                    extra={"graph_distance": graph.distance, "schedule": "idle"},
                )
            )
    return records


@torch.no_grad()
def evaluate_interleaved_time(
    model,
    *,
    model_name: str,
    seed: int,
    ticks: int,
    episodes: int,
    symbol_count: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """E: equal event/step counts, different placement of internal computation."""
    model.eval()
    rng = make_generator(seed + 50_000)
    facts = sample_facts(episodes, 2, symbol_count, rng, device)
    records: list[dict[str, Any]] = []
    terminal_states: dict[str, torch.Tensor] = {}
    for schedule in ("think_before_interruption", "think_after_interruption"):
        state = model.initial_state(episodes, device=device)
        state, _ = model.step(state, fact_event(facts.keys[:, 0], facts.values[:, 0]))
        if schedule == "think_before_interruption":
            for _ in range(ticks):
                state, _ = model.step(state, None)
        state, _ = model.step(state, interruption_event(facts.keys[:, 1]))
        if schedule == "think_after_interruption":
            for _ in range(ticks):
                state, _ = model.step(state, None)
        terminal_states[schedule] = state.H.detach().clone()
        state = model.reset_active(state)
        state, output, _ = _query_with_ticks(state=state, model=model, keys=facts.keys[:, 0], ticks=2)
        target_vectors = _normalize(model.symbol_embedding(facts.values[:, 0]), dim=-1)
        retention = _safe_cosine(_slow_read(model, state, facts.keys[:, 0]), target_vectors)
        records.extend(
            _base_rows(
                experiment="E_interleaved_time",
                model_name=model_name,
                seed=seed,
                episode_offset=(0 if schedule.startswith("think_before") else episodes),
                state=state,
                output=output,
                target=facts.values[:, 0],
                internal_tick=ticks,
                event_index=2,
                phase="final",
                state_bytes=model.persistent_state_bytes(),
                parameter_count=model.trainable_parameters(),
                compute_budget=ticks + 4,
                notes="matched external events and transition count; only tick placement differs",
                extra={"schedule": schedule, "slow_retention": retention},
            )
        )
    distance = _state_norm(
        terminal_states["think_before_interruption"]
        - terminal_states["think_after_interruption"]
    )
    for row in records:
        row["matched_schedule_state_distance"] = float(distance[row["episode"] % episodes])
    return records


@torch.no_grad()
def evaluate_consolidate_before_interference(
    model,
    *,
    model_name: str,
    seed: int,
    ticks: int,
    interference: int,
    episodes: int,
    symbol_count: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """F: matched compute before versus after the interference stream."""
    model.eval()
    rng = make_generator(seed + 60_000)
    target = sample_facts(episodes, 1, symbol_count, rng, device)
    distractors = sample_facts(episodes, interference, symbol_count, rng, device)
    records: list[dict[str, Any]] = []
    for schedule in ("before_interference", "after_interference"):
        state = model.initial_state(episodes, device=device)
        state, _ = model.step(state, fact_event(target.keys[:, 0], target.values[:, 0]))
        if schedule == "before_interference":
            state = model.reset_active(state)
            state, _, _ = _query_with_ticks(model, state, target.keys[:, 0], ticks)
        state, _ = _write_facts(model, state, distractors)
        if schedule == "after_interference":
            state = model.reset_active(state)
            state, _, _ = _query_with_ticks(model, state, target.keys[:, 0], ticks)
        slow = _slow_read(model, state, target.keys[:, 0])
        target_vector = _normalize(model.symbol_embedding(target.values[:, 0]), dim=-1)
        retention = _safe_cosine(slow, target_vector)
        state = model.reset_active(state)
        state, output, _ = _query_with_ticks(model, state, target.keys[:, 0], 2)
        records.extend(
            _base_rows(
                experiment="F_consolidate_before_interference",
                model_name=model_name,
                seed=seed,
                episode_offset=(0 if schedule.startswith("before") else episodes),
                state=state,
                output=output,
                target=target.values[:, 0],
                internal_tick=ticks,
                event_index=1 + interference,
                phase="final",
                state_bytes=model.persistent_state_bytes(),
                parameter_count=model.trainable_parameters(),
                compute_budget=1 + interference + ticks + 4,
                notes="same facts and compute; consolidation/query block moved across interference",
                extra={"schedule": schedule, "slow_retention": retention},
            )
        )
    return records


@torch.no_grad()
def evaluate_reason_before_interruption(
    model,
    *,
    model_name: str,
    seed: int,
    ticks: int,
    episodes: int,
    symbol_count: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """G: graph reasoning before versus after a matched interruption event."""
    model.eval()
    rng = make_generator(seed + 70_000)
    graph = sample_graphs(episodes, 5, 5, symbol_count, rng, device)
    interruption_keys = torch.randint(0, symbol_count, (episodes,), generator=rng).to(device)
    records: list[dict[str, Any]] = []
    for schedule in ("reason_before_interruption", "reason_after_interruption"):
        state = model.initial_state(episodes, device=device)
        for edge in range(5):
            state, _ = model.step(state, graph_edge_event(graph, edge))
        state = model.reset_active(state)
        state, _ = model.step(state, query_event(graph.start, target=graph.target))
        if schedule == "reason_before_interruption":
            for _ in range(ticks):
                state, _ = model.step(state, None)
        state, output = model.step(state, interruption_event(interruption_keys))
        if schedule == "reason_after_interruption":
            for _ in range(ticks):
                state, output = model.step(state, None)
        records.extend(
            _binary_rows(
                experiment="G_reason_before_interruption",
                model_name=model_name,
                seed=seed,
                episode_offset=(0 if schedule.startswith("reason_before") else episodes),
                state=state,
                output=output,
                target=graph.label,
                ticks=ticks,
                event_index=7,
                state_bytes=model.persistent_state_bytes(),
                parameter_count=model.trainable_parameters(),
                compute_budget=7 + ticks,
                notes="matched graph, interruption and transition count; order alone differs",
                extra={"schedule": schedule, "graph_distance": graph.distance},
            )
        )
    return records


@torch.no_grad()
def evaluate_unknowable(
    model,
    *,
    model_name: str,
    seed: int,
    tick_counts: list[int],
    episodes: int,
    symbol_count: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """H: absent random bit plus a matched knowable parity control."""
    model.eval()
    rng = make_generator(seed + 80_000)
    unknown_keys = torch.randint(0, symbol_count, (episodes,), generator=rng).to(device)
    labels = torch.randint(0, 2, (episodes,), generator=rng).float().to(device)
    facts = sample_facts(episodes, 1, symbol_count, rng, device)
    records: list[dict[str, Any]] = []
    for control in (False, True):
        base = model.initial_state(episodes, device=device)
        if control:
            base, _ = model.step(base, fact_event(facts.keys[:, 0], facts.values[:, 0]))
            base = model.reset_active(base)
            base, output = model.step(base, query_event(facts.keys[:, 0]))
            target = facts.values[:, 0].remainder(2).float()
        else:
            base, output = model.step(base, unknowable_event(unknown_keys))
            target = labels
        state = base
        for ticks in range(max(tick_counts) + 1):
            if ticks > 0:
                state, output = model.step(state, None)
            if ticks in tick_counts:
                records.extend(
                    _binary_rows(
                        experiment="H_unknowable_bit",
                        model_name=model_name,
                        seed=seed,
                        episode_offset=(ticks + (100 if control else 0)) * episodes,
                        state=state,
                        output=output,
                        target=target,
                        ticks=ticks,
                        event_index=1,
                        state_bytes=model.persistent_state_bytes(),
                        parameter_count=model.trainable_parameters(),
                        compute_budget=1 + ticks,
                        notes="random target is sampled independently and never encoded" if not control else "knowable parity control",
                        extra={"control": "knowable" if control else "unknowable"},
                    )
                )
    return records


@torch.no_grad()
def evaluate_revision(
    model,
    *,
    model_name: str,
    seed: int,
    old_exposures: list[int],
    new_exposures: list[int],
    new_reuses: list[int],
    episodes: int,
    symbol_count: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """I: preserve every preregistered old/new/reuse grid cell."""
    model.eval()
    rng = make_generator(seed + 90_000)
    facts = sample_facts(episodes, 3, symbol_count, rng, device)
    key = facts.keys[:, 0]
    old_value = facts.values[:, 0]
    new_value = (old_value + 1 + facts.values[:, 1]).remainder(symbol_count)
    unrelated_key, unrelated_value = facts.keys[:, 2], facts.values[:, 2]
    records: list[dict[str, Any]] = []
    condition = 0
    for old_count in old_exposures:
        for new_count in new_exposures:
            for reuse in new_reuses:
                state = model.initial_state(episodes, device=device)
                for _ in range(old_count):
                    state, _ = model.step(state, fact_event(key, old_value))
                for _ in range(4):
                    state = model.reset_active(state)
                    state, _, _ = _query_with_ticks(model, state, key, 1)
                state, _ = model.step(state, fact_event(unrelated_key, unrelated_value))
                for _ in range(new_count):
                    state, _ = model.step(state, fact_event(key, new_value))
                for _ in range(reuse):
                    state = model.reset_active(state)
                    state, _, _ = _query_with_ticks(model, state, key, 1)
                state = model.reset_active(state)
                state, output, _ = _query_with_ticks(model, state, key, 2)
                probabilities = torch.softmax(output["symbol_logits"], dim=-1)
                new_probability = probabilities.gather(1, new_value[:, None]).squeeze(1)
                old_probability = probabilities.gather(1, old_value[:, None]).squeeze(1)
                records.extend(
                    _base_rows(
                        experiment="I_memory_revision",
                        model_name=model_name,
                        seed=seed,
                        episode_offset=condition * episodes,
                        state=state,
                        output=output,
                        target=new_value,
                        internal_tick=reuse,
                        event_index=old_count + new_count + 1,
                        phase="final",
                        state_bytes=model.persistent_state_bytes(),
                        parameter_count=model.trainable_parameters(),
                        compute_budget=old_count + new_count + 1 + 8 + 2 * reuse + 3,
                        notes="old association is consolidated before genuine contradictory evidence",
                        extra={
                            "old_exposures": old_count,
                            "new_exposures": new_count,
                            "new_reuses": reuse,
                            "old_probability": old_probability,
                            "new_probability": new_probability,
                        },
                    )
                )
                condition += 1
    return records
