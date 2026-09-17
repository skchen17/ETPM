"""Machine-readable Stage-1.2 formal evaluations."""

from __future__ import annotations

import json
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
from etrcm.stage1_1.evaluation import _safe_cosine, _slow_read
from etrcm.stage1_1.model import LearnedState, _normalize
from etrcm.stage1_2.data import autonomous_labels, fixed_autonomous_keys, goal_event, path_key_at
from etrcm.stage1_2.interventions import decompose_query
from etrcm.stage1_2.model import Stage12ETRCM


def _norm(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return torch.zeros(value.shape[0], device=value.device)
    return torch.linalg.vector_norm(value.flatten(1), dim=-1)


def _json(value: torch.Tensor) -> str:
    return json.dumps([round(float(item), 6) for item in value.detach().cpu()])


def _entropy(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-7, 1 - 1e-7)
    return -(probability * probability.log() + (1 - probability) * (1 - probability).log())


def _symbol_rows(
    *,
    experiment: str,
    condition: str,
    model_name: str,
    seed: int,
    state,
    output: dict[str, Any],
    target: torch.Tensor,
    episode_offset: int,
    event_step: int,
    internal_tick: int,
    model,
    compute_budget: int,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    probability = torch.softmax(output["symbol_logits"], dim=-1)
    prediction = probability.argmax(-1)
    loss = Fnn.cross_entropy(output["symbol_logits"], target, reduction="none")
    confidence = probability.max(-1).values
    entropy = -(probability * probability.clamp_min(1e-8).log()).sum(-1)
    accuracy = prediction.eq(target).float()
    rows = []
    for index in range(target.shape[0]):
        row = {
            "experiment": experiment,
            "condition": condition,
            "model": model_name,
            "seed": seed,
            "episode": episode_offset + index,
            "event_step": event_step,
            "internal_tick": internal_tick,
            "query": _json(output["query"][index]) if "query" in output else None,
            "target_projection": None,
            "nontarget_projection": None,
            "access_strength": float(output.get("access", torch.zeros_like(accuracy))[index]),
            "h_norm": float(_norm(state.H)[index]),
            "f_norm": float(_norm(state.F)[index]),
            "m_norm": float(_norm(state.M)[index]),
            "transfer_norm": float(_norm(output.get("transfer", torch.zeros_like(state.F)))[index]),
            "retention": None,
            "accuracy": float(accuracy[index]),
            "loss": float(loss[index]),
            "confidence": float(confidence[index]),
            "entropy": float(entropy[index]),
            "ece": None,
            "brier": None,
            "persistent_state_bytes": model.persistent_state_bytes(),
            "parameter_count": model.trainable_parameters(),
            "compute_budget": compute_budget,
            "notes": "",
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
    condition: str,
    model_name: str,
    seed: int,
    state,
    output: dict[str, Any],
    target: torch.Tensor,
    episode_offset: int,
    event_step: int,
    internal_tick: int,
    model,
    compute_budget: int,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    probability = torch.sigmoid(output["binary_logit"])
    prediction = probability.ge(0.5)
    accuracy = prediction.eq(target.bool()).float()
    confidence = torch.maximum(probability, 1 - probability)
    loss = Fnn.binary_cross_entropy_with_logits(output["binary_logit"], target.float(), reduction="none")
    entropy = _entropy(probability)
    brier = (probability - target.float()).pow(2)
    ece = abs(float(confidence.mean()) - float(accuracy.mean()))
    rows = []
    for index in range(target.shape[0]):
        row = {
            "experiment": experiment,
            "condition": condition,
            "model": model_name,
            "seed": seed,
            "episode": episode_offset + index,
            "event_step": event_step,
            "internal_tick": internal_tick,
            "query": _json(output["query"][index]) if "query" in output else None,
            "target_projection": None,
            "nontarget_projection": None,
            "access_strength": float(output.get("access", torch.zeros_like(accuracy))[index]),
            "h_norm": float(_norm(state.H)[index]),
            "f_norm": float(_norm(state.F)[index]),
            "m_norm": float(_norm(state.M)[index]),
            "transfer_norm": float(_norm(output.get("transfer", torch.zeros_like(state.F)))[index]),
            "retention": None,
            "accuracy": float(accuracy[index]),
            "loss": float(loss[index]),
            "confidence": float(confidence[index]),
            "entropy": float(entropy[index]),
            "ece": ece,
            "brier": float(brier[index]),
            "positive_probability": float(probability[index]),
            "persistent_state_bytes": model.persistent_state_bytes(),
            "parameter_count": model.trainable_parameters(),
            "compute_budget": compute_budget,
            "notes": "",
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


def _write(model, state, facts):
    for index in range(facts.keys.shape[1]):
        state, _ = model.step(state, fact_event(facts.keys[:, index], facts.values[:, index]))
    return state


def _query_ticks(model, state, keys, ticks: int):
    state, output = model.step(state, query_event(keys))
    for _ in range(ticks):
        state, output = model.step(state, None)
    return state, output


def _zero_fast(state):
    if state.F.numel() == 0:
        return state
    return LearnedState(state.H, torch.zeros_like(state.F), state.M, state.tau, state.external_time)


@torch.no_grad()
def functional_query_intervention(
    model: Stage12ETRCM,
    *, seed: int, episodes: int, interference: int, symbol_count: int, device: torch.device
) -> list[dict[str, Any]]:
    model.eval()
    rng = make_generator(seed + 110_000)
    facts = sample_facts(episodes, 8, symbol_count, rng, device)
    state = _write(model, model.initial_state(episodes, device=device), facts)
    target_keys, targets = facts.keys[:, 0], facts.values[:, 0]
    for _ in range(4):
        state = model.reset_active(state)
        state, _ = _query_ticks(model, state, target_keys, 1)
    distractors = sample_facts(episodes, interference, symbol_count, rng, device)
    state = _write(model, state, distractors)
    state = model.reset_active(state)
    event = query_event(target_keys)
    query, _, _ = model.proposed_query(state, event)
    target_vector = model.target_key(target_keys)
    non_target = torch.stack([model.target_key(facts.keys[:, index]) for index in range(1, 8)], dim=1)
    interventions = decompose_query(query, target_vector, non_target, generator=rng)
    target_value_vector = _normalize(model.symbol_embedding(targets), dim=-1)
    records = []
    for condition in ("original", "parallel", "perpendicular", "signflip", "random", "target_zero", "strongest_nontarget_zero"):
        forced = getattr(interventions, condition)
        condition_state, forced_output = model.step_with_forced_query(state.clone(), event, forced)
        read_score = _safe_cosine(forced_output["read"], target_value_vector)
        h_change = _norm(condition_state.H - state.H)
        condition_state, output = model.step(condition_state, None)
        rows = _symbol_rows(
            experiment="A_functional_addressing", condition=condition, model_name="B6_full",
            seed=seed, state=condition_state, output=output, target=targets,
            episode_offset=0, event_step=8 + interference, internal_tick=1, model=model,
            compute_budget=8 + 8 + interference + 2,
            extra={
                "target_projection": interventions.signed_cosine,
                "absolute_cosine": interventions.absolute_cosine,
                "squared_projection": interventions.squared_projection,
                "nontarget_projection": interventions.target_nontarget_margin,
                "target_retrieval_score": read_score,
                "downstream_hidden_change": h_change,
                "forced_query_norm": torch.linalg.vector_norm(forced, dim=-1),
            },
        )
        records.extend(rows)
    return records


@torch.no_grad()
def exposure_reuse_phase(
    model: Stage12ETRCM,
    *, seed: int, exposures: list[int], reuses: list[int], episodes: int,
    interference: int, symbol_count: int, device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rng = make_generator(seed + 120_000)
    target = sample_facts(episodes, 1, symbol_count, rng, device)
    distractors = sample_facts(episodes, interference, symbol_count, rng, device)
    target_vector = _normalize(model.symbol_embedding(target.values[:, 0]), dim=-1)
    records = []
    cell = 0
    for exposure_count in exposures:
        for reuse_count in reuses:
            state = model.initial_state(episodes, device=device)
            transfer_mass = torch.zeros(episodes, device=device)
            for _ in range(exposure_count):
                state, output = model.step(state, fact_event(target.keys[:, 0], target.values[:, 0]))
                transfer_mass += _norm(output["transfer"])
            for _ in range(reuse_count):
                state = model.reset_active(state)
                state, output = _query_ticks(model, state, target.keys[:, 0], 1)
                transfer_mass += _norm(output["transfer"])
            state = _write(model, state, distractors)
            slow_retention = _safe_cosine(_slow_read(model, state, target.keys[:, 0]), target_vector)
            state = _zero_fast(model.reset_active(state))
            state, output = _query_ticks(model, state, target.keys[:, 0], 2)
            records += _symbol_rows(
                experiment="B_exposure_reuse_phase", condition="slow_only", model_name="B6_full",
                seed=seed, state=state, output=output, target=target.values[:, 0],
                episode_offset=cell * episodes, event_step=exposure_count + interference,
                internal_tick=reuse_count, model=model,
                compute_budget=exposure_count + 2 * reuse_count + interference + 3,
                extra={"exposure_count": exposure_count, "reuse_count": reuse_count,
                       "retention": slow_retention, "transfer_mass": transfer_mass},
            )
            cell += 1
    return records


@torch.no_grad()
def selective_scaling(
    model,
    *, model_name: str, seed: int, distractor_counts: list[int], useful_count: int,
    episodes: int, symbol_count: int, device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rng = make_generator(seed + 130_000)
    useful = sample_facts(episodes, useful_count, symbol_count, rng, device)
    state = _write(model, model.initial_state(episodes, device=device), useful)
    for index in range(useful_count):
        state = model.reset_active(state)
        state, _ = _query_ticks(model, state, useful.keys[:, index], 1)
    distractors = sample_facts(episodes, max(distractor_counts), symbol_count, rng, device)
    records = []
    cursor = 0
    for count in distractor_counts:
        while cursor < count:
            state, _ = model.step(state, fact_event(distractors.keys[:, cursor], distractors.values[:, cursor]))
            cursor += 1
        for target_index in range(useful_count):
            keys, targets = useful.keys[:, target_index], useful.values[:, target_index]
            target_vector = _normalize(model.symbol_embedding(targets), dim=-1)
            retention = _safe_cosine(_slow_read(model, state, keys), target_vector)
            eval_state = model.reset_active(state.clone())
            eval_state, output = _query_ticks(model, eval_state, keys, 2)
            rows = _symbol_rows(
                experiment="C_selective_scaling", condition="intact", model_name=model_name,
                seed=seed, state=eval_state, output=output, target=targets,
                episode_offset=(count * useful_count + target_index) * episodes,
                event_step=useful_count + count, internal_tick=2, model=model,
                compute_budget=useful_count + 2 * useful_count + count + 3,
                extra={
                    "distractor_count": count,
                    "useful_index": target_index,
                    "retention": retention,
                    "interference": 1.0 - retention,
                },
            )
            for row in rows:
                row["selective_persistence_efficiency"] = row["accuracy"] / row["persistent_state_bytes"]
            records += rows
            if count == max(distractor_counts) and isinstance(model, Stage12ETRCM):
                lesion = model.lesion(state.clone(), "both")
                post_retention = _safe_cosine(_slow_read(model, lesion, keys), target_vector)
                lesion = model.reset_active(lesion)
                lesion, post_output = _query_ticks(model, lesion, keys, 2)
                post_accuracy = post_output["symbol_logits"].argmax(-1).eq(targets).float()
                pre_accuracy = output["symbol_logits"].argmax(-1).eq(targets).float()
                records += _symbol_rows(
                    experiment="C_selective_scaling_lesion", condition="both_lesioned",
                    model_name=model_name, seed=seed, state=lesion, output=post_output,
                    target=targets, episode_offset=target_index * episodes,
                    event_step=useful_count + count, internal_tick=2, model=model,
                    compute_budget=useful_count + 2 * useful_count + count + 3,
                    extra={
                        "distractor_count": count,
                        "pre_lesion_slow_retention": retention,
                        "post_lesion_slow_retention": post_retention,
                        "pre_lesion_accuracy": pre_accuracy,
                        "post_lesion_accuracy": post_accuracy,
                    },
                )
    return records


@torch.no_grad()
def autonomous_memory_selection(
    model: Stage12ETRCM,
    *, seed: int, tick_counts: list[int], episodes: int, symbol_count: int, device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rng = make_generator(seed + 140_000)
    keys = fixed_autonomous_keys(episodes, device)
    values = torch.randint(0, symbol_count, (episodes, 3), generator=rng).to(device)
    operation = torch.randint(0, 3, (episodes,), generator=rng).to(device)
    labels = autonomous_labels(values, operation)
    state = model.initial_state(episodes, device=device)
    for index in range(3):
        state, _ = model.step(state, fact_event(keys[:, index], values[:, index]))
    state = model.reset_active(state)
    state, output = model.step(state, goal_event(episodes, operation, device))
    records = []
    key_vectors = torch.stack([model.target_key(keys[:, index]) for index in range(3)], dim=1)
    for tick in range(max(tick_counts) + 1):
        if tick > 0:
            state, output = model.step(state, None)
        if tick in tick_counts:
            projections = torch.einsum("bd,bnd->bn", output["query"], key_vectors)
            records += _binary_rows(
                experiment="D_autonomous_selection", condition="goal_then_null_only",
                model_name="B6_full", seed=seed, state=state, output=output, target=labels,
                episode_offset=tick * episodes, event_step=4, internal_tick=tick,
                model=model, compute_budget=4 + tick,
                extra={"operation": operation, "projection_A": projections[:, 0],
                       "projection_B": projections[:, 1], "projection_C": projections[:, 2],
                       "goal_contains_key_ids": False},
            )
    return records


@torch.no_grad()
def sequential_computation(
    model: Stage12ETRCM,
    *, seed: int, path_lengths: list[int], tick_counts: list[int], episodes: int,
    train_max_length: int, symbol_count: int, device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rng = make_generator(seed + 150_000)
    records = []
    for length in path_lengths:
        graph = sample_graphs(episodes, length, length, symbol_count, rng, device)
        state = model.initial_state(episodes, device=device)
        for edge in range(length):
            state, _ = model.step(state, graph_edge_event(graph, edge))
        state = model.reset_active(state)
        state, output = model.step(state, query_event(graph.start, target=graph.target))
        for tick in range(max(tick_counts) + 1):
            if tick > 0:
                state, output = model.step(state, None)
            if tick in tick_counts:
                expected = model.target_key(path_key_at(graph.sources, graph.destinations, graph.distance, tick))
                alignment = _safe_cosine(output["query"], expected)
                records += _binary_rows(
                    experiment="E_sequential_computation", condition="hard_one_query_bottleneck",
                    model_name="B6_full", seed=seed, state=state, output=output,
                    target=graph.label, episode_offset=(length * 100 + tick) * episodes,
                    event_step=length + 1, internal_tick=tick, model=model,
                    compute_budget=length + 1 + tick,
                    extra={"path_length": length, "length_regime": "train_support" if length <= train_max_length else "ood_longer",
                           "query_path_alignment": alignment, "queries_per_tick": 1,
                           "active_graph_history_scrubbed": True, "label_in_event": False},
                )
    return records


@torch.no_grad()
def endogenous_time_necessity(
    model: Stage12ETRCM,
    *, seed: int, ticks: int, interference: int, episodes: int,
    symbol_count: int, device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rng = make_generator(seed + 160_000)
    target = sample_facts(episodes, 1, symbol_count, rng, device)
    distractors = sample_facts(episodes, interference, symbol_count, rng, device)
    target_vector = _normalize(model.symbol_embedding(target.values[:, 0]), dim=-1)
    records = []
    for schedule in ("before_interference", "after_interference"):
        state = model.initial_state(episodes, device=device)
        state, _ = model.step(state, fact_event(target.keys[:, 0], target.values[:, 0]))
        state = model.reset_active(state)
        state, _ = model.step(state, query_event(target.keys[:, 0]))
        if schedule == "before_interference":
            for _ in range(ticks):
                state, _ = model.step(state, None)
        state = _write(model, state, distractors)
        if schedule == "after_interference":
            for _ in range(ticks):
                state, _ = model.step(state, None)
        retention = _safe_cosine(_slow_read(model, state, target.keys[:, 0]), target_vector)
        state = _zero_fast(model.reset_active(state))
        state, output = _query_ticks(model, state, target.keys[:, 0], 2)
        records += _symbol_rows(
            experiment="F_endogenous_time", condition=schedule, model_name="B6_full",
            seed=seed, state=state, output=output, target=target.values[:, 0],
            episode_offset=(0 if schedule.startswith("before") else episodes),
            event_step=2 + interference, internal_tick=ticks, model=model,
            compute_budget=2 + ticks + interference + 3,
            extra={"interference_count": interference, "retention": retention,
                   "matched_event_content": True, "matched_transition_count": True},
        )
    return records


@torch.no_grad()
def no_self_evidence(
    model: Stage12ETRCM,
    *, seed: int, tick_counts: list[int], episodes: int, symbol_count: int, device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rng = make_generator(seed + 170_000)
    query_keys = torch.randint(0, symbol_count, (2 * episodes,), generator=rng).to(device)
    values = torch.randint(0, symbol_count, (2 * episodes,), generator=rng).to(device)
    write_keys = query_keys.clone()
    offsets = 1 + torch.randint(0, symbol_count - 1, (episodes,), generator=rng).to(device)
    write_keys[episodes:] = (query_keys[episodes:] + offsets).remainder(symbol_count)
    targets = values.remainder(2).float()
    targets[episodes:] = torch.randint(0, 2, (episodes,), generator=rng).float().to(device)
    state = model.initial_state(2 * episodes, device=device)
    state, _ = model.step(state, fact_event(write_keys, values))
    state = model.reset_active(state)
    state, output = model.step(state, query_event(query_keys))
    records = []
    for tick in range(max(tick_counts) + 1):
        if tick > 0:
            state, output = model.step(state, None)
        if tick not in tick_counts:
            continue
        for start, stop, condition in ((0, episodes, "knowable"), (episodes, 2 * episodes, "unknowable")):
            sliced_state = type(state)(state.H[start:stop], state.F[start:stop], state.M[start:stop], state.tau, state.external_time)
            sliced_output = {key: value[start:stop] if isinstance(value, torch.Tensor) and value.shape[:1] == (2 * episodes,) else value for key, value in output.items()}
            records += _binary_rows(
                experiment="G_no_self_evidence", condition=condition, model_name="B6_full",
                seed=seed, state=sliced_state, output=sliced_output, target=targets[start:stop],
                episode_offset=(tick + (100 if condition == "knowable" else 200)) * episodes,
                event_step=2, internal_tick=tick, model=model, compute_budget=2 + tick,
                extra={"same_event_types": True, "event_sequence": "FACT,QUERY,NULL", "explicit_condition_cue": False},
            )
    return records


@torch.no_grad()
def storage_vs_use(
    model: Stage12ETRCM,
    *, seed: int, episodes: int, interference: int, symbol_count: int, device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rng = make_generator(seed + 180_000)
    facts = sample_facts(episodes, 8, symbol_count, rng, device)
    state = _write(model, model.initial_state(episodes, device=device), facts)
    keys, targets = facts.keys[:, 0], facts.values[:, 0]
    for _ in range(8):
        state = model.reset_active(state)
        state, _ = _query_ticks(model, state, keys, 1)
    distractors = sample_facts(episodes, interference, symbol_count, rng, device)
    state = _write(model, state, distractors)
    target_value = _normalize(model.symbol_embedding(targets), dim=-1)
    pre_retention = _safe_cosine(_slow_read(model, state, keys), target_value)
    base = model.reset_active(state)
    event = query_event(keys)
    query, _, _ = model.proposed_query(base, event)
    non_target = torch.stack([model.target_key(facts.keys[:, index]) for index in range(1, 8)], dim=1)
    decomposition = decompose_query(query, model.target_key(keys), non_target, generator=rng)
    records = []
    conditions = {
        "query_original": ("none", decomposition.original),
        "query_target_zero": ("none", decomposition.target_zero),
        "lesion_fast": ("fast", decomposition.original),
        "lesion_slow": ("slow", decomposition.original),
        "lesion_both": ("both", decomposition.original),
    }
    for condition, (lesion, forced) in conditions.items():
        current = model.lesion(base.clone(), lesion)
        post_retention = _safe_cosine(_slow_read(model, current, keys), target_value)
        current, output = model.step_with_forced_query(current, event, forced)
        rows = _symbol_rows(
            experiment="H_storage_vs_use", condition=condition, model_name="B6_full",
            seed=seed, state=current, output=output, target=targets,
            episode_offset=len(records), event_step=8 + interference,
            internal_tick=0, model=model, compute_budget=8 + 16 + interference + 1,
            extra={"pre_lesion_slow_retention": pre_retention,
                   "post_lesion_slow_retention": post_retention,
                   "pre_lesion_accuracy": torch.full((episodes,), float("nan"), device=device),
                   "post_lesion_accuracy": output["symbol_logits"].argmax(-1).eq(targets).float(),
                   "lesion_component": lesion,
                   "target_projection": decomposition.signed_cosine},
        )
        records += rows
    # Fill pre-lesion accuracy from the original condition without changing historical rows.
    original_accuracy = torch.tensor([row["accuracy"] for row in records[:episodes]])
    for row_index, row in enumerate(records):
        row["pre_lesion_accuracy"] = float(original_accuracy[row_index % episodes])
    return records

