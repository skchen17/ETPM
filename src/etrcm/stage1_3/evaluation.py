"""Formal Stage-1.3 streaming experiments and compact diagnostics."""

from __future__ import annotations

import copy
from types import MethodType
from typing import Any

import numpy as np
import torch

from etrcm.stage1_1.model import LearnedState, _normalize
from etrcm.stage1_3.events import ContinuousEvent, Stage13EventKind, context_event, evidence_event
from etrcm.stage1_3.model import ContinuousETRCM


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _ids(batch: int, count: int, rng: torch.Generator, device: torch.device) -> torch.Tensor:
    return torch.randint(0, count, (batch,), generator=rng).to(device)


def _norm(value: torch.Tensor) -> torch.Tensor:
    if value.ndim <= 1:
        return value.abs()
    return torch.linalg.vector_norm(value, dim=tuple(range(1, value.ndim)))


def _event_type(event: ContinuousEvent | None) -> str:
    if event is None:
        return "NULL"
    kinds = event.kind.unique().tolist()
    return Stage13EventKind(int(kinds[0])).name if len(kinds) == 1 else "MIXED_EXTERNAL"


def _diagnostic_rows(
    *,
    experiment: str,
    condition: str,
    model_name: str,
    seed: int,
    state: LearnedState,
    output: dict[str, torch.Tensor | bool],
    event: ContinuousEvent | None,
    threshold: float,
    tick: int,
    episode_offset: int = 0,
    target: torch.Tensor | None = None,
    extra: dict[str, Any] | None = None,
    aggregate: bool = False,
) -> list[dict[str, Any]]:
    batch = state.H.shape[0]
    content_logits = output["content_logits"]
    probability = content_logits.softmax(-1)
    prediction = content_logits.argmax(-1)
    confidence = probability.amax(-1)
    accuracy = (
        prediction.eq(target).float() if target is not None else torch.full((batch,), float("nan"), device=state.H.device)
    )
    values: dict[str, Any] = {
        "h_norm": _norm(state.H),
        "f_norm": _norm(state.F),
        "m_norm": _norm(state.M),
        "r_fast_norm": _norm(output["r_fast"]),
        "r_slow_norm": _norm(output["r_slow"]),
        "arbitration_gate": output["arbitration_gate"],
        "consolidation_access": output["access"],
        "transfer_norm": _norm(output["transfer"]),
        "expression_score": output["expression_score"],
        "emitted": output["emitted"].float(),
        "emitted_content_id": output["emitted_content_id"].float(),
        "accuracy": accuracy,
        "confidence": confidence,
        "external_write_flag": output["external_write_flag"].float(),
        "self_output_flag": output["emitted"].float(),
        "self_output_memory_update_norm": _norm(output["self_output_memory_update"]),
        "conservation_error": output["conservation_error"],
    }
    if extra:
        values.update(extra)
    indices = [None] if aggregate else list(range(batch))
    rows: list[dict[str, Any]] = []
    for index in indices:
        row: dict[str, Any] = {
            "experiment": experiment,
            "condition": condition,
            "model": model_name,
            "seed": seed,
            "episode": -1 if index is None else episode_offset + index,
            "external_event_index": state.external_time,
            "internal_tick": tick,
            "event_type": _event_type(event),
            "query_ref": f"tensor_samples.pt::{experiment}/{condition}/{tick}",
            "threshold": threshold,
            "persistent_state_bytes": 4 * (
                state.H.shape[1] * state.H.shape[2] + state.F.shape[1] * state.F.shape[2] + state.M.shape[1] * state.M.shape[2]
            ),
        }
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                selected = value.float().mean() if index is None else value[index]
                row[name] = float(selected.detach().cpu())
            else:
                row[name] = value
        rows.append(row)
    return rows


def _mixed_stream_event(
    keys: torch.Tensor,
    values: torch.Tensor,
    support: torch.Tensor,
    rng: torch.Generator,
    symbols: int,
) -> ContinuousEvent:
    random_keys = _ids(keys.shape[0], symbols, rng, keys.device)
    random_values = _ids(keys.shape[0], symbols, rng, keys.device)
    event_keys = torch.where(support, keys, random_keys)
    event_values = torch.where(support, values, random_values)
    scalars = torch.zeros(keys.shape[0], 4, device=keys.device)
    scalars[:, 0] = support.float() * 0.25
    return ContinuousEvent.create(
        kind=Stage13EventKind.EVIDENCE,
        key_id=event_keys,
        value_id=event_values,
        write=True,
        scalars=scalars,
    )


@torch.no_grad()
def evidence_accumulation(
    model: ContinuousETRCM,
    *, model_name: str, seed: int, threshold: float, episodes: int, length: int,
    sufficient_count: int, insufficient_count: int, device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    model.eval()
    rng = _generator(seed + 310_000)
    half = episodes // 2
    sufficient = torch.arange(episodes, device=device).lt(half)
    counts = torch.where(
        sufficient,
        torch.full((episodes,), sufficient_count, device=device),
        torch.full((episodes,), insufficient_count, device=device),
    )
    keys = _ids(episodes, model.config.symbol_count, rng, device)
    values = _ids(episodes, model.config.symbol_count, rng, device)
    order = torch.rand(episodes, length, generator=rng).argsort(1).to(device)
    support = torch.zeros(episodes, length, dtype=torch.bool, device=device)
    for index in range(length):
        support.scatter_(1, order[:, index : index + 1], (counts > index)[:, None])
    state = model.initial_state(episodes, device=device)
    cumulative = torch.zeros(episodes, dtype=torch.long, device=device)
    rows: list[dict[str, Any]] = []
    samples: dict[str, torch.Tensor] = {}
    for tick in range(length):
        event = _mixed_stream_event(keys, values, support[:, tick], rng, model.config.symbol_count)
        state, output = model.step(state, event, threshold=threshold)
        cumulative += support[:, tick].long()
        enough = sufficient & cumulative.ge(sufficient_count)
        newly_sufficient = enough & support[:, tick]
        emitted_correct = output["emitted"] & output["emitted_content_id"].eq(values)
        rows += _diagnostic_rows(
            experiment="A_evidence_accumulation",
            condition="sufficient" if tick < 0 else "mixed_strata",
            model_name=model_name,
            seed=seed,
            state=state,
            output=output,
            event=event,
            threshold=threshold,
            tick=tick,
            target=values,
            extra={
                "is_sufficient_arm": sufficient.float(),
                "support_count": cumulative.float(),
                "enough_evidence": enough.float(),
                "newly_sufficient": newly_sufficient.float(),
                "correct_emission": emitted_correct.float(),
                "false_emission": (output["emitted"] & (~enough | ~output["emitted_content_id"].eq(values))).float(),
            },
        )
        samples[f"A/{tick}/query"] = output["query"][0].detach().cpu()
    return rows, samples


@torch.no_grad()
def silence_under_noise(
    model: ContinuousETRCM,
    *, model_name: str, seed: int, threshold: float, length: int, streams: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    model.eval()
    rng = _generator(seed + 320_000 + length)
    state = model.initial_state(streams, device=device)
    rows: list[dict[str, Any]] = []
    samples: dict[str, torch.Tensor] = {}
    for tick in range(length):
        keys = _ids(streams, model.config.symbol_count, rng, device)
        values = _ids(streams, model.config.symbol_count, rng, device)
        event = evidence_event(keys, values, support=0.0)
        state, output = model.step(state, event, threshold=threshold)
        rows += _diagnostic_rows(
            experiment="D_silence_noise", condition=f"noise_{length}", model_name=model_name,
            seed=seed, state=state, output=output, event=event, threshold=threshold,
            tick=tick, aggregate=True,
            extra={"false_emission_rate_tick": output["emitted"].float().mean().expand(streams)},
        )
        if tick in {0, length // 2, length - 1}:
            samples[f"D/{length}/{tick}/query"] = output["query"][0].detach().cpu()
    return rows, samples


def _retention(model: ContinuousETRCM, state: LearnedState, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    key = model.target_key(keys)
    read = torch.einsum("bvk,bk->bv", state.M, key)
    target = model.value_vector(values)
    return (read * target).sum(-1) / (
        torch.linalg.vector_norm(read, dim=-1) * torch.linalg.vector_norm(target, dim=-1)
    ).clamp_min(1e-8)


@torch.no_grad()
def arbitration_sweep(
    model: ContinuousETRCM,
    *, model_name: str, seed: int, threshold: float, episodes: int,
    distractor_counts: list[int], device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    model.eval()
    rng = _generator(seed + 330_000)
    keys = _ids(episodes, model.config.symbol_count, rng, device)
    values = _ids(episodes, model.config.symbol_count, rng, device)
    state = model.initial_state(episodes, device=device)
    for _ in range(2):
        state, _ = model.step(state, evidence_event(keys, values, support=0.5))
    state, _ = model.step(state, context_event(keys))
    for _ in range(4):
        state, _ = model.step(state, None)
    rows: list[dict[str, Any]] = []
    samples: dict[str, torch.Tensor] = {}
    cursor = 0
    for count in distractor_counts:
        while cursor < count:
            noise_keys = _ids(episodes, model.config.symbol_count, rng, device)
            noise_values = _ids(episodes, model.config.symbol_count, rng, device)
            state, _ = model.step(state, evidence_event(noise_keys, noise_values, support=0.0))
            cursor += 1
        retained = _retention(model, state, keys, values)
        eval_state = model.reset_active(state.clone())
        event = context_event(keys)
        eval_state, output = model.step(eval_state, event, threshold=threshold)
        correct = output["emitted"] & output["emitted_content_id"].eq(values)
        rows += _diagnostic_rows(
            experiment="H_memory_arbitration", condition="target_context", model_name=model_name,
            seed=seed, state=eval_state, output=output, event=event, threshold=threshold,
            tick=0, target=values,
            extra={"distractor_count": torch.full((episodes,), count, device=device),
                   "stored_target_retention": retained,
                   "useful_emission": correct.float(),
                   "false_emission": (output["emitted"] & ~output["emitted_content_id"].eq(values)).float()},
        )
        absent_keys = (keys + 7).remainder(model.config.symbol_count)
        negative = model.reset_active(state.clone())
        negative_event = context_event(absent_keys)
        negative, negative_output = model.step(negative, negative_event, threshold=threshold)
        rows += _diagnostic_rows(
            experiment="H_memory_arbitration", condition="absent_context", model_name=model_name,
            seed=seed, state=negative, output=negative_output, event=negative_event,
            threshold=threshold, tick=0,
            extra={"distractor_count": torch.full((episodes,), count, device=device),
                   "stored_target_retention": retained,
                   "useful_emission": torch.zeros(episodes, device=device),
                   "false_emission": negative_output["emitted"].float()},
        )
        samples[f"H/{count}/query"] = output["query"][0].detach().cpu()
    return rows, samples


@torch.no_grad()
def self_output_audit(
    model: ContinuousETRCM,
    *, model_name: str, seed: int, threshold: float, episodes: int,
    tick_counts: list[int], device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    model.eval()
    rng = _generator(seed + 340_000)
    keys = _ids(episodes, model.config.symbol_count, rng, device)
    values = _ids(episodes, model.config.symbol_count, rng, device)
    state = model.initial_state(episodes, device=device)
    output = None
    for _ in range(4):
        state, output = model.step(state, evidence_event(keys, values, support=0.25), threshold=threshold)
    baseline_read = torch.einsum("bvk,bk->bv", state.F + state.M, model.target_key(keys))
    baseline_strength = (baseline_read * model.value_vector(values)).sum(-1).abs().clamp_min(1e-8)
    baseline_score = output["expression_score"].clone()
    rows: list[dict[str, Any]] = []
    samples: dict[str, torch.Tensor] = {}
    external_writes = torch.zeros(episodes, device=device)
    for tick in range(max(tick_counts) + 1):
        if tick > 0:
            state, output = model.step(state, None, threshold=threshold)
            external_writes += output["external_write_flag"].float()
        if tick in tick_counts:
            read = torch.einsum("bvk,bk->bv", state.F + state.M, model.target_key(keys))
            strength = (read * model.value_vector(values)).sum(-1).abs()
            rows += _diagnostic_rows(
                experiment="E_self_output_audit", condition="post_expression_no_external",
                model_name=model_name, seed=seed, state=state, output=output, event=None,
                threshold=threshold, tick=tick, target=values,
                extra={"post_output_external_write_count": external_writes,
                       "memory_evidence_strength": strength,
                       "memory_magnitude_relative_change": strength / baseline_strength - 1.0,
                       "expression_score_change": output["expression_score"] - baseline_score},
            )
            samples[f"E/{tick}/query"] = output["query"][0].detach().cpu()
    return rows, samples


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort()
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(len(values), device=values.device, dtype=torch.float32)
    return ranks


@torch.no_grad()
def thought_driven_persistence(
    model: ContinuousETRCM,
    *, model_name: str, seed: int, threshold: float, episodes: int, fact_count: int,
    internal_ticks: int, interference: int, device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    model.eval()
    rng = _generator(seed + 350_000)
    keys = torch.randint(0, model.config.symbol_count, (episodes, fact_count), generator=rng).to(device)
    values = torch.randint(0, model.config.symbol_count, (episodes, fact_count), generator=rng).to(device)
    state = model.initial_state(episodes, device=device)
    for fact in range(fact_count):
        state, _ = model.step(state, evidence_event(keys[:, fact], values[:, fact], support=0.25))
    usage = torch.zeros(episodes, fact_count, device=device)
    samples: dict[str, torch.Tensor] = {}
    for tick in range(internal_ticks):
        state, output = model.step(state, None, threshold=threshold)
        key_vectors = torch.stack([model.target_key(keys[:, fact]) for fact in range(fact_count)], 1)
        projection = torch.einsum("bd,bnd->bn", output["query"], key_vectors).abs()
        usage += projection * _norm(output["r_fast"])[:, None]
        if tick in {0, internal_ticks - 1}:
            samples[f"I/{tick}/query"] = output["query"][0].detach().cpu()
    for _ in range(interference):
        state, _ = model.step(
            state,
            evidence_event(
                _ids(episodes, model.config.symbol_count, rng, device),
                _ids(episodes, model.config.symbol_count, rng, device),
                support=0.0,
            ),
        )
    state = LearnedState(state.H, torch.zeros_like(state.F), state.M, state.tau, state.external_time)
    retention = torch.stack(
        [_retention(model, state, keys[:, fact], values[:, fact]) for fact in range(fact_count)], 1
    )
    correlations = []
    for episode in range(episodes):
        u, r = _rank(usage[episode]), _rank(retention[episode])
        correlations.append(torch.corrcoef(torch.stack([u, r]))[0, 1])
    correlation = torch.stack(correlations)
    rows: list[dict[str, Any]] = []
    for episode in range(episodes):
        for fact in range(fact_count):
            rows.append(
                {
                    "experiment": "I_thought_driven_persistence",
                    "condition": "matched_exposure_no_query",
                    "model": model_name,
                    "seed": seed,
                    "episode": episode,
                    "fact_index": fact,
                    "external_event_index": state.external_time,
                    "internal_tick": internal_ticks,
                    "event_type": "NULL_THEN_INTERFERENCE",
                    "query_ref": "tensor_samples.pt::I",
                    "h_norm": float(_norm(state.H)[episode]),
                    "f_norm": 0.0,
                    "m_norm": float(_norm(state.M)[episode]),
                    "usage_attribution": float(usage[episode, fact]),
                    "slow_retention": float(retention[episode, fact]),
                    "usage_retention_spearman": float(correlation[episode]),
                    "threshold": threshold,
                    "external_write_flag": 0.0,
                    "self_output_flag": 0.0,
                }
            )
    return rows, samples


@torch.no_grad()
def pattern_discovery(
    model: ContinuousETRCM,
    *, model_name: str, seed: int, threshold: float, episodes: int, device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    model.eval()
    rng = _generator(seed + 360_000)
    conditions = ["stable", "random_frequency", "accidental", "disappears", "reverses", "distribution_shift"]
    rows: list[dict[str, Any]] = []
    samples: dict[str, torch.Tensor] = {}
    length = 24
    for cindex, condition in enumerate(conditions):
        keys = _ids(episodes, model.config.symbol_count, rng, device)
        first = _ids(episodes, model.config.symbol_count, rng, device)
        second = (first + 1 + _ids(episodes, model.config.symbol_count - 1, rng, device)).remainder(model.config.symbol_count)
        state = model.initial_state(episodes, device=device)
        for tick in range(length):
            if condition == "stable":
                active, current = tick % 3 == 0, first
            elif condition == "random_frequency":
                active, current = tick % 2 == 0, _ids(episodes, model.config.symbol_count, rng, device)
            elif condition == "accidental":
                active, current = tick < 3, first
            elif condition == "disappears":
                active, current = tick < 5, first
            else:
                active, current = tick % 3 == 0, first if tick < 12 else second
            support = torch.full((episodes,), active, dtype=torch.bool, device=device)
            event = _mixed_stream_event(keys, current, support, rng, model.config.symbol_count)
            state, output = model.step(state, event, threshold=threshold)
            valid = torch.full((episodes,), condition == "stable" or (condition in {"reverses", "distribution_shift"} and tick >= 12), device=device)
            target = second if condition in {"reverses", "distribution_shift"} and tick >= 12 else first
            rows += _diagnostic_rows(
                experiment="B_pattern_discovery", condition=condition, model_name=model_name,
                seed=seed, state=state, output=output, event=event, threshold=threshold,
                tick=tick, target=target,
                extra={"valid_pattern_state": valid.float(),
                       "correct_emission": (output["emitted"] & output["emitted_content_id"].eq(target) & valid).float(),
                       "false_emission": (output["emitted"] & (~valid | ~output["emitted_content_id"].eq(target))).float()},
            )
        samples[f"B/{condition}/query"] = output["query"][0].detach().cpu()
    return rows, samples


@torch.no_grad()
def cross_time_association(
    model: ContinuousETRCM,
    *, model_name: str, seed: int, threshold: float, episodes: int, distractors: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    model.eval(); rng = _generator(seed + 370_000)
    a = _ids(episodes, model.config.symbol_count, rng, device)
    b = (a + 1).remainder(model.config.symbol_count)
    c = (b + 1).remainder(model.config.symbol_count)
    noise_keys = [_ids(episodes, model.config.symbol_count, rng, device) for _ in range(distractors)]
    noise_values = [_ids(episodes, model.config.symbol_count, rng, device) for _ in range(distractors)]

    def run_variant(current: ContinuousETRCM, condition: str):
        state = current.initial_state(episodes, device=device)
        state, _ = current.step(state, evidence_event(a, b))
        for keys, values in zip(noise_keys, noise_values):
            state, _ = current.step(state, evidence_event(keys, values, support=0.0))
        state, output = current.step(state, evidence_event(b, c), threshold=threshold)
        for _ in range(8):
            state, output = current.step(state, None, threshold=threshold)
        rows = _diagnostic_rows(
            experiment="C_cross_time_association", condition=condition, model_name=model_name,
            seed=seed, state=state, output=output, event=None, threshold=threshold, tick=8,
            target=c,
            extra={"cross_time_correct_emission": (output["emitted"] & output["emitted_content_id"].eq(c)).float(),
                   "distractor_count": torch.full((episodes,), distractors, device=device)},
        )
        return state, output, rows

    state, output, rows = run_variant(model, "A_to_B_gap_B_to_C")
    lesion = model.lesion(state, "slow")
    lesion, lesion_output = model.step(lesion, None, threshold=threshold)
    rows += _diagnostic_rows(
        experiment="C_cross_time_association", condition="M_lesion", model_name=model_name,
        seed=seed, state=lesion, output=lesion_output, event=None, threshold=threshold, tick=9,
        target=c,
        extra={"cross_time_correct_emission": (lesion_output["emitted"] & lesion_output["emitted_content_id"].eq(c)).float(),
               "distractor_count": torch.full((episodes,), distractors, device=device)},
    )
    if model_name == "B6_arbitration":
        gamma_zero = copy.deepcopy(model)

        def no_consolidation(self, fast, slow, query, access):
            return fast, slow, torch.zeros_like(fast)

        gamma_zero._consolidate = MethodType(no_consolidation, gamma_zero)
        _, _, variant_rows = run_variant(gamma_zero, "gamma_zero_intervention")
        rows += variant_rows

        random_query = copy.deepcopy(model)
        query_rng = torch.Generator(device=device).manual_seed(seed + 371_000)

        def random_query_fn(self, hidden):
            query = torch.randn(
                hidden.shape[0], self.config.key_dim,
                generator=query_rng, device=hidden.device, dtype=hidden.dtype,
            )
            return _normalize(query, dim=-1)

        random_query.query = MethodType(random_query_fn, random_query)
        _, _, variant_rows = run_variant(random_query, "random_query_intervention")
        rows += variant_rows
    return rows, {"C/query": output["query"][0].detach().cpu()}


@torch.no_grad()
def revision_after_expression(
    model: ContinuousETRCM,
    *, model_name: str, seed: int, threshold: float, episodes: int, device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    model.eval(); rng = _generator(seed + 380_000)
    keys = _ids(episodes, model.config.symbol_count, rng, device)
    old = _ids(episodes, model.config.symbol_count, rng, device)
    new = (old + 1 + _ids(episodes, model.config.symbol_count - 1, rng, device)).remainder(model.config.symbol_count)
    state = model.initial_state(episodes, device=device)
    rows: list[dict[str, Any]] = []
    for tick in range(4):
        state, output = model.step(state, evidence_event(keys, old, support=0.25), threshold=threshold)
    rows += _diagnostic_rows(
        experiment="F_revision", condition="old_supported", model_name=model_name, seed=seed,
        state=state, output=output, event=None, threshold=threshold, tick=4, target=old,
        extra={"revision_correct": torch.zeros(episodes, device=device), "new_evidence_count": torch.zeros(episodes, device=device)},
    )
    for count in range(1, 9):
        state, output = model.step(state, evidence_event(keys, new, support=0.25), threshold=threshold)
        rows += _diagnostic_rows(
            experiment="F_revision", condition="new_evidence", model_name=model_name, seed=seed,
            state=state, output=output, event=None, threshold=threshold, tick=4 + count, target=new,
            extra={"revision_correct": (output["emitted"] & output["emitted_content_id"].eq(new)).float(),
                   "new_evidence_count": torch.full((episodes,), count, device=device)},
        )
    return rows, {"F/query": output["query"][0].detach().cpu()}


@torch.no_grad()
def interleaved_streaming(
    model: ContinuousETRCM,
    *, model_name: str, seed: int, threshold: float, episodes: int, intervals: list[int],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    model.eval(); rng = _generator(seed + 390_000)
    keys = _ids(episodes, model.config.symbol_count, rng, device)
    values = _ids(episodes, model.config.symbol_count, rng, device)
    rows: list[dict[str, Any]] = []
    total_null = sum(intervals)
    for condition in ("block", "interleaved"):
        state = model.initial_state(episodes, device=device)
        if condition == "block":
            for _ in intervals:
                state, output = model.step(state, evidence_event(keys, values, support=0.25), threshold=threshold)
            for _ in range(total_null):
                state, output = model.step(state, None, threshold=threshold)
        else:
            for gap in intervals:
                state, output = model.step(state, evidence_event(keys, values, support=0.25), threshold=threshold)
                for _ in range(gap):
                    state, output = model.step(state, None, threshold=threshold)
        rows += _diagnostic_rows(
            experiment="G_interleaved_streaming", condition=condition, model_name=model_name,
            seed=seed, state=state, output=output, event=None, threshold=threshold,
            tick=len(intervals) + total_null, target=values,
            extra={"matched_event_content": torch.ones(episodes, device=device),
                   "matched_transition_count": torch.ones(episodes, device=device)},
        )
    return rows, {"G/query": output["query"][0].detach().cpu()}
