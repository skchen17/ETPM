"""Paired Stage 1.4 world-prediction and causal-intervention measurements."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.nn import functional as Fnn

from etrcm.stage1_1.model import LearnedState
from .interventions import (
    causal_usage_loss, lesion_direction, peripheral_swap, prediction_js,
    restore_workspace, pathological_self_output_write,
)
from etrcm.stage1_3.events import self_output_event
from .model import PredictiveETRCM
from .training import HORIZONS, prefix_index, run_prefix
from .world import FAMILIES, WorldBatch, generate_world


NULL_COUNTS = (0, 1, 2, 4, 8, 16)
CAUSAL_HORIZONS = (1, 2, 4, 8)


def _mean_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return Fnn.cross_entropy(logits, target, reduction="none")


def _row(
    model: PredictiveETRCM, state: LearnedState, output: dict[str, torch.Tensor] | None,
    *, run_id: str, seed: int, experiment: str, episode: int, family: str,
    external_step: int, internal_tick: int, horizon: int, event_type: str,
    target: int | None, prediction_loss: float | None, condition: str,
    compute_budget: int, **extra: object,
) -> dict[str, object]:
    index = episode
    source = output or {}

    def scalar(name: str) -> float | None:
        item = source.get(name)
        if item is None:
            return None
        return float(item[index].detach().cpu())

    def vector(name: str) -> list[float] | None:
        item = source.get(name)
        if item is None:
            return None
        return [float(value) for value in item[index].detach().cpu().tolist()]

    gates = source.get("gates")
    row: dict[str, object] = {
        "run_id": run_id, "model": model.mode, "seed": seed, "episode": episode,
        "world_family": family, "external_step": external_step,
        "internal_tick": internal_tick, "horizon": horizon, "event_type": event_type,
        "future_target": target, "prediction_loss": prediction_loss,
        "H_norm": float(state.H[index].detach().norm().cpu()),
        "F_norm": float(state.F[index].detach().norm().cpu()),
        "M_norm": float(state.M[index].detach().norm().cpu()),
        "q_F": vector("q_F"), "q_M": vector("q_M"),
        "r_F_norm": scalar("r_F_norm"), "r_M_norm": scalar("r_M_norm"),
        "normalized_r_F_norm": scalar("normalized_r_F_norm"),
        "normalized_r_M_norm": scalar("normalized_r_M_norm"),
        "g_F": float(gates[index, 0].detach().cpu()) if gates is not None else None,
        "g_M": float(gates[index, 1].detach().cpu()) if gates is not None else None,
        "effective_F_norm": scalar("effective_F_norm"),
        "effective_M_norm": scalar("effective_M_norm"),
        "transfer_norm": (
            float(source["transfer"][index].detach().norm().cpu()) if "transfer" in source else None
        ),
        "memory_item_id": None, "read_usage": None, "causal_usage": None,
        "retention": None, "intervention_condition": condition,
        "H_restoration_flag": False, "future_H_difference": None,
        "future_prediction_difference": None,
        "parameter_count": model.trainable_parameters(),
        "persistent_state_bytes": model.persistent_state_bytes(),
        "compute_budget": compute_budget,
    }
    row.update(extra)
    return row


def _roll_null(
    model: PredictiveETRCM, state: LearnedState, ticks: int,
    *, frozen: bool = False, random: bool = False,
) -> tuple[LearnedState, dict[str, torch.Tensor] | None]:
    output = None
    for _ in range(ticks):
        state, output = model.step(state, None, freeze_null_H=frozen, random_null_H=random)
    return state, output


@torch.no_grad()
def evaluate_A(
    model: PredictiveETRCM, *, seed: int, run_id: str, device: torch.device,
    batch: int = 32,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family_index, family in enumerate(FAMILIES):
        world = generate_world(
            family, batch=batch, length=16, seed=seed * 100_000 + 10_001 + family_index,
            device=device,
        )
        prefix = prefix_index(world, 5)
        base, diag = run_prefix(model, world, prefix=prefix, null_ticks=0)
        for condition in ("learned_NULL", "frozen_H_NULL", "random_NULL"):
            for k in NULL_COUNTS:
                state, output = _roll_null(
                    model, base.clone(), k, frozen=condition == "frozen_H_NULL",
                    random=condition == "random_NULL",
                )
                output = output or diag[-1]
                logits = model.predict_logits(state)
                for horizon in HORIZONS:
                    if prefix + horizon >= len(world.events):
                        continue
                    target = world.future(prefix, horizon)
                    loss = _mean_loss(logits[horizon], target)
                    for episode in range(batch):
                        rows.append(_row(
                            model, state, output, run_id=run_id, seed=seed,
                            experiment="A", episode=episode, family=family,
                            external_step=prefix, internal_tick=k, horizon=horizon,
                            event_type="NULL_EVENT" if k else "EXTERNAL_PREFIX",
                            target=int(target[episode]), prediction_loss=float(loss[episode]),
                            condition=condition, compute_budget=prefix + 1 + k,
                        ))
        if prefix + 2 < len(world.events):
            for k in NULL_COUNTS:
                before, before_diag = _roll_null(model, base.clone(), k)
                before, before_event = model.step(before, world.events[prefix + 1])
                after, after_event = model.step(base.clone(), world.events[prefix + 1])
                after, after_diag = _roll_null(model, after, k)
                target = world.future(prefix + 1, 1)
                for condition, state, output in (
                    ("idle_pre_event_then_event", before, before_event),
                    ("post_event_ticks", after, after_diag or after_event),
                ):
                    loss = _mean_loss(model.predict_logits(state)[1], target)
                    for episode in range(batch):
                        rows.append(_row(
                            model, state, output, run_id=run_id, seed=seed,
                            experiment="A_matched_compute", episode=episode, family=family,
                            external_step=prefix + 1, internal_tick=k, horizon=1,
                            event_type="EXTERNAL_THEN_NULL" if condition == "post_event_ticks" else "NULL_THEN_EXTERNAL",
                            target=int(target[episode]), prediction_loss=float(loss[episode]),
                            condition=condition, compute_budget=prefix + 2 + k,
                            future_latency_ticks=k if condition == "post_event_ticks" else 0,
                        ))
    return rows


def _before_bridge(
    model: PredictiveETRCM, world: WorldBatch,
) -> tuple[LearnedState, torch.Tensor]:
    state = model.initial_state(world.targets.shape[0], device=world.targets.device)
    usage = torch.zeros(world.targets.shape[0], device=world.targets.device)
    key = model.target_key(world.target_key)
    for step in range(world.bridge_index):
        state, output = model.step(state, world.events[step])
        similarity = (output["q_M"] * key).sum(-1).abs()
        usage = usage + similarity * output["effective_M_norm"]
    return state, usage


@torch.no_grad()
def evaluate_B(
    model: PredictiveETRCM, *, seed: int, run_id: str, device: torch.device,
    gap_counts: Iterable[int] = (128, 512, 2048), batch: int = 8,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for gap in gap_counts:
        world = generate_world(
            "long_gap_relation", batch=batch, length=16,
            seed=seed * 100_000 + 20_001 + gap, device=device, gap=gap,
        )
        base, usage = _before_bridge(model, world)
        conditions = ["full"]
        if model.mode == "B5_separate":
            conditions += ["M_lesion", "F_lesion", "random_q_M", "shuffled_M"]
        full_H = None
        for condition in conditions:
            state = base.clone()
            if condition == "M_lesion":
                state = model.lesion(state, "slow")
            elif condition == "F_lesion":
                state = model.lesion(state, "fast")
            elif condition == "shuffled_M":
                state = peripheral_swap(state, LearnedState(
                    state.H, state.F, state.M.roll(1, dims=0), state.tau, state.external_time
                ), channels="M")
            state, output = model.step(
                state, world.events[world.bridge_index],
                random_M_query=condition == "random_q_M",
            )
            # Re-activate for four idle ticks without any target-key hint.
            for _ in range(4):
                state, output = model.step(state, None, random_M_query=condition == "random_q_M")
            if condition == "full":
                full_H = state.H.clone()
            target = world.future(world.bridge_index, 1)
            loss = _mean_loss(model.predict_logits(state)[1], target)
            for episode in range(batch):
                rows.append(_row(
                    model, state, output, run_id=run_id, seed=seed,
                    experiment="B", episode=episode, family=world.family,
                    external_step=world.bridge_index, internal_tick=4, horizon=1,
                    event_type="BRIDGE_THEN_NULL", target=int(target[episode]),
                    prediction_loss=float(loss[episode]), condition=condition,
                    compute_budget=world.bridge_index + 5, distractor_count=gap,
                    memory_item_id=int(world.target_key[episode]),
                    read_usage=float(usage[episode]),
                    future_H_difference=(
                        float((state.H[episode] - full_H[episode]).norm())
                        if full_H is not None else 0.0
                    ),
                    target_key_available_to_model=False,
                ))
    return rows


@torch.no_grad()
def evaluate_CD(
    model: PredictiveETRCM, *, seed: int, run_id: str, device: torch.device,
    batch: int = 32,
) -> list[dict[str, object]]:
    if model.mode != "B5_separate":
        return []
    world = generate_world("long_gap_relation", batch=batch, length=16,
                           seed=seed * 100_000 + 30_001, device=device)
    base, _ = run_prefix(model, world, prefix=world.bridge_index, null_ticks=0)
    donor = LearnedState(base.H, base.F.roll(1, 0), base.M.roll(1, 0), base.tau, base.external_time)
    swapped = peripheral_swap(base, donor, channels="FM")
    assert torch.equal(base.H, swapped.H)
    a, b = base.clone(), swapped
    a_after_one, a_output = model.step(a, None)
    b_after_one, b_output = model.step(b, None)
    restored = restore_workspace(a_after_one, b_after_one)
    tracks: dict[str, tuple[LearnedState, dict[str, torch.Tensor]]] = {
        "intact": (a_after_one, a_output),
        "peripheral_swap": (b_after_one, b_output),
        "H_restored": (restored, b_output),
    }
    rows: list[dict[str, object]] = []
    for tick in range(1, 9):
        if tick > 1:
            tracks = {name: model.step(state, None) for name, (state, _) in tracks.items()}
        if tick not in CAUSAL_HORIZONS:
            continue
        intact = tracks["intact"][0]
        intact_logits = model.predict_logits(intact)[1]
        target = world.future(world.bridge_index, 1)
        for condition in ("peripheral_swap", "H_restored"):
            state, output = tracks[condition]
            logits = model.predict_logits(state)[1]
            loss = _mean_loss(logits, target)
            h_diff = (state.H - intact.H).norm(dim=(-2, -1))
            js = prediction_js(intact_logits, logits)
            for episode in range(batch):
                rows.append(_row(
                    model, state, output, run_id=run_id, seed=seed,
                    experiment="C" if condition == "peripheral_swap" else "D",
                    episode=episode, family=world.family,
                    external_step=world.bridge_index, internal_tick=tick, horizon=1,
                    event_type="NULL_EVENT", target=int(target[episode]),
                    prediction_loss=float(loss[episode]), condition=condition,
                    compute_budget=world.bridge_index + 1 + tick,
                    H_restoration_flag=condition == "H_restored",
                    future_H_difference=float(h_diff[episode]),
                    future_prediction_difference=float(js[episode]),
                    baseline_intact_loss=float(_mean_loss(intact_logits, target)[episode]),
                    same_H_initial_exact=True,
                ))
    return rows


@torch.no_grad()
def evaluate_EF(
    model: PredictiveETRCM, *, seed: int, run_id: str, device: torch.device,
    batch: int = 32, gap: int = 128,
) -> list[dict[str, object]]:
    if model.mode != "B5_separate":
        return []
    world = generate_world("long_gap_relation", batch=batch, length=16,
                           seed=seed * 100_000 + 40_001, device=device, gap=gap)
    base, usage = _before_bridge(model, world)
    key = model.target_key(world.target_key)
    lesion = lesion_direction(base, key, channel="M")
    target = world.future(world.bridge_index, 1)
    full, full_output = model.step(base, world.events[world.bridge_index])
    ablated, lesion_output = model.step(lesion, world.events[world.bridge_index])
    for _ in range(2):
        full, full_output = model.step(full, None)
        ablated, lesion_output = model.step(ablated, None)
    full_logits = model.predict_logits(full)[1]
    lesion_logits = model.predict_logits(ablated)[1]
    cu = causal_usage_loss(full_logits, lesion_logits, target)
    future_h = (full.H - ablated.H).norm(dim=(-2, -1))
    full_loss = _mean_loss(full_logits, target)
    # F is scrubbed after the entire interference stream; retention is M-only.
    m_read = torch.einsum("bvk,bk->bv", base.M, key)
    ideal = model.value_vector(world.metadata["A"])
    retention = Fnn.cosine_similarity(m_read, ideal, dim=-1, eps=1e-8)
    rows: list[dict[str, object]] = []
    for episode in range(batch):
        rows.append(_row(
            model, full, full_output, run_id=run_id, seed=seed,
            experiment="E_F", episode=episode, family=world.family,
            external_step=world.bridge_index, internal_tick=2, horizon=1,
            event_type="BRIDGE_THEN_NULL", target=int(target[episode]),
            prediction_loss=float(full_loss[episode]), condition="M_direction_lesion_paired",
            compute_budget=world.bridge_index + 3,
            memory_item_id=int(world.target_key[episode]),
            exposure_count=int(world.exposure_count[episode]),
            read_usage=float(usage[episode]), causal_usage=float(cu[episode]),
            causal_usage_H=float(future_h[episode]),
            retention=float(retention[episode]),
            lesion_prediction_loss=float(_mean_loss(lesion_logits, target)[episode]),
            memory_item_key_overlap_caveat=True,
        ))
    return rows


@torch.no_grad()
def evaluate_G(
    model: PredictiveETRCM, *, seed: int, run_id: str, device: torch.device,
    causal_rows: list[dict[str, object]], batch: int = 32, gap: int = 128,
) -> list[dict[str, object]]:
    """Direction-specific consolidation block after offline high/low matching.

    Exact exposure strata are matched first; nearest read-usage and memory
    magnitude is second. A control item may be reused to reach 16 pairs;
    therefore seed, not pair, remains the replication unit.
    """
    if model.mode != "B5_separate":
        return []
    world = generate_world("long_gap_relation", batch=batch, length=16,
                           seed=seed * 100_000 + 40_001, device=device, gap=gap)
    key = model.target_key(world.target_key)
    state_full = model.initial_state(batch, device=device)
    state_block = model.initial_state(batch, device=device)
    for step in range(world.bridge_index + 1):
        state_full, _ = model.step(state_full, world.events[step])
        state_block, _ = model.step(
            state_block, world.events[step], block_transfer_key=key
        )
    for _ in range(2):
        state_full, _ = model.step(state_full, None)
        state_block, _ = model.step(state_block, None, block_transfer_key=key)
    target = world.future(world.bridge_index, 1)
    normal_loss = _mean_loss(model.predict_logits(state_full)[1], target)
    blocked_loss = _mean_loss(model.predict_logits(state_block)[1], target)
    items = [row for row in causal_rows if row["experiment"] == "E_F"]
    if len(items) != batch:
        raise ValueError("causal item count does not match consolidation intervention")
    ordered = sorted(range(batch), key=lambda i: float(items[i]["causal_usage"]))
    lows, highs = ordered[:batch // 2], ordered[batch // 2:]
    pairs: list[tuple[int, int]] = []
    for high in reversed(highs):
        same_exposure = [low for low in lows if items[low]["exposure_count"] == items[high]["exposure_count"]]
        candidates = same_exposure or lows
        low = min(candidates, key=lambda i: (
            abs(float(items[i]["read_usage"]) - float(items[high]["read_usage"])),
            abs(float(items[i]["M_norm"]) - float(items[high]["M_norm"])),
        ))
        pairs.append((high, low))
    rows: list[dict[str, object]] = []
    for pair, (high, low) in enumerate(pairs):
        for stratum, episode in (("high_CU", high), ("low_CU", low)):
            rows.append(_row(
                model, state_block, None, run_id=run_id, seed=seed,
                experiment="G", episode=episode, family=world.family,
                external_step=world.bridge_index, internal_tick=2, horizon=1,
                event_type="CONSOLIDATION_BLOCKED", target=int(target[episode]),
                prediction_loss=float(blocked_loss[episode]), condition=stratum,
                compute_budget=world.bridge_index + 3,
                pair_id=pair, exposure_count=int(world.exposure_count[episode]),
                read_usage=items[episode]["read_usage"],
                causal_usage=items[episode]["causal_usage"],
                full_prediction_loss=float(normal_loss[episode]),
                consolidation_block_harm=float(blocked_loss[episode] - normal_loss[episode]),
                matched_exposure_exact=items[high]["exposure_count"] == items[low]["exposure_count"],
                matched_with_replacement=True,
            ))
    return rows


@torch.no_grad()
def evaluate_safety(
    model: PredictiveETRCM, *, seed: int, run_id: str, device: torch.device,
    batch: int = 32,
) -> list[dict[str, object]]:
    if model.mode != "B5_separate":
        return []
    content = torch.arange(batch, device=device).remainder(model.config.symbol_count)
    event = self_output_event(content)
    safe = model.initial_state(batch, device=device)
    bad = model.initial_state(batch, device=device)
    key = model.target_key(content)
    rows: list[dict[str, object]] = []
    for tick in range(1, 9):
        safe, safe_out = model.step(safe, event)
        bad, update = pathological_self_output_write(model, bad, content)
        bad, bad_out = model.step(bad, event)
        for condition, state, output, write_norm in (
            ("safe_SELF_OUTPUT", safe, safe_out, 0.0),
            ("B_bad_external_write", bad, bad_out, float(update.norm(dim=(-2, -1)).mean())),
        ):
            read = torch.einsum("bvk,bk->bv", state.F + state.M, key).norm(dim=-1)
            propensity = torch.sigmoid(4 * read - 2)
            for episode in range(batch):
                rows.append(_row(
                    model, state, output, run_id=run_id, seed=seed,
                    experiment="safety", episode=episode, family="self_output_control",
                    external_step=0, internal_tick=tick, horizon=1,
                    event_type="SELF_OUTPUT", target=None, prediction_loss=None,
                    condition=condition, compute_budget=tick,
                    self_write_norm=write_norm,
                    memory_readout_norm=float(read[episode]),
                    repeat_output_propensity_proxy=float(propensity[episode]),
                    external_write_count=int(output["external_write_flag"][episode]),
                    propensity_is_fixed_diagnostic_proxy=True,
                ))
    return rows
