"""Frozen Stage 2D behavioral, causal, NULL and continuous evaluation."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_3.events import self_output_event
from etrcm.stage2c.world import NUISANCE, tensor_ids, unrelated_token
from etrcm.stage2d.model import (Stage2DModel, behavioral_metrics, parameter_hash,
                                 play, probe_prob, state_intervention, state_norms)
from etrcm.stage2d.protocol import (CONTINUOUS_TICKS, DELAYS, FORMATION_N, FROZEN,
                                    HANDOFF_TIMES, NULL_TICKS, OPPOSE_FRACTIONS,
                                    P_LEVELS, REVERSAL_R)
from etrcm.stage2d.world import bayes_p_z0, paired_experiences


def load_model(path: Path, device: str) -> tuple[Stage2DModel, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = Stage2DModel(variant=checkpoint["variant"], gamma=checkpoint["gamma"],
                         rho_fast=checkpoint["rho_fast"], rho_slow=checkpoint["rho_slow"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def probe_rows(seed: int, reps: int, index: int = 99999):
    return paired_experiences(seed + 700001, reps, index, 0.65, split="novel")


def measure(model, state, seed, reps, *, clamp: str = "none") -> dict:
    probability = probe_prob(model, state, probe_rows(seed, reps), read_clamp=clamp)
    result = behavioral_metrics(probability, reps)
    result.pop("prob", None)
    result["state_norms"] = state_norms(state)
    return result


def advance_experience(model, state, seed, reps, index, p, *, noise=False,
                       oppose_fraction=0.0, clamp="none"):
    rows = paired_experiences(seed, reps, index, p, matched_noise=noise,
                              oppose_fraction=oppose_fraction)
    return play(model, state, rows, read_clamp=clamp)[0], rows


def delay_one(model, state, seed, reps, index, *, clamp="none"):
    rng = random.Random(seed * 1000003 + index * 97 + 31)
    values = [rng.choice(NUISANCE) for _ in range(reps)]
    ids = tensor_ids(values + values, state.H.device)
    return model.step(state, unrelated_token(ids), read_clamp=clamp)


def policy_a0(prob: torch.Tensor) -> torch.Tensor:
    entropy = -(prob.clamp_min(1e-10) * prob.clamp_min(1e-10).log()).sum(-1)
    return torch.softmax(-entropy / 0.35, -1)[:, 0]


@torch.no_grad()
def formation(model, seed, reps) -> dict:
    all_levels = {}
    for p in P_LEVELS:
        state = model.initial_state(2 * reps, next(model.parameters()).device)
        supports = [[] for _ in range(2 * reps)]
        curve = {}
        for index in range(max(FORMATION_N) + 1):
            if index in FORMATION_N:
                prob = probe_prob(model, state, probe_rows(seed, reps, index))
                metrics = behavioral_metrics(prob, reps)
                pa0 = policy_a0(prob).cpu()
                posterior = torch.tensor([bayes_p_z0(bits, p) for bits in supports])
                curve[str(index)] = {
                    "BS": metrics["behavioral_separation_entropy"],
                    "p_action_A_zA": float(pa0[:reps].mean()),
                    "p_action_A_zB": float(pa0[reps:].mean()),
                    "bayes_p_zA_mean_zA": float(posterior[:reps].mean()),
                    "bayes_p_zA_mean_zB": float(posterior[reps:].mean()),
                    "calibration_brier": float(((pa0 - posterior) ** 2).mean()),
                    "state_norms": state_norms(state),
                }
            if index == max(FORMATION_N):
                break
            state, rows = advance_experience(model, state, seed + int(p * 1000), reps,
                                             index, p)
            for row_index, row in enumerate(rows):
                supports[row_index].append(row.support_bit)
        threshold = FROZEN.acquisition_threshold
        acquire = next((n for n in FORMATION_N if abs(curve[str(n)]["BS"]) >= threshold), None)
        all_levels[f"{p:.2f}"] = {"curve": curve, "T_acquire": acquire}
    return all_levels


def formed_state(model, seed, reps, p=0.65, n=None):
    n = FROZEN.formation_length if n is None else n
    state = model.initial_state(2 * reps, next(model.parameters()).device)
    for index in range(n):
        state, _ = advance_experience(model, state, seed, reps, index, p)
    return state


@torch.no_grad()
def training_health(model, checkpoint, seed, reps) -> dict:
    state = formed_state(model, seed + 170003, reps)
    prob = probe_prob(model, state, probe_rows(seed, reps))
    aggregate = torch.stack([prob[:reps].mean(0), prob[reps:].mean(0)])
    target = torch.empty_like(aggregate)
    p = 0.65
    for z in (0, 1):
        for action in (0, 1):
            p0 = p if action == z else 1 - p
            target[z, action, 0] = p0
            target[z, action, 1:] = (1 - p0) / 3
    ce = float(-(target * aggregate.clamp_min(1e-9).log()).sum(-1).mean())
    marginal_ce = -(.5 * math.log(.5) + .5 * math.log(1 / 6))
    metrics = behavioral_metrics(prob, reps)
    def centroid_accuracy(value):
        flat = value.flatten(1).float().cpu()
        half = max(1, reps // 2)
        c0, c1 = flat[:half].mean(0), flat[reps:reps + half].mean(0)
        test = torch.cat([flat[half:reps], flat[reps + half:]])
        labels = torch.cat([torch.zeros(reps - half), torch.ones(reps - half)])
        pred = ((test - c1).square().sum(-1) < (test - c0).square().sum(-1)).float()
        return float((pred == labels).float().mean()) if len(labels) else float("nan")
    _, trace = model.step(state.clone(), None)
    return {"action_TV": sum(metrics["tv_action"]) / 2,
            "history_TV": sum(metrics["tv_history"]) / 2,
            "interaction_y0": metrics["interaction_y0"],
            "conditional_CE": ce, "marginal_CE": marginal_ce, "CFA": marginal_ce - ce,
            "evaluator_hash": checkpoint["protected_evaluator_hash"],
            "latent_probe_accuracy": {name: centroid_accuracy(getattr(state, name))
                                      for name in ("H", "F", "M")},
            "read_gate_mean": trace["gates"].mean(0).cpu().tolist(),
            "q_F_norm": float(trace["q_F"].norm(dim=-1).mean()),
            "q_M_norm": float(trace["q_M"].norm(dim=-1).mean()),
            "state_norms": state_norms(state)}


@torch.no_grad()
def persistence(model, seed, reps) -> dict:
    state = formed_state(model, seed, reps)
    curve = {}
    previous = 0
    for target in DELAYS:
        for index in range(previous, target):
            state, _ = delay_one(model, state, seed + 200003, reps, index)
        curve[str(target)] = measure(model, state, seed + target, reps)
        previous = target
    bs0 = abs(curve["0"]["behavioral_separation_entropy"])
    half = next((d for d in DELAYS if d > 0 and
                 abs(curve[str(d)]["behavioral_separation_entropy"]) <= 0.5 * bs0), None)
    return {"curve": curve, "half_life": half if half is not None else f">{max(DELAYS)}",
            "BS0": bs0}


@torch.no_grad()
def revision(model, seed, reps) -> dict:
    original = formed_state(model, seed, reps)
    state = original.clone()
    curve = {"0": measure(model, state, seed, reps)}
    previous = 0
    for target in REVERSAL_R[1:]:
        for index in range(previous, target):
            state, _ = advance_experience(model, state, seed + 300007, reps, index,
                                          0.65, oppose_fraction=1.0)
        curve[str(target)] = measure(model, state, seed + target, reps)
        previous = target
    initial = curve["0"]["behavioral_separation_entropy"]
    change = next((r for r in REVERSAL_R[1:] if
                   abs(curve[str(r)]["behavioral_separation_entropy"]) <= 0.5 * abs(initial)), None)
    neutral = next((r for r in REVERSAL_R[1:] if
                    curve[str(r)]["behavioral_separation_entropy"] * initial <= 0), None)
    reverse = next((r for r in REVERSAL_R[1:] if
                    curve[str(r)]["behavioral_separation_entropy"] * initial < 0 and
                    abs(curve[str(r)]["behavioral_separation_entropy"]) >= FROZEN.formed_threshold), None)
    fractions = {}
    for fraction in OPPOSE_FRACTIONS:
        mixed = original.clone()
        for index in range(64):
            mixed, _ = advance_experience(model, mixed, seed + 330013, reps, index,
                                          0.65, oppose_fraction=fraction)
        fractions[str(fraction)] = measure(model, mixed, seed + int(100 * fraction), reps)
    return {"curve": curve, "T_change": change, "T_neutral": neutral,
            "T_reverse": reverse if reverse is not None else "NOT_REVERSED_WITHIN_RANGE",
            "contradictory_fraction": fractions}


def run_history(model, seed, reps, n, *, noise=False, p=0.65):
    state = model.initial_state(2 * reps, next(model.parameters()).device)
    for index in range(n):
        state, _ = advance_experience(model, state, seed, reps, index, p, noise=noise)
    return state


@torch.no_grad()
def selectivity(model, seed, reps) -> dict:
    result = {"matched": {}, "rare_vs_frequent": {}}
    for noise in (False, True):
        state = run_history(model, seed + 400009, reps, 32, noise=noise)
        immediate = measure(model, state, seed, reps)
        for index in range(FROZEN.persistence_delay):
            state, _ = delay_one(model, state, seed + 410009, reps, index)
        delayed = measure(model, state, seed + 1, reps)
        result["matched"]["noise" if noise else "predictive"] = {
            "immediate": immediate, "delayed": delayed}
    for n in (4, 8):
        state = run_history(model, seed + 420001, reps, n, noise=False, p=0.70)
        result["rare_vs_frequent"][f"useful_{n}"] = measure(model, state, seed, reps)
    for n in (32, 64, 128):
        state = run_history(model, seed + 420001, reps, n, noise=True, p=0.70)
        result["rare_vs_frequent"][f"noise_{n}"] = measure(model, state, seed, reps)
    return result


def _handoff_trajectory(model, seed, reps, *, read_segment=None, clamp="none",
                        state_point=None, operation=None, which=None):
    state = model.initial_state(2 * reps, next(model.parameters()).device)
    for index in range(32):
        active = ((read_segment == "early_formation" and index < 8) or
                  (read_segment == "late_formation" and index >= 24))
        state, _ = advance_experience(model, state, seed, reps, index, 0.65,
                                      clamp=clamp if active else "none")
        if state_point == "early_formation" and index == 7:
            state = state_intervention(state, which, operation, reps)
        if state_point == "late_formation" and index == 27:
            state = state_intervention(state, which, operation, reps)
    for k in range(4):
        active = read_segment == "post_formation"
        state, _ = model.step(state, None, read_clamp=clamp if active else "none")
    if state_point == "post_formation":
        state = state_intervention(state, which, operation, reps)
    for index in range(FROZEN.persistence_delay):
        active = ((read_segment == "mid_delay" and 248 <= index < 252) or
                  (read_segment == "late_delay" and 496 <= index < 500))
        state, _ = delay_one(model, state, seed + 500009, reps, index,
                             clamp=clamp if active else "none")
        if state_point == "mid_delay" and index == 249:
            state = state_intervention(state, which, operation, reps)
        if state_point == "late_delay" and index == 499:
            state = state_intervention(state, which, operation, reps)
    if state_point == "probe":
        state = state_intervention(state, which, operation, reps)
    probe_clamp = clamp if read_segment == "probe" else "none"
    return measure(model, state, seed + 500, reps, clamp=probe_clamp)


@torch.no_grad()
def handoff(model, seed, reps) -> dict:
    baseline = _handoff_trajectory(model, seed, reps)
    base_bs = baseline["behavioral_separation_entropy"]
    reads, states = {}, {}
    for time_name in HANDOFF_TIMES:
        reads[time_name] = {}
        states[time_name] = {}
        for which in ("F", "M", "FM"):
            item = _handoff_trajectory(model, seed, reps, read_segment=time_name, clamp=which)
            item["causal_effect"] = base_bs - item["behavioral_separation_entropy"]
            reads[time_name][which] = item
            states[time_name][which] = {}
            for operation in ("zero", "swap"):
                strong = _handoff_trajectory(model, seed, reps, state_point=time_name,
                                             operation=operation, which=which)
                strong["causal_effect"] = base_bs - strong["behavioral_separation_entropy"]
                states[time_name][which][operation] = strong
    return {"baseline": baseline, "read_mediation": reads, "state_destruction": states}


@torch.no_grad()
def null_consolidation(model, seed, reps) -> dict:
    formed = formed_state(model, seed + 600011, reps)
    result = {}
    for target in NULL_TICKS:
        state = formed.clone()
        transfer = 0.0
        for _ in range(target):
            state, trace = model.step(state, None)
            transfer += float(trace["transfer"].norm(dim=(-2, -1)).mean())
            if bool(trace["external_write_flag"].any()):
                raise AssertionError("NULL produced external write")
        immediate = measure(model, state, seed, reps)
        causal = {}
        if model.variant not in {"no_memory", "gru"}:
            for clamp in ("F", "M", "FM"):
                val = measure(model, state, seed, reps, clamp=clamp)
                causal[clamp] = immediate["behavioral_separation_entropy"] - val["behavioral_separation_entropy"]
        for index in range(FROZEN.persistence_delay):
            state, _ = delay_one(model, state, seed + 610001, reps, index)
        result[str(target)] = {"immediate": immediate,
                               "delayed": measure(model, state, seed + 1, reps),
                               "cumulative_transfer_norm": transfer,
                               "probe_read_causal_effects": causal}
    return result


@torch.no_grad()
def continuous(model, seed, reps=2) -> dict:
    state = model.initial_state(2 * reps, next(model.parameters()).device)
    result = {}; first_h100 = None; first_h1000 = None; nonfinite = None
    writes = {"predictive": 0, "noise": 0, "unrelated": 0, "null": 0,
              "probe": 0, "self_output": 0}
    previous_h = state.H.clone()
    delta_sum = 0.0
    for tick in range(1, max(CONTINUOUS_TICKS) + 1):
        kind = tick % 20
        if kind < 5:
            state, _ = advance_experience(model, state, seed + 700001, reps, tick, 0.65)
            writes["predictive"] += 1
        elif kind < 10:
            state, _ = advance_experience(model, state, seed + 700001, reps, tick, 0.65, noise=True)
            writes["noise"] += 1
        elif kind < 14:
            state, _ = model.step(state, None); writes["null"] += 1
        elif kind < 18:
            state, _ = delay_one(model, state, seed + 700003, reps, tick); writes["unrelated"] += 1
        elif kind == 18:
            # A read-only abstract probe is computed on a cloned state.
            measure(model, state, seed + tick, reps); writes["probe"] += 1
        else:
            ids = torch.full((2 * reps,), tick % 24, dtype=torch.long, device=state.H.device)
            state, trace = model.step(state, self_output_event(ids)); writes["self_output"] += 1
            if bool(trace["external_write_flag"].any()):
                raise AssertionError("SELF_OUTPUT produced external write")
        delta_sum += float((state.H - previous_h).norm(dim=(-2, -1)).mean())
        previous_h = state.H.clone()
        hmax = float(state.H.norm(dim=(-2, -1)).max())
        first_h100 = tick if first_h100 is None and hmax > 100 else first_h100
        first_h1000 = tick if first_h1000 is None and hmax > 1000 else first_h1000
        finite = all(bool(torch.isfinite(getattr(state, name)).all()) for name in ("H", "F", "M"))
        if not finite:
            nonfinite = tick; break
        if tick in CONTINUOUS_TICKS:
            result[str(tick)] = {"behavior": measure(model, state, seed + tick, reps),
                                 "state_norms": state_norms(state),
                                 "internal_tau": state.tau,
                                 "external_time": state.external_time,
                                 "mean_H_delta": delta_sum / tick,
                                 "stream_counts": writes.copy()}
    return {"milestones": result, "first_H_gt_100": first_h100,
            "first_H_gt_1000": first_h1000, "first_nonfinite": nonfinite,
            "completed_ticks": tick}


@torch.no_grad()
def evaluate(args):
    torch.set_num_threads(1)
    model, checkpoint = load_model(args.checkpoint, args.device)
    before = parameter_hash(model)
    start = time.monotonic()
    result = {
        "seed": checkpoint["seed"], "arm": checkpoint["arm"],
        "variant": checkpoint["variant"], "scope": args.scope,
        "config": {key: checkpoint[key] for key in ("gamma", "rho_fast", "rho_slow", "train_p")},
        "protocol": FROZEN.to_dict(),
        "training_health": training_health(model, checkpoint, args.seed, args.replicates),
        "formation": formation(model, args.seed, args.replicates),
        "persistence": persistence(model, args.seed, args.replicates),
        "revision": revision(model, args.seed, args.replicates),
        "selectivity": selectivity(model, args.seed, args.replicates),
    }
    if args.scope in {"full", "phase"} and model.variant not in {"no_memory", "gru"}:
        result["handoff"] = handoff(model, args.seed, args.replicates)
        if args.scope == "full":
            result["null_consolidation"] = null_consolidation(model, args.seed, args.replicates)
            result["continuous"] = continuous(model, args.seed)
    elif args.scope == "control":
        result["null_consolidation"] = null_consolidation(model, args.seed, args.replicates)
        result["continuous"] = continuous(model, args.seed)
    result["parameter_hash_before"] = before
    result["parameter_hash_after"] = parameter_hash(model)
    result["evaluation_parameter_frozen"] = before == result["parameter_hash_after"]
    result["elapsed_s"] = time.monotonic() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"COMPLETED": {"seed": args.seed, "variant": checkpoint["variant"],
                                    "scope": args.scope, "elapsed_s": result["elapsed_s"]}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replicates", type=int, default=32)
    parser.add_argument("--scope", choices=("dev", "control", "phase", "full"), default="full")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    evaluate(parser.parse_args())
