"""Pure routing interventions and diagnostic measurements for Stage 2D.1.

This module deliberately reuses the frozen Stage 2D transition primitives.
The only optional changes are read-path interventions; parameters and the
stored input state are never mutated in place.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import torch
from torch.nn import functional as Fnn

from etrcm.stage1_1.model import LearnedState
from etrcm.stage2c.world import ABSTRACT, ACTION, context_token, outcome_token, tensor_ids
from etrcm.stage2d.model import Stage2DModel, behavioral_metrics
from etrcm.stage2d.world import NoisyExperience, observed_only, paired_experiences


def routed_step(model: Stage2DModel, state: LearnedState, event, *,
                alpha_f: float = 1.0, alpha_m: float = 1.0,
                gate_m: float | None = None, gate_floor_m: float | None = None):
    """Frozen transition with a read-only routing intervention.

    ``alpha_f``/``alpha_m`` scale raw reads. ``gate_m`` replaces the normalized
    F/M mixture; ``gate_floor_m`` smoothly lifts the learned M gate and is used
    only by preregistered training warm-ups. Memory laws remain unchanged.
    """
    core = model.core
    state.validate()
    batch = state.H.shape[0]
    encoded = torch.zeros(batch, core.config.hidden_dim, device=state.H.device, dtype=state.H.dtype)
    fast, slow = state.F, state.M
    external_update = torch.zeros_like(fast)
    external_write_flag = torch.zeros(batch, dtype=torch.bool, device=state.H.device)
    external_event = 0
    if event is not None:
        event.validate(batch)
        encoded = core.event_encoder(event).to(state.H.dtype)
        if bool(event.write_mask.any()):
            fast, slow, external_update = core._external_write(fast, slow, event)
        external_write_flag = event.write_mask
        external_event = int((~event.self_output_mask).any())
    event_slots = (core.event_to_slots(encoded).view(batch, core.config.latent_slots, core.config.hidden_dim)
                   if event is not None else torch.zeros_like(state.H))
    h_pre = state.H + event_slots
    q_f, q_m = core._queries(h_pre)
    memory_r_f = torch.einsum("bvk,bk->bv", fast, q_f)
    memory_r_m = torch.einsum("bvk,bk->bv", slow, q_m)
    r_f, r_m = alpha_f * memory_r_f, alpha_m * memory_r_m
    n_f, n_m = core.fast_read_norm(r_f), core.slow_read_norm(r_m)
    learned_gates = torch.softmax(core.two_way_gate(torch.cat([core._pool(h_pre), encoded], -1)), -1)
    gates = learned_gates
    if gate_m is not None:
        gm = torch.full_like(learned_gates[:, 1:2], float(gate_m))
        gates = torch.cat([1.0 - gm, gm], -1)
    elif gate_floor_m is not None:
        gm = float(gate_floor_m) + (1.0 - float(gate_floor_m)) * learned_gates[:, 1:2]
        gates = torch.cat([1.0 - gm, gm], -1)
    read = gates[:, :1] * n_f + gates[:, 1:] * n_m
    core_input = torch.cat([core.norm(h_pre),
                            read[:, None, :].expand(-1, core.config.latent_slots, -1),
                            encoded[:, None, :].expand(-1, core.config.latent_slots, -1)], -1)
    h_new = h_pre + torch.sigmoid(core.core_gate(core_input)) * core.core_out(Fnn.silu(core.core_in(core_input)))
    access = torch.sigmoid(core.access_head(core._pool(h_new))).squeeze(-1)
    total_before = fast + slow
    fast, slow, transfer = core._consolidate(fast, slow, q_f, access)
    conservation_error = (fast + slow - total_before).norm(dim=(-2, -1))
    fast, slow = core.config.rho_fast * fast, core.config.rho_slow * slow
    next_state = LearnedState(h_new, fast, slow, state.tau + 1, state.external_time + external_event)
    trace = {
        "read": read, "q_F": q_f, "q_M": q_m, "r_F": r_f, "r_M": r_m,
        "memory_r_F": memory_r_f, "memory_r_M": memory_r_m,
        "normalized_r_F": n_f, "normalized_r_M": n_m,
        "gates": gates, "learned_gates": learned_gates,
        "r_F_norm": r_f.norm(dim=-1), "r_M_norm": r_m.norm(dim=-1),
        "effective_F_norm": (gates[:, :1] * n_f).norm(dim=-1),
        "effective_M_norm": (gates[:, 1:] * n_m).norm(dim=-1),
        "access": access, "transfer": transfer, "external_update": external_update,
        "external_write_flag": external_write_flag, "conservation_error": conservation_error,
        "H_delta_norm": (h_new - state.H).norm(dim=(-2, -1)),
        "decay_F_norm": ((1.0 - core.config.rho_fast) * fast).norm(dim=(-2, -1)),
        "decay_M_norm": ((1.0 - core.config.rho_slow) * slow).norm(dim=(-2, -1)),
    }
    return next_state, trace


def diagnostic_step(model, state, event, **route):
    return routed_step(model, state, event, **route)


def play_routed(model: Stage2DModel, state: LearnedState, rows: list[NoisyExperience], **route):
    items = observed_only(rows)
    device = state.H.device
    traces = []
    fields = (
        torch.full((len(items),), ABSTRACT, dtype=torch.long, device=device),
        tensor_ids([x.color for x in items], device), tensor_ids([x.shape for x in items], device),
        tensor_ids([x.nuisance for x in items], device),
    )
    for label, ids in zip(("abstract", "color", "shape", "nuisance"), fields):
        state, trace = diagnostic_step(model, state, context_token(ids), **route)
        traces.append((label, trace))
    action = tensor_ids([x.action for x in items], device)
    action_ids = tensor_ids([ACTION[x.action] for x in items], device)
    state, trace = diagnostic_step(model, state, context_token(action_ids), **route)
    traces.append(("action", trace))
    logits = model.logits(state, action)
    target = tensor_ids([x.outcome - 4 for x in items], device)
    loss = Fnn.cross_entropy(logits, target)
    state, trace = diagnostic_step(model, state,
                                   outcome_token(tensor_ids([x.outcome for x in items], device), write=True),
                                   **route)
    traces.append(("outcome", trace))
    return state, loss, traces


def probe_prob_routed(model, state, rows, **route):
    items = observed_only(rows); device = state.H.device; context = state.clone()
    fields = (torch.full((len(items),), ABSTRACT, dtype=torch.long, device=device),
              tensor_ids([x.color for x in items], device), tensor_ids([x.shape for x in items], device),
              tensor_ids([x.nuisance for x in items], device))
    for ids in fields:
        context, _ = diagnostic_step(model, context, context_token(ids), **route)
    logits = []
    for action in (0, 1):
        ids = torch.full((len(items),), ACTION[action], dtype=torch.long, device=device)
        branch, _ = diagnostic_step(model, context.clone(), context_token(ids), **route)
        logits.append(model.logits(branch, torch.full((len(items),), action, dtype=torch.long, device=device)))
    return torch.stack(logits, 1).softmax(-1)


def centroid_accuracy(value: torch.Tensor, reps: int) -> float:
    flat = value.flatten(1).float().cpu(); half = max(1, reps // 2)
    c0, c1 = flat[:half].mean(0), flat[reps:reps + half].mean(0)
    test = torch.cat([flat[half:reps], flat[reps + half:]])
    labels = torch.cat([torch.zeros(reps - half), torch.ones(reps - half)])
    if not len(labels): return float("nan")
    pred = ((test - c1).square().sum(-1) < (test - c0).square().sum(-1)).float()
    return float((pred == labels).float().mean())


def _stats(values: Iterable[float]) -> dict:
    x = torch.tensor(list(values), dtype=torch.float64)
    if not x.numel(): return {"n": 0}
    q = torch.quantile(x, torch.tensor([.10, .25, .5, .75, .90], dtype=x.dtype))
    return {"n": int(x.numel()), "mean": float(x.mean()), "sum": float(x.sum()), "sd": float(x.std(unbiased=False)),
            "p10": float(q[0]), "p25": float(q[1]), "median": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4])}


def summarize_traces(labeled_traces) -> dict:
    buckets = defaultdict(lambda: defaultdict(list))
    for phase, label, trace in labeled_traces:
        key = f"{phase}:{label}"
        for name in ("r_F_norm", "r_M_norm", "effective_F_norm", "effective_M_norm",
                     "access", "H_delta_norm", "conservation_error", "decay_F_norm", "decay_M_norm"):
            if name in trace:
                buckets[key][name].extend(trace[name].detach().cpu().flatten().tolist())
        buckets[key]["gate_F"].extend(trace["gates"][:, 0].detach().cpu().tolist())
        buckets[key]["gate_M"].extend(trace["gates"][:, 1].detach().cpu().tolist())
        buckets[key]["q_cos"].extend(Fnn.cosine_similarity(trace["q_F"], trace["q_M"]).detach().cpu().tolist())
        buckets[key]["r_cos"].extend(Fnn.cosine_similarity(trace["r_F"], trace["r_M"]).detach().cpu().tolist())
        buckets[key]["write_norm"].extend(trace["external_update"].norm(dim=(-2,-1)).detach().cpu().tolist())
        buckets[key]["transfer_norm"].extend(trace["transfer"].norm(dim=(-2,-1)).detach().cpu().tolist())
        batch = trace["gates"].shape[0]
        if batch % 2 == 0:
            half = batch // 2
            for name, value in (("gate_M", trace["gates"][:,1]), ("q_M_norm", trace["q_M"].norm(dim=-1)),
                                ("r_M_norm_conditioned", trace["r_M"].norm(dim=-1))):
                buckets[key][f"latent_difference_{name}"].append(float(value[:half].mean()-value[half:].mean()))
    return {key: {name: _stats(vals) for name, vals in metrics.items()} for key, metrics in buckets.items()}


@torch.no_grad()
def health_audit(model: Stage2DModel, seed: int, reps: int = 16, *, p: float = .70, n: int = 32,
                 route: dict | None = None) -> dict:
    route = route or {}; device = next(model.parameters()).device
    state = model.initial_state(2 * reps, device); traces = []; losses = []
    for index in range(n):
        rows = paired_experiences(seed, reps, index, p)
        state, loss, step_traces = play_routed(model, state, rows, **route)
        losses.append(float(loss)); phase = "early" if index < n // 2 else "late"
        traces.extend((phase, label, trace) for label, trace in step_traces)
    probe_rows = paired_experiences(seed + 700001, reps, 99999, .65, split="novel")
    prob = probe_prob_routed(model, state, probe_rows, **route)
    metrics = behavioral_metrics(prob, reps); metrics.pop("prob", None)
    agg = torch.stack([prob[:reps].mean(0), prob[reps:].mean(0)])
    target = torch.empty_like(agg)
    for z in (0, 1):
        for action in (0, 1):
            p0 = .65 if action == z else .35
            target[z, action, 0] = p0; target[z, action, 1:] = (1 - p0) / 3
    conditional_ce = float(-(target * agg.clamp_min(1e-9).log()).sum(-1).mean())
    marginal_ce = -(.5 * math.log(.5) + .5 * math.log(1/6))
    h, f, m = state.H, state.F, state.M
    probes = {"H": centroid_accuracy(h, reps), "F": centroid_accuracy(f, reps),
              "M": centroid_accuracy(m, reps), "FM": centroid_accuracy(torch.cat([f.flatten(1), m.flatten(1)], -1), reps)}
    return {
        "action_TV": sum(metrics["tv_action"]) / 2,
        "history_TV": sum(metrics["tv_history"]) / 2,
        "I_HA": abs(float(metrics["interaction_y0"])),
        "interaction_signed": float(metrics["interaction_y0"]),
        "BS": float(metrics["behavioral_separation_entropy"]),
        "observed_CE": sum(losses) / len(losses), "conditional_CE": conditional_ce,
        "marginal_CE": marginal_ce, "CFA": marginal_ce - conditional_ce,
        "z_probe": probes,
        "state_norms": {k: float(getattr(state, k).norm(dim=(-2,-1)).mean()) for k in ("H","F","M")},
        "routing": summarize_traces(traces),
        "representation": {k: [getattr(state, k)[:reps].flatten(1).mean(0).cpu().tolist(),
                               getattr(state, k)[reps:].flatten(1).mean(0).cpu().tolist()] for k in ("H","F","M")},
    }


def classify(health: dict, tv=.10, iha=.10, bs=.10) -> str:
    flags = (health["action_TV"] >= tv, health["I_HA"] >= iha, abs(health["BS"]) >= bs)
    return "healthy" if all(flags) else ("partial" if any(flags) else "shortcut")


@torch.no_grad()
def auxiliary_routing_audit(model: Stage2DModel, seed: int, reps: int = 8) -> dict:
    """Matched predictive/noise/NULL traces and branch-specific decay summaries."""
    device = next(model.parameters()).device; labeled = []
    result = {}
    for noise in (False, True):
        state = model.initial_state(2 * reps, device)
        for index in range(16):
            rows = paired_experiences(seed + 33001, reps, index, .70, matched_noise=noise)
            state, _, traces = play_routed(model, state, rows)
            labeled.extend(("noise" if noise else "predictive", label, trace) for label, trace in traces)
        result["noise" if noise else "predictive"] = {
            k: float(getattr(state, k).norm(dim=(-2,-1)).mean()) for k in ("H","F","M")}
    state = model.initial_state(2 * reps, device); before = {k: float(getattr(state,k).norm()) for k in ("H","F","M")}
    for _ in range(16):
        state, trace = diagnostic_step(model, state, None); labeled.append(("NULL", "null", trace))
    after = {k: float(getattr(state,k).norm()) for k in ("H","F","M")}
    result["NULL"] = {"before": before, "after": after}
    result["distributions"] = summarize_traces(labeled)
    return result
