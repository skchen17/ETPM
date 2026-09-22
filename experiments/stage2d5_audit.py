"""Frozen per-run Stage 2D.5 interaction-flow and finite intervention audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from etrcm.stage2d.model import behavioral_metrics, head_hash, parameter_hash
from etrcm.stage2d.world import paired_experiences
from etrcm.stage2d3.interaction import factorial_components
from etrcm.stage2d4.model import Stage2D4Model, classify, context_state, health_audit
from etrcm.stage2d5.flow import (ORDER, candidate_flow, cell_means,
                                 factorial_injection, interaction_metrics,
                                 norm_match, transmission)
from stage2d4_audit import formed_state


REPAIR_NODES = ("q_F", "q_M", "r_F", "r_M", "normalized_F", "normalized_M",
                "gated_read", "read_projection", "core_preactivation",
                "gate_activation", "candidate_update", "temporary_H",
                "evaluator_H_input", "fusion_input", "fusion_pre", "fusion_post")
NECESSITY_NODES = REPAIR_NODES
EDGE_ORDER = ("q_F", "r_F", "normalized_F", "gated_read", "read_projection",
              "core_preactivation", "core_nonlinearity", "candidate_update",
              "temporary_H", "evaluator_H_input", "fusion_input", "fusion_pre",
              "fusion_post", "logits", "probabilities", "policy_score")


def load_checkpoint(path: Path, arm: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Stage2D4Model(arm)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def context(model, state, seed, reps):
    rows = paired_experiences(seed + 700001, reps, 99999, .65, split="novel")
    return context_state(model, state.clone(), rows)


def traces(model, ctx, reps, *, override_node=None, override_values=None,
           mixing="native", residual_alpha=1.):
    out = []
    for action in (0, 1):
        a = torch.full((2*reps,), action, dtype=torch.long)
        override = {} if override_node is None else {override_node: override_values[action]}
        out.append(candidate_flow(model, ctx, a, override=override,
                                  mixing=mixing, residual_alpha=residual_alpha))
    return out


def behavior_from_traces(rows, reps):
    probabilities = torch.stack([r["probabilities"] for r in rows], 1)
    metrics = behavioral_metrics(probabilities, reps)
    return {"action_TV": sum(metrics["tv_action"])/2,
            "I_HA": abs(float(metrics["interaction_y0"])),
            "I_HA_signed": float(metrics["interaction_y0"]),
            "BS": float(metrics["behavioral_separation_entropy"])}


def full_map(rows, reps):
    result = {}
    for node in ORDER:
        cells = cell_means([r[node] for r in rows], reps)
        result[node] = interaction_metrics(cells)
        result[node]["activation_norm"] = float(torch.stack([r[node].flatten(1).norm(dim=-1).mean() for r in rows]).mean())
    ratios = {f"{a}->{b}": transmission(result[a], result[b]) for a,b in zip(EDGE_ORDER, EDGE_ORDER[1:])}
    cosine = F.cosine_similarity(
        torch.tensor(result["r_F"]["interaction_vector"])[None],
        torch.tensor(result["r_M"]["interaction_vector"])[None]).item()
    return result, ratios, cosine


def read_mixing(model, ctx, reps):
    variants = {}
    for name in ("native", "F_only", "M_only", "equal", "norm_matched_sum"):
        with torch.no_grad(): rows = traces(model, ctx, reps, mixing=name)
        flow, _, _ = full_map(rows, reps)
        variants[name] = {"gated_read_interaction": flow["gated_read"]["norm"],
                          "candidate_update_interaction": flow["candidate_update"]["norm"],
                          "temporary_H_interaction": flow["temporary_H"]["norm"],
                          "behavior": behavior_from_traces(rows, reps)}
    return variants


def residual_scaling(model, ctx, reps):
    result = {}
    for alpha in (0., .25, .5, 1., 1.5, 2.):
        with torch.no_grad(): rows = traces(model, ctx, reps, residual_alpha=alpha)
        flow, _, _ = full_map(rows, reps)
        result[str(alpha)] = {"alpha": alpha,
                              "candidate_update_I_over_activation": flow["candidate_update"]["norm"]/
                                  (flow["candidate_update"]["activation_norm"]+1e-12),
                              "temporary_H_I_over_activation": flow["temporary_H"]["norm"]/
                                  (flow["temporary_H"]["activation_norm"]+1e-12),
                              "fusion_post_interaction": flow["fusion_post"]["norm"],
                              "behavior": behavior_from_traces(rows, reps)}
    return result


def aligned_direction(model, ctx, reps, node, healthy_output_template):
    # Local Jacobian aligns coordinates through the target model's output;
    # there is no cross-model raw-neuron transfer.
    with torch.enable_grad():
        rows = traces(model, ctx, reps)
        logits = cell_means([r["logits"] for r in rows], reps)
        template=torch.tensor(healthy_output_template,dtype=logits.dtype,device=logits.device)
        template=template/template.norm().clamp_min(1e-12)
        output_i=(factorial_components(logits)["interaction"]*template).sum()
        nodes = [r[node] for r in rows]
        grads = torch.autograd.grad(output_i, nodes, allow_unused=True)
        if any(g is None for g in grads): return None
        direction = 0
        signs = ((1., -1.), (-1., 1.))
        for action, grad in enumerate(grads):
            direction = direction + signs[0][action]*grad[:reps].mean(0) + signs[1][action]*grad[reps:].mean(0)
        if not torch.isfinite(direction).all() or direction.norm() < 1e-12: return None
        return direction.detach()/direction.detach().norm()


def restoration(model, ctx, reps, node, healthy_interaction_norm, healthy_output_template, seed):
    direction = aligned_direction(model, ctx, reps, node, healthy_output_template)
    if direction is None: return {"status": "NO_LOCAL_GRADIENT"}
    healthy_norm = max(float(healthy_interaction_norm), 1e-12)
    # Each factorial cell receives 1/4 of the healthy interaction norm;
    # the four-cell interaction increment is lambda times that norm.
    base_direction = direction * healthy_norm / 4
    generator = torch.Generator().manual_seed(seed + len(node)*97)
    random = torch.randn(base_direction.shape, generator=generator)
    random = norm_match(random, base_direction)
    with torch.no_grad(): native = behavior_from_traces(traces(model, ctx, reps), reps)
    interventions = {}
    for kind, vector in (("interaction", base_direction), ("state", base_direction),
                         ("action", base_direction), ("random_interaction", random),
                         ("shuffled", base_direction), ("generic", base_direction)):
        interventions[kind] = {}
        sign_kind = "interaction" if kind == "random_interaction" else kind
        for lam in (.25, .5, 1.):
            with torch.no_grad(): original = traces(model, ctx, reps)
            changed = [factorial_injection(original[a][node], reps, a, vector, sign_kind, lam)
                       for a in (0, 1)]
            with torch.no_grad(): outputs = traces(model, ctx, reps, override_node=node, override_values=changed)
            interventions[kind][str(lam)] = behavior_from_traces(outputs, reps)
    return {"native": native, "healthy_norm_reference": healthy_norm,
            "per_cell_intervention_norm_at_lambda1": float(base_direction.norm()),
            "cap_ratio": float(4*base_direction.norm()/healthy_norm),
            "interventions": interventions,
            "coordinate_alignment": "healthy same-architecture logit-interaction template mapped by target-model local Jacobian"}


def destruction(model, ctx, reps, node):
    with torch.no_grad(): original = traces(model, ctx, reps)
    cells = cell_means([r[node] for r in original], reps)
    parts = factorial_components(cells)
    result = {"native": behavior_from_traces(original, reps), "interaction_norm": float(parts["interaction"].norm())}
    for name, kind, vector in (
        ("remove", "interaction", -parts["interaction"]/4),
        ("reverse", "interaction", -parts["interaction"]/2),
        ("state_main_control", "state", -parts["state"]/2),
        ("action_main_control", "action", -parts["action"]/2),
    ):
        changed = [factorial_injection(original[a][node], reps, a, vector, kind, 1.) for a in (0, 1)]
        with torch.no_grad(): outputs = traces(model, ctx, reps, override_node=node, override_values=changed)
        result[name] = behavior_from_traces(outputs, reps)
    return result


def one(path, arm, cohort, out, healthy_norms=None, causal=False, reps=16):
    torch.set_num_threads(1)
    model, ckpt = load_checkpoint(path, arm)
    seed = ckpt.get("eval_seed", 16601)
    health = health_audit(model, seed, reps=reps)
    label = classify(health)
    state = formed_state(model, seed, reps)
    ctx = context(model, state, seed, reps)
    state_before = (state.H.clone(), state.F.clone(), state.M.clone(),state.tau,state.external_time)
    protected = head_hash(model); params = parameter_hash(model)
    with torch.no_grad(): rows = traces(model, ctx, reps)
    flow, ratios, cosine = full_map(rows, reps)
    result = {"checkpoint": str(path), "run": path.parent.name, "cohort": cohort, "arm": arm,
              "seed": seed, "class": label, "health": health,
              "flow": flow, "ratios": ratios, "F_M_raw_interaction_cosine": cosine,
              "behavior": behavior_from_traces(rows, reps)}
    result["mixing"] = read_mixing(model, ctx, reps)
    result["residual"] = residual_scaling(model, ctx, reps)
    if causal:
        result["restoration"] = {node: restoration(model, ctx, reps, node,
            healthy_norms[arm][node], healthy_norms[arm]["output_template"], seed)
            for node in REPAIR_NODES} if label != "healthy" else {}
        result["destruction"] = {node: destruction(model, ctx, reps, node)
                                  for node in NECESSITY_NODES} if label == "healthy" else {}
    result["integrity"] = {"parameters_unchanged": parameter_hash(model) == params,
                            "protected_head_unchanged": head_hash(model) == protected,
                            "persistent_state_unchanged": all(torch.equal(a,b) for a,b in zip(state_before[:3],
                                (state.H,state.F,state.M))) and state_before[3:] == (state.tau,state.external_time)}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result))
    print(json.dumps({"run": result["run"], "class": label, "causal": causal,
                      "Ir": flow["gated_read"]["norm"], "IdH": flow["candidate_update"]["norm"]}), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser();p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--arm", choices=("A0","A1"), required=True);p.add_argument("--cohort", required=True)
    p.add_argument("--out", type=Path, required=True);p.add_argument("--causal", action="store_true")
    p.add_argument("--healthy-norms", type=Path);p.add_argument("--reps", type=int, default=16)
    args=p.parse_args();norms=json.loads(args.healthy_norms.read_text()) if args.healthy_norms else None
    one(args.checkpoint,args.arm,args.cohort,args.out,norms,args.causal,args.reps)
