"""Frozen Stage 2D.7 projection/SVD, finite intervention, and trajectory audit."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from etrcm.stage2d.model import head_hash, parameter_hash
from etrcm.stage2d3.interaction import factorial_components
from etrcm.stage2d4.model import Stage2D4Model, classify, health_audit
from etrcm.stage2d5.flow import candidate_flow, cell_means, factorial_injection
from etrcm.stage2d6.fusion import behavior
from etrcm.stage2d7.projection import (
    aligned_direction, components, intervention_metrics, low_sensitivity_direction,
    override_main, response_matrix, visibility,
)
from stage2d4_audit import formed_state
from stage2d5_audit import context


STEPS = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500)


def load(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = Stage2D4Model("A0")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def traces(model, ctx, reps):
    return [candidate_flow(model, ctx, torch.full((2*reps,), a, dtype=torch.long))
            for a in (0, 1)]


def phenotype(health, label):
    if label == "healthy":
        return "healthy"
    high = max(health["z_probe"]["H"], health["z_probe"]["M"]) >= .75
    if high and abs(health["BS"]) < .10:
        return "F2_stored_unused"
    return "F1_state_formation" if not high else "other_partial"


@torch.no_grad()
def basic(path, reps=16, *, health_source=None):
    model, checkpoint = load(path)
    seed = checkpoint.get("eval_seed", 16601)
    health = health_audit(model, seed, reps=reps) if health_source is None else health_source
    label = classify(health)
    state = formed_state(model, seed, reps)
    ctx = context(model, state, seed, reps)
    rows = traces(model, ctx, reps)
    vis = visibility(model, rows, reps)
    result = {"checkpoint": str(path), "step": checkpoint["steps"],
              "init_seed": checkpoint.get("init_seed"),
              "data_seed": checkpoint.get("data_seed"), "eval_seed": seed,
              "class": label, "phenotype": phenotype(health, label),
              "health": health, "visibility": vis,
              "fusion_post_I": float(components(rows, "fusion_post", reps)["interaction"].norm()),
              "behavior": behavior(rows, reps),
              "protected_head_matches_checkpoint": head_hash(model) == checkpoint["protected_evaluator_hash"]}
    return result, model, ctx, rows


def make_template(paths, out, reps=16):
    vectors, state_norms, post_norms = [], [], []
    source = []
    with torch.no_grad():
        for path in paths:
            result, model, _, rows = basic(path, reps)
            if result["class"] != "healthy":
                raise AssertionError(f"template source is not healthy: {path}")
            vectors.append(components(rows, "logits", reps)["interaction"])
            state_norms.append(result["visibility"]["D_S"])
            post_norms.append(result["fusion_post_I"])
            source.append(str(path))
    template = torch.stack(vectors).mean(0)
    payload = {"source_checkpoints": source, "logit_interaction_template": template.tolist(),
               "healthy_state_median": statistics.median(state_norms),
               "healthy_post_interaction_median": statistics.median(post_norms),
               "alignment": "target local 4-by-32 logit-interaction Jacobian adjoint; no cross-model neuron swap",
               "cap": 1.25, "intervention_fraction": 1.0,
               "random_seed_rule": "eval_seed + 27183"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return payload


def positive_control(model, ctx, rows, reps, template, post_norm):
    # Four-cell interaction injection is only a known post-fusion ceiling.
    target = template / template.norm().clamp_min(1e-12)
    mapped = model.action_head.head[2].weight.T @ target
    vector = mapped / mapped.norm().clamp_min(1e-12) * (post_norm / 4)
    changed = []
    for action in (0, 1):
        signs = torch.cat([torch.ones(reps), -torch.ones(reps)]) * (1 if action == 0 else -1)
        delta = signs[:, None] * vector[None]
        ids = torch.full((2*reps,), action, dtype=torch.long)
        changed.append(candidate_flow(model, ctx, ids,
                       override={"fusion_post": rows[action]["fusion_post"] + delta}))
    return {"behavior": behavior(changed, reps),
            "fusion_post_I": float(components(changed, "fusion_post", reps)["interaction"].norm())}


def failed_interventions(model, ctx, rows, reps, template, seed):
    response = response_matrix(model, ctx, rows, reps)
    norm = min(template["healthy_state_median"], 1.25 * template["healthy_state_median"])
    target = torch.tensor(template["logit_interaction_template"], dtype=response.dtype)
    aligned = aligned_direction(response, target, norm)
    low = low_sensitivity_direction(response, norm)
    generator = torch.Generator().manual_seed(seed + 27183)
    random = torch.randn(aligned.shape, generator=generator)
    random = random / random.norm().clamp_min(1e-12) * norm
    interventions = {"native": {"behavior": behavior(rows, reps),
                                     "fusion_post_I": float(components(rows, "fusion_post", reps)["interaction"].norm())},
                     "response_singular_values": torch.linalg.svdvals(response).tolist(),
                     "norm": norm, "cap_ratio": norm / template["healthy_state_median"]}
    for key, vector, kind in (("aligned_state", aligned, "state"),
                              ("random_state", random, "state"),
                              ("low_sensitivity_state", low, "state"),
                              ("action_main", aligned, "action"),
                              ("common_shift", aligned, "common")):
        interventions[key] = {str(lam): intervention_metrics(model, ctx, rows, reps,
                                  vector * lam, kind=kind) for lam in (.25, .5, 1.)}
    interventions["fusion_post_positive"] = positive_control(model, ctx, rows, reps,
                                        target, template["healthy_post_interaction_median"])
    return interventions


def healthy_interventions(model, ctx, rows, reps):
    state = components(rows, "fusion_H_projection", reps)["state"]
    result = {"native": {"behavior": behavior(rows, reps),
                          "fusion_post_I": float(components(rows, "fusion_post", reps)["interaction"].norm())}}
    for beta in (0., .25, .5, .75, 1.):
        result[str(beta)] = intervention_metrics(model, ctx, rows, reps, (beta-1.)*state)
    result["common_shift"] = intervention_metrics(model, ctx, rows, reps, -state/2, kind="common")
    result["action_main_control"] = intervention_metrics(model, ctx, rows, reps, -state, kind="action")
    return result


def one(path, out, template, reps=16, trajectory=False, health_source=None):
    torch.set_num_threads(1)
    endpoint, model, ctx, rows = basic(path, reps, health_source=health_source)
    original_hash = parameter_hash(model)
    with torch.no_grad():
        if endpoint["class"] == "healthy":
            endpoint["healthy_destruction"] = healthy_interventions(model, ctx, rows, reps)
        else:
            # response matrix itself enables gradients; no checkpoint weights change.
            pass
    if endpoint["class"] != "healthy":
        endpoint["state_main_rescue"] = failed_interventions(model, ctx, rows, reps, template,
                                                                endpoint["eval_seed"])
    endpoint["parameters_unchanged"] = parameter_hash(model) == original_hash
    result = {"run": path.parent.name, "endpoint": endpoint}
    if trajectory:
        result["trajectory"] = []
        for step in STEPS:
            if step == 1500:
                result["trajectory"].append({k: endpoint[k] for k in (
                    "step", "class", "phenotype", "health", "visibility", "fusion_post_I", "behavior")})
            else:
                item, _, _, _ = basic(path.parent / f"checkpoint_{step:04d}.pt", reps)
                result["trajectory"].append({k: item[k] for k in (
                    "step", "class", "phenotype", "health", "visibility", "fusion_post_I", "behavior")})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    print(json.dumps({"run": result["run"], "class": endpoint["class"],
                      "phenotype": endpoint["phenotype"], "D_H": endpoint["visibility"]["D_H"],
                      "D_S": endpoint["visibility"]["D_S"]}), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--template", type=Path)
    p.add_argument("--make-template", nargs="*", type=Path)
    p.add_argument("--reps", type=int, default=16)
    p.add_argument("--trajectory", action="store_true")
    args = p.parse_args()
    if args.make_template is not None:
        make_template(args.make_template, args.out, args.reps)
    else:
        if not args.checkpoint or not args.template:
            p.error("--checkpoint and --template required")
        one(args.checkpoint, args.out, json.loads(args.template.read_text()),
            args.reps, args.trajectory)
