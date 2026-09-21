"""Frozen activation rescue and necessity tests at preregistered fusion_post."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from stage2d_evaluate import load_model
from etrcm.stage2d.model import parameter_hash
from etrcm.stage2d2.geometry import phase_states
from etrcm.stage2d3.interaction import (
    destroy_native_interaction, factorial_components, factorial_layers,
    intervention_metrics, output_aligned_template, output_metrics, signed_delta,
)


PREREGISTERED_LAYER = "fusion_post"


def final_audits(path: Path):
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]


def select(audits, label, n=8):
    return sorted((x for x in audits if x["final_class"] == label),
                  key=lambda x: (x["init_seed"], x["data_seed"]))[:n]


def matched_pairs(audits, n=8):
    healthy = [x for x in audits if x["final_class"] == "healthy"]
    failed = [x for x in audits if x["final_class"] == "shortcut"]
    edges = []
    for h in healthy:
        for f in failed:
            cost = 0 if h["init_seed"] == f["init_seed"] else (1 if h["data_seed"] == f["data_seed"] else 2)
            edges.append((cost, h["run"], f["run"], h, f))
    used_h, used_f, pairs = set(), set(), []
    for _, hname, fname, h, f in sorted(edges, key=lambda x: x[:3]):
        if hname in used_h or fname in used_f:
            continue
        pairs.append((h, f)); used_h.add(hname); used_f.add(fname)
        if len(pairs) == n:
            break
    return pairs


@torch.no_grad()
def healthy_template(run: Path, eval_seed: int, reps: int):
    model, _ = load_model(run / "checkpoint_1500.pt", "cpu")
    state = phase_states(model, eval_seed, reps)["post"]
    cells, _ = factorial_layers(model, state, eval_seed, reps)
    post_i = factorial_components(cells["fusion_post"])["interaction"]
    logit_i = factorial_components(cells["logits"])["interaction"]
    return {"logit": logit_i, "post_scale": float(post_i.norm())}


@torch.no_grad()
def failed_one(audit, checkpoint_root, template, eval_seed, reps):
    model, _ = load_model(checkpoint_root / audit["run"] / "checkpoint_1500.pt", "cpu")
    before = parameter_hash(model)
    state = phase_states(model, eval_seed, reps)["post"]
    native_cells, _ = factorial_layers(model, state, eval_seed, reps)
    native = output_metrics(native_cells)
    direction = output_aligned_template(template["logit"], model, template["post_scale"])
    gen = torch.Generator(device=direction.device).manual_seed(eval_seed + audit["init_seed"] * 101 + audit["data_seed"])
    random_direction = torch.randn(direction.shape, generator=gen, device=direction.device)
    random_direction = random_direction / random_direction.norm().clamp_min(1e-12) * direction.norm()
    variants = {"native": native}
    for lam in (.25, .50, 1.0):
        variants[f"interaction_{lam}"] = intervention_metrics(model, state, eval_seed, reps, direction, lam, "interaction")
        variants[f"state_{lam}"] = intervention_metrics(model, state, eval_seed, reps, direction, lam, "state")
        variants[f"action_{lam}"] = intervention_metrics(model, state, eval_seed, reps, direction, lam, "action")
        variants[f"random_{lam}"] = intervention_metrics(model, state, eval_seed, reps, random_direction, lam, "interaction")
        # Fixed permutation of the factorial signs; it has zero factorial interaction.
        delta = signed_delta(direction * lam, "shuffled")
        changed, _ = factorial_layers(model, state, eval_seed, reps, delta)
        variants[f"shuffled_{lam}"] = output_metrics(changed)
    if parameter_hash(model) != before:
        raise AssertionError("frozen rescue mutated parameters")
    return {"run": audit["run"], "class": audit["final_class"], "variants": variants,
            "template_cell_norm": float(direction.norm()), "parameters_unchanged": True}


@torch.no_grad()
def healthy_one(audit, checkpoint_root, eval_seed, reps):
    model, _ = load_model(checkpoint_root / audit["run"] / "checkpoint_1500.pt", "cpu")
    before = parameter_hash(model)
    state = phase_states(model, eval_seed, reps)["post"]
    cells, _ = factorial_layers(model, state, eval_seed, reps)
    result = {"native": output_metrics(cells)}
    for mode in ("remove", "reverse", "state_control"):
        result[mode] = destroy_native_interaction(model, state, eval_seed, reps, mode)
    if parameter_hash(model) != before:
        raise AssertionError("healthy destruction mutated parameters")
    return {"run": audit["run"], "class": audit["final_class"], "variants": result,
            "parameters_unchanged": True}


def main(args):
    audits = final_audits(args.audit)
    pairs = matched_pairs(audits)
    healthy = [x[0] for x in pairs]
    failed = [x[1] for x in pairs]
    if len(healthy) < 8 or len(failed) < 8:
        raise RuntimeError(f"need >=8 healthy and shortcut, got {len(healthy)}, {len(failed)}")
    templates = [healthy_template(args.checkpoints / x["run"], args.eval_seed, args.reps) for x in healthy]
    jobs = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for index, audit in enumerate(failed):
            jobs.append(pool.submit(failed_one, audit, args.checkpoints, templates[index % len(templates)],
                                    args.eval_seed, args.reps))
        for audit in healthy:
            jobs.append(pool.submit(healthy_one, audit, args.checkpoints, args.eval_seed, args.reps))
        rows = [f.result() for f in as_completed(jobs)]
    failed_rows = [x for x in rows if x["class"] == "shortcut"]
    healthy_rows = [x for x in rows if x["class"] == "healthy"]
    rescue_counts = {}
    for lam in (.25, .50, 1.0):
        key = f"interaction_{lam}"
        count = 0
        for row in failed_rows:
            v, native = row["variants"][key], row["variants"]["native"]
            controls = [row["variants"][f"{name}_{lam}"] for name in ("state", "action", "random", "shuffled")]
            if (v["I_HA"] >= .10 and abs(v["BS"]) >= .10 and
                v["I_HA"] > native["I_HA"] and abs(v["BS"]) > abs(native["BS"]) and
                all(v["I_HA"] > c["I_HA"] and abs(v["BS"]) > abs(c["BS"]) for c in controls)):
                count += 1
        rescue_counts[str(lam)] = count
    best_lam, best_rescue = max(rescue_counts.items(), key=lambda x: x[1])
    destruction = 0
    reversal = 0
    for row in healthy_rows:
        v = row["variants"]
        control = v["state_control"]
        if (v["remove"]["I_HA"] < .5 * v["native"]["I_HA"] and
            abs(v["remove"]["BS"]) < .5 * abs(v["native"]["BS"]) and
            v["remove"]["I_HA"] < control["I_HA"] and abs(v["remove"]["BS"]) < abs(control["BS"])):
            destruction += 1
        if (v["reverse"]["I_HA_signed"] * v["native"]["I_HA_signed"] < 0 and
            v["reverse"]["BS"] * v["native"]["BS"] < 0):
            reversal += 1
    payload = {
        "preregistered_layer": PREREGISTERED_LAYER,
        "selection": {"healthy": [x["run"] for x in healthy], "shortcut": [x["run"] for x in failed]},
        "runs": rows,
        "G81": {"counts": rescue_counts, "best_lambda": best_lam, "count": best_rescue,
                "denominator": 8, "pass": best_rescue >= 6},
        "G82": {"removal_count": destruction, "reversal_count": reversal, "denominator": 8,
                "pass": max(destruction, reversal) >= 6},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    print(json.dumps({"G81": payload["G81"], "G82": payload["G82"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--eval-seed", type=int, required=True)
    p.add_argument("--reps", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
