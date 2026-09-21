"""Checkpoint trajectories, retrieval anatomy, and query interventions."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from etrcm.stage2d.model import behavioral_metrics
from etrcm.stage2d.world import paired_experiences
from etrcm.stage2d3.interaction import factorial_components
from etrcm.stage2d4.model import (Stage2D4Model, aggregate, candidate_prob, classify,
                                  context_state, health_audit, play)


CHECKPOINTS = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500)
LAYERS = ("incoming_H", "candidate_read", "temporary_H", "fusion_input",
          "fusion_post", "pre_logit", "logits", "probabilities")


def load_model(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Stage2D4Model(ckpt["arm"])
    model.load_state_dict(ckpt["model"]); model.eval()
    return model, ckpt


def formed_state(model, seed, reps=16, n=32, p=.70, noise=False):
    state = model.initial_state(2 * reps, "cpu")
    for index in range(n):
        rows = paired_experiences(seed, reps, index, p, matched_noise=noise)
        state, _, _ = play(model, state, rows)
    return state


def cell_means(values: list[torch.Tensor], reps: int):
    stacked = torch.stack(values)  # [action,batch,...]
    return torch.stack([torch.stack([stacked[a, :reps].mean(0),
                                     stacked[a, reps:].mean(0)]) for a in range(2)], 1)


@torch.no_grad()
def anatomy(model, state, seed, reps=16, **kwargs):
    rows = paired_experiences(seed + 700001, reps, 99999, .65, split="novel")
    context = context_state(model, state.clone(), rows)
    traces = []
    for action in (0, 1):
        declared = torch.full((2 * reps,), action, dtype=torch.long)
        call = dict(kwargs)
        if call.pop("swap_query", False): call["query_action"] = 1 - declared
        _, trace = model.candidate(context, declared, **call); traces.append(trace)
    values = {}
    for layer in LAYERS:
        if layer == "candidate_read":
            items = [torch.cat([t["r_F"], t["r_M"]], -1) for t in traces]
        else:
            items = [t[layer] for t in traces]
        cells = cell_means(items, reps); parts = factorial_components(cells)
        s, a, i = (float(parts[k].flatten().norm()) for k in ("state", "action", "interaction"))
        values[layer] = {"state_norm": s, "action_norm": a, "interaction_norm": i,
                         "normalized_interaction": i / (s + a + 1e-12)}
    retrieval = {}
    for name in ("q_F", "q_M", "r_F", "r_M"):
        cells = cell_means([t[name] for t in traces], reps)
        interaction = factorial_components(cells)["interaction"]
        retrieval[name] = {"interaction_norm": float(interaction.norm()),
                           "interaction_vector": interaction.flatten().tolist(),
                           "action_distance": float((traces[0][name] - traces[1][name]).norm(dim=-1).mean())}
    return {"layers": values, "retrieval": retrieval}


@torch.no_grad()
def behavior(model, state, seed, reps=16, **kwargs):
    rows = paired_experiences(seed + 700001, reps, 99999, .65, split="novel")
    prob, _ = candidate_prob(model, state, rows, **kwargs)
    metrics = behavioral_metrics(prob, reps)
    return {"action_TV": sum(metrics["tv_action"]) / 2,
            "I_HA": abs(float(metrics["interaction_y0"])),
            "I_HA_signed": float(metrics["interaction_y0"]),
            "BS": float(metrics["behavioral_separation_entropy"])}


def one(run: Path, out: Path, reps: int):
    torch.set_num_threads(1); trajectory = []; metadata = None
    for step in CHECKPOINTS:
        model, metadata = load_model(run / f"checkpoint_{step:04d}.pt")
        health = health_audit(model, metadata["eval_seed"], reps=reps)
        state = formed_state(model, metadata["eval_seed"], reps)
        trajectory.append({"step": step, "health": health, "class": classify(health),
                           "anatomy": anatomy(model, state, metadata["eval_seed"], reps)})
    model, metadata = load_model(run / "checkpoint_1500.pt")
    state = formed_state(model, metadata["eval_seed"], reps)
    native = behavior(model, state, metadata["eval_seed"], reps)
    interventions = {"correct": native}
    if metadata["arm"] == "A1":
        interventions.update({
            "swapped": behavior(model, state, metadata["eval_seed"], reps, swap_query=True),
            "both_neutral": behavior(model, state, metadata["eval_seed"], reps, neutral_f=True, neutral_m=True),
            "shared": behavior(model, state, metadata["eval_seed"], reps, shared_query=0),
            "F_neutral": behavior(model, state, metadata["eval_seed"], reps, neutral_f=True),
            "M_neutral": behavior(model, state, metadata["eval_seed"], reps, neutral_m=True),
            "F_read_clamp": behavior(model, state, metadata["eval_seed"], reps, read_clamp="F"),
            "M_read_clamp": behavior(model, state, metadata["eval_seed"], reps, read_clamp="M"),
            "FM_read_clamp": behavior(model, state, metadata["eval_seed"], reps, read_clamp="FM"),
        })
    payload = {"run": run.name, "arm": metadata["arm"], "init_seed": metadata["init_seed"],
               "data_seed": metadata["data_seed"], "eval_seed": metadata["eval_seed"],
               "final_class": trajectory[-1]["class"], "trajectory": trajectory,
               "interventions": interventions}
    out.mkdir(parents=True, exist_ok=True); target = out / f"{run.name}.json"
    target.write_text(json.dumps(payload)); return target


def main(args):
    runs = sorted(p for p in args.checkpoints.glob("*/*")
                  if p.is_dir() and (p / "checkpoint_1500.pt").is_file())
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(one, run, args.out / run.parent.name, args.reps) for run in runs]
        for f in as_completed(futures): print("COMPLETED", f.result(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--reps", type=int, default=16)
    p.add_argument("--jobs", type=int, default=16); main(p.parse_args())
