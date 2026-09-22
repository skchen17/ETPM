"""Read-only per-checkpoint fusion anatomy and finite causal-unit audit."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from etrcm.stage2d.model import head_hash, parameter_hash
from etrcm.stage2d4.model import Stage2D4Model, classify, health_audit
from etrcm.stage2d5.flow import candidate_flow, cell_means
from etrcm.stage2d3.interaction import factorial_components
from etrcm.stage2d6.fusion import anatomy, behavior, top_k, unit_removal
from stage2d4_audit import formed_state
from stage2d5_audit import context


STEPS = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500)


def load(path, arm):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = Stage2D4Model(arm)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def rows_for(model, ctx, reps):
    return [candidate_flow(model, ctx, torch.full((2*reps,), a, dtype=torch.long))
            for a in (0, 1)]


def unit_causality(model, rows, reps, fusion, seed):
    interaction = torch.tensor(fusion["post_I_vector"])
    native = fusion["behavior"]
    generator = random.Random(seed)
    result = {}
    for k in (1, 2, 4, 8, 16):
        chosen = top_k(fusion, k)
        removed = behavior(unit_removal(model, rows, reps, interaction, chosen), reps)
        random_controls = []
        for _ in range(16):
            selected = generator.sample(range(interaction.numel()), k)
            random_controls.append(behavior(unit_removal(model, rows, reps, interaction, selected), reps))
        result[str(k)] = {"units": chosen, "removed": removed,
                          "random_controls": random_controls,
                          "IHA_loss": native["IHA"]-removed["IHA"],
                          "random_mean_IHA_loss": native["IHA"]-
                              sum(x["IHA"] for x in random_controls)/len(random_controls),
                          "BS_loss": abs(native["BS"])-abs(removed["BS"])}
    return result


@torch.no_grad()
def checkpoint_audit(path, arm, reps=16, *, causal=True):
    model, ckpt = load(path, arm)
    seed = ckpt.get("eval_seed", 16601)
    initial_hash = parameter_hash(model); protected = head_hash(model)
    health = health_audit(model, seed, reps=reps)
    state = formed_state(model, seed, reps)
    before = (state.H.clone(), state.F.clone(), state.M.clone(), state.tau, state.external_time)
    ctx = context(model, state, seed, reps)
    rows = rows_for(model, ctx, reps)
    fusion = anatomy(model, rows, reps, unit_causality=causal)
    native = [model.candidate(ctx, torch.full((2*reps,), a, dtype=torch.long))[0]
              for a in (0, 1)]
    match = max(float((rows[a]["logits"]-native[a]).abs().max()) for a in (0, 1))
    result = {"step": ckpt["steps"], "health": health, "class": classify(health),
              "fusion": fusion, "native_logit_max_error": match,
              "integrity": {"parameters_unchanged": parameter_hash(model)==initial_hash,
                            "protected_head_unchanged": head_hash(model)==protected,
                            "protected_head_matches_checkpoint": protected==ckpt["protected_evaluator_hash"],
                            "persistent_state_unchanged": all(torch.equal(x,y) for x,y in zip(
                                before[:3], (state.H,state.F,state.M))) and before[3:]==(
                                state.tau,state.external_time)}}
    if causal:
        result["unit_causality"] = unit_causality(model, rows, reps, fusion, seed)
    return result


def one(path, arm, cohort, out, reps=16, trajectory=False):
    torch.set_num_threads(1)
    endpoint = checkpoint_audit(path, arm, reps)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    result = {"run": path.parent.name, "cohort": cohort, "arm": arm,
              "checkpoint": str(path), "init_seed": ckpt.get("init_seed"),
              "data_seed": ckpt.get("data_seed"), "eval_seed": ckpt.get("eval_seed"),
              "endpoint": endpoint}
    if trajectory:
        result["trajectory"] = []
        for step in STEPS:
            item = endpoint if step==1500 else checkpoint_audit(
                path.parent / f"checkpoint_{step:04d}.pt", arm, reps, causal=False)
            result["trajectory"].append(item)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result))
    print(json.dumps({"run": result["run"], "class": endpoint["class"],
                      "pre_I": endpoint["fusion"]["pre_I"],
                      "post_I": endpoint["fusion"]["post_I"]}), flush=True)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--arm", choices=("A0","A1"), default="A0")
    p.add_argument("--cohort", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=16)
    p.add_argument("--trajectory", action="store_true")
    args=p.parse_args()
    one(args.checkpoint,args.arm,args.cohort,args.out,args.reps,args.trajectory)
