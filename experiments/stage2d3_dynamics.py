"""Reduced memory dynamics, peripheral mediation and continuous diagnostics."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from stage2d_evaluate import advance_experience, delay_one, load_model, measure
from etrcm.stage2d.model import state_norms


FORMATION = (0, 1, 4, 16, 64)
DELAYS = (0, 100, 500, 1000)
REVISION = (0, 8, 32, 128)


def formed(model, seed, reps, n=64, *, noise=False, clamp="none"):
    state = model.initial_state(2 * reps, "cpu")
    for index in range(n):
        state, _ = advance_experience(model, state, seed, reps, index, .65,
                                      noise=noise, clamp=clamp)
    return state


@torch.no_grad()
def one(run, seed, reps, continuous):
    torch.set_num_threads(1)
    model, _ = load_model(run / "checkpoint_1500.pt", "cpu")
    state = model.initial_state(2 * reps, "cpu"); formation = {}
    for index in range(max(FORMATION) + 1):
        if index in FORMATION:
            formation[str(index)] = measure(model, state, seed + index, reps)
        if index < max(FORMATION):
            state, _ = advance_experience(model, state, seed, reps, index, .65)
    persistence = {}; delayed = state.clone(); previous = 0
    for target in DELAYS:
        for index in range(previous, target):
            delayed, _ = delay_one(model, delayed, seed + 200003, reps, index)
        persistence[str(target)] = measure(model, delayed, seed + target, reps); previous = target
    revision = {"0": measure(model, state, seed, reps)}; revised = state.clone(); previous = 0
    for target in REVISION[1:]:
        for index in range(previous, target):
            revised, _ = advance_experience(model, revised, seed + 300007, reps, index, .65,
                                             oppose_fraction=1.)
        revision[str(target)] = measure(model, revised, seed + target, reps); previous = target
    predictive = measure(model, formed(model, seed + 400009, reps), seed, reps)
    noise = measure(model, formed(model, seed + 400009, reps, noise=True), seed, reps)
    clamps = {name: measure(model, formed(model, seed + 500009, reps, clamp=name), seed, reps)
              for name in ("none", "F", "M", "FM")}
    continuous_result = {}
    if continuous:
        mixed = model.initial_state(2 * reps, "cpu")
        milestones = {1000, 5000, 10000}
        for index in range(10000):
            mixed, _ = advance_experience(model, mixed, seed + 600011, reps, index, .65,
                                          noise=(index % 4 == 3))
            if index + 1 in milestones:
                m = measure(model, mixed, seed + index, reps)
                m["state_norms"] = state_norms(mixed)
                _, trace = model.step(mixed.clone(), None)
                m["read_gate_mean"] = trace["gates"].mean(0).cpu().tolist()
                m["all_finite"] = all(torch.isfinite(getattr(mixed, k)).all().item() for k in ("H", "F", "M"))
                continuous_result[str(index + 1)] = m
    def bs(x): return abs(x["behavioral_separation_entropy"])
    fvals = [bs(formation[str(n)]) for n in FORMATION]
    gradual = fvals[-1] >= .10 and sum(b > a for a, b in zip(fvals, fvals[1:])) >= 2
    persistence_ok = bs(persistence["500"]) >= .05 and bs(persistence["500"]) >= .25 * bs(persistence["0"])
    initial = revision["0"]["behavioral_separation_entropy"]
    final = revision["128"]["behavioral_separation_entropy"]
    revision_ok = abs(final) <= .5 * abs(initial) or final * initial < 0
    selectivity_ok = bs(predictive) >= bs(noise) + .02
    native = bs(clamps["none"])
    harms = {name: native - bs(clamps[name]) for name in ("F", "M", "FM")}
    mediation = max(harms.values()) >= max(.02, .25 * native)
    return {"run": run.name, "formation": formation, "persistence": persistence,
            "revision": revision, "selectivity": {"predictive": predictive, "noise": noise},
            "memory_interventions": clamps, "continuous": continuous_result,
            "criteria": {"gradual": gradual, "persistence": persistence_ok, "revision": revision_ok,
                         "selectivity": selectivity_ok, "mediation": mediation, "clamp_harms": harms}}


def main(args):
    runs = sorted(p for p in args.checkpoints.iterdir() if p.is_dir() and p.name.startswith("C1_") and
                  (p / "checkpoint_1500.pt").exists())
    with ProcessPoolExecutor(max_workers=min(8, len(runs))) as pool:
        rows = list(pool.map(one, runs,
                             [args.eval_seed + i * 101 for i in range(len(runs))],
                             [args.reps] * len(runs), [args.continuous] * len(runs)))
    joint = sum(all(x["criteria"][k] for k in ("gradual", "persistence", "revision", "selectivity")) for x in rows)
    mediation = sum(x["criteria"]["mediation"] for x in rows)
    payload = {"rows": rows,
               "G86": {"pass": joint >= 6, "count": joint, "required": 6},
               "G87": {"pass": mediation >= 6, "count": mediation, "required": 6,
                       "formation_window": 64}}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload))
    print(json.dumps({"G86": payload["G86"], "G87": payload["G87"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--eval-seed", type=int, required=True); p.add_argument("--reps", type=int, default=8)
    p.add_argument("--continuous", action="store_true"); p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
