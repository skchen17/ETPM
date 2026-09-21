"""Stage 2D.1 high-resolution training with separated initialization/data seeds."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from etrcm.stage2d.model import Stage2DModel, head_hash, parameter_hash, pretrain_evaluator
from etrcm.stage2d.world import training_experiences
from etrcm.stage2d1.engine import play_routed
from etrcm.stage2d1.protocol import CHECKPOINT_STEPS, FROZEN_D1, evidence_schedule, gate_floor


def grad_norm(parameters) -> float:
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None))


def gradient_groups(model):
    return {
        "all": model.parameters(),
        "H_core": list(model.core.core_in.parameters()) + list(model.core.core_out.parameters()) + list(model.core.core_gate.parameters()),
        "q_F": model.core.q_fast_projection.parameters(), "q_M": model.core.q_slow_projection.parameters(),
        "read_gate": model.core.two_way_gate.parameters(), "external_write": model.core.event_encoder.parameters(),
        "access_consolidation": model.core.access_head.parameters(),
        "action_branch": model.action_head.parameters(), "evaluator": model.action_head.parameters(),
    }


def save_checkpoint(model, args, step, out, protected_hash, gradients, schedule_log):
    item = {
        "model": model.state_dict(), "seed": args.init_seed, "init_seed": args.init_seed,
        "data_seed": args.data_seed, "eval_seed": args.eval_seed, "steps": step,
        "arm": "A2", "variant": "full", "gamma": args.gamma,
        "rho_fast": args.rho_fast, "rho_slow": args.rho_slow,
        "train_p": args.train_p, "curriculum": args.curriculum,
        "gate_floor": args.gate_floor, "gate_window": args.gate_window,
        "protected_evaluator_hash": protected_hash, "gradients_preceding_update": gradients,
        "schedule_prefix": schedule_log.copy(),
    }
    torch.save(item, out / f"checkpoint_{step:04d}.pt")


def train(args):
    torch.set_num_threads(1); torch.manual_seed(args.init_seed); random.seed(args.init_seed)
    model = Stage2DModel(variant="full", gamma=args.gamma, rho_fast=args.rho_fast,
                         rho_slow=args.rho_slow).to(args.device)
    pre_initial_hash = parameter_hash(model)
    pretrain = pretrain_evaluator(model, args.init_seed, args.device, args.evaluator_steps)
    for parameter in model.action_head.parameters(): parameter.requires_grad_(False)
    protected_hash = head_hash(model); initialized_hash = parameter_hash(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.01)
    args.out.mkdir(parents=True, exist_ok=True); checkpoints = set(args.checkpoints)
    logs = []; schedule_log = []; gradients = {name: 0.0 for name in gradient_groups(model)}
    save_checkpoint(model, args, 0, args.out, protected_hash, gradients, schedule_log)
    start = time.monotonic(); lengths = tuple(args.lengths)
    for step in range(args.steps):
        model.train(); optimizer.zero_grad(set_to_none=True); state = model.initial_state(args.batch, args.device)
        p = evidence_schedule(args.curriculum, step, args.steps) if args.curriculum != "C0" else args.train_p
        floor = gate_floor(args.curriculum, step, floor=args.gate_floor, window=args.gate_window)
        schedule_log.append({"step": step, "p": p, "gate_floor_m": floor})
        losses, penalties = [], []; length = lengths[(step + args.data_seed) % len(lengths)]
        for index in range(length):
            rows = training_experiences(args.data_seed * 100003 + step, args.batch, index, p)
            route = {"gate_floor_m": floor} if floor is not None else {}
            state, loss, _ = play_routed(model, state, rows, **route)
            losses.append(loss); penalties.append(state.H.square().mean())
        ce = torch.stack(losses).mean(); objective = ce + .001 * torch.stack(penalties).mean()
        if not torch.isfinite(objective): raise RuntimeError("nonfinite training objective")
        objective.backward(); gradients = {name: grad_norm(params) for name, params in gradient_groups(model).items()}
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); optimizer.step()
        if head_hash(model) != protected_hash: raise AssertionError("protected evaluator changed")
        completed = step + 1
        if completed in checkpoints:
            save_checkpoint(model, args, completed, args.out, protected_hash, gradients, schedule_log)
            row = {"step": completed, "CE": float(ce.detach()), "objective": float(objective.detach()),
                   "H_norm": float(state.H.detach().norm(dim=(-2,-1)).mean()),
                   "F_norm": float(state.F.detach().norm(dim=(-2,-1)).mean()),
                   "M_norm": float(state.M.detach().norm(dim=(-2,-1)).mean()),
                   "gradients_preclip": gradients, "p": p, "gate_floor_m": floor,
                   "elapsed_s": time.monotonic() - start}
            logs.append(row); print(json.dumps(row), flush=True)
    summary = {
        "init_seed": args.init_seed, "data_seed": args.data_seed, "eval_seed": args.eval_seed,
        "pre_initial_hash": pre_initial_hash, "initialized_hash": initialized_hash,
        "final_hash": parameter_hash(model), "evaluator_pretraining": pretrain,
        "evaluator_hash_unchanged": head_hash(model) == protected_hash,
        "seeds_strictly_separated": len({args.init_seed, args.data_seed, args.eval_seed}) == 3,
        "architecture_change": "none", "memory_law_change": "none",
        "privileged_lifetime_inputs": [], "curriculum": args.curriculum,
        "gate_warmup_removed_before_final_evaluation": gate_floor(args.curriculum, args.steps - 1,
                                                                   floor=args.gate_floor,
                                                                   window=args.gate_window) is None,
        "checkpoint_steps": sorted(checkpoints), "training_log": logs,
        "elapsed_s": time.monotonic() - start,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"COMPLETED": summary | {"training_log": "omitted"}}), flush=True)


def parser():
    p = argparse.ArgumentParser(); p.add_argument("--init-seed", type=int, required=True)
    p.add_argument("--data-seed", type=int, required=True); p.add_argument("--eval-seed", type=int, default=15101)
    p.add_argument("--steps", type=int, default=FROZEN_D1.steps); p.add_argument("--evaluator-steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16); p.add_argument("--train-p", type=float, default=.70)
    p.add_argument("--lengths", type=int, nargs="+", default=list(FROZEN_D1.lengths))
    p.add_argument("--checkpoints", type=int, nargs="+", default=list(CHECKPOINT_STEPS))
    p.add_argument("--gamma", type=float, default=.50); p.add_argument("--rho-fast", type=float, default=.97)
    p.add_argument("--rho-slow", type=float, default=.9995); p.add_argument("--curriculum", choices=("C0","C1","C2","C3","C4"), default="C0")
    p.add_argument("--gate-floor", type=float, default=.65); p.add_argument("--gate-window", type=int, default=100)
    p.add_argument("--device", default="cpu"); p.add_argument("--out", type=Path, required=True)
    return p


if __name__ == "__main__": train(parser().parse_args())
