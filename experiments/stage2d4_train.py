"""Train one independent Stage 2D.4 run."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from etrcm.stage2d.model import head_hash, parameter_hash, pretrain_evaluator
from etrcm.stage2d.world import training_experiences
from etrcm.stage2d3.curriculum import paired_action_loss, paired_observational_targets
from etrcm.stage2d4.model import Stage2D4Model, play


CHECKPOINTS = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500)


def grad_norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None))


def save(model, args, step, out, protected, gradients):
    torch.save({
        "model": model.state_dict(), "arm": args.arm, "init_seed": args.init_seed,
        "data_seed": args.data_seed, "eval_seed": args.eval_seed, "steps": step,
        "gamma": .50, "rho_fast": .97, "rho_slow": .9995,
        "protected_evaluator_hash": protected, "gradients_preceding_update": gradients,
        "training_objective": "observed_only_consequence",
        "paired_warmup_steps": 50 if args.arm == "A4" else 0,
    }, out / f"checkpoint_{step:04d}.pt")


def train(args):
    torch.set_num_threads(1); torch.manual_seed(args.init_seed); random.seed(args.init_seed)
    model = Stage2D4Model(args.arm).to(args.device)
    pre_initial_hash = parameter_hash(model)
    pretrain = pretrain_evaluator(model, args.init_seed, args.device, args.evaluator_steps)
    for parameter in model.action_head.parameters(): parameter.requires_grad_(False)
    protected = head_hash(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=.001, weight_decay=.01)
    args.out.mkdir(parents=True, exist_ok=True); checkpoints = set(args.checkpoints)
    save(model, args, 0, args.out, protected, {"all": 0., "core": 0.})
    logs = []; start = time.monotonic(); lengths = (4, 6, 8)
    for step in range(args.steps):
        model.train(); optimizer.zero_grad(set_to_none=True)
        state = model.initial_state(args.batch, args.device)
        losses, penalties, paired = [], [], []
        length = lengths[(step + args.data_seed) % len(lengths)]
        for index in range(length):
            rows = training_experiences(args.data_seed * 100003 + step,
                                        args.batch, index, .70)
            if args.arm == "A4" and step < 50:
                targets = paired_observational_targets(rows, args.data_seed * 31 + step, index)
                paired.append(paired_action_loss(model, state, rows, targets, "C1"))
            state, loss, _ = play(model, state, rows)
            losses.append(loss); penalties.append(state.H.square().mean())
        ce = torch.stack(losses).mean()
        auxiliary = torch.stack(paired).mean() if paired else ce.detach() * 0.
        objective = ce + (auxiliary if args.arm == "A4" and step < 50 else 0.) + \
                    .001 * torch.stack(penalties).mean()
        if not torch.isfinite(objective): raise RuntimeError("nonfinite objective")
        objective.backward()
        gradients = {"all": grad_norm(model.parameters()), "core": grad_norm(model.core.parameters())}
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        if head_hash(model) != protected: raise AssertionError("protected evaluator changed")
        completed = step + 1
        if completed in checkpoints:
            save(model, args, completed, args.out, protected, gradients)
            row = {"step": completed, "CE": float(ce.detach()),
                   "paired_CE": float(auxiliary.detach()), "objective": float(objective.detach()),
                   "H_norm": float(state.H.detach().norm(dim=(-2, -1)).mean()),
                   "F_norm": float(state.F.detach().norm(dim=(-2, -1)).mean()),
                   "M_norm": float(state.M.detach().norm(dim=(-2, -1)).mean()),
                   "elapsed_s": time.monotonic() - start}
            logs.append(row); print(json.dumps(row), flush=True)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {
        "arm": args.arm, "init_seed": args.init_seed, "data_seed": args.data_seed,
        "eval_seed": args.eval_seed, "elapsed_s": time.monotonic() - start,
        "pre_initial_hash": pre_initial_hash, "final_hash": parameter_hash(model),
        "evaluator_pretraining": pretrain, "evaluator_hash_unchanged": head_hash(model) == protected,
        "parameter_counts": {"total": total, "trainable": trainable,
                             "architecture_added": model.added_parameters},
        "architecture_change": {"A0": "none", "A1": "candidate-query-only",
            "A2": "candidate-gate-only", "A3": "downstream-capacity-only",
            "A4": "none; training-only 50-step scaffold"}[args.arm],
        "memory_law_change": "none", "model_inputs_added": [], "latent_z_input": False,
        "correct_action_labels": False, "memory_labels": False,
        "protected_evaluator": True, "training_log": logs,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"COMPLETED": args.out.name, "elapsed_s": summary["elapsed_s"]}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--arm", choices=("A0","A1","A2","A3","A4"), required=True)
    p.add_argument("--init-seed", type=int, required=True); p.add_argument("--data-seed", type=int, required=True)
    p.add_argument("--eval-seed", type=int, default=16601); p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=16); p.add_argument("--evaluator-steps", type=int, default=1000)
    p.add_argument("--checkpoints", type=int, nargs="+", default=list(CHECKPOINTS))
    p.add_argument("--device", default="cpu"); p.add_argument("--out", type=Path, required=True)
    train(p.parse_args())
