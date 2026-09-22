"""Conditionally authorized C0–C5 observed-only compatibility training."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from etrcm.stage2d.model import parameter_hash, pretrain_evaluator
from etrcm.stage2d.world import training_experiences
from etrcm.stage2d1.engine import play_routed
from etrcm.stage2d7.compat import CompatibilityModel, configure_training, protected_exact


STEPS = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500)


def grad_norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().square().sum())
                         for p in parameters if p.grad is not None))


def save(model, args, step, out, pretrain, snapshot, gradients):
    torch.save({"model": model.state_dict(), "arm": args.arm,
                "rank": args.rank, "init_seed": args.init_seed,
                "data_seed": args.data_seed, "eval_seed": args.eval_seed,
                "steps": step, "gamma": .50, "rho_fast": .97,
                "rho_slow": .9995, "evaluator_pretraining": pretrain,
                "compatibility_lr_ratio": args.lr_ratio,
                "progressive_freeze_step": args.freeze_step,
                "protected_evaluator_snapshot": snapshot,
                "gradients_preceding_update": gradients,
                "objective": "observed consequence CE + 0.001 H penalty",
                "latent_z_input": False, "correct_action_label": False,
                "memory_label": False, "paired_warmup": False,
                "memory_law_change": "none"},
               out / f"checkpoint_{step:04d}.pt")


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.init_seed)
    random.seed(args.init_seed)
    model = CompatibilityModel(args.arm, rank=args.rank)
    pre_initial_hash = parameter_hash(model)
    pretrain = pretrain_evaluator(model, args.init_seed, args.device, args.evaluator_steps)
    groups, protected = configure_training(model, args.arm, lr_ratio=args.lr_ratio)
    optimizer = torch.optim.AdamW(groups)
    args.out.mkdir(parents=True, exist_ok=True)
    save(model, args, 0, args.out, pretrain, protected, {"all": 0., "core": 0.})
    logs = []
    start = time.monotonic()
    freeze_snapshot = None
    for step in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        state = model.initial_state(args.batch, args.device)
        losses, penalties = [], []
        length = (4, 6, 8)[(step + args.data_seed) % 3]
        for index in range(length):
            rows = training_experiences(args.data_seed * 100003 + step,
                                        args.batch, index, .70)
            state, loss, _ = play_routed(model, state, rows)
            losses.append(loss)
            penalties.append(state.H.square().mean())
        ce = torch.stack(losses).mean()
        objective = ce + .001 * torch.stack(penalties).mean()
        if not torch.isfinite(objective):
            raise RuntimeError("nonfinite objective")
        objective.backward()
        gradients = {"all": grad_norm(model.parameters()),
                     "core": grad_norm(model.core.parameters())}
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        completed = step + 1
        if args.arm == "C2" and completed == args.freeze_step:
            freeze_snapshot = {k: v.detach().clone()
                               for k, v in model.action_head.state_dict().items()}
            model.action_head.head[0].weight.requires_grad_(False)
            model.action_head.head[0].bias.requires_grad_(False)
        reference = freeze_snapshot if freeze_snapshot is not None else protected
        if not protected_exact(model, args.arm, reference,
                               after_freeze=freeze_snapshot is not None):
            raise AssertionError("undeclared protected evaluator parameter changed")
        if completed in STEPS:
            save(model, args, completed, args.out, pretrain, protected, gradients)
            row = {"step": completed, "CE": float(ce.detach()),
                   "objective": float(objective.detach()),
                   "H_norm": float(state.H.detach().norm(dim=(-2, -1)).mean()),
                   "F_norm": float(state.F.detach().norm(dim=(-2, -1)).mean()),
                   "M_norm": float(state.M.detach().norm(dim=(-2, -1)).mean()),
                   "elapsed_s": time.monotonic() - start}
            logs.append(row)
            print(json.dumps(row), flush=True)
    summary = {"arm": args.arm, "rank": args.rank,
               "init_seed": args.init_seed, "data_seed": args.data_seed,
               "eval_seed": args.eval_seed, "pre_initial_hash": pre_initial_hash,
               "final_hash": parameter_hash(model), "evaluator_pretraining": pretrain,
               "protected_exact": protected_exact(model, args.arm,
                   freeze_snapshot if freeze_snapshot is not None else protected,
                   after_freeze=freeze_snapshot is not None),
               "objective": "observed-only consequence CE + original H penalty",
               "architecture_change": args.arm,
               "memory_law_change": "none",
               "model_inputs_added": [], "correct_action_labels": False,
               "memory_labels": False, "latent_z_input": False,
               "paired_observational_scaffold": False,
               "training_log": logs, "elapsed_s": time.monotonic() - start}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"COMPLETED": {k: v for k, v in summary.items()
                                    if k != "training_log"}}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("C0", "C1", "C2", "C3", "C4", "C5"), required=True)
    p.add_argument("--rank", type=int, choices=(0, 2, 4), default=4)
    p.add_argument("--lr-ratio", type=float, choices=(.01, .1, .25), default=.1)
    p.add_argument("--freeze-step", type=int, choices=(100, 300, 500), default=300)
    p.add_argument("--init-seed", type=int, required=True)
    p.add_argument("--data-seed", type=int, required=True)
    p.add_argument("--eval-seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--evaluator-steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    train(p.parse_args())
