"""Stage 2D.3 temporary paired-action conditional-warmup training."""

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
from etrcm.stage2d1.protocol import CHECKPOINT_STEPS, FROZEN_D1
from etrcm.stage2d3.curriculum import PairedWarmup, paired_action_loss, paired_observational_targets


def grad_norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None))


def save_checkpoint(model, args, step, out, protected_hash, gradients, aux_weight):
    torch.save({
        "model": model.state_dict(), "seed": args.init_seed, "init_seed": args.init_seed,
        "data_seed": args.data_seed, "eval_seed": args.eval_seed, "steps": step,
        "arm": "A2", "variant": "full", "gamma": args.gamma,
        "rho_fast": args.rho_fast, "rho_slow": args.rho_slow,
        "train_p": args.train_p, "curriculum": args.arm,
        "paired_warmup_steps": args.warmup_steps,
        "protected_evaluator_hash": protected_hash,
        "gradients_preceding_update": gradients, "auxiliary_weight": aux_weight,
    }, out / f"checkpoint_{step:04d}.pt")


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.init_seed); random.seed(args.init_seed)
    model = Stage2DModel(variant="full", gamma=args.gamma, rho_fast=args.rho_fast,
                         rho_slow=args.rho_slow).to(args.device)
    pre_initial_hash = parameter_hash(model)
    pretrain = pretrain_evaluator(model, args.init_seed, args.device, args.evaluator_steps)
    for parameter in model.action_head.parameters():
        parameter.requires_grad_(False)
    protected_hash = head_hash(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.01)
    protocol = PairedWarmup(args.warmup_steps, args.arm) if args.arm != "C0" else None
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints = set(args.checkpoints)
    zero_gradients = {"all": 0.0, "core": 0.0}
    save_checkpoint(model, args, 0, args.out, protected_hash, zero_gradients,
                    0.0 if protocol is None else protocol.auxiliary_weight(0))
    logs = []; start = time.monotonic(); lengths = tuple(args.lengths)
    for step in range(args.steps):
        model.train(); optimizer.zero_grad(set_to_none=True)
        state = model.initial_state(args.batch, args.device)
        losses, penalties, pair_losses = [], [], []
        length = lengths[(step + args.data_seed) % len(lengths)]
        aux_weight = 0.0 if protocol is None else protocol.auxiliary_weight(step)
        for index in range(length):
            rows = training_experiences(args.data_seed * 100003 + step, args.batch, index, args.train_p)
            if aux_weight:
                targets = paired_observational_targets(rows, args.data_seed * 31 + step, index)
                pair_losses.append(paired_action_loss(model, state, rows, targets, args.arm))
            state, loss, _ = play_routed(model, state, rows)
            losses.append(loss); penalties.append(state.H.square().mean())
        ce = torch.stack(losses).mean()
        pair = torch.stack(pair_losses).mean() if pair_losses else ce.detach() * 0.
        objective = ce + aux_weight * pair + .001 * torch.stack(penalties).mean()
        if not torch.isfinite(objective):
            raise RuntimeError("nonfinite objective")
        objective.backward()
        gradients = {"all": grad_norm(model.parameters()), "core": grad_norm(model.core.parameters())}
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if head_hash(model) != protected_hash:
            raise AssertionError("protected evaluator changed")
        completed = step + 1
        if completed in checkpoints:
            next_aux = 0.0 if protocol is None else protocol.auxiliary_weight(completed)
            save_checkpoint(model, args, completed, args.out, protected_hash, gradients, next_aux)
            row = {"step": completed, "CE": float(ce.detach()), "pair_CE": float(pair.detach()),
                   "objective": float(objective.detach()), "auxiliary_weight": aux_weight,
                   "H_norm": float(state.H.detach().norm(dim=(-2, -1)).mean()),
                   "F_norm": float(state.F.detach().norm(dim=(-2, -1)).mean()),
                   "M_norm": float(state.M.detach().norm(dim=(-2, -1)).mean()),
                   "elapsed_s": time.monotonic() - start}
            logs.append(row); print(json.dumps(row), flush=True)
    summary = {
        "init_seed": args.init_seed, "data_seed": args.data_seed, "eval_seed": args.eval_seed,
        "arm": args.arm, "warmup_steps": args.warmup_steps,
        "pre_initial_hash": pre_initial_hash, "final_hash": parameter_hash(model),
        "evaluator_pretraining": pretrain, "evaluator_hash_unchanged": head_hash(model) == protected_hash,
        "architecture_change": "none", "memory_law_change": "none",
        "model_inputs_added": [], "correct_action_labels": False, "memory_labels": False,
        "paired_observational_scaffold": args.arm != "C0",
        "scaffold_generation": "two legal simulator consequence samples; simulator latent never enters model",
        "auxiliary_exactly_zero_after_warmup": protocol is None or protocol.auxiliary_weight(args.warmup_steps) == 0.,
        "evaluation_scaffold": False, "original_objective_fraction": (args.steps - args.warmup_steps) / args.steps if protocol else 1.,
        "training_log": logs, "elapsed_s": time.monotonic() - start,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"COMPLETED": summary | {"training_log": "omitted"}}), flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--init-seed", type=int, required=True); p.add_argument("--data-seed", type=int, required=True)
    p.add_argument("--eval-seed", type=int, default=15601); p.add_argument("--arm", choices=("C0", "C1", "C2", "C3"), required=True)
    p.add_argument("--warmup-steps", type=int, choices=(50, 100, 200, 300), default=100)
    p.add_argument("--steps", type=int, default=FROZEN_D1.steps); p.add_argument("--evaluator-steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=16); p.add_argument("--train-p", type=float, default=.70)
    p.add_argument("--lengths", type=int, nargs="+", default=list(FROZEN_D1.lengths))
    p.add_argument("--checkpoints", type=int, nargs="+", default=list(CHECKPOINT_STEPS))
    p.add_argument("--gamma", type=float, default=.50); p.add_argument("--rho-fast", type=float, default=.97)
    p.add_argument("--rho-slow", type=float, default=.9995); p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, required=True)
    return p


if __name__ == "__main__":
    train(parser().parse_args())
