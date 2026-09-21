"""Independent Stage 2D training for protected and comparison arms."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from etrcm.stage2d.model import (Stage2DModel, head_hash, parameter_hash, play,
                                 pretrain_evaluator)
from etrcm.stage2d.world import training_experiences


def grad_norm(parameters) -> float:
    return math.sqrt(sum(float(p.grad.detach().square().sum()) for p in parameters
                         if p.grad is not None))


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = Stage2DModel(variant=args.variant, gamma=args.gamma,
                         rho_fast=args.rho_fast, rho_slow=args.rho_slow).to(args.device)
    initial_hash = parameter_hash(model)
    pretrain = None
    if args.arm == "A2":
        pretrain = pretrain_evaluator(model, args.seed, args.device, args.evaluator_steps)
        for parameter in model.action_head.parameters():
            parameter.requires_grad_(False)
    protected_hash = head_hash(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=0.001, weight_decay=0.01)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    logs = []
    last_gradients = {}
    for step in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        state = model.initial_state(args.batch, args.device)
        losses, penalties = [], []
        lengths = tuple(int(item) for item in args.lengths.split(","))
        if not lengths or any(item < 1 for item in lengths):
            raise ValueError("--lengths must contain positive integers")
        length = lengths[(step + args.seed) % len(lengths)]
        for index in range(length):
            rows = training_experiences(args.seed * 100003 + step, args.batch, index,
                                        args.train_p)
            state, loss, _ = play(model, state, rows)
            losses.append(loss)
            penalties.append(state.H.square().mean())
        ce = torch.stack(losses).mean()
        objective = ce + 0.001 * torch.stack(penalties).mean()
        if not torch.isfinite(objective):
            raise RuntimeError("nonfinite training objective")
        objective.backward()
        last_gradients = {
            "all": grad_norm(model.parameters()),
            "H_core": grad_norm(list(model.core.core_in.parameters()) +
                                list(model.core.core_out.parameters()) +
                                list(model.core.core_gate.parameters())),
            "q_F": grad_norm(model.core.q_fast_projection.parameters()),
            "q_M": grad_norm(model.core.q_slow_projection.parameters()),
            "read_gate": grad_norm(model.core.two_way_gate.parameters()),
            "action_head": grad_norm(model.action_head.parameters()),
        }
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if args.arm == "A2" and head_hash(model) != protected_hash:
            raise AssertionError("protected evaluator changed")
        if step == 0 or (step + 1) % 100 == 0:
            row = {"step": step + 1, "CE": float(ce.detach()),
                   "objective": float(objective.detach()),
                   "H_norm": float(state.H.detach().norm(dim=(-2, -1)).mean()),
                   "F_norm": float(state.F.detach().norm(dim=(-2, -1)).mean()),
                   "M_norm": float(state.M.detach().norm(dim=(-2, -1)).mean()),
                   "gradients_preclip": last_gradients,
                   "elapsed_s": time.monotonic() - start}
            logs.append(row)
            print(json.dumps(row), flush=True)
    checkpoint = {
        "model": model.state_dict(), "seed": args.seed, "steps": args.steps,
        "arm": args.arm, "variant": args.variant, "gamma": args.gamma,
        "rho_fast": args.rho_fast, "rho_slow": args.rho_slow,
        "train_p": args.train_p, "protected_evaluator_hash": protected_hash,
    }
    torch.save(checkpoint, args.out / "checkpoint.pt")
    summary = {
        **{key: checkpoint[key] for key in checkpoint if key != "model"},
        "initial_model_hash": initial_hash,
        "final_model_hash": parameter_hash(model),
        "evaluator_pretraining": pretrain,
        "evaluator_frozen": args.arm == "A2",
        "evaluator_hash_final": head_hash(model),
        "evaluator_hash_unchanged": head_hash(model) == protected_hash,
        "latent_z_lifetime_input": False,
        "correct_action_label_used": False,
        "objective": "observed consequence CE + 0.001 H penalty",
        "architecture_change": "none; existing Stage2C.3 transition selected by variant",
        "training_lengths": [int(item) for item in args.lengths.split(",")],
        "last_gradients_preclip": last_gradients,
        "training_log": logs,
        "elapsed_s": time.monotonic() - start,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"COMPLETED": {"arm": args.arm, "variant": args.variant,
                                    "seed": args.seed, "elapsed_s": summary["elapsed_s"]}}),
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("A0", "A2"), required=True)
    parser.add_argument("--variant", choices=("full", "gamma_zero", "f_only", "no_memory", "gru"),
                        default="full")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--evaluator-steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--train-p", type=float, default=0.65)
    parser.add_argument("--lengths", default="16,32,64")
    parser.add_argument("--gamma", type=float, default=0.12)
    parser.add_argument("--rho-fast", type=float, default=0.97)
    parser.add_argument("--rho-slow", type=float, default=0.9995)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    train(parser.parse_args())
