"""Frozen-upstream observed-only R0–R4 refit ceiling, no latent labels."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from etrcm.stage2c.world import ACTION, context_token, outcome_token, tensor_ids
from etrcm.stage2d.world import observed_only, training_experiences
from etrcm.stage2d4.model import classify, health_audit
from stage2d7_diagnostic import load


class ResidualAdapter(nn.Module):
    """Generic post-fusion 32x32 residual adapter; 1056 parameters."""

    def __init__(self, width: int):
        super().__init__()
        self.base = nn.SiLU()
        self.adapter = nn.Linear(width, width)
        nn.init.zeros_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)

    def forward(self, x):
        z = self.base(x)
        return z + self.adapter(z)


@torch.no_grad()
def training_features(model, *, seed, episodes=200, batch=16, lengths=(4, 6, 8)):
    """Cache candidate features from frozen dynamics; observed outcomes only."""
    features, targets = [], []
    for step in range(episodes):
        state = model.initial_state(batch, "cpu")
        length = lengths[(step + seed) % len(lengths)]
        for index in range(length):
            rows = training_experiences(seed * 100003 + step, batch, index, .70)
            items = observed_only(rows)
            ids = (
                torch.full((batch,), 0, dtype=torch.long),
                tensor_ids([x.color for x in items], "cpu"),
                tensor_ids([x.shape for x in items], "cpu"),
                tensor_ids([x.nuisance for x in items], "cpu"),
            )
            from etrcm.stage2c.world import ABSTRACT
            ids = (torch.full((batch,), ABSTRACT, dtype=torch.long), *ids[1:])
            for item in ids:
                state, _ = model.step(state, context_token(item))
            action = tensor_ids([x.action for x in items], "cpu")
            _, trace = model.candidate(state, action)
            features.append(trace["fusion_input"].detach().clone())
            targets.append(tensor_ids([x.outcome - 4 for x in items], "cpu"))
            action_ids = tensor_ids([ACTION[x.action] for x in items], "cpu")
            state, _ = model.step(state, context_token(action_ids))
            state, _ = model.step(state, outcome_token(
                tensor_ids([x.outcome for x in items], "cpu"), write=True))
    return torch.cat(features), torch.cat(targets)


def configure(model, arm):
    for p in model.parameters():
        p.requires_grad_(False)
    first = model.action_head.head[0]
    hdim = model.config.hidden_dim
    if arm == "R0":
        return [], 0
    if arm == "R4":
        width = first.out_features
        model.action_head.head[1] = ResidualAdapter(width)
        return list(model.action_head.head[1].parameters()), width * width + width
    if arm in {"R1", "R2"}:
        first.weight.requires_grad_(True)
        mask = torch.zeros_like(first.weight)
        if arm == "R1":
            mask[:, :hdim] = 1.
            first.bias.requires_grad_(True)
        else:
            mask[:, hdim:] = 1.
        first.weight.register_hook(lambda grad: grad * mask)
        params = [first.weight] + ([first.bias] if arm == "R1" else [])
        count = int(mask.sum()) + (first.bias.numel() if arm == "R1" else 0)
        return params, count
    if arm == "R3":
        first.bias.requires_grad_(True)
        return [first.bias], first.bias.numel()
    raise ValueError(arm)


def changed_parameters(before, model):
    after = model.state_dict()
    changed = []
    for key, old in before.items():
        if key not in after or not torch.equal(old, after[key]):
            changed.append(key)
    changed += [key for key in after if key not in before]
    return sorted(changed)


def fit_arm(original, feature, target, *, arm, eval_seed, steps=500, lr=.001,
            train_seed=0, reps=16):
    model = copy.deepcopy(original)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    parameters, count = configure(model, arm)
    if arm == "R0":
        health = health_audit(model, eval_seed, reps=reps)
        return {"arm": arm, "trainable_parameters": count, "heldout_health": health,
                "heldout_class": classify(health), "changed_parameters": [], "loss_log": []}
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.)
    generator = torch.Generator().manual_seed(train_seed + 113)
    logs = []
    model.train()
    for step in range(steps):
        index = torch.randint(feature.shape[0], (256,), generator=generator)
        logits = model.action_head.head[2](model.action_head.head[1](
            model.action_head.head[0](feature[index])))
        loss = F.cross_entropy(logits, target[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step + 1 in (1, 25, 100, 250, steps):
            logs.append({"step": step + 1, "observed_CE": float(loss.detach())})
    model.eval()
    health = health_audit(model, eval_seed, reps=reps)
    changed = changed_parameters(before, model)
    permitted = {
        "R1": {"action_head.head.0.weight", "action_head.head.0.bias"},
        "R2": {"action_head.head.0.weight"},
        "R3": {"action_head.head.0.bias"},
        "R4": {"action_head.head.1.adapter.weight", "action_head.head.1.adapter.bias"},
    }[arm]
    # R4 replacement adds new keys but does not remove trainable old coordinates.
    if arm != "R4" and not set(changed) <= permitted:
        raise AssertionError((arm, changed))
    if arm == "R4" and (set(changed) != permitted or any(
            not torch.equal(old, model.state_dict()[key]) for key, old in before.items())):
        raise AssertionError((arm, changed))
    hdim = model.config.hidden_dim
    first = model.action_head.head[0]
    mask_exact = True
    if arm == "R1":
        mask_exact = torch.equal(first.weight[:, hdim:], before["action_head.head.0.weight"][:, hdim:])
    if arm == "R2":
        mask_exact = torch.equal(first.weight[:, :hdim], before["action_head.head.0.weight"][:, :hdim])
    if not mask_exact:
        raise AssertionError(f"{arm} protected slice changed")
    return {"arm": arm, "trainable_parameters": count, "heldout_health": health,
            "heldout_class": classify(health), "changed_parameters": changed,
            "parameter_mask_exact": mask_exact, "loss_log": logs}


def one(path, out, *, episodes=200, steps=500, reps=16, seed=37101):
    torch.set_num_threads(1)
    model, checkpoint = load(path)
    eval_seed = checkpoint["eval_seed"]
    if seed == eval_seed or seed + 700001 == eval_seed:
        raise AssertionError("training/evaluation seed collision")
    with torch.no_grad():
        feature, target = training_features(model, seed=seed, episodes=episodes)
    results = {arm: fit_arm(model, feature, target, arm=arm, eval_seed=eval_seed,
                            steps=steps, train_seed=seed, reps=reps)
               for arm in ("R0", "R1", "R2", "R3", "R4")}
    payload = {"checkpoint": str(path), "run": path.parent.name,
               "protocol": {"frozen_upstream": True, "train_seed": seed,
                            "eval_seed": eval_seed, "training_split": "train",
                            "heldout_split": "novel", "episodes": episodes,
                            "batch": 16, "lengths": [4, 6, 8], "steps_per_arm": steps,
                            "optimizer": "AdamW", "lr": .001, "weight_decay": 0.,
                            "objective": "observed consequence cross entropy",
                            "latent_z_input": False, "correct_action_label": False,
                            "memory_label": False},
               "training_examples": int(feature.shape[0]), "arms": results}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(json.dumps({"run": payload["run"], "classes": {k: v["heldout_class"]
                                                  for k, v in results.items()}}), flush=True)
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--reps", type=int, default=16)
    p.add_argument("--train-seed", type=int, default=37101)
    args = p.parse_args()
    one(args.checkpoint, args.out, episodes=args.episodes, steps=args.steps,
        reps=args.reps, seed=args.train_seed)
