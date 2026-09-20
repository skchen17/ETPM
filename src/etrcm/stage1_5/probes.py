"""Train-only frozen-state probes; these are observational, not causal claims."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from torch.nn import functional as Fnn

from etrcm.stage1_3.events import evidence_event
from etrcm.stage1_4.training import prefix_index, run_prefix
from etrcm.stage1_4.world import FAMILIES, generate_world
from .model import AnatomicalETRCM


def _ridge_predict(train: np.ndarray, test: np.ndarray,
                   train_target: np.ndarray, *, ridge: float = 1.0) -> np.ndarray:
    """Dual ridge with intercept/standardization fitted on train only."""
    mean = train.mean(0, keepdims=True)
    scale = train.std(0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    x = np.clip((train - mean) / scale, -10, 10).astype(np.float64)
    z = np.clip((test - mean) / scale, -10, 10).astype(np.float64)
    target_mean = train_target.mean(0, keepdims=True)
    centered = train_target - target_mean
    kernel = x @ x.T
    alpha = np.linalg.solve(kernel + ridge * np.eye(len(x)), centered)
    return target_mean + (z @ x.T) @ alpha


def _ridge_test_weights(train: np.ndarray, test: np.ndarray, ridge: float) -> np.ndarray:
    mean = train.mean(0, keepdims=True)
    scale = train.std(0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    x = np.clip((train - mean) / scale, -10, 10).astype(np.float64)
    z = np.clip((test - mean) / scale, -10, 10).astype(np.float64)
    kernel = x @ x.T + ridge * np.eye(len(x))
    return np.linalg.solve(kernel, x @ z.T).T


def _probe_scores(train: np.ndarray, test: np.ndarray,
                  y_train: np.ndarray, y_test: np.ndarray,
                  *, classes: int, ridge: float) -> tuple[float, float, float]:
    onehot = np.eye(classes, dtype=np.float64)[y_train]
    scores = _ridge_predict(train, test, onehot, ridge=ridge)
    logits = torch.from_numpy(scores * 8.0)
    target = torch.from_numpy(y_test)
    ce = float(Fnn.cross_entropy(logits, target))
    accuracy = float((logits.argmax(-1) == target).double().mean())
    counts = np.bincount(y_train, minlength=classes).astype(np.float64) + 1.0
    frequency = counts / counts.sum()
    constant_ce = float(-np.log(frequency[y_test]).mean())
    return ce, accuracy, constant_ce


def _probe_scores_with_weights(weights: np.ndarray,
                               y_train: np.ndarray, y_test: np.ndarray,
                               classes: int) -> tuple[float, float, float]:
    onehot = np.eye(classes, dtype=np.float64)[y_train]
    mean = onehot.mean(0, keepdims=True)
    scores = mean + weights @ (onehot - mean)
    logits = torch.from_numpy(scores * 8.0)
    target = torch.from_numpy(y_test)
    ce = float(Fnn.cross_entropy(logits, target))
    accuracy = float((logits.argmax(-1) == target).double().mean())
    counts = np.bincount(y_train, minlength=classes).astype(np.float64) + 1
    frequency = counts / counts.sum()
    baseline = float(-np.log(frequency[y_test]).mean())
    return ce, accuracy, baseline


@torch.no_grad()
def evaluate_timescales(model: AnatomicalETRCM, *, seed: int, run_id: str,
                        device: torch.device, train_episodes: int = 512,
                        test_episodes: int = 256, ridge: float = 1.0,
                        lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
                        ) -> list[dict[str, object]]:
    batch = train_episodes + test_episodes
    length = max(lags) + 1
    gen = torch.Generator(device="cpu").manual_seed(seed * 100_000 + 20001)
    keys = torch.randint(0, 16, (length, batch), generator=gen).to(device)
    values = torch.randint(0, model.config.symbol_count, (length, batch), generator=gen).to(device)
    original = model.config
    variants = (
        ("primary", original.rho_fast, original.rho_slow, original.gamma),
        ("equal_fast_decay", original.rho_fast, original.rho_fast, original.gamma),
        ("equal_slow_decay", original.rho_slow, original.rho_slow, original.gamma),
        ("no_transfer", original.rho_fast, original.rho_slow, 0.0),
    )
    rows: list[dict[str, object]] = []
    try:
        for variant, rho_f, rho_m, gamma in variants:
            model.config = replace(original, rho_fast=rho_f, rho_slow=rho_m, gamma=gamma)
            state = model.initial_state(batch, device=device)
            for tick in range(length):
                state, _ = model.step(state, evidence_event(keys[tick], values[tick]))
            features = {
                component: getattr(state, component).flatten(1).detach().cpu().numpy()
                for component in "HFM"
            }
            weights = {
                component: _ridge_test_weights(feature[:train_episodes], feature[train_episodes:], ridge)
                for component, feature in features.items()
            }
            for lag in lags:
                target = values[length - lag].detach().cpu().numpy()
                for component, feature in features.items():
                    ce, accuracy, baseline = _probe_scores_with_weights(
                        weights[component], target[:train_episodes], target[train_episodes:],
                        classes=model.config.symbol_count,
                    )
                    rows.append({
                        "run_id": run_id, "experiment": "A3_timescale", "model": model.mode,
                        "seed": seed, "world_family": "iid_external_history",
                        "decay_variant": variant, "rho_fast": rho_f, "rho_slow": rho_m,
                        "gamma": gamma, "component": component, "history_lag": lag,
                        "probe_train_episodes": train_episodes,
                        "probe_test_episodes": test_episodes,
                        "ridge": ridge, "heldout_CE": ce, "constant_CE": baseline,
                        "heldout_decoding_gain": baseline - ce,
                        "heldout_accuracy": accuracy,
                        "prediction_target": "historical_visible_value",
                        "mutual_information_claim": False,
                        "parameter_count": model.trainable_parameters(),
                        "state_bytes": model.persistent_state_bytes(),
                        "precision_mode": "fp32",
                    })
    finally:
        model.config = original
    return rows


@torch.no_grad()
def evaluate_observability(model: AnatomicalETRCM, *, seed: int, run_id: str,
                           device: torch.device, train_per_family: int = 64,
                           test_per_family: int = 32,
                           ridge: float = 1.0) -> list[dict[str, object]]:
    features: dict[str, list[np.ndarray]] = {name: [] for name in "HFM"}
    outcomes: dict[str, list[np.ndarray]] = {name: [] for name in ("future_H", "future_logits", "event")}
    partitions: list[np.ndarray] = []
    family_names: list[str] = []
    for family_id, family in enumerate(FAMILIES):
        batch = train_per_family + test_per_family
        world = generate_world(family, batch=batch, length=16,
                               seed=seed * 100_000 + 21001 + family_id, device=device)
        prefix = prefix_index(world, 5)
        state, _ = run_prefix(model, world, prefix=prefix, null_ticks=0)
        future = state.clone()
        for _ in range(4):
            future, _ = model.step(future, None)
        for component in "HFM":
            features[component].append(getattr(state, component).flatten(1).cpu().numpy())
        outcomes["future_H"].append(future.H.flatten(1).cpu().numpy())
        outcomes["future_logits"].append(model.predict_logits(future)[1].cpu().numpy())
        outcomes["event"].append(world.future(prefix, 1).cpu().numpy())
        partitions.append(np.arange(batch) < train_per_family)
        family_names.extend([family] * batch)
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in features.items()}
    targets = {name: np.concatenate(parts, axis=0) for name, parts in outcomes.items()}
    train_mask = np.concatenate(partitions)
    family_names_array = np.asarray(family_names)
    rows: list[dict[str, object]] = []
    for condition in ("H", "HF", "HM", "HFM"):
        x = np.concatenate([arrays[name] for name in condition], axis=1)
        yh = _ridge_predict(x[train_mask], x[~train_mask], targets["future_H"][train_mask], ridge=ridge)
        yl = _ridge_predict(x[train_mask], x[~train_mask], targets["future_logits"][train_mask], ridge=ridge)
        event_scores = _ridge_predict(
            x[train_mask], x[~train_mask],
            np.eye(model.config.symbol_count)[targets["event"][train_mask]], ridge=ridge,
        )
        event_logits = torch.from_numpy(event_scores * 8.0)
        event_targets = torch.from_numpy(targets["event"][~train_mask])
        per_ce = Fnn.cross_entropy(event_logits, event_targets, reduction="none").numpy()
        per_acc = (event_logits.argmax(-1) == event_targets).numpy()
        test_families = family_names_array[~train_mask]
        for family in FAMILIES:
            mask = test_families == family
            rows.append({
                "run_id": run_id, "experiment": "A6_observability", "model": model.mode,
                "seed": seed, "world_family": family, "feature_condition": condition,
                "future_H_MSE": float(np.mean((yh[mask] - targets["future_H"][~train_mask][mask]) ** 2)),
                "future_logits_MSE": float(np.mean((yl[mask] - targets["future_logits"][~train_mask][mask]) ** 2)),
                "future_event_CE": float(per_ce[mask].mean()),
                "future_event_accuracy": float(per_acc[mask].mean()),
                "train_episodes": int(train_mask.sum()), "test_episodes": int((~train_mask).sum()),
                "ridge": ridge, "observational_only": True,
                "parameter_count": model.trainable_parameters(),
                "state_bytes": model.persistent_state_bytes(), "precision_mode": "fp32",
            })
    return rows
