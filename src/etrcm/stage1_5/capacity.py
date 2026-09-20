"""Sparse scaling: finite associative capacity and trained predictive gaps."""

from __future__ import annotations

import torch
from torch.nn import functional as Fnn

from etrcm.stage1_4.world import generate_world
from .model import AnatomicalETRCM


COUNTS = (1, 2, 4, 8, 16, 32, 64, 128)
DISTRACTORS = (0, 32, 128, 512, 2048, 8192)


def _unit(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def _delta_write(F: torch.Tensor, M: torch.Tensor, key: torch.Tensor,
                 value: torch.Tensor, eta: float) -> torch.Tensor:
    current = torch.einsum("...vk,...k->...v", F + M, key)
    return F + eta * torch.einsum("...v,...k->...vk", value - current, key)


@torch.no_grad()
def evaluate_associative_capacity(model: AnatomicalETRCM, *, seed: int, run_id: str,
                                  device: torch.device, counts: tuple[int, ...] = COUNTS,
                                  distractor_grid: tuple[int, ...] = DISTRACTORS) -> list[dict[str, object]]:
    """Synthetic continuous vectors isolate cell capacity, not learned routing."""
    m = model.config.key_dim
    generator = torch.Generator(device=device).manual_seed(seed * 100_000 + 22001 + m)
    keys = _unit(torch.randn(128, m, generator=generator, device=device))
    values = _unit(torch.randn(128, m, generator=generator, device=device))
    F = torch.zeros(m, m, device=device)
    M = torch.zeros_like(F)
    snapshots: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for item in range(128):
        F = _delta_write(F, M, keys[item], values[item], model.config.eta_external)
        fast_read = F @ keys[item]
        transfer = model.config.gamma * torch.outer(fast_read, keys[item])
        F, M = F - transfer, M + transfer
        F, M = model.config.rho_fast * F, model.config.rho_slow * M
        if item + 1 in counts:
            snapshots[item + 1] = (F.clone(), M.clone())
    F = torch.stack([snapshots[count][0] for count in counts])
    M = torch.stack([snapshots[count][1] for count in counts])
    rows: list[dict[str, object]] = []

    def record(distractors: int) -> None:
        full_reads = torch.einsum("cvk,ik->civ", F + M, keys)
        slow_reads = torch.einsum("cvk,ik->civ", M, keys)
        full_cos = Fnn.cosine_similarity(full_reads, values[None], dim=-1)
        slow_cos = Fnn.cosine_similarity(slow_reads, values[None], dim=-1)
        similarity = torch.einsum("civ,jv->cij", _unit(full_reads), values)
        for index, count in enumerate(counts):
            scores = similarity[index, :count, :count]
            correct = torch.arange(count, device=device)
            rows.append({
                "run_id": run_id, "experiment": "A4_associative_capacity", "model": model.mode,
                "seed": seed, "hidden_dim": model.config.hidden_dim, "memory_dim": m,
                "stored_item_count": count, "distractor_count": distractors,
                "retrieval_top1_accuracy": float((scores.argmax(-1) == correct).float().mean()),
                "retrieval_cosine": float(full_cos[index, :count].mean()),
                "M_only_cosine": float(slow_cos[index, :count].mean()),
                "F_norm": float(F[index].norm()), "M_norm": float(M[index].norm()),
                "state_bytes": model.persistent_state_bytes(),
                "parameter_count": model.trainable_parameters(),
                "compute_budget": count + distractors,
                "approx_memory_multiply_adds": (count + distractors) * 2 * m * m,
                "precision_mode": "fp32", "probe_type": "mechanistic_untrained_vectors",
                "learned_prediction_claim": False,
            })

    record(0)
    for tick in range(1, max(distractor_grid) + 1):
        key = _unit(torch.randn(m, generator=generator, device=device))
        value = _unit(torch.randn(m, generator=generator, device=device))
        F = _delta_write(F, M, key, value, model.config.eta_external)
        F, M = model.config.rho_fast * F, model.config.rho_slow * M
        if tick in distractor_grid:
            record(tick)
    return rows


@torch.no_grad()
def evaluate_capacity_prediction(model: AnatomicalETRCM, *, seed: int, run_id: str,
                                 device: torch.device, batch: int = 8,
                                 max_gap: int = 8192,
                                 distractor_grid: tuple[int, ...] = DISTRACTORS) -> list[dict[str, object]]:
    """One paired distractor stream; clone prefixes at registered N values."""
    world = generate_world("long_gap_relation", batch=batch, length=16,
                           seed=seed * 100_000 + 23001, device=device, gap=max_gap)
    state = model.initial_state(batch, device=device)
    for index in range(4):
        state, _ = model.step(state, world.events[index])
    target = world.future(world.bridge_index, 1)
    target_key = model.target_key(world.events[world.bridge_index].value_id)
    rows: list[dict[str, object]] = []
    for distractors in range(max_gap + 1):
        if distractors in distractor_grid:
            snapshot = state.clone()
            retained_M = torch.einsum("bvk,bk->bv", snapshot.M, target_key)
            historical_value = model.value_vector(target)
            retention = Fnn.cosine_similarity(retained_M, historical_value, dim=-1)
            future, _ = model.step(snapshot, world.events[world.bridge_index])
            for _ in range(4):
                future, _ = model.step(future, None)
            logits = model.predict_logits(future)[1]
            ce = Fnn.cross_entropy(logits, target, reduction="none")
            for ep in range(batch):
                rows.append({
                    "run_id": run_id, "experiment": "A4_capacity_prediction",
                    "model": model.mode, "seed": seed, "episode": ep,
                    "world_family": world.family,
                    "hidden_dim": model.config.hidden_dim, "memory_dim": model.config.key_dim,
                    "distractor_count": distractors, "stored_item_count": int(world.exposure_count[ep]),
                    "future_CE": float(ce[ep]),
                    "behavioral_accuracy": float(logits[ep].argmax() == target[ep]),
                    "M_only_retention_cosine": float(retention[ep]),
                    "F_norm": float(snapshot.F[ep].norm()),
                    "M_norm": float(snapshot.M[ep].norm()),
                    "H_norm": float(snapshot.H[ep].norm()),
                    "state_bytes": model.persistent_state_bytes(),
                    "parameter_count": model.trainable_parameters(),
                    "compute_budget": 4 + distractors + 5,
                    "precision_mode": "fp32", "probe_type": "trained_prediction",
                })
        if distractors < max_gap:
            state, _ = model.step(state, world.events[4 + distractors])
    return rows
