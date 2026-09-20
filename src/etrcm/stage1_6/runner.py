"""Paired training and finite read/lesion evaluation; no query supervision."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass

import torch
from torch.nn import functional as Fnn

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_4.model import Stage14Config
from etrcm.stage1_5.model import AnatomicalETRCM
from .world import ComposedWorld


ARMS = {"learned": "B5_separate", "oracle": "B5_separate",
        "curriculum": "B5_separate", "no_memory": "B0_no_memory",
        "gru": "B1_gru", "single_memory": "B2_single_memory"}
CONDITIONS = ("learned", "oracle", "zero", "random", "shuffled", "M_lesion", "F_lesion")


def oracle_probability(step: int, total_steps: int) -> float:
    if total_steps <= 0 or step < 0:
        raise ValueError("invalid training step")
    return (1.0, 0.75, 0.5, 0.25, 0.0)[min(4, 5 * step // total_steps)]


def choose_oracle(step: int, total_steps: int, *, seed: int) -> bool:
    probability = oracle_probability(step, total_steps)
    # Independent deterministic Bernoulli, not dependent on a future label.
    digest = hashlib.sha256(f"stage16-read-{seed}-{step}".encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2**64 < probability


def make_model(config: Stage14Config, arm: str, *, seed: int, device: torch.device) -> AnatomicalETRCM:
    if arm not in ARMS:
        raise ValueError(arm)
    # Same initial parameters for the three B5 arms of a seed.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return AnatomicalETRCM(config, ARMS[arm]).to(device)


def scrub(model: AnatomicalETRCM, state: LearnedState) -> LearnedState:
    after = model.reset_active(state)
    if not (torch.equal(after.F, state.F) and torch.equal(after.M, state.M)):
        raise AssertionError("H scrub modified memory")
    return after


def historical_oracle(model: AnatomicalETRCM, state: LearnedState,
                      world: ComposedWorld) -> torch.Tensor:
    """Read only the observed past B->A at the early-history boundary."""
    key = model.target_key(world.key)
    return torch.einsum("bvk,bk->bv", state.F + state.M, key)


def derangement(batch: int, device: torch.device) -> torch.Tensor:
    if batch < 2:
        raise ValueError("shuffle requires at least two episodes")
    return torch.arange(batch, device=device).roll(1)


def norm_matched_random(read: torch.Tensor, *, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(read.shape, generator=gen, dtype=read.dtype).to(read.device)
    return Fnn.normalize(noise, dim=-1) * read.norm(dim=-1, keepdim=True)


def _step_bridge(model: AnatomicalETRCM, state: LearnedState, world: ComposedWorld,
                 oracle: torch.Tensor, condition: str,
                 *, random_seed: int) -> tuple[LearnedState, dict[str, torch.Tensor]]:
    event = world.events[world.bridge_index]
    if condition in {"learned", "M_lesion", "F_lesion"}:
        return model.step(state, event)
    if model.mode != "B5_separate":
        raise ValueError("read clamps are defined only for B5")
    if condition == "zero":
        slow = torch.zeros_like(oracle)
    elif condition == "oracle":
        slow = oracle
    elif condition == "random":
        slow = norm_matched_random(oracle, seed=random_seed)
    elif condition == "shuffled":
        slow = oracle[derangement(oracle.shape[0], oracle.device)]
    else:
        raise ValueError(condition)
    return model.step_with_read(state, event, fast_read_override=torch.zeros_like(slow),
                                slow_read_override=slow)


def run_episode(model: AnatomicalETRCM, world: ComposedWorld, *, condition: str,
                random_seed: int = 0, collect_gradient: bool = False):
    state = model.initial_state(world.key.shape[0], device=world.key.device)
    for t in range(world.early_exposures):
        state, _ = model.step(state, world.events[t])
    oracle = historical_oracle(model, state, world)
    state = scrub(model, state)
    # Lesion at the scrub boundary, before any post-scrub read can copy the
    # lesioned component into H. The other component and H are untouched.
    if condition in {"M_lesion", "F_lesion"}:
        state = LearnedState(state.H,
                             torch.zeros_like(state.F) if condition == "F_lesion" else state.F,
                             torch.zeros_like(state.M) if condition == "M_lesion" else state.M,
                             state.tau, state.external_time)
    for t in range(world.early_exposures, world.bridge_index):
        state, _ = model.step(state, world.events[t])
    state, output = _step_bridge(model, state, world, oracle, condition,
                                 random_seed=random_seed)
    logits = model.predict_logits(state)[1]
    ce = Fnn.cross_entropy(logits, world.target, reduction="none")
    prediction = logits.argmax(-1)
    gate = output.get("gates", torch.zeros(world.key.shape[0], 2, device=world.key.device))
    core_width = model.config.hidden_dim
    diagnostics = {
        "ce": ce,
        "correct": prediction.eq(world.target).float(),
        "read_norm": output["read"].norm(dim=-1),
        "read_gate_slow": gate[:, 1] if gate.ndim == 2 else gate,
        "H_norm": state.H.norm(dim=(-2, -1)),
        "F_norm": state.F.norm(dim=(-2, -1)),
        "M_norm": state.M.norm(dim=(-2, -1)),
        "candidate_update_norm": output["H_delta_norm"],
        "access": output["access"],
        "oracle_norm": oracle.norm(dim=-1),
    }
    return ce.mean(), diagnostics


def gradient_diagnostics(model: AnatomicalETRCM, world: ComposedWorld,
                         *, condition: str, random_seed: int) -> dict[str, float]:
    """Secondary diagnostics only; never a causal gate."""
    model.zero_grad(set_to_none=True)
    loss, _ = run_episode(model, world, condition=condition, random_seed=random_seed)
    loss.backward()
    names = ("q_fast_projection", "q_slow_projection", "fast_read_norm",
             "slow_read_norm", "two_way_gate", "access_head")
    branch = [p.grad.detach().norm().item() for n, p in model.named_parameters()
              if p.grad is not None and any(n.startswith(prefix) for prefix in names)]
    core = [p.grad.detach().norm().item() for n, p in model.named_parameters()
            if p.grad is not None and n.startswith(("core_in", "core_out", "core_gate"))]
    result = {"memory_grad_norm": math.sqrt(sum(x*x for x in branch)),
              "core_grad_norm": math.sqrt(sum(x*x for x in core))}
    model.zero_grad(set_to_none=True)
    return result
