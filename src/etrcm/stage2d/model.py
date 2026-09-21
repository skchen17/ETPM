"""Stage 2D model wrapper and training primitives.

The wrapper selects already-existing Stage 2C core modes.  It does not alter
the external write, consolidation, decay, NULL, or SELF_OUTPUT laws.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_4.model import Stage14Config
from etrcm.stage1_5.model import AnatomicalETRCM
from etrcm.stage2c.world import ABSTRACT, ACTION, tensor_ids, context_token, outcome_token
from etrcm.stage2c1.diagnostic import ActionHead, OracleLatent, balanced_batch, forecast_metrics
from etrcm.stage2d.world import NoisyExperience, observed_only


VARIANT_MODE = {
    "full": "B5_separate",
    "gamma_zero": "B6_gamma_zero",
    "f_only": "B5_separate",
    "no_memory": "B0_no_memory",
    "gru": "B1_gru",
}


def core_config(gamma: float = 0.12, rho_fast: float = 0.97,
                rho_slow: float = 0.9995) -> Stage14Config:
    return Stage14Config(hidden_dim=32, latent_slots=1, symbol_count=24,
                         key_dim=8, value_dim=8, event_type_dim=8,
                         gamma=gamma, rho_fast=rho_fast, rho_slow=rho_slow,
                         eta_external=0.6)


class Stage2DModel(nn.Module):
    def __init__(self, *, variant: str = "full", gamma: float = 0.12,
                 rho_fast: float = 0.97, rho_slow: float = 0.9995):
        super().__init__()
        if variant not in VARIANT_MODE:
            raise ValueError(variant)
        self.variant = variant
        self.config = core_config(gamma, rho_fast, rho_slow)
        self.core = AnatomicalETRCM(self.config, mode=VARIANT_MODE[variant])
        self.action_head = ActionHead("late_concat")

    def initial_state(self, batch: int, device: str | torch.device) -> LearnedState:
        return self.core.initial_state(batch, device=device)

    def logits(self, state: LearnedState, action: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.core._pool(state.H), action)

    def step(self, state: LearnedState, event, *, read_clamp: str = "none"):
        if read_clamp not in {"none", "F", "M", "FM"}:
            raise ValueError(read_clamp)
        if self.variant in {"no_memory", "gru"}:
            if read_clamp != "none":
                raise ValueError("memory clamp requested for no-memory baseline")
            return self.core.step(state, event)
        batch = state.H.shape[0]
        zero = torch.zeros(batch, self.config.value_dim, device=state.H.device, dtype=state.H.dtype)
        fast = zero if read_clamp in {"F", "FM"} else None
        slow = zero if read_clamp in {"M", "FM"} or self.variant == "f_only" else None
        if fast is not None or slow is not None:
            return self.core.step_with_read(state, event, fast_read_override=fast,
                                            slow_read_override=slow)
        return self.core.step(state, event)


def parameter_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def head_hash(model: Stage2DModel) -> str:
    return parameter_hash(model.action_head)


def pretrain_evaluator(model: Stage2DModel, seed: int, device: str, steps: int = 1000) -> dict:
    """The same protected deterministic evaluator used by Stage 2C.3 A2."""
    torch.manual_seed(seed + 300_000)
    oracle = OracleLatent("late_concat").to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed + 300_137)
    optimizer = torch.optim.AdamW(oracle.parameters(), lr=0.003, weight_decay=0.0)
    for _ in range(steps):
        z, action, outcome = balanced_batch(64, device, generator)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(oracle(z, action), outcome)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        z = torch.tensor([0, 0, 1, 1], device=device)
        action = torch.tensor([0, 1, 0, 1], device=device)
        metrics = forecast_metrics(oracle(z, action).softmax(-1).view(2, 2, 4))
    model.action_head.load_state_dict(oracle.action_head.state_dict())
    return {"steps": steps, "oracle_evaluator_metrics": metrics,
            "transferred_head_hash": head_hash(model),
            "oracle_z_scope": "evaluator pretraining only; absent from lifetime input"}


def context_state(model: Stage2DModel, state: LearnedState, rows: list[NoisyExperience],
                  *, read_clamp: str = "none") -> LearnedState:
    items = observed_only(rows)
    device = state.H.device
    fields = (
        torch.full((len(items),), ABSTRACT, dtype=torch.long, device=device),
        tensor_ids([x.color for x in items], device),
        tensor_ids([x.shape for x in items], device),
        tensor_ids([x.nuisance for x in items], device),
    )
    for ids in fields:
        state, _ = model.step(state, context_token(ids), read_clamp=read_clamp)
    return state


def play(model: Stage2DModel, state: LearnedState, rows: list[NoisyExperience],
         *, read_clamp: str = "none", write: bool = True):
    items = observed_only(rows)
    device = state.H.device
    state = context_state(model, state, rows, read_clamp=read_clamp)
    action = tensor_ids([x.action for x in items], device)
    action_ids = tensor_ids([ACTION[x.action] for x in items], device)
    state, _ = model.step(state, context_token(action_ids), read_clamp=read_clamp)
    logits = model.logits(state, action)
    target = tensor_ids([x.outcome - 4 for x in items], device)
    loss = F.cross_entropy(logits, target)
    event = outcome_token(tensor_ids([x.outcome for x in items], device), write=write)
    state, trace = model.step(state, event, read_clamp=read_clamp)
    return state, loss, trace


def probe_prob(model: Stage2DModel, state: LearnedState, rows: list[NoisyExperience],
               *, read_clamp: str = "none") -> torch.Tensor:
    items = observed_only(rows)
    device = state.H.device
    context = context_state(model, state.clone(), rows, read_clamp=read_clamp)
    logits = []
    for action in (0, 1):
        ids = torch.full((len(items),), ACTION[action], dtype=torch.long, device=device)
        branch, _ = model.step(context.clone(), context_token(ids), read_clamp=read_clamp)
        actions = torch.full((len(items),), action, dtype=torch.long, device=device)
        logits.append(model.logits(branch, actions))
    return torch.stack(logits, 1).softmax(-1)


def aggregate_prob(prob: torch.Tensor, replicates: int) -> torch.Tensor:
    if prob.shape[0] != 2 * replicates:
        raise ValueError((prob.shape, replicates))
    return torch.stack([prob[:replicates].mean(0), prob[replicates:].mean(0)])


def behavioral_metrics(prob: torch.Tensor, replicates: int) -> dict:
    return forecast_metrics(aggregate_prob(prob, replicates))


def state_intervention(state: LearnedState, which: str, operation: str,
                       replicates: int) -> LearnedState:
    if which not in {"F", "M", "FM"} or operation not in {"zero", "swap"}:
        raise ValueError((which, operation))
    def change(name: str, value: torch.Tensor) -> torch.Tensor:
        if name not in which:
            return value.clone()
        if operation == "zero":
            return torch.zeros_like(value)
        if value.shape[0] != 2 * replicates:
            raise ValueError("paired swap requires 2*replicates batch")
        return torch.cat([value[replicates:], value[:replicates]], 0).clone()
    return LearnedState(state.H.clone(), change("F", state.F), change("M", state.M),
                        state.tau, state.external_time)


def state_norms(state: LearnedState) -> dict[str, float]:
    return {name: float(getattr(state, name).norm(dim=(-2, -1)).mean())
            for name in ("H", "F", "M")}
