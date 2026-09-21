"""Action/outcome binding ladder on the frozen Stage 2C world.

Only the behavioral interface and diagnostic state sources are new. The
inherited AnatomicalETRCM transition, F write, F->M and decay are untouched.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_5.model import AnatomicalETRCM
from etrcm.stage2c.world import SYMBOL_COUNT
from etrcm.stage1_4.model import Stage14Config


def core_config() -> Stage14Config:
    return Stage14Config(hidden_dim=32, latent_slots=1, symbol_count=SYMBOL_COUNT,
                         key_dim=8, value_dim=8, event_type_dim=8, gamma=0.12,
                         rho_fast=0.97, rho_slow=0.9995, eta_external=0.6)


def oracle_matrices(device: str | torch.device = "cpu") -> torch.Tensor:
    """Fixed, equal-norm, mutually orthogonal matrices unrelated to action IDs."""
    gen = torch.Generator(device="cpu").manual_seed(271828)
    q, _ = torch.linalg.qr(torch.randn(64, 2, generator=gen))
    return q.T.reshape(2, 8, 8).to(device)


class ActionHead(nn.Module):
    def __init__(self, kind: str, hidden: int = 32, action_dim: int = 8):
        super().__init__()
        if kind not in {"late_concat", "modulation"}:
            raise ValueError(kind)
        self.kind = kind
        self.action_embedding = nn.Embedding(2, action_dim)
        self.action_projection = nn.Linear(action_dim, hidden, bias=False) if kind == "modulation" else None
        width = hidden + action_dim if kind == "late_concat" else hidden
        self.head = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, 4))

    def forward(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a = self.action_embedding(action)
        if self.kind == "late_concat":
            return self.head(torch.cat([h, a], dim=-1))
        return self.head(h + self.action_projection(a))


class OracleLatent(nn.Module):
    """L0: z is only a context index; it never supplies action or outcome labels."""
    def __init__(self, kind: str):
        super().__init__()
        self.latent_embedding = nn.Embedding(2, 32)
        self.action_head = ActionHead(kind)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.latent_embedding(latent), action)


class HistoryEncoder(nn.Module):
    """Reads only past observed (surface, action, outcome) records."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(SYMBOL_COUNT, 12)
        self.gru = nn.GRU(12 * 5, 32, batch_first=True)
        self.to_memory = nn.Linear(32, 64)

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        if past.ndim != 3 or past.shape[-1] != 5:
            raise ValueError("past must be [batch,time,5] of observed tokens")
        if past.shape[1] == 0:
            raise ValueError("history cannot be empty")
        embedded = self.embedding(past).flatten(-2)
        _, h = self.gru(embedded)
        return self.to_memory(h[-1]).reshape(-1, 8, 8)


class PersistentInterface(nn.Module):
    """L1/L2: M can affect forecasts only through existing M read -> H path."""
    def __init__(self, kind: str, *, history: bool):
        super().__init__()
        self.core = AnatomicalETRCM(core_config(), mode="B5_separate")
        self.action_head = ActionHead(kind)
        self.history_encoder = HistoryEncoder() if history else None
        self.register_buffer("oracle_M", oracle_matrices(), persistent=True)

    def source_memory(self, latent: torch.Tensor | None = None,
                      past: torch.Tensor | None = None) -> torch.Tensor:
        if self.history_encoder is not None:
            if latent is not None or past is None:
                raise ValueError("L2 uses past only, never latent")
            return self.history_encoder(past)
        if latent is None or past is not None:
            raise ValueError("L1 uses oracle matrix only")
        return self.oracle_M[latent]

    def states(self, memory: torch.Tensor) -> tuple[LearnedState, LearnedState, dict]:
        state = self.core.initial_state(memory.shape[0], device=memory.device)
        state = LearnedState(state.H, state.F, memory, state.tau, state.external_time)
        # No event: no F external write. Same H0 and present for both latent states.
        next_state, trace = self.core.step(state, None)
        return state, next_state, trace

    def forward(self, action: torch.Tensor, *, latent: torch.Tensor | None = None,
                past: torch.Tensor | None = None, clamp_m: bool = False) -> torch.Tensor:
        memory = self.source_memory(latent, past)
        before = self.core.initial_state(memory.shape[0], device=memory.device)
        state = LearnedState(before.H, before.F, memory, before.tau, before.external_time)
        if clamp_m:
            state, _ = self.core.step_with_read(
                state, None, slow_read_override=torch.zeros(memory.shape[0], 8, device=memory.device))
        else:
            state, _ = self.core.step(state, None)
        return self.action_head(self.core._pool(state.H), action)


def balanced_batch(batch: int, device: str, generator: torch.Generator):
    """Exactly balanced four z/action cells; only sampled consequence is target."""
    if batch % 4:
        raise ValueError("batch must be divisible by four")
    pairs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long).repeat(batch // 4, 1)
    order = torch.randperm(batch, generator=generator)
    pairs = pairs[order]
    latent, action = pairs[:, 0], pairs[:, 1]
    wrong = torch.randint(1, 4, (batch,), generator=generator)
    outcome = torch.where(action.eq(latent), torch.zeros_like(wrong), wrong)
    return latent.to(device), action.to(device), outcome.to(device)


def legal_history(batch: int, length: int, split: str, device: str,
                  generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Only Stage 2C-observable past: color, shape, nuisance, action, outcome."""
    if batch % 2 or length % 2:
        raise ValueError("batch and length must be even")
    latent = torch.arange(batch) % 2
    parity = 0 if split in {"train", "seen"} else 1
    combos = [(8 + c, 14 + s) for c in range(4) for s in range(4) if (c+s)%2 == parity]
    if split == "hard_ood":
        combos = [(12+c, 18+s) for c in range(2) for s in range(2)]
    combo_ids = torch.randint(len(combos), (batch, length), generator=generator)
    color = torch.tensor([x[0] for x in combos])[combo_ids]
    shape = torch.tensor([x[1] for x in combos])[combo_ids]
    nuisance = 20 + torch.randint(4, (batch, length), generator=generator)
    # Paired balanced schedules, shuffled independently of z.
    action = torch.tensor([0, 1] * (length // 2)).repeat(batch, 1)
    for i in range(batch):
        action[i] = action[i, torch.randperm(length, generator=generator)]
    wrong = 1 + torch.randint(3, (batch, length), generator=generator)
    outcome = torch.where(action.eq(latent[:, None]), torch.zeros_like(wrong), wrong) + 4
    observed = torch.stack([color, shape, nuisance, action + 2, outcome], -1)
    return observed.to(device), latent.to(device)


def forecast_metrics(prob: torch.Tensor, temperature: float = 0.35) -> dict:
    """prob[z,a,y]; includes all direct action interventions and policies."""
    p = prob.detach().float().cpu().clamp_min(1e-10)
    entropy = -(p * p.log()).sum(-1)
    entropy_policy = torch.softmax(-entropy / temperature, -1)
    q_policy = torch.softmax(p[:, :, 0] / temperature, -1)
    tv_action = 0.5 * (p[:, 0] - p[:, 1]).abs().sum(-1)
    kl01 = (p[:, 0] * (p[:, 0].log() - p[:, 1].log())).sum(-1)
    kl10 = (p[:, 1] * (p[:, 1].log() - p[:, 0].log())).sum(-1)
    mid = (p[:, 0] + p[:, 1]) / 2
    js = 0.5 * ((p[:, 0] * (p[:, 0].log() - mid.log())).sum(-1) +
                (p[:, 1] * (p[:, 1].log() - mid.log())).sum(-1))
    tv_history = 0.5 * (p[0] - p[1]).abs().sum(-1)
    interaction = (p[0, 0] - p[0, 1]) - (p[1, 0] - p[1, 1])
    return {
        "prob": p.tolist(), "tv_action": tv_action.tolist(),
        "tv_history": tv_history.tolist(), "kl_action_01": kl01.tolist(),
        "kl_action_10": kl10.tolist(), "js_action": js.tolist(),
        "entropy_difference_a0_minus_a1": (entropy[:, 0] - entropy[:, 1]).tolist(),
        "p0_difference_a0_minus_a1": (p[:, 0, 0] - p[:, 1, 0]).tolist(),
        "delta_q_a": float(p[0, 0, 0] - p[0, 1, 0]),
        "delta_q_b": float(p[1, 1, 0] - p[1, 0, 0]),
        "interaction_y0": float(interaction[0]), "interaction_vector": interaction.tolist(),
        "entropy_policy": entropy_policy.tolist(), "q_policy": q_policy.tolist(),
        "entropy_correct_mean": float((entropy_policy[0, 0] + entropy_policy[1, 1]) / 2),
        "q_correct_mean": float((q_policy[0, 0] + q_policy[1, 1]) / 2),
        "behavioral_separation_entropy": float(entropy_policy[0, 0] - entropy_policy[1, 0]),
        "behavioral_separation_q": float(q_policy[0, 0] - q_policy[1, 0]),
    }
