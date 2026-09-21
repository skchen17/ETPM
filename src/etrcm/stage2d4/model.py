"""Read-only candidate branches for Stage 2D.4.

Persistent transitions are inherited unchanged from :class:`Stage2DModel`.
Only ephemeral candidate evaluation can add an explicit action-dependent
query, action-dependent gate, or parameter-matched downstream projection.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from etrcm.stage1_1.model import LearnedState
from etrcm.stage2c.world import ABSTRACT, ACTION, context_token, outcome_token, tensor_ids
from etrcm.stage2d.model import Stage2DModel, behavioral_metrics
from etrcm.stage2d.world import NoisyExperience, observed_only, paired_experiences
from etrcm.stage2d1.engine import centroid_accuracy


Arm = Literal["A0", "A1", "A2", "A3", "A4"]


class Stage2D4Model(Stage2DModel):
    """Frozen Stage 2D dynamics plus a pure candidate-evaluation branch."""

    def __init__(self, arm: Arm = "A0", *, gamma: float = .50,
                 rho_fast: float = .97, rho_slow: float = .9995):
        super().__init__(variant="full", gamma=gamma, rho_fast=rho_fast, rho_slow=rho_slow)
        if arm not in {"A0", "A1", "A2", "A3", "A4"}:
            raise ValueError(arm)
        self.arm = arm
        action_dim = self.action_head.action_embedding.embedding_dim
        key_dim = self.config.key_dim
        if arm == "A1":
            self.query_action_F = nn.Linear(action_dim, key_dim, bias=False)
            self.query_action_M = nn.Linear(action_dim, key_dim, bias=False)
            nn.init.zeros_(self.query_action_F.weight)
            nn.init.zeros_(self.query_action_M.weight)
        else:
            self.query_action_F = self.query_action_M = None
        if arm == "A2":
            self.gate_action = nn.Linear(action_dim, 2, bias=False)
            nn.init.zeros_(self.gate_action.weight)
        else:
            self.gate_action = None
        if arm == "A3":
            # 32x4 = 128 parameters, exactly matching A1's two 8x8 maps.
            self.capacity_projection = nn.Linear(self.config.hidden_dim, 4, bias=False)
            nn.init.zeros_(self.capacity_projection.weight)
        else:
            self.capacity_projection = None

    @property
    def added_parameters(self) -> int:
        names = ("query_action_F", "query_action_M", "gate_action", "capacity_projection")
        return sum(p.numel() for name in names for p in
                   ([] if getattr(self, name, None) is None else getattr(self, name).parameters()))

    def candidate(
        self,
        state: LearnedState,
        evaluator_action: torch.Tensor,
        *,
        query_action: torch.Tensor | None = None,
        neutral_f: bool = False,
        neutral_m: bool = False,
        shared_query: int | None = None,
        read_clamp: str = "none",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate a declared action without changing H/F/M or either clock.

        ``query_action`` is intentionally distinct from ``evaluator_action`` so
        that query-action swaps intervene on memory access alone.
        """
        if read_clamp not in {"none", "F", "M", "FM"}:
            raise ValueError(read_clamp)
        state.validate()
        before = (state.H.clone(), state.F.clone(), state.M.clone(), state.tau, state.external_time)
        batch = state.H.shape[0]
        if evaluator_action.shape != (batch,):
            raise ValueError("evaluator_action must have shape [batch]")
        qa = evaluator_action if query_action is None else query_action
        if shared_query is not None:
            qa = torch.full_like(evaluator_action, int(shared_query))
        ids = torch.tensor(ACTION, device=state.H.device, dtype=torch.long)[evaluator_action]
        event = context_token(ids)
        encoded = self.core.event_encoder(event).to(state.H.dtype)
        slots = self.core.event_to_slots(encoded).view(batch, self.config.latent_slots,
                                                        self.config.hidden_dim)
        h_pre = state.H + slots
        query_h_pre = h_pre
        if shared_query is not None:
            shared_ids = torch.full_like(evaluator_action, ACTION[int(shared_query)])
            shared_event = context_token(shared_ids)
            shared_encoded = self.core.event_encoder(shared_event).to(state.H.dtype)
            shared_slots = self.core.event_to_slots(shared_encoded).view(
                batch, self.config.latent_slots, self.config.hidden_dim)
            query_h_pre = state.H + shared_slots
        q_f_base, q_m_base = self.core._queries(query_h_pre)
        q_f, q_m = q_f_base, q_m_base
        query_emb = self.action_head.action_embedding(qa)
        if self.arm == "A1":
            if not neutral_f:
                q_f = F.normalize(q_f_base + self.query_action_F(query_emb), dim=-1)
            if not neutral_m:
                q_m = F.normalize(q_m_base + self.query_action_M(query_emb), dim=-1)
        r_f = torch.einsum("bvk,bk->bv", state.F, q_f)
        r_m = torch.einsum("bvk,bk->bv", state.M, q_m)
        if read_clamp in {"F", "FM"}:
            r_f = torch.zeros_like(r_f)
        if read_clamp in {"M", "FM"}:
            r_m = torch.zeros_like(r_m)
        n_f, n_m = self.core.fast_read_norm(r_f), self.core.slow_read_norm(r_m)
        gate_logits = self.core.two_way_gate(torch.cat([self.core._pool(h_pre), encoded], -1))
        if self.arm == "A2":
            gate_logits = gate_logits + self.gate_action(query_emb)
        gates = torch.softmax(gate_logits, -1)
        read = gates[:, :1] * n_f + gates[:, 1:] * n_m
        core_input = torch.cat([
            self.core.norm(h_pre),
            read[:, None, :].expand(-1, self.config.latent_slots, -1),
            encoded[:, None, :].expand(-1, self.config.latent_slots, -1),
        ], -1)
        temporary_h = h_pre + torch.sigmoid(self.core.core_gate(core_input)) * self.core.core_out(
            F.silu(self.core.core_in(core_input)))
        if self.arm == "A3":
            low = self.capacity_projection(self.core._pool(h_pre))
            padded = F.pad(low, (0, self.config.hidden_dim - low.shape[-1]))
            temporary_h = temporary_h + padded[:, None, :]
        pooled = self.core._pool(temporary_h)
        head_emb = self.action_head.action_embedding(evaluator_action)
        fusion_input = torch.cat([pooled, head_emb], -1)
        fusion_pre = self.action_head.head[0](fusion_input)
        fusion_post = self.action_head.head[1](fusion_pre)
        logits = self.action_head.head[2](fusion_post)
        after = (state.H, state.F, state.M, state.tau, state.external_time)
        if not (torch.equal(before[0], after[0]) and torch.equal(before[1], after[1]) and
                torch.equal(before[2], after[2]) and before[3:] == after[3:]):
            raise AssertionError("candidate branch mutated persistent state")
        return logits, {
            "q_F_base": q_f_base, "q_M_base": q_m_base, "q_F": q_f, "q_M": q_m,
            "r_F": r_f, "r_M": r_m, "read": read, "gates": gates,
            "incoming_H": self.core._pool(h_pre), "temporary_H": pooled,
            "fusion_input": fusion_input, "fusion_pre": fusion_pre,
            "fusion_post": fusion_post, "pre_logit": fusion_post,
            "logits": logits, "probabilities": logits.softmax(-1),
            "evaluator_action": evaluator_action, "query_action": qa,
        }


def context_state(model: Stage2D4Model, state: LearnedState,
                  rows: list[NoisyExperience]) -> LearnedState:
    items = observed_only(rows); device = state.H.device
    fields = (
        torch.full((len(items),), ABSTRACT, dtype=torch.long, device=device),
        tensor_ids([x.color for x in items], device),
        tensor_ids([x.shape for x in items], device),
        tensor_ids([x.nuisance for x in items], device),
    )
    for ids in fields:
        state, _ = model.step(state, context_token(ids))
    return state


def play(model: Stage2D4Model, state: LearnedState, rows: list[NoisyExperience]):
    """Observed-only loss with the original persistent transition committed."""
    items = observed_only(rows); device = state.H.device
    state = context_state(model, state, rows)
    action = tensor_ids([x.action for x in items], device)
    logits, trace = model.candidate(state, action)
    target = tensor_ids([x.outcome - 4 for x in items], device)
    loss = F.cross_entropy(logits, target)
    action_ids = tensor_ids([ACTION[x.action] for x in items], device)
    state, _ = model.step(state, context_token(action_ids))
    state, persistent_trace = model.step(
        state, outcome_token(tensor_ids([x.outcome for x in items], device), write=True))
    return state, loss, {"candidate": trace, "persistent": persistent_trace}


def candidate_prob(model: Stage2D4Model, state: LearnedState,
                   rows: list[NoisyExperience], **kwargs) -> tuple[torch.Tensor, list[dict]]:
    items = observed_only(rows); device = state.H.device
    context = context_state(model, state.clone(), rows)
    logits, traces = [], []
    for action in (0, 1):
        declared = torch.full((len(items),), action, dtype=torch.long, device=device)
        call = dict(kwargs)
        if call.pop("swap_query", False):
            call["query_action"] = 1 - declared
        out, trace = model.candidate(context, declared, **call)
        logits.append(out); traces.append(trace)
    return torch.stack(logits, 1).softmax(-1), traces


def aggregate(prob: torch.Tensor, reps: int) -> torch.Tensor:
    return torch.stack([prob[:reps].mean(0), prob[reps:].mean(0)])


@torch.no_grad()
def health_audit(model: Stage2D4Model, seed: int, reps: int = 16,
                 *, p: float = .70, n: int = 32, probe_kwargs: dict | None = None) -> dict:
    device = next(model.parameters()).device
    state = model.initial_state(2 * reps, device); losses = []
    transfer, writes = [], []
    for index in range(n):
        rows = paired_experiences(seed, reps, index, p)
        state, loss, trace = play(model, state, rows)
        losses.append(float(loss)); transfer.append(float(trace["persistent"]["transfer"].norm()))
        writes.append(float(trace["persistent"]["external_update"].norm()))
    rows = paired_experiences(seed + 700001, reps, 99999, .65, split="novel")
    prob, traces = candidate_prob(model, state, rows, **(probe_kwargs or {}))
    metrics = behavioral_metrics(prob, reps); metrics.pop("prob", None)
    agg = aggregate(prob, reps)
    target = torch.empty_like(agg)
    for z in (0, 1):
        for action in (0, 1):
            p0 = .65 if action == z else .35
            target[z, action, 0] = p0; target[z, action, 1:] = (1 - p0) / 3
    ce = float(-(target * agg.clamp_min(1e-9).log()).sum(-1).mean())
    marginal_ce = -(.5 * math.log(.5) + .5 * math.log(1 / 6))
    dqf = float((traces[0]["q_F"] - traces[1]["q_F"]).norm(dim=-1).mean())
    dqm = float((traces[0]["q_M"] - traces[1]["q_M"]).norm(dim=-1).mean())
    drf = float((traces[0]["r_F"] - traces[1]["r_F"]).norm(dim=-1).mean())
    drm = float((traces[0]["r_M"] - traces[1]["r_M"]).norm(dim=-1).mean())
    return {
        "action_TV": sum(metrics["tv_action"]) / 2,
        "history_TV": sum(metrics["tv_history"]) / 2,
        "I_HA": abs(float(metrics["interaction_y0"])),
        "interaction_signed": float(metrics["interaction_y0"]),
        "BS": float(metrics["behavioral_separation_entropy"]),
        "observed_CE": sum(losses) / len(losses), "conditional_CE": ce,
        "marginal_CE": marginal_ce, "CFA": marginal_ce - ce,
        "z_probe": {name: centroid_accuracy(getattr(state, name), reps) for name in ("H", "F", "M")},
        "state_norms": {name: float(getattr(state, name).norm(dim=(-2, -1)).mean()) for name in ("H", "F", "M")},
        "retrieval": {"DqF": dqf, "DqM": dqm, "DrF": drf, "DrM": drm},
        "write_mean": sum(writes) / len(writes), "transfer_mean": sum(transfer) / len(transfer),
    }


def classify(health: dict) -> str:
    healthy = (health["action_TV"] >= .10 and health["I_HA"] >= .10 and
               abs(health["BS"]) >= .10 and health["CFA"] > 0)
    if healthy:
        return "healthy"
    flags = (health["action_TV"] >= .10, health["I_HA"] >= .10, abs(health["BS"]) >= .10)
    return "partial" if any(flags) else "shortcut"
