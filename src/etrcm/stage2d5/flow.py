"""Exact, side-effect-free decomposition of the Stage 2D.4 candidate path.

No parameters or persistent transition equations are changed. Overrides are
finite *diagnostic* activations, never committed to H/F/M or either clock.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from etrcm.stage2c.world import ACTION, context_token
from etrcm.stage2d3.interaction import factorial_components


ORDER = (
    "q_F", "q_M", "r_F", "r_M", "normalized_F", "normalized_M",
    "gated_read", "read_integration_input", "H_projection", "read_projection",
    "event_projection", "core_preactivation", "core_nonlinearity",
    "gate_preactivation", "gate_activation", "candidate_update",
    "temporary_H", "evaluator_H_input", "fusion_input", "fusion_H_projection",
    "fusion_action_projection", "fusion_additive_merge", "fusion_pre",
    "fusion_post", "pre_logit", "logits", "probabilities", "policy_score",
)


def candidate_flow(model, state, action, *, override: dict | None = None,
                   mixing: str = "native", residual_alpha: float = 1.) -> dict:
    """Return all candidate activations; verify equality to the frozen model.

    The `override` dictionary replaces *one* named activation in the local
    forward path. It is not a write to any persistent state.
    """
    if mixing not in {"native", "F_only", "M_only", "equal", "norm_matched_sum"}:
        raise ValueError(mixing)
    override = override or {}; core = model.core; batch = action.shape[0]
    if action.shape != (state.H.shape[0],): raise ValueError("batch mismatch")
    before = (state.H.clone(), state.F.clone(), state.M.clone(), state.tau, state.external_time)
    trace = {}
    def put(name, value):
        if name in override:
            new = override[name]
            if new.shape != value.shape: raise ValueError((name, new.shape, value.shape))
            value = new
        trace[name] = value
        return value
    ids = torch.tensor(ACTION, device=state.H.device, dtype=torch.long)[action]
    encoded = core.event_encoder(context_token(ids)).to(state.H.dtype)
    event_slots = core.event_to_slots(encoded).view(batch, core.config.latent_slots, core.config.hidden_dim)
    h_pre = state.H + event_slots
    trace["incoming_H"] = core._pool(h_pre)
    q_f, q_m = core._queries(h_pre)
    if model.arm == "A1":
        emb = model.action_head.action_embedding(action)
        q_f = F.normalize(q_f + model.query_action_F(emb), dim=-1)
        q_m = F.normalize(q_m + model.query_action_M(emb), dim=-1)
    q_f = put("q_F", q_f); q_m = put("q_M", q_m)
    r_f = put("r_F", torch.einsum("bvk,bk->bv", state.F, q_f))
    r_m = put("r_M", torch.einsum("bvk,bk->bv", state.M, q_m))
    n_f = put("normalized_F", core.fast_read_norm(r_f))
    n_m = put("normalized_M", core.slow_read_norm(r_m))
    gate_logits = core.two_way_gate(torch.cat([core._pool(h_pre), encoded], -1))
    if model.arm == "A2": gate_logits = gate_logits + model.gate_action(model.action_head.action_embedding(action))
    gates = torch.softmax(gate_logits, -1); trace["read_gates"] = gates
    if mixing == "native": read = gates[:, :1] * n_f + gates[:, 1:] * n_m
    elif mixing == "F_only": read = n_f
    elif mixing == "M_only": read = n_m
    elif mixing == "equal": read = .5 * (n_f + n_m)
    else:
        combined = n_f + n_m
        native = gates[:, :1] * n_f + gates[:, 1:] * n_m
        read = combined / combined.norm(dim=-1, keepdim=True).clamp_min(1e-12) * native.norm(dim=-1, keepdim=True)
    read = put("gated_read", read)
    read = put("read_integration_input", read)
    norm_h = core.norm(h_pre)
    x = torch.cat([norm_h, read[:, None, :].expand(-1, core.config.latent_slots, -1),
                   encoded[:, None, :].expand(-1, core.config.latent_slots, -1)], -1)
    hdim, vdim = core.config.hidden_dim, core.config.value_dim
    w = core.core_in.weight
    hp = put("H_projection", F.linear(norm_h, w[:, :hdim]))
    rp = put("read_projection", F.linear(x[..., hdim:hdim+vdim], w[:, hdim:hdim+vdim]))
    ep = put("event_projection", F.linear(x[..., hdim+vdim:], w[:, hdim+vdim:]))
    pre = put("core_preactivation", hp + rp + ep + core.core_in.bias)
    nonlinear = put("core_nonlinearity", F.silu(pre))
    gate_pre = put("gate_preactivation", core.core_gate(x))
    gate = put("gate_activation", torch.sigmoid(gate_pre))
    update = put("candidate_update", gate * core.core_out(nonlinear))
    temporary = put("temporary_H", h_pre + residual_alpha * update)
    if model.arm == "A3":
        low = model.capacity_projection(core._pool(h_pre))
        temporary = temporary + F.pad(low, (0, hdim-low.shape[-1]))[:, None, :]
    pooled = put("evaluator_H_input", core._pool(temporary))
    head = model.action_head
    action_emb = head.action_embedding(action)
    fusion_input = put("fusion_input", torch.cat([pooled, action_emb], -1))
    linear = head.head[0]
    fusion_h = put("fusion_H_projection", F.linear(fusion_input[:, :hdim], linear.weight[:, :hdim]))
    fusion_a = put("fusion_action_projection", F.linear(fusion_input[:, hdim:], linear.weight[:, hdim:]))
    merged = put("fusion_additive_merge", fusion_h + fusion_a)
    fusion_pre = put("fusion_pre", merged + linear.bias)
    fusion_post = put("fusion_post", head.head[1](fusion_pre))
    prelogit = put("pre_logit", fusion_post)
    logits = put("logits", head.head[2](prelogit))
    prob = put("probabilities", logits.softmax(-1))
    entropy = -(prob.clamp_min(1e-9) * prob.clamp_min(1e-9).log()).sum(-1, keepdim=True)
    put("policy_score", -entropy / .35)
    if not (torch.equal(state.H, before[0]) and torch.equal(state.F, before[1]) and
            torch.equal(state.M, before[2]) and (state.tau, state.external_time) == before[3:]):
        raise AssertionError("diagnostic branch modified persistent state")
    return trace


def cell_means(actions: list[torch.Tensor], reps: int) -> torch.Tensor:
    stacked = torch.stack(actions)
    if stacked.shape[:2] != (2, 2*reps): raise ValueError((stacked.shape, reps))
    return torch.stack([torch.stack([stacked[a, :reps].mean(0), stacked[a, reps:].mean(0)])
                        for a in range(2)], dim=1)


def interaction_metrics(cells: torch.Tensor, eps=1e-12) -> dict:
    parts = factorial_components(cells)
    norms = {k: float(v.flatten().norm()) for k, v in parts.items()}
    ratio = norms["interaction"] / (norms["state"] + norms["action"] + eps)
    return {"norm": norms["interaction"], "normalized": ratio,
            "state_norm": norms["state"], "action_norm": norms["action"],
            "interaction_vector": parts["interaction"].flatten().tolist(),
            "grand_norm": norms["grand"]}


def transmission(upstream: dict, downstream: dict, eps=1e-12) -> dict:
    return {"absolute": downstream["norm"] / (upstream["norm"] + eps),
            "normalized": downstream["normalized"] / (upstream["normalized"] + eps)}


def signed_pattern(kind: str, device, dtype):
    patterns = {
        "interaction": ((1., -1.), (-1., 1.)),
        "state": ((1., 1.), (-1., -1.)),
        "action": ((1., -1.), (1., -1.)),
        "shuffled": ((1., -1.), (1., -1.)),
        "generic": ((1., 1.), (1., 1.)),
    }
    if kind not in patterns: raise ValueError(kind)
    return torch.tensor(patterns[kind], device=device, dtype=dtype)


def factorial_injection(value: torch.Tensor, reps: int, action: int,
                        direction: torch.Tensor, kind: str, lam: float) -> torch.Tensor:
    """Apply a norm-controlled four-cell perturbation to [batch, features]."""
    if direction.shape != value.shape[1:]: raise ValueError((direction.shape, value.shape))
    signs = signed_pattern(kind, value.device, value.dtype)
    one = torch.cat([signs[0, action] * direction.expand(reps, *direction.shape),
                     signs[1, action] * direction.expand(reps, *direction.shape)], 0)
    return value + float(lam) * one


def norm_match(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value / value.norm().clamp_min(1e-12) * reference.norm()
