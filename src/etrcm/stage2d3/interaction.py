"""Layerwise state-by-action interaction measurements and frozen interventions.

All routines are diagnostic: model parameters and input states are never
mutated.  The statistical unit is an independently trained model; replicate
rows are averaged before run-level quantities are returned.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch.nn import functional as F

from etrcm.stage1_1.model import LearnedState
from etrcm.stage2c.world import ACTION, context_token
from etrcm.stage2d.model import Stage2DModel, behavioral_metrics
from etrcm.stage2d2.geometry import context_for_probe


LAYER_ORDER = (
    "pre_action_H", "incoming_H", "projected_H", "action_embedding",
    "fusion_input", "fusion_pre", "fusion_post", "pre_logit", "logits",
    "probabilities", "entropy_policy_score",
)


def factorial_components(cells: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decompose cells ordered [history=2, action=2, ...]."""
    if cells.shape[:2] != (2, 2):
        raise ValueError("factorial cells must begin [2,2]")
    grand = cells.mean(dim=(0, 1))
    state = .5 * (cells[0, 0] + cells[0, 1]) - .5 * (cells[1, 0] + cells[1, 1])
    action = .5 * (cells[0, 0] + cells[1, 0]) - .5 * (cells[0, 1] + cells[1, 1])
    interaction = cells[0, 0] - cells[0, 1] - cells[1, 0] + cells[1, 1]
    return {"grand": grand, "state": state, "action": action, "interaction": interaction}


def factorial_reconstruct(parts: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Exactly reconstruct the four cells from grand/S/A/I components."""
    rows = []
    for hs in (1., -1.):
        row = []
        for ac in (1., -1.):
            row.append(parts["grand"] + hs * parts["state"] / 2 +
                       ac * parts["action"] / 2 + hs * ac * parts["interaction"] / 4)
        rows.append(torch.stack(row))
    return torch.stack(rows)


def _cell_means(value: torch.Tensor, reps: int) -> torch.Tensor:
    """Convert [action,2*reps,...] to [history,action,...]."""
    if value.shape[0] != 2 or value.shape[1] != 2 * reps:
        raise ValueError((value.shape, reps))
    return torch.stack([
        torch.stack([value[a, :reps].mean(0), value[a, reps:].mean(0)])
        for a in range(2)
    ], dim=1)


def _head_forward(head, h: torch.Tensor, action: int,
                  post_delta: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    if head.kind != "late_concat":
        raise ValueError("Stage 2D.3 preregisters the frozen late-concat evaluator")
    ids = torch.full((h.shape[0],), action, dtype=torch.long, device=h.device)
    emb = head.action_embedding(ids)
    fusion = torch.cat([h, emb], -1)
    pre = head.head[0](fusion)
    post = head.head[1](pre)
    if post_delta is not None:
        post = post + post_delta
    logits = head.head[2](post)
    prob = logits.softmax(-1)
    score = -(prob.clamp_min(1e-9) * prob.clamp_min(1e-9).log()).sum(-1, keepdim=True) / .35
    return {"projected_H": h, "action_embedding": emb, "fusion_input": fusion,
            "fusion_pre": pre, "fusion_post": post, "pre_logit": post,
            "logits": logits, "probabilities": prob, "entropy_policy_score": score}


@torch.no_grad()
def factorial_layers(model: Stage2DModel, state: LearnedState, seed: int, reps: int,
                     post_deltas: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run the legal 2x2 history/action factorial without mutating state.

    ``post_deltas`` is [history,action,hidden] and is used only for frozen
    activation interventions after the first fusion nonlinearity.
    """
    before = (state.H.clone(), state.F.clone(), state.M.clone())
    context = context_for_probe(model, state, seed, reps)
    action_layers: dict[str, list[torch.Tensor]] = {name: [] for name in LAYER_ORDER}
    for action in (0, 1):
        ids = torch.full((2 * reps,), ACTION[action], dtype=torch.long, device=state.H.device)
        branch, _ = model.step(context.clone(), context_token(ids))
        incoming = model.core._pool(branch.H)
        delta = None
        if post_deltas is not None:
            delta = torch.cat([post_deltas[0, action].expand(reps, -1),
                               post_deltas[1, action].expand(reps, -1)], 0)
        values = _head_forward(model.action_head, incoming, action, delta)
        action_layers["pre_action_H"].append(model.core._pool(context.H))
        action_layers["incoming_H"].append(incoming)
        for name, value in values.items():
            action_layers[name].append(value)
    cells = {name: _cell_means(torch.stack(values), reps) for name, values in action_layers.items()}
    if not all(torch.equal(x, y) for x, y in zip(before, (state.H, state.F, state.M))):
        raise AssertionError("layer hooks mutated the input state")
    prob_rows = torch.stack(action_layers["probabilities"])
    return cells, prob_rows


def _norm(value: torch.Tensor) -> float:
    return float(value.flatten().norm())


def layer_metrics(cells: Mapping[str, torch.Tensor]) -> dict:
    result = {}
    for name in LAYER_ORDER:
        parts = factorial_components(cells[name])
        s, a, i = (_norm(parts[k]) for k in ("state", "action", "interaction"))
        result[name] = {"state_norm": s, "action_norm": a, "interaction_norm": i,
                        "normalized_interaction": i / (s + a + 1e-12),
                        "interaction_vector": parts["interaction"].flatten().cpu().tolist()}
    return result


def output_metrics(cells: Mapping[str, torch.Tensor]) -> dict:
    prob = cells["probabilities"]
    inter = factorial_components(prob)["interaction"]
    entropy = -(prob.clamp_min(1e-9) * prob.clamp_min(1e-9).log()).sum(-1)
    bs = float((entropy[0, 1] - entropy[0, 0]) - (entropy[1, 1] - entropy[1, 0]))
    tv = float(.25 * ((prob[:, 0] - prob[:, 1]).abs().sum(-1)).sum())
    return {"I_HA_signed": float(inter[0]), "I_HA": abs(float(inter[0])),
            "I_dist": inter.cpu().tolist(), "I_dist_norm": _norm(inter),
            "BS": bs, "action_TV": tv}


@torch.no_grad()
def anatomy(model: Stage2DModel, state: LearnedState, seed: int, reps: int) -> dict:
    cells, _ = factorial_layers(model, state, seed, reps)
    return {"layers": layer_metrics(cells), "output": output_metrics(cells)}


def signed_delta(direction: torch.Tensor, kind: str) -> torch.Tensor:
    """Return [history,action,hidden] intervention with exact sign pattern."""
    patterns = {
        "interaction": ((1., -1.), (-1., 1.)),
        "state": ((1., 1.), (-1., -1.)),
        "action": ((1., -1.), (1., -1.)),
        "shuffled": ((1., -1.), (1., -1.)),
    }
    if kind not in patterns:
        raise ValueError(kind)
    signs = torch.tensor(patterns[kind], device=direction.device, dtype=direction.dtype)
    return signs[..., None] * direction


def output_aligned_template(source_interaction: torch.Tensor, target_model: Stage2DModel,
                            source_scale: float) -> torch.Tensor:
    """Map a healthy logit interaction into target post-activation coordinates.

    Alignment is through the common four-dimensional protected logit space,
    never by directly swapping raw hidden coordinates across models.
    """
    weight = target_model.action_head.head[2].weight.detach()  # [4,32]
    direction = torch.linalg.pinv(weight) @ source_interaction.to(weight)
    direction = direction / direction.norm().clamp_min(1e-12)
    return direction * (source_scale / 4.)


@torch.no_grad()
def intervention_metrics(model: Stage2DModel, state: LearnedState, seed: int, reps: int,
                         direction: torch.Tensor, lam: float, kind: str) -> dict:
    deltas = signed_delta(direction * lam, kind)
    cells, _ = factorial_layers(model, state, seed, reps, deltas)
    result = output_metrics(cells)
    result["intervention_cell_norm"] = float((direction * lam).norm())
    return result


@torch.no_grad()
def destroy_native_interaction(model: Stage2DModel, state: LearnedState, seed: int,
                               reps: int, mode: str) -> dict:
    cells, _ = factorial_layers(model, state, seed, reps)
    interaction = factorial_components(cells["fusion_post"])["interaction"]
    if mode == "remove":
        delta = signed_delta(-interaction / 4., "interaction")
    elif mode == "reverse":
        delta = signed_delta(-interaction / 2., "interaction")
    elif mode == "state_control":
        delta = signed_delta(interaction / 4., "state")
    else:
        raise ValueError(mode)
    changed, _ = factorial_layers(model, state, seed, reps, delta)
    out = output_metrics(changed)
    out["native_interaction_norm"] = float(interaction.norm())
    out["intervention_cell_norm"] = float(delta[0, 0].norm())
    return out


def _custom_prob(model: Stage2DModel, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    head = model.action_head
    x = torch.cat([h, emb.expand(h.shape[0], -1)], -1)
    return head.head[2](head.head[1](head.head[0](x))).softmax(-1)


@torch.no_grad()
def finite_cross_interaction(model: Stage2DModel, state: LearnedState, seed: int, reps: int,
                             directions: int = 64, scales=(.10, .25), direction_seed: int = 0) -> dict:
    """Finite evaluator H x real-action-embedding cross responses."""
    context = context_for_probe(model, state, seed, reps)
    h = model.core._pool(context.H)
    centers = torch.stack([h[:reps].mean(0), h[reps:].mean(0)])
    emb = model.action_head.action_embedding.weight.detach()
    action_delta = emb[1] - emb[0]
    gen = torch.Generator(device=h.device).manual_seed(direction_seed)
    dirs = F.normalize(torch.randn(directions, h.shape[-1], generator=gen, device=h.device), dim=-1)
    native = max(float(h.norm(dim=-1).median() / math.sqrt(h.shape[-1])), 1e-3)
    by_scale = {}
    for scale in scales:
        rows, state_rows = [], []
        amount = native * float(scale)
        for direction in dirs:
            delta_h = amount * direction
            base = _custom_prob(model, centers, emb[0])
            h_only = _custom_prob(model, centers + delta_h, emb[0])
            a_only = _custom_prob(model, centers, emb[0] + action_delta)
            both = _custom_prob(model, centers + delta_h, emb[0] + action_delta)
            rows.append(((both - h_only - a_only + base) / amount).flatten())
            state_rows.append(((h_only - base) / amount).flatten())
        matrix = torch.stack(rows)
        state_matrix = torch.stack(state_rows)
        singular = torch.linalg.svdvals(matrix)
        energy = singular.square()
        frac = energy / energy.sum().clamp_min(1e-12)
        cumulative = frac.cumsum(0)
        by_scale[str(scale)] = {
            "cross_norm": float(matrix.norm()), "state_response_norm": float(state_matrix.norm()),
            "cross_to_state": float(matrix.norm() / state_matrix.norm().clamp_min(1e-12)),
            "singular_values": singular.cpu().tolist(),
            "rank1_energy": float(cumulative[0]),
            "rank2_energy": float(cumulative[min(1, len(cumulative)-1)]),
            "rank4_energy": float(cumulative[min(3, len(cumulative)-1)]),
            "effective_rank": float(torch.exp(-(frac * frac.clamp_min(1e-12).log()).sum())),
        }
    return {"native_H_scale": native, "action_contrast_norm": float(action_delta.norm()),
            "directions": directions, "by_scale": by_scale}
