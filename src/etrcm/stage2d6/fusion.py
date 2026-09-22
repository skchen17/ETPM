"""Exact finite 2x2 fusion analysis; diagnostic branches never persist state."""

from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from etrcm.stage2d3.interaction import factorial_components
from etrcm.stage2d5.flow import candidate_flow, cell_means, factorial_injection
from etrcm.stage2d.model import behavioral_metrics


def silu_derivatives(x: torch.Tensor):
    sigmoid = torch.sigmoid(x)
    common = sigmoid * (1 - sigmoid)
    first = sigmoid + x * common
    second = 2 * common + x * common * (1 - 2 * sigmoid)
    return first, second


def behavior(rows: list[dict], reps: int):
    probabilities = torch.stack([row["probabilities"] for row in rows], 1)
    metrics = behavioral_metrics(probabilities, reps)
    return {"TV": sum(metrics["tv_action"]) / 2,
            "IHA": abs(float(metrics["interaction_y0"])),
            "signed_IHA": float(metrics["interaction_y0"]),
            "BS": float(metrics["behavioral_separation_entropy"])}


def prob_interaction(rows, reps):
    cells = cell_means([r["probabilities"] for r in rows], reps)
    return factorial_components(cells)["interaction"]


def with_post(model, rows, post):
    """Forward frozen final linear+softmax only; no model or state mutation."""
    result = []
    final = model.action_head.head[2]
    for row, value in zip(rows, post):
        logits = final(value)
        new = dict(row)
        new.update(fusion_post=value, logits=logits, probabilities=logits.softmax(-1))
        result.append(new)
    return result


def finite_response(model, rows, reps, eps=0.02):
    """Finite output-probability interaction response, not a gradient."""
    hidden = rows[0]["fusion_post"].shape[-1]
    base = [r["fusion_post"] for r in rows]
    columns = []
    for j in range(hidden):
        unit = torch.zeros(hidden, dtype=base[0].dtype, device=base[0].device)
        unit[j] = eps / 4
        plus = [factorial_injection(base[a], reps, a, unit, "interaction", 1.) for a in (0, 1)]
        minus = [factorial_injection(base[a], reps, a, -unit, "interaction", 1.) for a in (0, 1)]
        columns.append((prob_interaction(with_post(model, rows, plus), reps) -
                        prob_interaction(with_post(model, rows, minus), reps)) / (2 * eps))
    response = torch.stack(columns, dim=1)  # [4, hidden]
    return response


def unit_removal(model, rows, reps, interaction, indices, *, reverse=False):
    delta = torch.zeros_like(interaction)
    delta[list(indices)] = interaction[list(indices)] * (-.5 if reverse else -.25)
    post = [factorial_injection(rows[a]["fusion_post"], reps, a, delta,
                                "interaction", 1.) for a in (0, 1)]
    return with_post(model, rows, post)


def anatomy(model, rows, reps, *, unit_causality=True):
    """Per-run metrics after within-run replicate aggregation."""
    cells = {node: cell_means([r[node] for r in rows], reps)
             for node in ("fusion_H_projection", "fusion_action_projection",
                          "fusion_additive_merge", "fusion_pre", "fusion_post")}
    parts = {node: factorial_components(value) for node, value in cells.items()}
    pre = parts["fusion_pre"]["interaction"]
    post = parts["fusion_post"]["interaction"]
    state = parts["fusion_H_projection"]["state"]
    action = parts["fusion_action_projection"]["action"]
    first, second = silu_derivatives(cells["fusion_pre"])
    curvature = second.abs().mean((0, 1))
    overlap = state.abs() * action.abs() * curvature
    response = finite_response(model, rows, reps)
    _, singular, vh = torch.linalg.svd(response, full_matrices=False)
    quality = {str(rank): float((vh[:rank].T @ (vh[:rank] @ post)).norm() /
                                post.norm().clamp_min(1e-12)) for rank in (1, 2, 4)}
    native = behavior(rows, reps)
    contributions = []
    if unit_causality:
        for j in range(post.numel()):
            removed = behavior(unit_removal(model, rows, reps, post, (j,)), reps)
            contributions.append({"unit": j, "IHA_loss": native["IHA"]-removed["IHA"],
                                  "BS_loss": abs(native["BS"])-abs(removed["BS"])})
    return {"pre_I": float(pre.norm()), "post_I": float(post.norm()),
            "delta_NL": float(post.norm() - pre.norm()),
            "generated_vector_norm": float((post-pre).norm()),
            "pre_I_vector": pre.tolist(), "post_I_vector": post.tolist(),
            "state_main_norm": float(state.norm()), "action_main_norm": float(action.norm()),
            "fusion_norm": float(cells["fusion_post"].norm()),
            "state_main_vector": state.tolist(), "action_main_vector": action.tolist(),
            "mean_abs_curvature": float(curvature.mean()),
            "curvature_vector": curvature.tolist(),
            "mean_abs_slope": float(first.abs().mean()),
            "overlap": float(overlap.sum()), "overlap_vector": overlap.tolist(),
            "pre_mean": float(cells["fusion_pre"].mean()),
            "pre_sd": float(cells["fusion_pre"].std()),
            "pre_cell_means": cells["fusion_pre"].tolist(),
            "silu_slope_cells": first.tolist(),
            "silu_curvature_cells": second.tolist(),
            "Q": quality, "response_singular_values": singular.tolist(),
            "response_rank2_energy": float(singular[:2].square().sum() / singular.square().sum().clamp_min(1e-12)),
            "unit_contributions": contributions, "behavior": native,
            "pre_additivity_error": float((parts["fusion_pre"]["interaction"]-
                                            parts["fusion_H_projection"]["interaction"]-
                                            parts["fusion_action_projection"]["interaction"]).abs().max())}


def pre_override(model, ctx, rows, reps, delta):
    result = []
    for a in (0, 1):
        action = torch.full((2*reps,), a, dtype=torch.long, device=ctx.H.device)
        result.append(candidate_flow(model, ctx, action,
                      override={"fusion_pre": rows[a]["fusion_pre"]+delta}))
    return result


def scaled_main_override(model, ctx, rows, reps, which, scale=1.5):
    node = "fusion_H_projection" if which == "state" else "fusion_action_projection"
    result = []
    for a in (0, 1):
        action = torch.full((2*reps,), a, dtype=torch.long, device=ctx.H.device)
        value = rows[a][node] * scale
        result.append(candidate_flow(model, ctx, action, override={node: value}))
    return result


def top_k(anatomy_row, k):
    return [x["unit"] for x in sorted(anatomy_row["unit_contributions"],
            key=lambda x: (-x["IHA_loss"], -x["BS_loss"], x["unit"]))[:k]]


def common_offset_from_target(rows, indices, target, cap_sd=1.):
    pre = cell_means([r["fusion_pre"] for r in rows], rows[0]["fusion_pre"].shape[0]//2)
    mean = pre.mean((0, 1))
    sd = float(pre.flatten().std())
    delta = torch.zeros_like(mean)
    chosen = list(indices)
    delta[chosen] = (float(target) - mean[chosen]).clamp(-cap_sd*sd, cap_sd*sd)
    return delta


def low_curvature_offset(rows, indices, cap_sd=1.):
    pre = cell_means([r["fusion_pre"] for r in rows], rows[0]["fusion_pre"].shape[0]//2)
    sd = float(pre.flatten().std())
    delta = torch.zeros(pre.shape[-1], dtype=pre.dtype)
    for j in indices:
        left = silu_derivatives(pre[..., j] - cap_sd*sd)[1].abs().mean()
        right = silu_derivatives(pre[..., j] + cap_sd*sd)[1].abs().mean()
        delta[j] = (-cap_sd*sd) if left < right else cap_sd*sd
    return delta


def offset_metrics(model, ctx, rows, reps, delta):
    changed = pre_override(model, ctx, rows, reps, delta)
    before = cell_means([r["fusion_pre"] for r in rows], reps)
    after = cell_means([r["fusion_pre"] for r in changed], reps)
    p0, p1 = factorial_components(before), factorial_components(after)
    post = factorial_components(cell_means([r["fusion_post"] for r in changed], reps))["interaction"]
    return {"delta_norm": float(delta.norm()), "post_I": float(post.norm()),
            "behavior": behavior(changed, reps),
            "pre_state_effect_error": float((p0["state"]-p1["state"]).abs().max()),
            "pre_action_effect_error": float((p0["action"]-p1["action"]).abs().max()),
            "pre_interaction_error": float((p0["interaction"]-p1["interaction"]).abs().max())}
