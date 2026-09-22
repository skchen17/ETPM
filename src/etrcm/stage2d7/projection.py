"""Exact state-projection anatomy and finite, nonpersistent interventions."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from etrcm.stage2d3.interaction import factorial_components
from etrcm.stage2d5.flow import cell_means, candidate_flow
from etrcm.stage2d6.fusion import behavior


def components(rows, node, reps):
    return factorial_components(cell_means([r[node] for r in rows], reps))


def state_matrix(model):
    hdim = model.config.hidden_dim
    return model.action_head.head[0].weight[:, :hdim]


def visibility(model, rows, reps, ranks=(1, 2, 4, 8, 16)):
    """State contrast is action-averaged; per-action equality also checked."""
    w = state_matrix(model).detach()
    h = components(rows, "evaluator_H_input", reps)["state"].detach()
    s = components(rows, "fusion_H_projection", reps)["state"].detach()
    error = float((w @ h - s).abs().max())
    h_cells = cell_means([r["evaluator_H_input"] for r in rows], reps)
    s_cells = cell_means([r["fusion_H_projection"] for r in rows], reps)
    per_action_error = max(float((w @ (h_cells[0, a] - h_cells[1, a]) -
                                  (s_cells[0, a] - s_cells[1, a])).abs().max())
                           for a in (0, 1))
    u, sigma, vh = torch.linalg.svd(w, full_matrices=False)
    reconstruction = float((u @ torch.diag(sigma) @ vh - w).abs().max())
    denom = float(h.square().sum()) + 1e-12
    energies = {}
    for k in ranks:
        if k > vh.shape[0]:
            continue
        energies[str(k)] = {
            "top": float((vh[:k] @ h).square().sum()) / denom,
            "bottom": float((vh[-k:] @ h).square().sum()) / denom,
        }
    return {"D_H": float(h.norm()), "D_S": float(s.norm()),
            "G_proj": float(s.norm() / h.norm().clamp_min(1e-12)),
            "V_H": float(s.square().sum() / h.square().sum().clamp_min(1e-12)),
            "hook_error": error, "per_action_hook_error": per_action_error,
            "svd_reconstruction_error": reconstruction,
            "singular_values": sigma.tolist(), "matrix_rank": int(torch.linalg.matrix_rank(w)),
            "energy": energies, "state_vector_H": h.tolist(),
            "state_vector_S": s.tolist()}


def override_main(model, context, rows, reps, node, direction, *, kind="state", scale=1.):
    """Add ±direction/2 by history or action at one activation only."""
    if kind not in {"state", "action", "common"}:
        raise ValueError(kind)
    direction = direction * (float(scale) / 2 if kind != "common" else float(scale))
    outputs = []
    for action in (0, 1):
        source = rows[action][node]
        if kind == "state":
            delta = torch.cat([direction.expand(reps, -1), -direction.expand(reps, -1)], 0)
        elif kind == "action":
            delta = direction.expand(2*reps, -1) * (1 if action == 0 else -1)
        else:
            delta = direction.expand(2*reps, -1)
        ids = torch.full((2*reps,), action, dtype=torch.long, device=source.device)
        outputs.append(candidate_flow(model, context, ids, override={node: source + delta}))
    return outputs


def response_matrix(model, context, rows, reps, node="fusion_H_projection"):
    """Local 4-by-D logit-interaction response to unit state-main change."""
    with torch.enable_grad():
        direction = torch.zeros_like(rows[0][node][0], requires_grad=True)
        outputs = override_main(model, context, rows, reps, node, direction)
        interaction = components(outputs, "logits", reps)["interaction"]
        columns = [torch.autograd.grad(interaction[j], direction,
                                      retain_graph=j+1 < interaction.numel())[0]
                   for j in range(interaction.numel())]
    return torch.stack(columns).detach()


def aligned_direction(response, template, magnitude):
    """Healthy output template -> target coordinates via local Jacobian adjoint."""
    template = template / template.norm().clamp_min(1e-12)
    vector = response.T @ template
    return vector / vector.norm().clamp_min(1e-12) * float(magnitude)


def low_sensitivity_direction(response, magnitude):
    _, _, vh = torch.linalg.svd(response, full_matrices=True)
    vector = vh[-1]
    return vector / vector.norm().clamp_min(1e-12) * float(magnitude)


def intervention_metrics(model, context, rows, reps, direction, node="fusion_H_projection", kind="state"):
    changed = override_main(model, context, rows, reps, node, direction, kind=kind)
    before = components(rows, node, reps)
    after = components(changed, node, reps)
    return {"behavior": behavior(changed, reps),
            "fusion_post_I": float(components(changed, "fusion_post", reps)["interaction"].norm()),
            "node_state_delta": float((after["state"]-before["state"]).norm()),
            "node_action_delta": float((after["action"]-before["action"]).norm()),
            "node_interaction_delta": float((after["interaction"]-before["interaction"]).norm())}


def rotated_contrast(h: torch.Tensor, target: torch.Tensor, lam: float):
    """Spherical interpolation keeps the history-contrast norm exactly."""
    norm = h.norm()
    if norm < 1e-12:
        return h.clone()
    unit = h / norm
    target = target - (target @ unit) * unit
    if target.norm() < 1e-12:
        raise ValueError("target parallel to history contrast")
    target = target / target.norm()
    angle = torch.acos((unit @ target).clamp(-1., 1.)) * float(lam)
    return norm * (torch.cos(angle) * unit + torch.sin(angle) * target)


def rotate_h(model, context, rows, reps, target, lam):
    old = components(rows, "evaluator_H_input", reps)["state"]
    new = rotated_contrast(old, target, lam)
    return override_main(model, context, rows, reps, "evaluator_H_input", new-old)
