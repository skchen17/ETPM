"""Stage 2D.7 mathematical and parameter-isolation invariants."""

import sys
from pathlib import Path
from dataclasses import replace

import torch

from etrcm.stage2d3.interaction import factorial_components
from etrcm.stage2d4.model import Stage2D4Model
from etrcm.stage2d5.flow import candidate_flow, cell_means
from etrcm.stage2d7.projection import (
    components, override_main, rotated_contrast, state_matrix, visibility,
)


def fixture():
    torch.manual_seed(7)
    model = Stage2D4Model("A0").eval()
    state = model.initial_state(8, "cpu")
    state.H[:4] += .4
    state.H[4:] -= .2
    rows = [candidate_flow(model, state, torch.full((8,), a, dtype=torch.long))
            for a in (0, 1)]
    return model, state, rows


def test_hook_equality_and_svd():
    model, _, rows = fixture()
    v = visibility(model, rows, 4)
    assert v["hook_error"] < 1e-5
    assert v["per_action_hook_error"] < 1e-5
    assert v["svd_reconstruction_error"] < 1e-5
    w = state_matrix(model)
    _, _, vh = torch.linalg.svd(w, full_matrices=False)
    assert torch.allclose(vh[:4] @ vh[-4:].T, torch.zeros(4, 4), atol=1e-5)
    assert abs(v["V_H"] - v["G_proj"]**2) < 1e-5


def test_state_main_preserves_action_and_interaction():
    model, state, rows = fixture()
    direction = torch.randn_like(rows[0]["fusion_H_projection"][0])
    baseline = components(rows, "fusion_H_projection", 4)
    before = (state.H.clone(), state.F.clone(), state.M.clone(), state.tau, state.external_time)
    changed = override_main(model, state, rows, 4, "fusion_H_projection", direction)
    after = components(changed, "fusion_H_projection", 4)
    assert torch.allclose(after["state"] - baseline["state"], direction, atol=1e-5)
    assert torch.allclose(after["action"], baseline["action"], atol=1e-5)
    assert torch.allclose(after["interaction"], baseline["interaction"], atol=1e-5)
    assert all(torch.equal(x, y) for x, y in zip(before[:3], (state.H, state.F, state.M)))
    assert before[3:] == (state.tau, state.external_time)


def test_midpoint_destruction_and_partial_attenuation():
    model, state, rows = fixture()
    baseline = components(rows, "fusion_H_projection", 4)
    for beta in (0., .25, .5, .75, 1.):
        changed = override_main(model, state, rows, 4, "fusion_H_projection",
                                (beta-1)*baseline["state"])
        current = components(changed, "fusion_H_projection", 4)
        assert torch.allclose(current["state"], beta*baseline["state"], atol=1e-5)


def test_rotation_preserves_norm():
    h = torch.tensor([2., 1., -3., 4.])
    target = torch.tensor([-1., 5., 2., 1.])
    for lam in (0., .25, .5, .75, 1.):
        value = rotated_contrast(h, target, lam)
        assert torch.allclose(value.norm(), h.norm(), atol=1e-6)
    assert torch.allclose(rotated_contrast(h, target, 0), h, atol=1e-6)


def test_refit_parameter_masks():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from stage2d7_refit import configure
    for arm in ("R1", "R2", "R3", "R4"):
        model = Stage2D4Model("A0")
        before = {k: v.clone() for k, v in model.state_dict().items()}
        params, count = configure(model, arm)
        assert count == {"R1": 1056, "R2": 256, "R3": 32, "R4": 1056}[arm]
        for p in params:
            p.grad = None
        if arm != "R4":
            x = torch.randn(10, 40)
            loss = model.action_head.head[0](x).square().mean()
        else:
            x = torch.randn(10, 32)
            loss = model.action_head.head[1](x).square().mean()
        loss.backward()
        for p in params:
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-.01)
        current = model.state_dict()
        for name, value in before.items():
            if arm == "R1" and name in ("action_head.head.0.weight", "action_head.head.0.bias"):
                continue
            if arm == "R2" and name == "action_head.head.0.weight":
                continue
            if arm == "R3" and name == "action_head.head.0.bias":
                continue
            assert torch.equal(value, current[name])
        w = current["action_head.head.0.weight"]
        if arm == "R1":
            assert torch.equal(w[:, 32:], before["action_head.head.0.weight"][:, 32:])
        if arm == "R2":
            assert torch.equal(w[:, :32], before["action_head.head.0.weight"][:, :32])


def test_compatibility_arm_masks_and_parameter_match():
    from etrcm.stage2d7.compat import CompatibilityModel, configure_training, protected_exact
    counts = {}
    for arm in ("C0", "C1", "C2", "C3", "C4", "C5"):
        model = CompatibilityModel(arm, rank=4)
        groups, snapshot = configure_training(model, arm)
        extra = [p for group in groups[1:] for p in group["params"]]
        counts[arm] = sum(p.numel() for p in extra)
        assert protected_exact(model, arm, snapshot)
        state = model.initial_state(4, "cpu")
        logits = model.logits(state, torch.tensor([0, 1, 0, 1]))
        if arm == "C0":
            continue
        loss = logits.square().mean()
        loss.backward()
        optimizer = torch.optim.AdamW(groups)
        optimizer.step()
        assert protected_exact(model, arm, snapshot)
        if arm == "C2":
            freeze = {k: v.clone() for k, v in model.action_head.state_dict().items()}
            model.action_head.head[0].weight.requires_grad_(False)
            model.action_head.head[0].bias.requires_grad_(False)
            optimizer.zero_grad(set_to_none=True)
            model.logits(state, torch.tensor([0, 1, 0, 1])).square().mean().backward()
            optimizer.step()
            assert protected_exact(model, arm, freeze, after_freeze=True)
    assert counts["C3"] == counts["C4"] == 256
    assert counts["C0"] == 0
    dense = CompatibilityModel("C4", rank=0)
    dense_groups, _ = configure_training(dense, "C4")
    assert sum(p.numel() for p in dense_groups[1]["params"]) == 1056


def test_refit_uses_observed_fields_not_latent_labels(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import stage2d7_refit as refit
    from etrcm.stage2d.world import training_experiences
    model = Stage2D4Model("A0").eval()
    original = training_experiences
    with torch.no_grad():
        features_a, targets_a = refit.training_features(model, seed=4321,
                                                        episodes=1, batch=4, lengths=(4,))
    def changed_metadata(seed, batch, index, p):
        rows = original(seed, batch, index, p)
        return [replace(r, latent=1-r.latent, support_bit=1-r.support_bit,
                        predictive=not r.predictive, opposed=not r.opposed) for r in rows]
    monkeypatch.setattr(refit, "training_experiences", changed_metadata)
    with torch.no_grad():
        features_b, targets_b = refit.training_features(model, seed=4321,
                                                        episodes=1, batch=4, lengths=(4,))
    assert torch.equal(features_a, features_b)
    assert torch.equal(targets_a, targets_b)


def test_refit_train_and_novel_surface_sets_disjoint():
    from etrcm.stage2d.world import paired_experiences
    train = paired_experiences(37101, 8, 0, .70, split="train")
    novel = paired_experiences(26601 + 700001, 8, 99999, .65, split="novel")
    train_pairs = {(r.observed.color, r.observed.shape) for r in train}
    novel_pairs = {(r.observed.color, r.observed.shape) for r in novel}
    assert train_pairs.isdisjoint(novel_pairs)
    assert 37101 != 26601


def test_historical_stage2c_through_2d6_assets_unchanged():
    import json
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from stage2d7_integrity import snapshot
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "results/stage2d7/manifests/historical_baseline_sha256.json").read_text())
    assert snapshot() == baseline["files"]
