from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import torch

from etrcm.stage2d.model import Stage2DModel, head_hash, parameter_hash
from etrcm.stage2d3.curriculum import PairedWarmup, paired_observational_targets
from etrcm.stage2d3.interaction import (
    factorial_components, factorial_layers, factorial_reconstruct, signed_delta,
)
from etrcm.stage2d.world import paired_experiences


def test_factorial_pairing_and_interaction_algebra_exact():
    cells = torch.arange(24, dtype=torch.float64).reshape(2, 2, 6)
    parts = factorial_components(cells)
    assert torch.equal(parts["interaction"], cells[0, 0] - cells[0, 1] - cells[1, 0] + cells[1, 1])
    assert torch.allclose(factorial_reconstruct(parts), cells, atol=1e-12)


def test_main_effect_decomposition_known_example():
    grand = torch.tensor([3., 4.]); state = torch.tensor([1., 2.])
    action = torch.tensor([2., -1.]); interaction = torch.tensor([.5, -.5])
    cells = factorial_reconstruct({"grand": grand, "state": state,
                                   "action": action, "interaction": interaction})
    parts = factorial_components(cells)
    for name, expected in (("grand", grand), ("state", state),
                           ("action", action), ("interaction", interaction)):
        assert torch.allclose(parts[name], expected)


def test_interaction_only_signs_exact():
    direction = torch.tensor([2., -3.])
    delta = signed_delta(direction, "interaction")
    assert torch.equal(delta[0, 0], direction)
    assert torch.equal(delta[0, 1], -direction)
    assert torch.equal(delta[1, 0], -direction)
    assert torch.equal(delta[1, 1], direction)
    parts = factorial_components(delta)
    assert torch.equal(parts["state"], torch.zeros_like(direction))
    assert torch.equal(parts["action"], torch.zeros_like(direction))


def test_state_action_and_shuffled_controls_have_no_interaction():
    direction = torch.randn(7)
    for kind in ("state", "action", "shuffled"):
        parts = factorial_components(signed_delta(direction, kind))
        assert torch.allclose(parts["interaction"], torch.zeros_like(direction))


def test_random_control_can_be_norm_matched():
    gen = torch.Generator().manual_seed(3)
    reference = torch.randn(32, generator=gen)
    random = torch.randn(32, generator=gen)
    random = random / random.norm() * reference.norm()
    assert torch.allclose(random.norm(), reference.norm())


def test_hooks_do_not_mutate_state_or_parameters():
    torch.manual_seed(7)
    model = Stage2DModel(gamma=.50)
    state = model.initial_state(4, "cpu")
    state_before = state.clone(); params = parameter_hash(model)
    cells, _ = factorial_layers(model, state, 15501, 2)
    assert set(cells) >= {"incoming_H", "fusion_pre", "fusion_post", "logits", "probabilities"}
    assert torch.equal(state.H, state_before.H)
    assert torch.equal(state.F, state_before.F)
    assert torch.equal(state.M, state_before.M)
    assert parameter_hash(model) == params


def test_activation_intervention_norm_matched_and_no_parameter_mutation():
    model = Stage2DModel(gamma=.50); state = model.initial_state(4, "cpu")
    direction = torch.randn(32); direction = direction / direction.norm() * .25
    delta = signed_delta(direction, "interaction")
    before = parameter_hash(model)
    factorial_layers(model, state, 15501, 2, delta)
    assert all(torch.allclose(delta[h, a].norm(), direction.norm()) for h in range(2) for a in range(2))
    assert parameter_hash(model) == before


def test_paired_scaffold_exports_no_privileged_fields():
    rows = paired_experiences(17, 2, 0, .70)
    targets = paired_observational_targets(rows, 19, 0)
    assert targets.shape == (4, 2)
    assert targets.dtype == torch.long
    # Only sampled outcome IDs are exported: no z, correct-action, or memory label.
    assert targets.min() >= 0 and targets.max() <= 3


def test_paired_warmup_has_exact_zero_after_window_and_at_evaluation():
    protocol = PairedWarmup(100, "C1")
    assert protocol.auxiliary_weight(99) == 1.0
    assert protocol.auxiliary_weight(100) == 0.0
    assert protocol.auxiliary_weight(0, evaluation=True) == 0.0


def test_scaffold_api_has_no_z_correct_action_or_memory_argument():
    signature = inspect.signature(paired_observational_targets)
    forbidden = {"z", "latent", "correct_action", "memory", "H", "M", "F"}
    assert not forbidden.intersection(signature.parameters)


def test_evaluator_can_remain_protected():
    model = Stage2DModel(gamma=.50)
    before = head_hash(model)
    for parameter in model.action_head.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    sum(p.square().mean() for p in model.core.parameters()).backward()
    optimizer.step()
    assert head_hash(model) == before


def test_historical_stage2_assets_unchanged(tmp_path):
    manifest = Path("results/stage2d3/manifests/prior_manifest.json")
    assert manifest.is_file()
    subprocess.run([
        sys.executable, "experiments/stage2d3_integrity.py", "--mode", "verify",
        "--manifest", str(manifest), "--out", str(tmp_path / "verification.json"),
    ], check=True)
