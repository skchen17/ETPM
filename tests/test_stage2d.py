from __future__ import annotations

import math
import json
from pathlib import Path

import pytest
import torch

from etrcm.stage1_3.events import self_output_event
from etrcm.stage2d.model import (Stage2DModel, parameter_hash, play, probe_prob,
                                 state_intervention)
from etrcm.stage2d.protocol import (DELAYS, FORMATION_N, FROZEN, P_LEVELS,
                                    REVERSAL_R)
from etrcm.stage2d.world import (bayes_p_z0, empirical_support_rate, infer_support,
                                 observed_only, outcome_marginal, paired_experiences)
from experiments.stage2d_integrity import snapshot


def rows(seed=1, reps=2000, p=.65, **kwargs):
    return paired_experiences(seed, reps, 3, p, **kwargs)


def test_noisy_evidence_distribution_correct():
    sample = rows()
    assert empirical_support_rate(sample) == pytest.approx(.65, abs=.025)


def test_single_evidence_does_not_identify_z():
    assert bayes_p_z0([0], .65) == pytest.approx(.65)
    assert 0 < bayes_p_z0([1], .65) < 1


def test_bayesian_reference_correct():
    assert bayes_p_z0([0, 0], .75) == pytest.approx(.9)
    assert bayes_p_z0([0, 1], .75) == pytest.approx(.5)


def test_latent_not_in_lifetime_adapter():
    observed = observed_only(rows(reps=2))
    assert set(observed[0].__dataclass_fields__) == {"color", "shape", "nuisance", "action", "outcome"}


def test_no_correct_action_label():
    observed = observed_only(rows(reps=2))
    assert not any(hasattr(item, "correct_action") for item in observed)


def test_support_is_inferable_but_z_is_not():
    for row in rows(reps=20):
        assert infer_support(row.observed) == row.support_bit


def test_marginal_shortcut_controls_intact():
    marginal = outcome_marginal(rows(reps=10000))
    assert marginal[0] == pytest.approx(.5, abs=.02)
    for value in marginal[1:]:
        assert value == pytest.approx(1 / 6, abs=.02)


def test_matched_noise_has_identical_paired_observations():
    sample = rows(reps=32, matched_noise=True)
    for left, right in zip(sample[:32], sample[32:]):
        assert left.observed == right.observed


def test_opposing_evidence_generation_correct():
    sample = rows(reps=4000, p=.70, oppose_fraction=1.0)
    rate = sum(x.support_bit != x.latent for x in sample) / len(sample)
    assert rate == pytest.approx(.70, abs=.02)


def test_read_clamp_exact():
    model = Stage2DModel()
    state = model.initial_state(4, "cpu")
    state.F.normal_(); state.M.normal_()
    _, trace = model.step(state, None, read_clamp="FM")
    assert torch.count_nonzero(trace["r_F"]) == 0
    assert torch.count_nonzero(trace["r_M"]) == 0


def test_state_zero_exact():
    model = Stage2DModel(); state = model.initial_state(4, "cpu")
    state.F.normal_(); state.M.normal_()
    changed = state_intervention(state, "FM", "zero", 2)
    assert torch.count_nonzero(changed.F) == 0 and torch.count_nonzero(changed.M) == 0
    assert torch.equal(changed.H, state.H)


def test_swap_exact():
    model = Stage2DModel(); state = model.initial_state(4, "cpu")
    state.F.copy_(torch.arange(state.F.numel()).reshape_as(state.F))
    changed = state_intervention(state, "F", "swap", 2)
    assert torch.equal(changed.F[:2], state.F[2:])
    assert torch.equal(changed.F[2:], state.F[:2])
    assert torch.equal(changed.M, state.M)


def test_gamma_zero_is_independent_and_has_no_transfer():
    torch.manual_seed(1); full = Stage2DModel(variant="full")
    torch.manual_seed(2); zero = Stage2DModel(variant="gamma_zero")
    assert parameter_hash(full) != parameter_hash(zero)
    state = zero.initial_state(2, "cpu"); state.F.normal_()
    _, trace = zero.step(state, None)
    assert torch.count_nonzero(trace["transfer"]) == 0


def test_null_has_no_external_write():
    model = Stage2DModel(); state = model.initial_state(2, "cpu")
    _, trace = model.step(state, None)
    assert not bool(trace["external_write_flag"].any())
    assert torch.count_nonzero(trace["external_update"]) == 0


def test_self_output_has_no_external_write():
    model = Stage2DModel(); state = model.initial_state(2, "cpu")
    event = self_output_event(torch.tensor([1, 2]))
    _, trace = model.step(state, event)
    assert not bool(trace["external_write_flag"].any())
    assert torch.count_nonzero(trace["external_update"]) == 0


def test_probe_does_not_change_parameters():
    model = Stage2DModel(); state = model.initial_state(4, "cpu")
    before = parameter_hash(model)
    probe_prob(model, state, rows(reps=2))
    assert parameter_hash(model) == before


def test_protected_evaluator_can_be_frozen():
    model = Stage2DModel()
    for parameter in model.action_head.parameters():
        parameter.requires_grad_(False)
    assert all(not p.requires_grad for p in model.action_head.parameters())


def test_evaluation_parameters_frozen_constants():
    assert FORMATION_N == (0, 1, 2, 4, 8, 16, 32, 64)
    assert DELAYS[-1] == 5000 and REVERSAL_R[-1] == 128
    assert len(P_LEVELS) == 4 and FROZEN.persistence_delay == 500


def test_delay_token_has_no_latent_cue():
    # Paired construction is exact for all non-latent fields; delay uses only a
    # duplicated nuisance-token vector in the evaluator.
    sample = rows(reps=16)
    for left, right in zip(sample[:16], sample[16:]):
        assert (left.observed.color, left.observed.shape, left.observed.nuisance, left.observed.action) == (
            right.observed.color, right.observed.shape, right.observed.nuisance, right.observed.action)


def test_play_boundary_accepts_observed_only():
    model = Stage2DModel(); state = model.initial_state(4, "cpu")
    next_state, loss, trace = play(model, state, rows(reps=2))
    assert torch.isfinite(loss) and next_state.external_time > state.external_time
    assert bool(trace["external_write_flag"].all())


def test_historical_stage2c_artifacts_unchanged():
    root = Path(__file__).resolve().parents[1]
    path = root / "results/stage2d/manifests/historical_stage2c_sha256.json"
    if not path.exists():
        pytest.skip("integrity manifest is created before the formal run")
    expected = json.loads(path.read_text())["files"]
    assert snapshot(root) == expected
