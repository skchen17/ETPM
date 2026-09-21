from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import torch

from etrcm.stage2c.world import ACTION, context_token
from etrcm.stage2d.model import Stage2DModel, head_hash, parameter_hash
from etrcm.stage2d4.model import Stage2D4Model, play
from etrcm.stage2d.world import paired_experiences


def action(batch, value):
    return torch.full((batch,), value, dtype=torch.long)


def test_a0_exact_legacy_candidate_reproduction():
    torch.manual_seed(1); model = Stage2D4Model("A0")
    state = model.initial_state(4, "cpu"); a = action(4, 1)
    got, _ = model.candidate(state, a)
    branch, _ = model.step(state.clone(), context_token(torch.full((4,), ACTION[1])))
    expected = model.logits(branch, a)
    assert torch.equal(got, expected)


def test_candidate_query_depends_on_declared_query_action_only():
    model = Stage2D4Model("A1"); model.query_action_F.weight.data.fill_(.2)
    state = model.initial_state(4, "cpu"); evaluator = action(4, 0)
    _, left = model.candidate(state, evaluator, query_action=action(4, 0))
    _, right = model.candidate(state, evaluator, query_action=action(4, 1))
    assert not torch.equal(left["q_F"], right["q_F"])
    assert torch.equal(left["evaluator_action"], right["evaluator_action"])


def test_candidate_read_never_writes_fast_or_slow_or_h():
    model = Stage2D4Model("A1"); state = model.initial_state(4, "cpu")
    before = state.clone(); model.candidate(state, action(4, 0))
    assert torch.equal(state.H, before.H)
    assert torch.equal(state.F, before.F)
    assert torch.equal(state.M, before.M)


def test_candidate_has_no_consolidation_or_decay_and_never_advances_clocks():
    model = Stage2D4Model("A1"); state = model.initial_state(4, "cpu")
    state.F.normal_(); state.M.normal_(); tau, ext = state.tau, state.external_time
    before_f, before_m = state.F.clone(), state.M.clone()
    model.candidate(state, action(4, 1))
    assert torch.equal(state.F, before_f) and torch.equal(state.M, before_m)
    assert state.tau == tau and state.external_time == ext


def test_swapped_query_changes_only_query_action():
    model = Stage2D4Model("A1"); model.query_action_M.weight.data.normal_()
    state = model.initial_state(4, "cpu"); evaluator = action(4, 1)
    _, native = model.candidate(state, evaluator, query_action=evaluator)
    _, swapped = model.candidate(state, evaluator, query_action=1-evaluator)
    assert torch.equal(native["evaluator_action"], swapped["evaluator_action"])
    assert not torch.equal(native["q_M"], swapped["q_M"])


def test_shared_query_control_is_exact():
    model = Stage2D4Model("A1"); model.query_action_F.weight.data.normal_()
    state = model.initial_state(4, "cpu")
    _, a = model.candidate(state, action(4, 0), shared_query=0)
    _, b = model.candidate(state, action(4, 1), shared_query=0)
    assert torch.equal(a["q_F"], b["q_F"])
    assert torch.equal(a["q_M"], b["q_M"])
    assert torch.equal(a["query_action"], b["query_action"])
    assert torch.equal(a["query_action"], action(4, 0))


def test_neutral_query_exactly_removes_explicit_offset():
    model = Stage2D4Model("A1"); model.query_action_F.weight.data.normal_()
    state = model.initial_state(4, "cpu")
    _, trace = model.candidate(state, action(4, 0), neutral_f=True)
    assert torch.equal(trace["q_F"], trace["q_F_base"])


def test_a3_parameter_count_exactly_matches_a1():
    assert Stage2D4Model("A1").added_parameters == 128
    assert Stage2D4Model("A3").added_parameters == 128


def test_gate_only_has_legacy_queries():
    model = Stage2D4Model("A2"); model.gate_action.weight.data.normal_()
    _, trace = model.candidate(model.initial_state(4, "cpu"), action(4, 0))
    assert torch.equal(trace["q_F"], trace["q_F_base"])
    assert torch.equal(trace["q_M"], trace["q_M_base"])


def test_candidate_read_clamps_are_ephemeral():
    model = Stage2D4Model("A1"); state = model.initial_state(4, "cpu")
    state.F.normal_(); before = state.clone()
    _, trace = model.candidate(state, action(4, 0), read_clamp="F")
    assert torch.count_nonzero(trace["r_F"]) == 0
    assert torch.equal(state.F, before.F)


def test_observed_training_api_has_no_privileged_label_arguments():
    forbidden = {"z", "latent", "correct_action", "memory_label", "habit"}
    assert not forbidden.intersection(inspect.signature(play).parameters)


def test_training_rows_are_observed_only_at_model_boundary():
    source = inspect.getsource(play)
    assert "row.latent" not in source and "correct_action" not in source


def test_protected_evaluator_can_remain_bit_exact():
    model = Stage2D4Model("A1"); before = head_hash(model)
    for p in model.action_head.parameters(): p.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    state = model.initial_state(4, "cpu")
    rows = paired_experiences(9, 2, 0, .70)
    optimizer.zero_grad(); _, loss, _ = play(model, state, rows); loss.backward(); optimizer.step()
    assert head_hash(model) == before


def test_persistent_step_still_conserves_pre_decay_readout():
    model = Stage2D4Model("A1"); state = model.initial_state(4, "cpu")
    state.F.normal_(); state.M.normal_(); _, trace = model.step(state, None)
    assert float(trace["conservation_error"].max()) < 1e-6


def test_candidate_does_not_mutate_parameters():
    model = Stage2D4Model("A1"); before = parameter_hash(model)
    model.candidate(model.initial_state(4, "cpu"), action(4, 0))
    assert parameter_hash(model) == before


def test_historical_stage2_assets_unchanged(tmp_path):
    manifest = Path("results/stage2d4/manifests/prior_manifest.json")
    assert manifest.is_file()
    subprocess.run([sys.executable, "experiments/stage2d4_integrity.py", "--mode", "verify",
                    "--manifest", str(manifest), "--out", str(tmp_path / "verify.json")], check=True)
