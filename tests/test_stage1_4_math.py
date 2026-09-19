from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import torch

from etrcm.stage1_3.events import evidence_event, self_output_event
from etrcm.stage1_4.interventions import (
    causal_usage_loss, lesion_direction, pathological_self_output_write,
    peripheral_swap, prediction_js, restore_workspace,
)
from etrcm.stage1_4.model import PredictiveETRCM, Stage14Config
from etrcm.stage1_4.evaluation import evaluate_A, evaluate_B, evaluate_CD, evaluate_EF, evaluate_safety
from etrcm.stage1_4.training import run_prefix
from etrcm.stage1_4.world import FAMILIES, generate_world


ROOT = Path(__file__).resolve().parents[1]


def make_model(mode: str = "B5_separate") -> PredictiveETRCM:
    torch.manual_seed(1414)
    return PredictiveETRCM(Stage14Config(), mode)


def test_separate_queries_independent_and_reads_normalized() -> None:
    model = make_model()
    state = model.initial_state(4)
    state.H.normal_()
    state.F.normal_()
    state.M.normal_()
    _, out = model.step(state, None)
    assert not torch.allclose(out["q_F"], out["q_M"])
    assert not torch.allclose(out["r_F"], out["r_M"])
    assert torch.allclose(out["gates"].sum(-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(out["normalized_r_F_norm"], torch.full((4,), 4.0), atol=0.02)
    assert torch.allclose(out["normalized_r_M_norm"], torch.full((4,), 4.0), atol=0.02)


def test_same_h_swap_and_workspace_restore_are_exact() -> None:
    model = make_model()
    a, b = model.initial_state(2), model.initial_state(2)
    a.H.normal_(); a.F.normal_(); a.M.normal_()
    b.H.normal_(); b.F.normal_(); b.M.normal_()
    swapped = peripheral_swap(a, b, channels="M")
    assert torch.equal(swapped.H, a.H)
    assert torch.equal(swapped.F, a.F)
    assert torch.equal(swapped.M, b.M)
    a1, _ = model.step(a, None)
    b1, _ = model.step(swapped, None)
    restored = restore_workspace(a1, b1)
    assert torch.equal(restored.H, a1.H)
    assert torch.equal(restored.F, b1.F)
    assert torch.equal(restored.M, b1.M)


def test_direction_lesion_changes_one_channel_and_one_orthogonal_direction() -> None:
    model = make_model()
    state = model.initial_state(2)
    state.H.normal_(); state.F.normal_(); state.M.normal_()
    basis = torch.eye(model.config.key_dim)[:2]
    key = basis[0].expand(2, -1)
    orthogonal = basis[1].expand(2, -1)
    lesion = lesion_direction(state, key, channel="M")
    assert torch.equal(lesion.H, state.H)
    assert torch.equal(lesion.F, state.F)
    assert torch.allclose((lesion.M @ key.unsqueeze(-1)).squeeze(-1), torch.zeros(2, model.config.value_dim), atol=1e-6)
    assert torch.allclose((lesion.M @ orthogonal.unsqueeze(-1)).squeeze(-1),
                          (state.M @ orthogonal.unsqueeze(-1)).squeeze(-1), atol=1e-6)


def test_causal_usage_loss_sign_and_prediction_js() -> None:
    target = torch.tensor([0])
    full = torch.tensor([[8.0, -8.0]])
    lesion = torch.tensor([[-8.0, 8.0]])
    assert causal_usage_loss(full, lesion, target).item() > 10
    assert prediction_js(full, full).item() < 1e-7
    assert prediction_js(full, lesion).item() > 0.5


def test_external_only_write_and_conservation() -> None:
    model = make_model()
    state = model.initial_state(2)
    state, external = model.step(state, evidence_event(torch.tensor([1, 2]), torch.tensor([3, 4])))
    assert external["external_write_flag"].all()
    before_external_time = state.external_time
    state, null = model.step(state, None)
    assert not null["external_write_flag"].any()
    assert torch.count_nonzero(null["external_update"]) == 0
    state, self_out = model.step(state, self_output_event(torch.tensor([1, 2])))
    assert not self_out["external_write_flag"].any()
    assert torch.count_nonzero(self_out["external_update"]) == 0
    assert state.external_time == before_external_time
    assert external["conservation_error"].max().item() < 1e-6
    assert null["conservation_error"].max().item() < 1e-6


def test_direction_specific_consolidation_block_preserves_readout() -> None:
    model = make_model()
    state = model.initial_state(2)
    state.H.normal_(); state.F.normal_(); state.M.normal_()
    key = torch.eye(model.config.key_dim)[:2]
    standard, normal_output = model.step(state.clone(), None)
    blocked, block_output = model.step(state.clone(), None, block_transfer_key=key)
    assert block_output["conservation_error"].max().item() < 1e-6
    assert torch.linalg.vector_norm(block_output["transfer"]) <= torch.linalg.vector_norm(normal_output["transfer"])
    assert torch.allclose(standard.H, blocked.H)


def test_pathological_control_definitely_writes() -> None:
    model = make_model()
    state = model.initial_state(2)
    new_state, update = pathological_self_output_write(model, state, torch.tensor([3, 4]))
    assert torch.linalg.vector_norm(update).item() > 0.01
    assert torch.linalg.vector_norm(new_state.F).item() > 0.01
    assert new_state.external_time == state.external_time


def test_world_targets_are_future_shifted_and_not_in_event_schema() -> None:
    for family in FAMILIES:
        world = generate_world(family, batch=2, length=16, seed=77)
        assert len(world.events) == 16
        assert torch.equal(world.future(3, 4), world.targets[:, 7])
        assert not (set(world.events[0].__dataclass_fields__) & {"target", "answer", "history", "query"})
        assert all(event.key_id.shape == (2,) for event in world.events)
    model = make_model()
    world = generate_world("long_gap_relation", batch=2, length=16, seed=88)
    prefix = world.bridge_index
    state, _ = run_prefix(model, world, prefix=prefix, null_ticks=0)
    # The target event is beyond the executed prefix. Altering it cannot change state.
    modified = list(world.events)
    modified[-1] = evidence_event(torch.tensor([7, 7]), torch.tensor([7, 7]))
    from etrcm.stage1_4.world import WorldBatch
    altered = WorldBatch(world.family, tuple(modified), world.targets, world.target_key,
                         world.exposure_count, world.bridge_index, world.metadata)
    other, _ = run_prefix(model, altered, prefix=prefix, null_ticks=0)
    assert torch.equal(state.H, other.H)
    assert torch.equal(state.F, other.F)
    assert torch.equal(state.M, other.M)


def test_historical_frozen_assets_remain_immutable() -> None:
    freeze = json.loads((ROOT / "artifacts/stage1_4_prior_assets.freeze.json").read_text())
    amendment = json.loads((ROOT / "artifacts/stage1_4_amendment1.freeze.json").read_text())
    checks = {
        "reports/STAGE1_3_FINAL_REPORT.md": amendment["server_stage1_3_final_report_sha256"],
        "artifacts/stage1_3_protocol.freeze.json": amendment["server_stage1_3_protocol_freeze_sha256"],
    }
    for relative, expected in checks.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    prior_tree = subprocess.check_output(
        ["git", "rev-parse", f"{freeze['prior_commit']}^{{tree}}"], cwd=ROOT, text=True
    ).strip()
    assert prior_tree == freeze["prior_tree"]


def test_stage14_intervention_evaluation_smoke() -> None:
    model = make_model().eval()
    with torch.no_grad():
        a = evaluate_A(model, seed=91, run_id="smoke", device=torch.device("cpu"), batch=2)
        b = evaluate_B(model, seed=91, run_id="smoke", device=torch.device("cpu"),
                       gap_counts=(4,), batch=2)
        cd = evaluate_CD(model, seed=91, run_id="smoke", device=torch.device("cpu"), batch=2)
        ef = evaluate_EF(model, seed=91, run_id="smoke", device=torch.device("cpu"), batch=2, gap=4)
        safety = evaluate_safety(model, seed=91, run_id="smoke", device=torch.device("cpu"), batch=2)
    assert {"learned_NULL", "frozen_H_NULL", "random_NULL"} <= {row["intervention_condition"] for row in a}
    assert {"full", "M_lesion", "random_q_M", "shuffled_M"} <= {row["intervention_condition"] for row in b}
    assert {row["experiment"] for row in cd} == {"C", "D"}
    assert all(row["H_restoration_flag"] for row in cd if row["experiment"] == "D")
    assert len(ef) == 2 and all(row["causal_usage"] is not None for row in ef)
    assert any(row["self_write_norm"] > 0 for row in safety if row["intervention_condition"] == "B_bad_external_write")


def test_formal_selection_hashes_are_frozen() -> None:
    manifest = json.loads((ROOT / "artifacts/stage1_4_formal_selection.freeze.json").read_text())
    for relative, expected in manifest["files_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert len(manifest["formal_seeds"]) == 8
    assert not manifest["experiment_G_authorized"]
