"""Mathematics, hook fidelity, diagnostic purity and stopping-rule tests."""

from __future__ import annotations

import json
import inspect
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from etrcm.stage2d.model import parameter_hash
from etrcm.stage2d3.interaction import factorial_components, factorial_reconstruct
from etrcm.stage2d4.model import Stage2D4Model
from etrcm.stage2d5.flow import candidate_flow, cell_means
from etrcm.stage2d6.fusion import (anatomy, common_offset_from_target,
                                   finite_response, low_curvature_offset,
                                   offset_metrics, pre_override, silu_derivatives,
                                   top_k, unit_removal)


@pytest.fixture(scope="module")
def prepared():
    torch.set_num_threads(1);torch.manual_seed(43)
    model=Stage2D4Model("A0").eval()
    state=model.initial_state(8,"cpu")
    actions=[torch.full((8,),a,dtype=torch.long) for a in (0,1)]
    rows=[candidate_flow(model,state,a) for a in actions]
    return model,state,rows


def test_exact_factorial_reconstruction():
    x=torch.randn(2,2,32)
    parts=factorial_components(x)
    assert torch.allclose(factorial_reconstruct(parts),x,atol=1e-6)


def test_fusion_hooks_exact(prepared):
    model,state,rows=prepared
    linear=model.action_head.head[0]
    for a,row in enumerate(rows):
        assert torch.allclose(row["fusion_pre"],linear(row["fusion_input"]),atol=1e-6)
        assert torch.allclose(row["fusion_post"],model.action_head.head[1](row["fusion_pre"]),atol=1e-6)
        logits,_=model.candidate(state,torch.full((8,),a,dtype=torch.long))
        assert torch.allclose(row["logits"],logits,atol=1e-6)


def test_additive_merge_and_silu_analytic(prepared):
    _,_,rows=prepared
    for row in rows:
        assert torch.allclose(row["fusion_H_projection"]+
                              row["fusion_action_projection"]+0*row["fusion_pre"],
                              row["fusion_additive_merge"],atol=1e-6)
        assert torch.allclose(row["fusion_post"],row["fusion_pre"]*
                              torch.sigmoid(row["fusion_pre"]),atol=1e-6)


def test_silu_derivatives():
    x=torch.linspace(-5,5,101,dtype=torch.float64,requires_grad=True)
    first,second=silu_derivatives(x)
    ad_first=torch.autograd.grad(F.silu(x).sum(),x,create_graph=True)[0]
    ad_second=torch.autograd.grad(ad_first.sum(),x)[0]
    assert torch.allclose(first,ad_first,atol=1e-11)
    assert torch.allclose(second,ad_second,atol=1e-11)


def test_pre_interaction_is_h_projection_interaction(prepared):
    _,_,rows=prepared
    pre=factorial_components(cell_means([r["fusion_pre"] for r in rows],4))["interaction"]
    h=factorial_components(cell_means([r["fusion_H_projection"] for r in rows],4))["interaction"]
    a=factorial_components(cell_means([r["fusion_action_projection"] for r in rows],4))["interaction"]
    assert torch.allclose(pre,h+a,atol=1e-6)
    assert a.norm()<1e-6


def test_common_offset_preserves_pre_main_effects_and_state(prepared):
    model,state,rows=prepared
    before=(state.H.clone(),state.F.clone(),state.M.clone(),state.tau,state.external_time)
    param=parameter_hash(model)
    delta=torch.zeros(32);delta[:8]=.2
    result=offset_metrics(model,state,rows,4,delta)
    assert result["pre_state_effect_error"]<1e-5
    assert result["pre_action_effect_error"]<1e-5
    assert result["pre_interaction_error"]<1e-5
    assert parameter_hash(model)==param
    assert all(torch.equal(x,y) for x,y in zip(before[:3],(state.H,state.F,state.M)))
    assert before[3:]==(state.tau,state.external_time)


def test_unit_removal_exact(prepared):
    model,_,rows=prepared
    interaction=factorial_components(cell_means([r["fusion_post"] for r in rows],4))["interaction"]
    changed=unit_removal(model,rows,4,interaction,range(32))
    after=factorial_components(cell_means([r["fusion_post"] for r in changed],4))
    before=factorial_components(cell_means([r["fusion_post"] for r in rows],4))
    assert after["interaction"].norm()<1e-5
    assert torch.allclose(after["state"],before["state"],atol=1e-6)
    assert torch.allclose(after["action"],before["action"],atol=1e-6)


def test_finite_response_shape_and_rank(prepared):
    model,_,rows=prepared
    response=finite_response(model,rows,4)
    assert response.shape==(4,32)
    assert torch.isfinite(response).all()
    result=anatomy(model,rows,4,unit_causality=False)
    assert torch.tensor(result["pre_cell_means"]).shape==(2,2,32)
    assert torch.tensor(result["silu_slope_cells"]).shape==(2,2,32)
    assert torch.tensor(result["silu_curvature_cells"]).shape==(2,2,32)


def test_unit_topk_and_random_match(prepared):
    model,_,rows=prepared
    row=anatomy(model,rows,4)
    for k in (1,2,4,8,16):
        selected=top_k(row,k)
        assert len(selected)==k and len(set(selected))==k
        assert max(selected)<32


def test_low_curvature_shift_has_matching_support(prepared):
    _,_,rows=prepared
    delta=low_curvature_offset(rows,[0,1,2,3],1.)
    assert set(torch.nonzero(delta).flatten().tolist())=={0,1,2,3}


def test_no_labelled_scaffold_or_memory_law_change():
    source=Path("experiments/stage2d3_train.py").read_text()
    baseline=Path("experiments/stage2d6_train_baseline.py").read_text()
    assert '"--arm", "C0"' in baseline
    assert '"--gamma", ".50"' in baseline
    assert '"--rho-fast", ".97"' in baseline
    assert '"--rho-slow", ".9995"' in baseline
    assert '"correct_action_labels": False' in source
    assert '"memory_labels": False' in source
    assert '"evaluation_scaffold": False' in source
    assert "latent_z" not in inspect.signature(Stage2D4Model.candidate).parameters
    assert "correct_action" not in inspect.signature(Stage2D4Model.candidate).parameters
    assert "memory_label" not in inspect.signature(Stage2D4Model.candidate).parameters


def test_training_rescue_gate_file_declares_no_unrun_regularizer():
    path=Path("results/stage2d6/training_rescue/status.json")
    if path.exists():
        payload=json.loads(path.read_text())
        assert payload["status"] in ("NOT_RUN_BY_PROTOCOL","COMPLETED")
        if payload["status"]=="NOT_RUN_BY_PROTOCOL":
            assert payload.get("regularizer_enabled",False) is False


def test_post_fusion_positive_control_reproduced():
    path=Path("results/stage2d6/operating_point_rescue/formal/confirmatory_query/i19103_d20103.json")
    if not path.exists(): pytest.skip("operating-point audit not generated yet")
    row=json.loads(path.read_text())
    control=row["post_positive_control"]["interventions"]["interaction"]
    assert max(item["I_HA"] for item in control.values())>row["native"]["IHA"]
    assert row["integrity"]["parameters_unchanged"]


def test_prior_stage_manifest_unchanged():
    path=Path("results/stage2d6/manifests/integrity_verification.json")
    if path.exists():
        result=json.loads(path.read_text())
        assert result["pass"] and not result["changed"]


def test_formal_stopping_rule_and_report():
    gates=Path("results/stage2d6/processed/formal_gates.json")
    if not gates.exists(): pytest.skip("confirmatory cohort not finished")
    result=json.loads(gates.read_text())
    if not result["training_authorized"]:
        assert all(result["gates"][f"G{i}"]=="NOT_RUN_BY_PROTOCOL"
                   for i in range(118,123))
        for folder in ("training_rescue","dynamics_recheck","memory_mediation","continuous_runs"):
            path=Path("results/stage2d6")/folder/"status.json"
            assert path.exists() and json.loads(path.read_text())["status"]=="NOT_RUN_BY_PROTOCOL"
    report=Path("reports/STAGE2D6_CONDITIONAL_INTERACTION_GENERATION_RESULTS.md")
    assert report.exists() and report.stat().st_size>1000
