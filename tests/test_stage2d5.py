from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import torch

from etrcm.stage2c.world import context_token
from etrcm.stage2d.model import head_hash, parameter_hash
from etrcm.stage2d3.interaction import factorial_components, factorial_reconstruct
from etrcm.stage2d4.model import Stage2D4Model, play
from etrcm.stage2d5.flow import (ORDER, candidate_flow, cell_means,
                                 factorial_injection, interaction_metrics, norm_match,
                                 transmission)


def fixture():
    torch.manual_seed(31); model=Stage2D4Model("A1")
    state=model.initial_state(4,"cpu");state.F.normal_();state.M.normal_()
    return model,state,torch.tensor([0,0,1,1])


def test_factorial_decomposition_exact():
    x=torch.randn(2,2,8,dtype=torch.float64)
    assert torch.allclose(factorial_reconstruct(factorial_components(x)),x,atol=1e-12)


def test_transmission_ratios_exact():
    a={"norm":2.,"normalized":.4};b={"norm":.5,"normalized":.2}
    t=transmission(a,b,eps=0)
    assert t=={"absolute":.25,"normalized":.5}


def test_candidate_branch_never_writes_or_advances_clocks():
    model,state,action=fixture();before=state.clone()
    candidate_flow(model,state,action)
    assert torch.equal(state.H,before.H)
    assert torch.equal(state.F,before.F)
    assert torch.equal(state.M,before.M)
    assert state.tau==before.tau and state.external_time==before.external_time


def test_layer_hooks_exact_vs_frozen_candidate():
    model,state,action=fixture()
    with torch.no_grad():
        flow=candidate_flow(model,state,action);logits,reference=model.candidate(state,action)
    assert torch.allclose(flow["logits"],logits,atol=1e-6)
    assert torch.allclose(flow["gated_read"],reference["read"],atol=1e-6)
    assert torch.allclose(flow["q_F"],reference["q_F"],atol=1e-6)
    assert torch.allclose(flow["q_M"],reference["q_M"],atol=1e-6)


def test_all_required_internal_and_fusion_hooks_present():
    model,state,action=fixture();flow=candidate_flow(model,state,action)
    assert set(ORDER).issubset(flow)
    assert set(("H_projection","read_projection","event_projection","gate_preactivation",
                "gate_activation","candidate_update","fusion_H_projection",
                "fusion_action_projection","fusion_additive_merge")).issubset(flow)


def test_read_projection_decomposition_exact():
    model,state,action=fixture();flow=candidate_flow(model,state,action)
    encoded=model.core.event_encoder(context_token(torch.tensor([2,2,3,3])))
    h_pre=state.H+model.core.event_to_slots(encoded).view(4,1,32)
    expected=model.core.core_in(torch.cat([
        model.core.norm(h_pre),
        flow["gated_read"][:,None,:],
        encoded[:,None,:]],-1))
    assert torch.allclose(flow["H_projection"]+flow["read_projection"]+
                          flow["event_projection"]+model.core.core_in.bias,expected,atol=1e-5)


def test_f_only_m_only_read_decomposition_exact():
    model,state,action=fixture()
    f=candidate_flow(model,state,action,mixing="F_only")
    m=candidate_flow(model,state,action,mixing="M_only")
    assert torch.equal(f["gated_read"],f["normalized_F"])
    assert torch.equal(m["gated_read"],m["normalized_M"])


def test_injection_changes_only_factorial_component():
    x=torch.randn(4,8);d=torch.randn(8)
    outputs=[factorial_injection(x,2,a,d,"interaction",.5) for a in (0,1)]
    before=factorial_components(cell_means([x,x],2))
    after=factorial_components(cell_means(outputs,2))
    assert torch.allclose(after["state"],before["state"],atol=1e-6)
    assert torch.allclose(after["action"],before["action"],atol=1e-6)
    assert torch.allclose(after["interaction"]-before["interaction"],2*d,atol=1e-6)


def test_norm_matching_exact():
    a=torch.randn(32);b=torch.randn(32)
    assert torch.allclose(norm_match(a,b).norm(),b.norm(),atol=1e-6)


def test_residual_scaling_diagnostic_keeps_parameters_and_state():
    model,state,action=fixture();params=parameter_hash(model);before=state.clone()
    for scale in (0.,.25,.5,1.,1.5,2.):candidate_flow(model,state,action,residual_alpha=scale)
    assert parameter_hash(model)==params
    assert torch.equal(state.H,before.H) and torch.equal(state.F,before.F) and torch.equal(state.M,before.M)


def test_fusion_projection_recombines_exact():
    model,state,action=fixture();flow=candidate_flow(model,state,action)
    assert torch.allclose(flow["fusion_H_projection"]+flow["fusion_action_projection"],
                          flow["fusion_additive_merge"],atol=1e-6)
    assert torch.allclose(flow["fusion_pre"],model.action_head.head[0](flow["fusion_input"]),atol=1e-6)


def test_read_only_activation_override_does_not_mutate_model():
    model,state,action=fixture();params=parameter_hash(model);head=head_hash(model)
    base=candidate_flow(model,state,action)
    candidate_flow(model,state,action,override={"gated_read":torch.zeros_like(base["gated_read"])})
    assert parameter_hash(model)==params and head_hash(model)==head


def test_restoration_replay_is_not_persistent_update():
    model,state,action=fixture();before=state.clone()
    base=candidate_flow(model,state,action)
    candidate_flow(model,state,action,override={"fusion_post":base["fusion_post"]+.1})
    assert torch.equal(state.H,before.H) and torch.equal(state.F,before.F) and torch.equal(state.M,before.M)


def test_architecture_rescue_requires_gate_authorization():
    status=json.loads(Path("results/stage2d5/architecture_rescue/status.json").read_text())
    if not (status["G100_pass"] and (status["G102_pass"] or status["G103_pass"])):
        assert status["status"]=="NOT_AUTHORIZED_BY_PROTOCOL"


def test_observed_only_objective_preserved():
    source=inspect.getsource(play)
    assert "observed_only" in source and "row.latent" not in source
    for name in ("z","correct_action","memory_label"):
        assert name not in inspect.signature(play).parameters


def test_historical_stage2_assets_unchanged(tmp_path):
    manifest=Path("results/stage2d5/manifests/prior_manifest.json")
    assert manifest.is_file()
    subprocess.run([sys.executable,"experiments/stage2d5_integrity.py","--mode","verify",
                    "--manifest",str(manifest),"--out",str(tmp_path/"verify.json")],check=True)
