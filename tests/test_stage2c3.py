"""Stage 2C.3 shortcut, freeze, intervention and integrity invariants."""

from __future__ import annotations

import json

import pytest
import torch

from etrcm.stage1_1.model import LearnedState
from etrcm.stage2c.world import Experience,OUTCOME
from etrcm.stage2c3.controls import (MarginalPredictor,ActionOnlyPredictor,
    HistoryOnlyPredictor)
from etrcm.stage2c3.protocol import (CHECKPOINTS,MARGINAL_CE,arm_phase,
    evaluator_lr_ratio,target_distribution)
from experiments.stage2c1_lifetime import LifetimeModel,swap
from experiments.stage2c2_pathway import parameter_hash
from experiments.stage2c3_evaluate import (clamp_probe,formation_clamped_state,
    formation_zeroed_state)
from experiments.stage2c3_integrity import MANIFEST,inventory
from experiments.stage2c3_train import (context_state,branch_logits,head_hash,
    health_snapshot,training_event)
from experiments.stage2c3_analyze import gates


def test_marginal_predictor_restricted_interface():
    model=MarginalPredictor().fit(torch.tensor([0,0,1,2,3,0]))
    assert model.predict(2).shape==(2,4)
    with pytest.raises(TypeError):model.predict(2,torch.zeros(2))


def test_action_only_predictor_restricted_interface():
    model=ActionOnlyPredictor().fit(torch.tensor([0,0,1,1]),torch.tensor([0,1,0,2]))
    assert model.predict(torch.tensor([0,1])).shape==(2,4)
    with pytest.raises(TypeError):model.predict(torch.tensor([0]),torch.zeros(1))


def test_history_only_predictor_has_no_action_argument():
    model=HistoryOnlyPredictor(32)
    assert model(torch.zeros(2,32)).shape==(2,4)
    with pytest.raises(TypeError):model(torch.zeros(2,32),torch.zeros(2))


def test_counterfactual_target_distributions_are_exact():
    z=torch.tensor([[0,0],[1,1]])
    a=torch.tensor([[0,1],[0,1]])
    target=target_distribution(z,a)
    assert torch.equal(target[0,0],torch.tensor([1.,0.,0.,0.]))
    assert torch.allclose(target[0,1],torch.tensor([0.,1/3,1/3,1/3]))
    assert torch.allclose(target[1,0],target[0,1])
    assert torch.equal(target[1,1],target[0,0])
    assert 1.24<MARGINAL_CE<1.25


def test_counterfactual_branches_share_context_state():
    model=LifetimeModel()
    state=model.initial_state(2,"cpu")
    items=[Experience(8,14,20,0,OUTCOME[0]),Experience(8,14,20,1,OUTCOME[1])]
    context=context_state(model,state,items)
    before=context.H.clone()
    for action in (0,1):
        branch,logits=branch_logits(model,context.clone(),torch.full((2,),action))
        assert logits.shape==(2,4)
        assert branch.tau==context.tau+1
        assert torch.equal(context.H,before)


def test_latent_never_enters_lifetime_state_transition():
    model=LifetimeModel()
    initial=model.initial_state(2,"cpu")
    items=[Experience(8,14,20,0,OUTCOME[0]),Experience(9,15,21,1,OUTCOME[2])]
    left,loss_a=training_event(model,initial.clone(),items,torch.zeros(2,dtype=torch.long),"A1")
    right,loss_b=training_event(model,initial.clone(),items,torch.ones(2,dtype=torch.long),"A1")
    for name in ("H","F","M"):assert torch.equal(getattr(left,name),getattr(right,name))
    assert not torch.equal(loss_a,loss_b)


def test_evaluator_freeze_is_bit_exact():
    model=LifetimeModel()
    for p in model.action_head.parameters():p.requires_grad_(False)
    before=head_hash(model)
    opt=torch.optim.SGD(model.core.parameters(),lr=.001)
    state=model.initial_state(2,"cpu")
    items=[Experience(8,14,20,0,OUTCOME[0]),Experience(9,15,21,1,OUTCOME[2])]
    _,loss=training_event(model,state,items,torch.tensor([0,1]),"A2")
    loss.backward();opt.step()
    assert head_hash(model)==before


def test_gradual_unfreeze_schedule_exact():
    assert CHECKPOINTS==(0,25,50,100,200,300,500,1000)
    assert arm_phase("A2",1000)=="frozen_evaluator"
    assert arm_phase("A3",499)=="frozen_evaluator"
    assert arm_phase("A3",500)=="gradual_unfreeze"
    assert evaluator_lr_ratio("A3",499)==0
    assert evaluator_lr_ratio("A3",500)==.1
    assert evaluator_lr_ratio("A2",1000)==0


def test_h_f_m_swap_exact():
    model=LifetimeModel()
    state=model.initial_state(2,"cpu")
    h=state.H.clone();f=state.F.clone();m=state.M.clone()
    h[0]+=1;f[0]+=2;m[0]+=3
    state=LearnedState(h,f,m)
    for which in ("H","F","M","FM","HFM"):
        changed=swap(state,which)
        for name in ("H","F","M"):
            expected=getattr(state,name).flip(0) if name in which else getattr(state,name)
            assert torch.equal(getattr(changed,name),expected)


def test_read_clamp_exact():
    model=LifetimeModel()
    state=model.initial_state(2,"cpu")
    f=state.F.clone();m=state.M.clone()
    f[:,0,0]=1;m[:,0,0]=1
    state=LearnedState(state.H,f,m)
    zero=torch.zeros(2,model.core.config.value_dim)
    _,trace=model.core.step_with_read(state,None,fast_read_override=zero,slow_read_override=zero)
    assert torch.equal(trace["r_F"],zero)
    assert torch.equal(trace["r_M"],zero)
    p=clamp_probe(model,state,[(8,14,20),(8,14,20)],"FM")
    assert p.shape==(2,2,4)


def test_formation_window_clamp_is_frozen_and_preserves_external_time():
    model=LifetimeModel()
    before=parameter_hash(model)
    baseline=formation_clamped_state(model,77,2,"cpu","F")
    assert baseline.external_time==12
    assert parameter_hash(model)==before


def test_formation_window_state_zero_is_exact_and_frozen():
    model=LifetimeModel();before=parameter_hash(model)
    state=formation_zeroed_state(model,78,2,"cpu","FM")
    assert torch.count_nonzero(state.F)==0
    assert torch.count_nonzero(state.M)==0
    assert state.external_time==12
    assert parameter_hash(model)==before


def test_formal_snapshot_has_no_optimizer_or_parameter_mutation():
    model=LifetimeModel()
    before=parameter_hash(model)
    row=health_snapshot(model,7611,0,"cpu",reps=1)
    assert row["parameter_sha256_before"]==before
    assert row["parameter_sha256_after"]==before


def test_historical_stage2c_to_stage2c2_artifacts_unchanged():
    expected=json.loads(MANIFEST.read_text())["files"]
    assert inventory()==expected


def test_preregistered_gate_joint_seed_and_formation_clamp_logic():
    names=("swap_F","swap_M","swap_FM","clamp_F","clamp_M","clamp_FM",
           "formation_clamp_F","formation_clamp_M","formation_clamp_FM")
    data={}
    for arm in ("A0","A1","A2","A3"):
        data[arm]={}
        for seed in range(8):
            successful=arm!="A0" and seed<6
            health={"action_TV":.5 if successful else .01,
                    "interaction_y0":.8 if successful else 0.0,
                    "CFA":.3 if successful else 0.0}
            expo={"0":0.0,"1":.5 if successful else 0.0,
                  "16":.6 if successful else 0.0,"32":.6 if successful else 0.0}
            interventions={name:(-.2 if name=="formation_clamp_FM" and successful else 0.0)
                           for name in names}
            data[arm][seed]={"eval":{"behavior":{"aggregate":{
                "final_health":health,"exposure":expo,"interventions":interventions}}}}
    config={"gates":{"tau_tv":.2,"tau_interaction":.3,"delta_cfa":.1,
                     "tau_bs":.1,"tau_exposure_delta":.1,"tau_peripheral_change":.05}}
    result=gates(data,config)
    assert all(result["statuses"][key]=="PASS" for key in ("G57","G58","G59","G60","G61"))
    assert result["counts"]["A1"]["G60"]["formation_clamp_FM"]==6
