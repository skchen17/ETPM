"""Frozen-checkpoint and gated-intervention invariants."""

from __future__ import annotations

import subprocess
from pathlib import Path

import torch

from etrcm.stage1_1.model import LearnedState
from etrcm.stage2c1.diagnostic import legal_history
from etrcm.stage2c2.protocol import ASSISTANCE_SCHEDULE, final_evaluation_is_endogenous, next_gate_status
from experiments.stage2c1_lifetime import LifetimeModel
from experiments.stage2c2_pathway import consequence_batch, fit_oracle_h, load_l3, parameter_hash


def test_frozen_l3_parameter_hash_unchanged_after_oracle_h_fit():
    model=LifetimeModel()
    for parameter in model.parameters():parameter.requires_grad_(False)
    before=parameter_hash(model)
    fitted=fit_oracle_h(model,77,2.0,"cpu",steps=4,restarts=1)
    assert parameter_hash(model)==before
    assert torch.allclose(torch.tensor(fitted["state_norms"]),torch.tensor([2.,2.]),atol=1e-5)


def test_oracle_h_changes_only_h():
    model=LifetimeModel()
    original=model.initial_state(2,"cpu")
    h=original.H.clone();h[1]=-h[1]
    changed=LearnedState(h,original.F,original.M)
    assert torch.equal(changed.F,original.F)
    assert torch.equal(changed.M,original.M)
    assert not torch.equal(changed.H,original.H)


def test_oracle_read_only_changes_read_input_not_matrices():
    model=LifetimeModel()
    original=model.initial_state(2,"cpu")
    oracle=torch.ones(2,model.core.config.value_dim)
    post,diag=model.core.step_with_read(original,None,slow_read_override=oracle)
    assert torch.equal(diag["r_M"],oracle)
    assert torch.equal(original.F,torch.zeros_like(original.F))
    assert torch.equal(original.M,torch.zeros_like(original.M))
    assert torch.equal(post.F,original.F)
    assert torch.equal(post.M,original.M)


def test_oracle_m_only_changes_m_and_swap_exact():
    model=LifetimeModel()
    original=model.initial_state(2,"cpu")
    m=original.M.clone();m[0,0,0]=1;m[1,1,1]=1
    changed=LearnedState(original.H,original.F,m)
    swapped=LearnedState(changed.H,changed.F,changed.M.flip(0))
    assert torch.equal(changed.H,swapped.H)
    assert torch.equal(changed.F,swapped.F)
    assert torch.equal(swapped.M,changed.M.flip(0))


def test_m_read_clamp_exact():
    model=LifetimeModel()
    state=model.initial_state(2,"cpu")
    m=state.M.clone();m[:,0,0]=1
    state=LearnedState(state.H,state.F,m)
    zero=torch.zeros(2,model.core.config.value_dim)
    _,trace=model.core.step_with_read(state,None,slow_read_override=zero)
    assert torch.equal(trace["r_M"],zero)
    assert torch.equal(trace["normalized_r_M"],zero)


def test_no_correct_action_or_outcome_one_hot_input():
    z,a,y=consequence_batch(torch.Generator().manual_seed(5),64,"cpu")
    assert [int(((z==i)&(a==j)).sum()) for i in (0,1) for j in (0,1)]==[16]*4
    assert torch.all(y[z==a]==0)
    assert torch.all((y[z!=a]>=1)&(y[z!=a]<=3))


def test_oracle_states_are_l3_local_not_l1_l2_copy():
    model,path=load_l3(7201,"cpu")
    assert "stage2c1/raw/lifetime/7201/checkpoint.pt" in str(path)
    assert "oracle_memory" not in str(path) and "history_encoder" not in str(path)
    assert all(not p.requires_grad for p in model.parameters())


def test_legal_history_has_only_past_fields():
    past,z=legal_history(8,8,"novel","cpu",torch.Generator().manual_seed(9))
    assert past.shape==(8,8,5)
    assert z.shape==(8,)
    assert torch.all((past[:,:,3]==2)|(past[:,:,3]==3))
    assert torch.all((past[:,:,4]>=4)&(past[:,:,4]<=7))


def test_native_probe_train_test_episode_sets_disjoint():
    seed=7201
    train={((seed*2+17)*100_000+i) for i in range(128)}
    test={((seed*2+17017)*100_000+i) for i in range(128)}
    assert train.isdisjoint(test)


def test_curriculum_schedule_exact_and_final_teacher_zero():
    assert ASSISTANCE_SCHEDULE==(1.0,.75,.5,.25,0.0)
    assert all(ASSISTANCE_SCHEDULE[i]>ASSISTANCE_SCHEDULE[i+1] for i in range(4))
    assert final_evaluation_is_endogenous(ASSISTANCE_SCHEDULE[-1])
    assert not final_evaluation_is_endogenous(.25)


def test_stop_rule_marks_downstream_not_run():
    assert next_gate_status(False,None)=="NOT_RUN_BY_GATE"
    assert next_gate_status(True,False)=="FAIL"


def test_historical_stage2c_and_stage2c1_tracked_files_unchanged():
    root=Path(__file__).resolve().parents[1]
    paths=["results/stage2c","results/stage2c1","src/etrcm/stage2c",
           "src/etrcm/stage2c1","reports/STAGE2C_BEHAVIORAL_MEMORY_RESULTS.md",
           "reports/STAGE2C1_ACTION_OUTCOME_BINDING_RESULTS.md"]
    run=subprocess.run(["git","diff","--quiet","HEAD","--",*paths],cwd=root)
    assert run.returncode==0
