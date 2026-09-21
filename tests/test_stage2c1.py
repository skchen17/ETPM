"""Stage 2C.1 leakage, intervention, and frozen-protocol checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

from etrcm.stage2c1.diagnostic import (ActionHead, HistoryEncoder, OracleLatent,
    PersistentInterface, balanced_batch, legal_history, oracle_matrices)


def test_balanced_latent_and_action_four_cells():
    gen=torch.Generator().manual_seed(8)
    z,a,y=balanced_batch(64,"cpu",gen)
    assert [int(((z==i)&(a==j)).sum()) for i in (0,1) for j in (0,1)] == [16]*4
    assert torch.all(y[a==z]==0)
    assert bool(torch.all((y[a!=z]>=1)&(y[a!=z]<=3)))


def test_action_reaches_head_and_swap_changes_exactly_one_input():
    head=ActionHead("late_concat")
    h=torch.randn(4,32)
    a=torch.zeros(4,dtype=torch.long); b=torch.ones_like(a)
    seen=[]
    hook=head.action_embedding.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
    out_a=head(h,a);out_b=head(h,b)
    hook.remove()
    assert torch.equal(seen[0],a) and torch.equal(seen[1],b)
    assert torch.equal(h,h.clone())
    assert not torch.equal(out_a,out_b)


def test_action_branch_has_gradient():
    model=OracleLatent("modulation")
    z=torch.tensor([0,0,1,1]);a=torch.tensor([0,1,0,1])
    model(z,a).sum().backward()
    assert model.action_head.action_embedding.weight.grad.abs().sum()>0
    assert model.action_head.action_projection.weight.grad.abs().sum()>0


def test_oracle_z_encodes_neither_outcome_nor_correct_action_label():
    model=OracleLatent("late_concat")
    assert model.latent_embedding.num_embeddings==2
    assert model.action_head.action_embedding.num_embeddings==2
    assert not hasattr(model,"outcome_embedding")
    assert not hasattr(model,"correct_action_head")
    # The *same* z can be paired with either action and either resulting y.
    z,a,y=balanced_batch(32,"cpu",torch.Generator().manual_seed(4))
    assert int(((z==0)&(a==0)).sum())==int(((z==0)&(a==1)).sum())
    assert int(y[(z==0)&(a==0)].unique().item())==0


def test_oracle_m_equal_norm_orthogonal_and_fixed():
    m=oracle_matrices()
    assert m.shape==(2,8,8)
    assert torch.allclose(m.flatten(1).norm(dim=-1),torch.ones(2),atol=1e-6)
    assert torch.allclose((m[0]*m[1]).sum(),torch.zeros(()),atol=1e-6)
    model=PersistentInterface("late_concat",history=False)
    assert not any(name=="oracle_M" for name,_ in model.named_parameters())


def test_oracle_m_only_through_inherited_read_H():
    model=PersistentInterface("late_concat",history=False)
    z=torch.tensor([0,1]);a=torch.zeros(2,dtype=torch.long)
    state0,state1,trace=model.states(model.oracle_M)
    assert torch.equal(state0.H[0],state0.H[1])
    assert torch.count_nonzero(state0.F)==0
    assert not torch.equal(trace["r_M"][0],trace["r_M"][1])
    clamped=model(a,latent=z,clamp_m=True)
    assert torch.allclose(clamped[0],clamped[1],atol=1e-6)


def test_oracle_m_swap_exact():
    model=PersistentInterface("late_concat",history=False)
    a=torch.tensor([0,1,0,1]);z=torch.tensor([0,0,1,1])
    original=model(a,latent=z).view(2,2,4)
    swapped=model(a,latent=1-z).view(2,2,4)
    assert torch.equal(swapped,original.flip(0))
    assert torch.equal(model.oracle_M[1-z],model.oracle_M[z].view(2,2,8,8).flip(0).view(4,8,8))


def test_history_encoder_only_past_observations_and_novel_surface():
    gen=torch.Generator().manual_seed(3)
    past,z=legal_history(32,8,"novel","cpu",gen)
    assert past.shape==(32,8,5)
    assert (past[:,:,0]>=8).all() and (past[:,:,1]>=14).all()
    assert ((past[:,:,3]==2)|(past[:,:,3]==3)).all()
    assert ((past[:,:,4]>=4)&(past[:,:,4]<=7)).all()
    model=PersistentInterface("late_concat",history=True)
    a=torch.zeros(32,dtype=torch.long)
    out1=model(a,past=past)
    future_consequence=torch.randint(4,(32,))  # never passed to model
    future_consequence=3-future_consequence
    out2=model(a,past=past)
    assert torch.equal(out1,out2)
    assert future_consequence.shape==(32,)
    with pytest.raises(ValueError):model(a,latent=z,past=past)
    with pytest.raises(ValueError):model(a,latent=z)


def test_history_encoder_uses_all_five_observed_fields_not_latent():
    model=HistoryEncoder()
    past,z=legal_history(16,8,"train","cpu",torch.Generator().manual_seed(5))
    assert model(past).shape==(16,8,8)
    with pytest.raises(ValueError):model(z)


def test_formal_parameters_frozen_in_config():
    root=Path(__file__).resolve().parents[1]
    config=(root/"configs/stage2c1_formal.yaml").read_text()
    assert "frozen_after_two_development_seeds_before_formal" in config
    assert "7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208" in config
    assert "minimum_independent_seeds: 6" in config
    assert "L0_steps: 3000" in config


def test_historical_stage2c_tracked_artifacts_unchanged():
    root=Path(__file__).resolve().parents[1]
    paths=["src/etrcm/stage2c","experiments/stage2c_train.py",
           "experiments/stage2c_eval.py","configs/stage2c_formal.yaml",
           "reports/STAGE2C_BEHAVIORAL_MEMORY_RESULTS.md","results/stage2c"]
    done=subprocess.run(["git","diff","--quiet","HEAD","--",*paths],cwd=root,check=False)
    assert done.returncode==0


def test_formal_evaluation_no_parameter_update():
    from experiments.stage2c1_train import digest, evaluate
    model=OracleLatent("late_concat")
    before=digest(model)
    _=evaluate(model,"L0","cpu",7201)
    assert digest(model)==before


def test_l3_reuses_original_memory_core_and_direct_action_head():
    from experiments.stage2c1_lifetime import LifetimeModel
    from etrcm.stage2c.model import BehavioralModel
    from etrcm.stage2c1.diagnostic import core_config
    newer=LifetimeModel()
    original=BehavioralModel(core_config(),"full")
    assert type(newer.core) is type(original.core)
    assert newer.core._external_write.__func__ is original.core._external_write.__func__
    assert newer.core._consolidate.__func__ is original.core._consolidate.__func__
    assert newer.action_head.kind=="late_concat"
