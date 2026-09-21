from __future__ import annotations

import json
from pathlib import Path

import torch

from etrcm.stage1_1.model import LearnedState
from etrcm.stage2d.model import Stage2DModel, parameter_hash
from etrcm.stage2d.world import paired_experiences, training_experiences
from etrcm.stage2d1.engine import routed_step
from etrcm.stage2d1.protocol import evidence_schedule, gate_floor


def model(seed=3):
    torch.manual_seed(seed); return Stage2DModel(variant="full",gamma=.5,rho_fast=.97,rho_slow=.9995)


def nonzero_state(m,batch=4):
    s=m.initial_state(batch,"cpu")
    return LearnedState(torch.randn_like(s.H),torch.randn_like(s.F),torch.randn_like(s.M),s.tau,s.external_time)


def test_exact_baseline_transition_reproduction():
    m=model(); s=nonzero_state(m); a,_=m.step(s.clone(),None); b,_=routed_step(m,s.clone(),None)
    for name in ("H","F","M"): assert torch.equal(getattr(a,name),getattr(b,name))


def test_initialization_fixed_across_streams():
    assert parameter_hash(model(11))==parameter_hash(model(11))


def test_stream_fixed_across_initializations():
    a=training_experiences(222,8,4,.7); b=training_experiences(222,8,4,.7); assert a==b


def test_normalized_gate_override():
    m=model(); s=nonzero_state(m); _,t=routed_step(m,s,None,gate_m=.75)
    assert torch.allclose(t["gates"].sum(-1),torch.ones(4)); assert torch.allclose(t["gates"][:,1],torch.full((4,),.75))


def test_read_scaling_does_not_mutate_input_memory():
    m=model(); s=nonzero_state(m); f=s.F.clone(); mm=s.M.clone(); routed_step(m,s,None,alpha_f=0.,alpha_m=2.)
    assert torch.equal(s.F,f) and torch.equal(s.M,mm)


def test_frozen_parameter_intervention():
    m=model(); before=parameter_hash(m); routed_step(m,nonzero_state(m),None,gate_m=.2); assert parameter_hash(m)==before


def test_warmup_removed():
    assert gate_floor("C2",99,floor=.65,window=100)==.65 and gate_floor("C2",100,floor=.65,window=100) is None
    assert gate_floor("C2",1499,floor=.65,window=100) is None


def test_curriculum_schedule_exact():
    assert [evidence_schedule("C1",x) for x in (0,299,300,599,600,899,900,1199,1200,1499)]==[.9,.9,.8,.8,.75,.75,.7,.7,.65,.65]
    assert evidence_schedule("C3",299)==.9 and evidence_schedule("C3",300)==.7


def test_no_privileged_lifetime_fields_in_trainer():
    text=Path("experiments/stage2d1_train.py").read_text()
    assert ".latent" not in text and "correct_action" not in text and ".support_bit" not in text


def test_matched_noise_is_paired_and_latent_independent():
    rows=paired_experiences(9,8,2,.7,matched_noise=True)
    assert [x.support_bit for x in rows[:8]]==[x.support_bit for x in rows[8:]]


def test_eval_intervention_leaves_model_frozen():
    m=model(); h=parameter_hash(m)
    for gm in (0.,.5,1.): routed_step(m,nonzero_state(m),None,gate_m=gm)
    assert parameter_hash(m)==h


def test_gate_floor_is_normalized_and_removed_at_final():
    m=model(); _,t=routed_step(m,nonzero_state(m),None,gate_floor_m=.65)
    assert bool((t["gates"][:,1]>=.65).all()) and torch.allclose(t["gates"].sum(-1),torch.ones(4))
    assert gate_floor("C4",1499,window=200) is None


def test_null_has_no_external_write_under_override():
    m=model(); _,t=routed_step(m,nonzero_state(m),None,gate_m=.6); assert not bool(t["external_write_flag"].any())


def test_prior_stage_assets_unchanged_if_manifest_present():
    manifest=Path("results/stage2d1/manifests/prior_manifest.json")
    if not manifest.exists(): return
    import hashlib, subprocess
    expected=json.loads(manifest.read_text()); files=subprocess.check_output(["git","ls-files"],text=True).splitlines()
    current={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files if ("stage2c" in p.lower() or "stage2d" in p.lower()) and "stage2d1" not in p.lower()}
    assert current==expected
