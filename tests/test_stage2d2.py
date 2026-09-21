from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
import torch
from etrcm.stage2d.model import Stage2DModel,parameter_hash
from etrcm.stage2d2.geometry import (alignment,decompose,perturb_contrast,project,
    rotated_alignment,row_norm_match)
from etrcm.stage2d2.warmup import auxiliary_weight


def basis(h=32,r=4):
    q,_=torch.linalg.qr(torch.randn(h,r)); return q


def test_finite_h_perturbation_exact():
    m=Stage2DModel(gamma=.5); s=m.initial_state(4,"cpu"); d=torch.zeros(32); d[3]=1
    p=perturb_contrast(s,d,.25); delta=p.H-s.H
    assert torch.equal(delta[:2,:,3],torch.full((2,1),.25)); assert torch.equal(delta[2:,:,3],torch.full((2,1),-.25))


def test_norm_matched_perturbations():
    x=torch.randn(7,32); y=torch.randn(7,32); z=row_norm_match(x,y); assert torch.allclose(z.norm(dim=-1),y.norm(dim=-1),atol=1e-6)


def test_projection_idempotence():
    b=basis(); x=torch.randn(8,32); assert torch.allclose(project(project(x,b),b),project(x,b),atol=1e-5)


def test_orthogonal_decomposition_reconstructs():
    b=basis(); x=torch.randn(8,32); p,o=decompose(x,b); assert torch.allclose(p+o,x,atol=1e-6); assert float((o@b).abs().max())<1e-5


def test_random_control_rank_matched():
    a=basis(r=4); b=basis(r=4); assert a.shape==b.shape and torch.linalg.matrix_rank(a)==torch.linalg.matrix_rank(b)==4


def test_rotation_is_norm_matched():
    b=basis(); x=torch.randn(8,32); y=rotated_alignment(x,b,.75); assert torch.allclose(x.norm(dim=-1),y.norm(dim=-1),atol=1e-5)


def test_no_fm_mutation_by_geometry_operations():
    m=Stage2DModel(gamma=.5); s=m.initial_state(4,"cpu"); f=s.F.clone(); mm=s.M.clone(); perturb_contrast(s,torch.randn(32),.1); assert torch.equal(s.F,f) and torch.equal(s.M,mm)


def test_no_parameter_mutation_by_geometry_operations():
    m=Stage2DModel(gamma=.5); h=parameter_hash(m); alignment(torch.randn(4,32),basis()); assert parameter_hash(m)==h


def test_no_latent_correct_action_or_memory_target_in_warmup_source():
    text=Path("src/etrcm/stage2d2/warmup.py").read_text(); assert "latent" not in text and "correct_action" not in text and "memory_target" not in text


def test_evaluator_is_not_part_of_auxiliary_loss():
    text=Path("src/etrcm/stage2d2/warmup.py").read_text(); assert "action_head" not in text and "evaluator" not in text


def test_auxiliary_exactly_zero_after_warmup():
    assert auxiliary_weight(49,50,.1)==.1 and auxiliary_weight(50,50,.1)==0 and auxiliary_weight(1500,50,.1)==0


def test_at_least_80_percent_endogenous():
    assert sum(auxiliary_weight(s,200,.1)==0 for s in range(1500))/1500>=.8


def test_formal_evaluation_has_no_auxiliary_path():
    text=Path("experiments/stage2d2_audit.py").read_text(); assert "alignment_loss" not in text and "auxiliary_weight" not in text


def test_historical_assets_unchanged_if_manifest_present():
    p=Path("results/stage2d2/manifests/prior_manifest.json")
    if not p.exists(): return
    expected=json.loads(p.read_text()); files=subprocess.check_output(["git","ls-files"],text=True).splitlines()
    current={x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in files if any(k in x.lower() for k in ("stage2c","stage2d","stage2d1")) and "stage2d2" not in x.lower()}
    assert current==expected
