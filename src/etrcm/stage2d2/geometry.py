"""Finite state-to-behavior geometry and norm-matched causal interventions."""

from __future__ import annotations
import math,random
import numpy as np
import torch
from torch.nn import functional as F

from etrcm.stage1_1.model import LearnedState
from etrcm.stage2c.world import ACTION,NUISANCE,context_token,tensor_ids,unrelated_token
from etrcm.stage2d.model import Stage2DModel,aggregate_prob,behavioral_metrics,context_state,play,probe_prob
from etrcm.stage2d.world import paired_experiences
from etrcm.stage2d2.protocol import EPS_MULTIPLIERS,RANKS


def contrast(value:torch.Tensor)->torch.Tensor:
    half=value.shape[0]//2
    return value[:half].mean(0)-value[half:].mean(0)


def probe_rows(seed:int,reps:int):
    return paired_experiences(seed+700001,reps,99999,.65,split="novel")


def context_for_probe(model,state,seed,reps):
    return context_state(model,state.clone(),probe_rows(seed,reps))


def probability_from_context(model:Stage2DModel,state:LearnedState)->torch.Tensor:
    batch=state.H.shape[0]; logits=[]; device=state.H.device
    for action in (0,1):
        ids=torch.full((batch,),ACTION[action],dtype=torch.long,device=device)
        branch,_=model.step(state.clone(),context_token(ids))
        logits.append(model.logits(branch,torch.full((batch,),action,dtype=torch.long,device=device)))
    return torch.stack(logits,1).softmax(-1)


def response_vector(prob:torch.Tensor,reps:int)->torch.Tensor:
    agg=aggregate_prob(prob,reps); metrics=behavioral_metrics(prob,reps)
    qdiff=agg[:,0]-agg[:,1]
    entropy=-(agg.clamp_min(1e-9)*agg.clamp_min(1e-9).log()).sum(-1)
    extras=torch.tensor([metrics["interaction_y0"],metrics["behavioral_separation_entropy"]],
                        device=prob.device,dtype=prob.dtype)
    return torch.cat([agg.flatten(),qdiff.flatten(),(entropy[:,0]-entropy[:,1]).flatten(),extras])


def perturb_contrast(state:LearnedState,direction:torch.Tensor,amount:float)->LearnedState:
    out=state.clone(); half=out.H.shape[0]//2
    delta=(amount*direction).view(1,1,-1)
    out.H[:half]=out.H[:half]+delta; out.H[half:]=out.H[half:]-delta
    return out


@torch.no_grad()
def finite_behavioral_subspace(model:Stage2DModel,state:LearnedState,seed:int,reps:int,
                               directions:int=128,epsilons=EPS_MULTIPLIERS,direction_seed:int=0)->dict:
    context=context_for_probe(model,state,seed,reps); hdim=context.H.shape[-1]
    generator=torch.Generator(device=context.H.device).manual_seed(direction_seed)
    d=F.normalize(torch.randn(directions,hdim,generator=generator,device=context.H.device),dim=-1)
    native=float(context.H.norm(dim=-1).median()/math.sqrt(hdim)); native=max(native,1e-3)
    by_eps={}; primary_R=None; primary_J=None
    for multiplier in epsilons:
        amount=native*float(multiplier); rows=[]
        for direction in d:
            plus=response_vector(probability_from_context(model,perturb_contrast(context,direction,amount)),reps)
            minus=response_vector(probability_from_context(model,perturb_contrast(context,direction,-amount)),reps)
            rows.append((plus-minus)/(2*amount))
        R=torch.stack(rows); J=(hdim/directions)*(d.T@R)
        u,s,vh=torch.linalg.svd(J,full_matrices=False); energy=s.square(); cumulative=energy.cumsum(0)/energy.sum().clamp_min(1e-12)
        by_eps[str(multiplier)]={"singular_values":s.cpu().tolist(),"cumulative_energy":cumulative.cpu().tolist(),
                                 "effective_rank":float(torch.exp(-(energy/energy.sum().clamp_min(1e-12)*
                                     (energy/energy.sum().clamp_min(1e-12)).clamp_min(1e-12).log()).sum()))}
        if abs(float(multiplier)-.10)<1e-8: primary_R,primary_J=R,J
    if primary_J is None: primary_J=J; primary_R=R
    u,s,_=torch.linalg.svd(primary_J,full_matrices=False)
    return {"basis":u[:,:max(RANKS)].cpu().tolist(),"native_H_scale":native,"by_epsilon":by_eps,
            "response_matrix":primary_R.cpu().tolist(),"direction_seed":direction_seed}


def basis_tensor(subspace:dict,device="cpu")->torch.Tensor:
    return torch.tensor(subspace["basis"],dtype=torch.float32,device=device)


def project(value:torch.Tensor,basis:torch.Tensor)->torch.Tensor:
    return value@basis@basis.T


def alignment(value:torch.Tensor,basis:torch.Tensor,ranks=RANKS)->dict:
    flat=value.reshape(-1,value.shape[-1]); denom=flat.square().sum().clamp_min(1e-12)
    result={}
    for r in ranks:
        b=basis[:,:r]; result[str(r)]=float(project(flat,b).square().sum()/denom)
    vector=flat.mean(0); u=basis[:,0]
    result["cos_u1"]=float(F.cosine_similarity(vector[None],u[None])) if float(vector.norm())>0 else 0.
    result["norm"]=float(flat.norm())
    return result


def delay_one(model,state,seed,reps,index):
    rng=random.Random(seed*1000003+index*97+31); vals=[rng.choice(NUISANCE) for _ in range(reps)]
    return model.step(state,unrelated_token(tensor_ids(vals+vals,state.H.device)))[0]


@torch.no_grad()
def phase_states(model:Stage2DModel,seed:int,reps:int,p=.70)->dict[str,LearnedState]:
    state=model.initial_state(2*reps,next(model.parameters()).device); result={}
    for index in range(32):
        state,_,_=play(model,state,paired_experiences(seed,reps,index,p))
        if index==3: result["early"]=state.clone()
        if index==15: result["mid"]=state.clone()
    result["post"]=state.clone()
    for index in range(100): state=delay_one(model,state,seed+310001,reps,index)
    result["delay"]=state.clone(); result["probe"]=context_for_probe(model,state,seed+1,reps)
    return result


@torch.no_grad()
def memory_proposals(model:Stage2DModel,state:LearnedState)->dict[str,torch.Tensor]:
    base,_=model.step(state.clone(),None); result={}
    for which in ("F","M","FM"):
        clamped,_=model.step(state.clone(),None,read_clamp=which)
        result[which]=(base.H-clamped.H).squeeze(1)
    return result


@torch.no_grad()
def alignment_audit(model:Stage2DModel,subspace:dict,seed:int,reps:int=8)->dict:
    basis=basis_tensor(subspace,next(model.parameters()).device); phases=phase_states(model,seed,reps); result={}
    for name,state in phases.items():
        hist=contrast(state.H).reshape(1,-1); props=memory_proposals(model,state)
        result[name]={"history":alignment(hist,basis),
                      "memory":{k:alignment(v,basis) for k,v in props.items()},
                      "history_norm":float(hist.norm())}
    return result


def row_norm_match(value:torch.Tensor,reference:torch.Tensor)->torch.Tensor:
    target=reference.norm(dim=-1,keepdim=True); return value/value.norm(dim=-1,keepdim=True).clamp_min(1e-12)*target


def decompose(value:torch.Tensor,basis:torch.Tensor):
    parallel=project(value,basis); return parallel,value-parallel


def rotated_alignment(value:torch.Tensor,basis:torch.Tensor,lam:float)->torch.Tensor:
    parallel,_=decompose(value,basis); target=row_norm_match(parallel,value)
    return row_norm_match((1-lam)*value+lam*target,value)


def state_with_proposal(anchor:LearnedState,proposal:torch.Tensor,reference:LearnedState)->LearnedState:
    # Only H is changed; F/M are bit-exact copies of the frozen reference.
    return LearnedState(anchor.H+proposal[:,None,:],reference.F.clone(),reference.M.clone(),
                        reference.tau,reference.external_time)


def metric_summary(model,state,seed,reps):
    prob=probe_prob(model,state,probe_rows(seed,reps)); m=behavioral_metrics(prob,reps)
    return {"action_TV":sum(m["tv_action"])/2,"I_HA":abs(float(m["interaction_y0"])),
            "BS":float(m["behavioral_separation_entropy"])}


@torch.no_grad()
def frozen_alignment_interventions(model:Stage2DModel,subspace:dict,seed:int,reps:int=16)->dict:
    basis=basis_tensor(subspace,next(model.parameters()).device)[:,:4]; state=phase_states(model,seed,reps)["post"]
    native_next,_=model.step(state.clone(),None); clamp_next,_=model.step(state.clone(),None,read_clamp="M")
    proposal=(native_next.H-clamp_next.H).squeeze(1); parallel,orthogonal=decompose(proposal,basis)
    generator=torch.Generator(device=proposal.device).manual_seed(seed+99017)
    random_control=row_norm_match(torch.randn(proposal.shape,generator=generator,device=proposal.device),proposal)
    variants={"native":proposal,"useful_only":row_norm_match(parallel,proposal),
              "orthogonal_only":row_norm_match(orthogonal,proposal),"random_rotation":random_control}
    for lam in (.25,.5,.75,1.): variants[f"rotate_{lam}"]=rotated_alignment(proposal,basis,lam)
    anchor=LearnedState(clamp_next.H.clone(),state.F.clone(),state.M.clone(),state.tau,state.external_time)
    before_F,before_M=state.F.clone(),state.M.clone(); results={"baseline":metric_summary(model,state,seed,reps)}
    for name,value in variants.items():
        candidate=state_with_proposal(anchor,value,state); results[name]=metric_summary(model,candidate,seed,reps)
        results[name]["norm_ratio"]=float(value.norm()/proposal.norm().clamp_min(1e-12))
    if not torch.equal(state.F,before_F) or not torch.equal(state.M,before_M): raise AssertionError("frozen memory mutated")
    results["proposal_alignment"]=alignment(proposal,basis); return results


@torch.no_grad()
def finite_controllability(model:Stage2DModel,subspace:dict,seed:int,reps:int=4)->dict:
    basis=basis_tensor(subspace,next(model.parameters()).device)[:,:4]; columns={k:[] for k in ("external","F","M","combined")}
    for trial in range(16):
        initial=model.initial_state(2*reps,next(model.parameters()).device)
        row=paired_experiences(seed+trial*101,reps,0,.70)
        updated,_,_=play(model,initial,row); columns["external"].append(contrast(updated.H).flatten())
        formed=phase_states(model,seed+trial*101,reps)["early"]
        base,_=model.step(formed.clone(),None); columns["combined"].append(contrast(base.H-formed.H).flatten())
        props=memory_proposals(model,formed)
        columns["F"].append(contrast(props["F"]).flatten()); columns["M"].append(contrast(props["M"]).flatten())
    result={}
    for name,items in columns.items():
        matrix=torch.stack(items); result[name]=float(project(matrix,basis).square().sum()/matrix.square().sum().clamp_min(1e-12))
    return result
