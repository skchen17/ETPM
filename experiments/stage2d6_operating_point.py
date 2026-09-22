"""Frozen common-preactivation-shift tests with matched negative controls."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from etrcm.stage2d.model import head_hash, parameter_hash
from etrcm.stage2d4.model import Stage2D4Model
from etrcm.stage2d5.flow import candidate_flow, cell_means, factorial_injection
from etrcm.stage2d3.interaction import factorial_components
from etrcm.stage2d6.fusion import (behavior, low_curvature_offset, offset_metrics,
                                   pre_override, scaled_main_override, top_k)
from stage2d4_audit import formed_state
from stage2d5_audit import context, restoration


GRID=(-1.,-.5,-.25,0.,.25,.5,1.)


def load(checkpoint, arm):
    ckpt=torch.load(checkpoint,map_location="cpu",weights_only=False)
    model=Stage2D4Model(arm);model.load_state_dict(ckpt["model"]);model.eval()
    return model,ckpt


@torch.no_grad()
def one(audit_path, out, *, mode, offset_scale=0.5, k=8, reps=16,
        healthy_norms=None):
    torch.set_num_threads(1)
    source=json.loads(audit_path.read_text());checkpoint=Path(source["checkpoint"])
    model,ckpt=load(checkpoint,source["arm"]);seed=ckpt["eval_seed"]
    before_params=parameter_hash(model);before_head=head_hash(model)
    state=formed_state(model,seed,reps)
    before_state=(state.H.clone(),state.F.clone(),state.M.clone(),state.tau,state.external_time)
    ctx=context(model,state,seed,reps)
    rows=[candidate_flow(model,ctx,torch.full((2*reps,),a,dtype=torch.long)) for a in (0,1)]
    fusion=source["endpoint"]["fusion"]
    units=top_k(fusion,k)
    sd=fusion["pre_sd"]
    result={"run":source["run"],"cohort":source["cohort"],"class":source["endpoint"]["class"],
            "arm":source["arm"],"checkpoint":source["checkpoint"],"mode":mode,
            "selected_units":units,"k":k,"native":fusion["behavior"],
            "native_post_I":fusion["post_I"],"native_pre_I":fusion["pre_I"]}
    def run_scalar(scale):
        delta=torch.zeros(rows[0]["fusion_pre"].shape[-1])
        delta[units]=float(scale)*sd
        return offset_metrics(model,ctx,rows,reps,delta)
    if mode=="grid":
        result["offset_grid"]={str(scale):run_scalar(scale) for scale in GRID}
    else:
        result["selected_offset_scale"]=offset_scale
        result["common_offset"]=run_scalar(offset_scale)
        base=torch.zeros(rows[0]["fusion_pre"].shape[-1]);base[units]=offset_scale*sd
        generator=random.Random(seed+601)
        random_delta=base.clone()
        for j in units: random_delta[j]*=generator.choice((-1.,1.))
        result["random_offset"]=offset_metrics(model,ctx,rows,reps,random_delta)
        low=low_curvature_offset(rows,units,cap_sd=abs(offset_scale) if offset_scale else 1.)
        result["low_curvature_offset"]=offset_metrics(model,ctx,rows,reps,low)
        result["state_scale"]=behavior(scaled_main_override(model,ctx,rows,reps,"state"),reps)
        result["action_scale"]=behavior(scaled_main_override(model,ctx,rows,reps,"action"),reps)
        if result["class"]=="healthy":
            destroy=low_curvature_offset(rows,units,cap_sd=3.)
            result["healthy_destruction"]=offset_metrics(model,ctx,rows,reps,destroy)
            matched=torch.zeros_like(destroy)
            for j in units: matched[j]=3.*sd*generator.choice((-1.,1.))
            result["healthy_matched_shift"]=offset_metrics(model,ctx,rows,reps,matched)
        elif healthy_norms is not None:
            ref=healthy_norms[source["arm"]]
            result["post_positive_control"]=restoration(model,ctx,reps,"fusion_post",
                        ref["fusion_post"],ref["output_template"],seed)
    result["integrity"]={"parameters_unchanged":parameter_hash(model)==before_params,
            "protected_head_unchanged":head_hash(model)==before_head,
            "persistent_state_unchanged":all(torch.equal(a,b) for a,b in zip(before_state[:3],
                (state.H,state.F,state.M))) and before_state[3:]==(state.tau,state.external_time)}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result))
    print(json.dumps({"run":result["run"],"class":result["class"],"mode":mode}),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--mode",choices=("grid","formal"),required=True)
    p.add_argument("--offset-scale",type=float,default=.5)
    p.add_argument("--k",type=int,default=8)
    p.add_argument("--reps",type=int,default=16)
    p.add_argument("--healthy-norms",type=Path)
    a=p.parse_args()
    norms=json.loads(a.healthy_norms.read_text()) if a.healthy_norms else None
    one(a.audit,a.out,mode=a.mode,offset_scale=a.offset_scale,k=a.k,reps=a.reps,
        healthy_norms=norms)
