"""Continuous-stream M-gate audit at 1k/5k/10k ticks."""

from __future__ import annotations
import argparse,json
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import torch
from etrcm.stage1_3.events import self_output_event
from etrcm.stage2c.world import tensor_ids
from stage2d_evaluate import advance_experience,delay_one,load_model,measure


@torch.no_grad()
def one(run,seed):
    model,_=load_model(run/"checkpoint_1500.pt","cpu"); reps=2; state=model.initial_state(4,"cpu"); result={}; nonfinite=None
    for tick in range(1,10001):
        kind=tick%20
        if kind<5: state,_=advance_experience(model,state,seed+700001,reps,tick,.65)
        elif kind<10: state,_=advance_experience(model,state,seed+700001,reps,tick,.65,noise=True)
        elif kind<14: state,_=model.step(state,None)
        elif kind<18: state,_=delay_one(model,state,seed+700003,reps,tick)
        elif kind==18: measure(model,state,seed+tick,reps)
        else: state,_=model.step(state,self_output_event(torch.full((4,),tick%24,dtype=torch.long)))
        if not all(bool(torch.isfinite(getattr(state,k)).all()) for k in ("H","F","M")): nonfinite=tick; break
        if tick in (1000,5000,10000):
            _,trace=model.step(state.clone(),None); b=measure(model,state,seed+tick,reps)
            result[str(tick)]={"gate_F":float(trace["gates"][:,0].mean()),"gate_M":float(trace["gates"][:,1].mean()),
                               "H":float(state.H.norm(dim=(-2,-1)).mean()),"F":float(state.F.norm(dim=(-2,-1)).mean()),
                               "M":float(state.M.norm(dim=(-2,-1)).mean()),"BS":b["behavioral_separation_entropy"]}
    return {"run":run.name,"first_nonfinite":nonfinite,"completed":tick,"milestones":result}


def main(args):
    runs=sorted(p for p in args.checkpoints.iterdir() if p.is_dir())
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(one,r,70000+i) for i,r in enumerate(runs)]; rows=[f.result() for f in as_completed(fs)]
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps({"runs":rows},indent=2)); print(args.out)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--checkpoints",type=Path,required=True); p.add_argument("--out",type=Path,required=True); main(p.parse_args())
