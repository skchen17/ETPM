"""Frozen failed-state rescue and healthy alignment destruction."""

from __future__ import annotations
import argparse,json
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import torch
from stage2d_evaluate import load_model
from etrcm.stage2d.model import parameter_hash
from etrcm.stage2d2.geometry import frozen_alignment_interventions


def one(audit,checkpoints,eval_seed,reps):
    model,_=load_model(checkpoints/audit["run"]/"checkpoint_1500.pt","cpu"); before=parameter_hash(model)
    sub=audit["trajectory"][-1]["subspace"]; result=frozen_alignment_interventions(model,sub,eval_seed,reps)
    if parameter_hash(model)!=before: raise AssertionError("frozen parameters mutated")
    return {"run":audit["run"],"class":audit["final_class"],"results":result,"parameters_unchanged":True}


def main(args):
    torch.set_num_threads(1); audits=[json.loads(p.read_text()) for p in sorted(args.audit.glob("*.json"))]
    failed=[x for x in audits if x["final_class"]!="healthy"][:8]; healthy=[x for x in audits if x["final_class"]=="healthy"][:8]
    with ThreadPoolExecutor(max_workers=16) as pool:
        fs=[pool.submit(one,x,args.checkpoints,args.eval_seed,args.reps) for x in failed+healthy]; rows=[f.result() for f in as_completed(fs)]
    failed_rows=[x for x in rows if x["class"]!="healthy"]
    names=("useful_only","rotate_0.25","rotate_0.5","rotate_0.75","rotate_1.0")
    rescue_counts={}
    for name in names:
        count=0
        for x in failed_rows:
            r=x["results"]; v=r[name]; native=r["native"]; random=r["random_rotation"]; orth=r["orthogonal_only"]
            if (v["I_HA"]>native["I_HA"] and abs(v["BS"])>abs(native["BS"])
                and v["I_HA"]>random["I_HA"] and abs(v["BS"])>abs(random["BS"])
                and v["I_HA"]>orth["I_HA"] and abs(v["BS"])>abs(orth["BS"])
                and v["norm_ratio"]<=1.25): count+=1
        rescue_counts[name]=count
    best_name,best_count=max(rescue_counts.items(),key=lambda x:x[1])
    healthy_rows=[x for x in rows if x["class"]=="healthy"]; destruction=0
    for x in healthy_rows:
        r=x["results"]; native=r["native"]; orth=r["orthogonal_only"]; useful=r["useful_only"]
        if (native["I_HA"]>orth["I_HA"] and abs(native["BS"])>abs(orth["BS"])
            and useful["I_HA"]>orth["I_HA"] and abs(useful["BS"])>abs(orth["BS"])): destruction+=1
    payload={"selection":{"failed":[x["run"] for x in failed],"healthy":[x["run"] for x in healthy]},"runs":rows,
             "G73":{"counts":rescue_counts,"best":best_name,"count":best_count,"pass":len(failed)==8 and best_count>=6},
             "G74":{"count":destruction,"denominator":len(healthy),"pass":len(healthy)==8 and destruction>=6}}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(payload)); print(json.dumps({"G73":payload["G73"],"G74":payload["G74"]}))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--audit",type=Path,required=True); p.add_argument("--checkpoints",type=Path,required=True)
    p.add_argument("--eval-seed",type=int,required=True); p.add_argument("--reps",type=int,default=16); p.add_argument("--out",type=Path,required=True); main(p.parse_args())
