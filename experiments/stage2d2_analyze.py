"""Development-frozen alignment/controllability analysis and confirmation."""

from __future__ import annotations
import argparse,json,statistics
from pathlib import Path


def auc(pos,neg): return sum(1 if a>b else .5 if a==b else 0 for a in pos for b in neg)/max(1,len(pos)*len(neg))
def signed_auc(pos,neg):
    raw=auc(pos,neg); return (raw,1) if raw>=.5 else (1-raw,-1)


def extract(rows,step,path):
    out=[]
    for r in rows:
        t=next(x for x in r["trajectory"] if x["step"]==step); value=t
        for key in path: value=value[key]
        out.append((r,value))
    return out


def group_summary(rows,step,path):
    values=extract(rows,step,path); h=[v for r,v in values if r["final_class"]=="healthy"]; f=[v for r,v in values if r["final_class"]!="healthy"]
    a,d=signed_auc(h,f); return {"healthy_mean":statistics.mean(h) if h else None,"failed_mean":statistics.mean(f) if f else None,
                                 "AUROC":a,"direction":d,"healthy_n":len(h),"failed_n":len(f)}


def main(args):
    old=[json.loads(p.read_text()) for p in sorted(args.cohort_a.glob("*.json"))]; new=[json.loads(p.read_text()) for p in sorted(args.cohort_b.glob("*.json"))]
    candidates=[]
    for step in (0,25):
        for name in ("external","F","M","combined"):
            path=("controllability",name); s=group_summary(old,step,path); candidates.append({"step":step,"feature":name,**s})
    selected=max(candidates,key=lambda x:x["AUROC"]); step=selected["step"]; feature=selected["feature"]; direction=selected["direction"]
    pairs=extract(new,step,("controllability",feature)); h=[direction*v for r,v in pairs if r["final_class"]=="healthy"]; f=[direction*v for r,v in pairs if r["final_class"]!="healthy"]
    confirm_auc=auc(h,f); by_stream={}
    for stream in sorted({r["data_seed"] for r in new}):
        subset=[(r,v) for r,v in pairs if r["data_seed"]==stream]; hp=[direction*v for r,v in subset if r["final_class"]=="healthy"]; fp=[direction*v for r,v in subset if r["final_class"]!="healthy"]
        by_stream[str(stream)]=auc(hp,fp) if hp and fp else None
    stream_pass=sum(v is not None and v>=.70 for v in by_stream.values())
    align={}; align_b={}
    for step0 in (0,25,50,100,200,300,500,750,1000,1500):
        align[str(step0)]={}
        for key,path in {
            "A_hist":("alignment","post","history","4"),"A_F":("alignment","post","memory","F","4"),
            "A_M":("alignment","post","memory","M","4"),"A_FM":("alignment","post","memory","FM","4")}.items():
            align[str(step0)][key]=group_summary(old,step0,path)
            align_b.setdefault(str(step0),{})[key]=group_summary(new,step0,path)
    ranks={}
    for step0 in (0,100,500,1500):
        vals=extract(old,step0,("subspace","by_epsilon","0.1","cumulative_energy")); ranks[str(step0)]={str(r):statistics.mean(v[r-1] for _,v in vals) for r in (1,2,4,8)}
    payload={"cohort_A":{"runs":len(old),"healthy":sum(r['final_class']=='healthy' for r in old)},
             "cohort_B":{"runs":len(new),"healthy":sum(r['final_class']=='healthy' for r in new)},
             "development_candidates":candidates,"frozen_predictor":selected,
             "confirmatory":{"AUROC":confirm_auc,"by_stream":by_stream,"streams_passing":stream_pass},
             "G75":{"pass":confirm_auc>=.75 and stream_pass>=2},"alignment_trajectory":align,
             "confirmatory_alignment_trajectory":align_b,"rank_energy":ranks}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(payload,indent=2)); print(json.dumps({"selected":selected,"confirmatory":payload["confirmatory"],"G75":payload["G75"],"cohort_B":payload["cohort_B"]}))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--cohort-a",type=Path,required=True); p.add_argument("--cohort-b",type=Path,required=True); p.add_argument("--out",type=Path,required=True); main(p.parse_args())
