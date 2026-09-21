"""Statistical analysis of the Stage 2D.1 factorial basin audit."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def cohen_d(a,b):
    a,b=np.asarray(a),np.asarray(b); pooled=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/max(1,len(a)+len(b)-2))
    return float((a.mean()-b.mean())/pooled) if pooled>0 else 0.


def rank_auc(pos,neg):
    return float(np.mean([1 if x>y else .5 if x==y else 0 for x in pos for y in neg]))


def bootstrap_diff(a,b,seed=1771,n=2000):
    rng=np.random.default_rng(seed); a=np.asarray(a); b=np.asarray(b)
    vals=[rng.choice(a,len(a),replace=True).mean()-rng.choice(b,len(b),replace=True).mean() for _ in range(n)]
    return [float(x) for x in np.quantile(vals,[.025,.5,.975])]


def variance_components(rows,key):
    inits=sorted({r["init_seed"] for r in rows}); streams=sorted({r["data_seed"] for r in rows})
    y=np.array([[next(r for r in rows if r["init_seed"]==i and r["data_seed"]==d)["trajectory"][-1]["health"][key] for d in streams] for i in inits])
    grand=y.mean(); ss_i=len(streams)*((y.mean(1)-grand)**2).sum(); ss_d=len(inits)*((y.mean(0)-grand)**2).sum()
    ss_e=((y-y.mean(1,keepdims=True)-y.mean(0,keepdims=True)+grand)**2).sum()
    ms_i=ss_i/max(1,len(inits)-1); ms_d=ss_d/max(1,len(streams)-1); ms_e=ss_e/max(1,(len(inits)-1)*(len(streams)-1))
    vi=max(0.,(ms_i-ms_e)/len(streams)); vd=max(0.,(ms_d-ms_e)/len(inits)); ve=max(0.,ms_e); total=vi+vd+ve
    return {"init":vi,"stream":vd,"interaction_residual":ve,"fractions":{k:v/total if total else 0. for k,v in {"init":vi,"stream":vd,"interaction_residual":ve}.items()}}


def linear_cka(x,y):
    x=x-x.mean(0); y=y-y.mean(0); num=np.linalg.norm(x.T@y,"fro")**2
    den=np.linalg.norm(x.T@x,"fro")*np.linalg.norm(y.T@y,"fro"); return float(num/den) if den else 0.


def trajectory_matrix(row,name):
    return np.asarray([np.asarray(t["health"]["representation"][name]).reshape(-1) for t in row["trajectory"]])


def representation_analysis(rows):
    result={}; healthy=[r for r in rows if r["trajectory"][-1]["class"]=="healthy"]
    failed=[r for r in rows if r["trajectory"][-1]["class"]!="healthy"]
    for name in ("H","F","M"):
        within=[linear_cka(trajectory_matrix(a,name),trajectory_matrix(b,name)) for a,b in combinations(healthy,2)]
        cross=[linear_cka(trajectory_matrix(a,name),trajectory_matrix(b,name)) for a in healthy for b in failed]
        angle=None
        if healthy and failed:
            x=np.stack([trajectory_matrix(r,name)[-1] for r in healthy]); y=np.stack([trajectory_matrix(r,name)[-1] for r in failed])
            _,_,vx=np.linalg.svd(x-x.mean(0),full_matrices=False); _,_,vy=np.linalg.svd(y-y.mean(0),full_matrices=False)
            k=min(3,len(vx),len(vy)); s=np.linalg.svd(vx[:k]@vy[:k].T,compute_uv=False).clip(0,1)
            angle=float(np.degrees(np.arccos(s.min())))
        result[name]={"within_healthy_CKA_mean":float(np.mean(within)) if within else None,
                      "healthy_failed_CKA_mean":float(np.mean(cross)) if cross else None,
                      "max_principal_angle_deg_k3":angle}
    return result


def main(args):
    rows=[json.loads(p.read_text()) for p in sorted(args.audit.glob("*.json"))]
    healthy=[r for r in rows if r["trajectory"][-1]["class"]=="healthy"]
    failed=[r for r in rows if r["trajectory"][-1]["class"]!="healthy"]
    def value(t,key):
        h=t["health"]
        if key in h: return h[key]
        if key=="M_probe": return h["z_probe"]["M"]
        if key=="H_probe": return h["z_probe"]["H"]
        if key=="gate_M": return h["routing"]["early:abstract"]["gate_M"]["mean"]
        if key=="q_M_norm": return h["routing"]["early:abstract"]["latent_difference_q_M_norm"]["mean"]
        if key=="r_M_difference": return h["routing"]["early:abstract"]["latent_difference_r_M_norm_conditioned"]["mean"]
        if key=="transfer_amount": return h["routing"]["early:outcome"]["transfer_norm"]["sum"]
        if key=="H_norm": return h["state_norms"]["H"]
        raise KeyError(key)
    metrics=("action_TV","history_TV","I_HA","BS","observed_CE","conditional_CE","CFA",
             "M_probe","H_probe","gate_M","q_M_norm","r_M_difference","transfer_amount","H_norm")
    divergence=[]
    for idx in range(len(rows[0]["trajectory"])):
        item={"step":rows[0]["trajectory"][idx]["step"],"metrics":{}}
        for key in metrics:
            a=[value(r["trajectory"][idx],key) for r in healthy]; b=[value(r["trajectory"][idx],key) for r in failed]
            item["metrics"][key]={"healthy_mean":float(np.mean(a)) if a else None,"failed_mean":float(np.mean(b)) if b else None,
                                  "cohen_d":cohen_d(a,b) if len(a)>1 and len(b)>1 else None,
                                  "rank_auc":rank_auc(a,b) if a and b else None,
                                  "bootstrap_diff_95":bootstrap_diff(a,b) if a and b else None}
        divergence.append(item)
    first_by_metric={}
    for key in metrics:
        first_by_metric[key]=next((item["step"] for item in divergence if item["metrics"][key]["cohen_d"] is not None
                                   and abs(item["metrics"][key]["cohen_d"])>=.5
                                   and item["metrics"][key]["bootstrap_diff_95"][0]*item["metrics"][key]["bootstrap_diff_95"][2]>0),None)
    earliest=min((x for x in first_by_metric.values() if x is not None),default=None)
    outcome_counts={c:sum(r["trajectory"][-1]["class"]==c for r in rows) for c in ("healthy","partial","shortcut")}
    payload={"design":{"runs":len(rows),"initializations":len({r['init_seed'] for r in rows}),"streams":len({r['data_seed'] for r in rows})},
             "outcome_counts":outcome_counts,"earliest_reliable_divergence_step":earliest,
             "earliest_by_metric":first_by_metric,"trajectory_statistics":divergence,
             "variance_components":{k:variance_components(rows,k) for k in ("action_TV","I_HA","BS","CFA")},
             "representation_geometry":representation_analysis(rows),
             "healthy_runs":[r["run"] for r in healthy],"failed_runs":[r["run"] for r in failed]}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(payload,indent=2)); print(args.out)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--audit",type=Path,required=True); p.add_argument("--out",type=Path,required=True); main(p.parse_args())
