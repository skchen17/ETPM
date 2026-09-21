"""Frozen-checkpoint routing rescue, necessity, and sufficiency audit."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from etrcm.stage2d.model import parameter_hash
from etrcm.stage2d1.engine import classify, health_audit
from etrcm.stage2d1.protocol import GATE_M_VALUES, ROUTING_SCALES
from stage2d_evaluate import HANDOFF_TIMES, _handoff_trajectory, load_model


def score(h): return (h["action_TV"], h["I_HA"], abs(h["BS"]))


def evaluate_run(run_dir, baseline, args):
    model, _ = load_model(run_dir / "checkpoint_1500.pt", "cpu"); before = parameter_hash(model)
    base = baseline["trajectory"][-1]["health"]; interventions = []
    routes=[]
    for x in ROUTING_SCALES: routes += [(f"alphaF_{x}",{"alpha_f":x}), (f"alphaM_{x}",{"alpha_m":x})]
    routes += [(f"gateM_{x}",{"gate_m":x}) for x in GATE_M_VALUES]
    routes += [("joint_0_0",{"alpha_f":0.,"alpha_m":0.}), ("joint_05_15",{"alpha_f":.5,"alpha_m":1.5}),
               ("joint_15_05",{"alpha_f":1.5,"alpha_m":.5}), ("joint_2_2",{"alpha_f":2.,"alpha_m":2.})]
    for name, route in routes:
        h=health_audit(model,args.eval_seed,reps=args.reps,route=route)
        b,s=score(base),score(h)
        interventions.append({"name":name,"route":route,"health":h,"class":classify(h),
                              "finite":all(map(lambda x: x==x and abs(x)<1e6,s)),
                              "improves_all":all(y>x for x,y in zip(b,s)),
                              "norm_blowup":h["state_norms"]["H"]>10*max(1.,base["state_norms"]["H"])})
    necessity={}
    if baseline["trajectory"][-1]["class"]=="healthy":
        b=_handoff_trajectory(model,args.eval_seed,args.reps)
        for window in HANDOFF_TIMES:
            necessity[window]={}
            for clamp in ("F","M","FM"):
                v=_handoff_trajectory(model,args.eval_seed,args.reps,read_segment=window,clamp=clamp)
                necessity[window][clamp]={"BS":v["behavioral_separation_entropy"],
                                          "BS_reduction":b["behavioral_separation_entropy"]-v["behavioral_separation_entropy"]}
    if parameter_hash(model)!=before: raise AssertionError("routing audit changed frozen parameters")
    return {"run":run_dir.name,"baseline_class":baseline["trajectory"][-1]["class"],
            "baseline":base,"interventions":interventions,"necessity":necessity,
            "parameter_hash_unchanged":True}


def main(args):
    import torch; torch.set_num_threads(1)
    audits={p.stem:json.loads(p.read_text()) for p in args.audit.glob("*.json")}
    failed=[v for v in audits.values() if v["trajectory"][-1]["class"]!="healthy"][:8]
    healthy=[v for v in audits.values() if v["trajectory"][-1]["class"]=="healthy"][:8]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(evaluate_run,args.checkpoints/a["run"],a,args) for a in failed+healthy]
        results=[f.result() for f in as_completed(futures)]
    failed_results=[r for r in results if r["baseline_class"]!="healthy"]
    names=sorted({x["name"] for r in failed_results for x in r["interventions"]})
    rescue_by_intervention={name:sum(next(x for x in r["interventions"] if x["name"]==name)["finite"]
                                             and next(x for x in r["interventions"] if x["name"]==name)["improves_all"]
                                             and not next(x for x in r["interventions"] if x["name"]==name)["norm_blowup"]
                                             for r in failed_results) for name in names}
    best_name,best_count=max(rescue_by_intervention.items(),key=lambda x:x[1],default=(None,0))
    payload={"selection":{"failed":[x["run"] for x in failed],"healthy":[x["run"] for x in healthy]},
             "runs":results,"G68":{"same_intervention_counts":rescue_by_intervention,
                                      "best_intervention":best_name,"rescued_failed":best_count,
                                      "denominator":len(failed),"pass":len(failed)==8 and best_count>=6}}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(payload)); print(args.out)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--checkpoints",type=Path,required=True); p.add_argument("--audit",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True); p.add_argument("--eval-seed",type=int,default=15101); p.add_argument("--reps",type=int,default=16); p.add_argument("--jobs",type=int,default=16)
    main(p.parse_args())
