"""Non-gating per-tick H/F/M and read/write traces for primary-model interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from etrcm.stage2c.rollout import play_experience, probe_policy, unrelated_delay
from etrcm.stage2c.world import paired_histories, surface
from experiments.stage2c_eval import load_model, common_surfaces, paired


def run(checkpoint: Path, out: Path, *, device="cpu", pairs=4):
    model,payload=load_model(checkpoint,device)
    assert payload["variant"]=="full"
    seed=payload["seed"]
    initial=model.initial_state(2*pairs,device=device)
    before=model.parameter_digest()
    histories=[paired(seed+1009*i,32) for i in range(pairs)]
    cases=[("none","none","none")]
    for window in ("W1","W2","W3","W4"):
        for clamp in ("F","M","FM"):
            cases.append((window,clamp,"read"))
    for window in ("W1","W2"):
        cases.append((window,"none","write"))
    scalars=[];tensors=[];summaries=[]
    for window,clamp,kind in cases:
        case=f"{kind}_{window}_{clamp}"
        state=initial.clone()
        local=[]
        def record(event_type,current,diag,forecast_logits=None,future_loss=None):
            nonlocal scalars,tensors
            entry={"case":case,"seed":seed,"tau":current.tau,
                   "external_time":current.external_time,"event_type":event_type,
                   "latent_environment":[0,1]*pairs,
                   "intervention_flags":{"kind":kind,"window":window,"clamp":clamp},
                   "H_norm":current.H.norm(dim=(-2,-1)).detach().cpu().tolist(),
                   "F_norm":current.F.norm(dim=(-2,-1)).detach().cpu().tolist(),
                   "M_norm":current.M.norm(dim=(-2,-1)).detach().cpu().tolist(),
                   "H_delta_norm":diag["H_delta_norm"].detach().cpu().tolist(),
                   "q_F":diag["q_F"].detach().cpu().tolist(),
                   "q_M":diag["q_M"].detach().cpu().tolist(),
                   "r_F":diag["r_F"].detach().cpu().tolist(),
                   "r_M":diag["r_M"].detach().cpu().tolist(),
                   "transfer_norm":diag["transfer"].norm(dim=(-2,-1)).detach().cpu().tolist(),
                   "write_norm":diag["external_update"].norm(dim=(-2,-1)).detach().cpu().tolist(),
                   "behavior_logits":None,
                   "forecast_logits":forecast_logits.detach().cpu().tolist() if forecast_logits is not None else None,
                   "future_loss":future_loss.detach().cpu().tolist() if future_loss is not None else None}
            scalars.append(entry)
            tensors.append({"case":case,"tau":current.tau,"event_type":event_type,
                            "H":current.H.detach().cpu(),"F":current.F.detach().cpu(),
                            "M":current.M.detach().cpu()})
            local.append(entry)
        for t in range(16):
            items=[histories[i][2*t+j] for i in range(pairs) for j in range(2)]
            active=window==("W1" if t<8 else "W2")
            state,_,_,_=play_experience(model,state,items,
                                        read_clamp=clamp if kind=="read" and active else "none",
                                        write_block=kind=="write" and active,
                                        trace=record)
        for tick in range(4):
            state,diag=model.step(state,None,
                                  read_clamp=clamp if kind=="read" and window=="W2" else "none")
            record("NULL",state,diag)
        def delay_callback(t,current,diag,phase):
            record(phase,current,diag)
        state=unrelated_delay(model,state,50,seed+1701,
                              read_clamp=clamp if kind=="read" and window=="W3" else "none",
                              trace=delay_callback)
        surfaces=common_surfaces(seed,pairs,"novel")
        probs,action_logits,forecasts,probe_state,action_states,diags=probe_policy(
            model,state,surfaces,read_clamp=clamp if kind=="read" and window=="W4" else "none")
        record("probe_context_end",probe_state,diags[3])
        for action in range(2):
            record(f"probe_action_{action}",action_states[action],diags[4+action],
                   forecast_logits=forecasts[:,action,:])
        scalars[-1]["behavior_logits"]=action_logits.detach().cpu().tolist()
        summaries.append({"case":case,"events_logged":len(local),
                          "BS":float((probs[:,0].reshape(-1,2)[:,0]-
                                      probs[:,0].reshape(-1,2)[:,1]).mean()),
                          "first_H_over_100":next((x["tau"] for x in local
                                                   if max(x["H_norm"])>100),None)})
    if model.parameter_digest()!=before:raise AssertionError("parameter changed")
    formal_rows=out.parents[2]/"raw"/"full"/str(seed)/"rows.jsonl"
    if formal_rows.exists():
        window_rows=[json.loads(x) for x in formal_rows.read_text().splitlines()
                     if '"section": "window"' in x]
        for item in summaries:
            name=item["case"]
            expected=("none" if name=="none_none_none" else
                      name+"/probe_read_"+name.split("_")[-1] if name.startswith("read_W4")
                      else name)
            matches=[r for r in window_rows if r["intervention"]==expected]
            if len(matches)!=1 or abs(matches[0]["BS"]-item["BS"])>1e-6:
                raise AssertionError(f"telemetry replay mismatch {seed} {name}")
    out.mkdir(parents=True,exist_ok=True)
    (out/"windows.jsonl").write_text("".join(json.dumps(x)+"\n" for x in scalars))
    torch.save(tensors,out/"window_states.pt")
    (out/"summary.json").write_text(json.dumps({"seed":seed,"parameter_hash_before":before,
                                                 "parameter_hash_after":model.parameter_digest(),
                                                 "cases":summaries},indent=2))
    return {"seed":seed,"cases":len(cases),"ticks":len(scalars)}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--device",default="cpu")
    args=p.parse_args()
    root=Path("results/stage2c")
    torch.set_num_threads(2)
    with torch.no_grad():
        result=run(root/"checkpoints"/"full"/str(args.seed)/"checkpoint.pt",
                   root/"trajectories"/"full"/str(args.seed),device=args.device)
    print(json.dumps(result))
