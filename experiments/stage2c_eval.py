"""Frozen-lifetime Stage 2C evaluation. One row per paired lifetime/condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as TF

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_4.model import Stage14Config
from etrcm.stage2c.model import BehavioralModel
from etrcm.stage2c.rollout import play_experience, probe_policy, unrelated_delay
from etrcm.stage2c.world import Experience, consequence, paired_histories, surface


def load_model(path: Path, device: str, *, posthoc_gamma_zero=False):
    payload = torch.load(path, map_location=device, weights_only=False)
    variant="gamma_zero" if posthoc_gamma_zero else payload["variant"]
    model = BehavioralModel(Stage14Config(**payload["config"]), variant).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    if posthoc_gamma_zero:
        payload=dict(payload)
        payload["variant"]="gamma_zero_posthoc"
    return model, payload


def frozen_snapshot(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.named_parameters()}


def assert_frozen(model, snapshot):
    for name, tensor in model.named_parameters():
        if not torch.equal(tensor.detach().cpu(), snapshot[name]):
            raise AssertionError(f"parameter changed in lifetime: {name}")


def paired(seed: int, n: int, *, noise=False):
    a, b = paired_histories(seed, n, noise=noise)
    return [item for pair in zip(a, b) for item in pair]


def common_surfaces(seed: int, pairs: int, split: str):
    rng = random.Random(seed * 823 + 701)
    return [item for i in range(pairs) for item in (surface(rng, split),) * 2]


def swap(state: LearnedState, fields: str):
    def changed(name, tensor):
        if name not in fields:
            return tensor.clone()
        shape = tensor.shape
        return tensor.reshape(-1, 2, *shape[1:]).flip(1).reshape(shape)
    return LearnedState(changed("H",state.H),changed("F",state.F),changed("M",state.M),
                        state.tau,state.external_time)


def mutate(state: LearnedState, intervention: str, initial: LearnedState):
    out=state.clone()
    if intervention.startswith("H_reset"):
        out.H=initial.H.clone()
    if intervention in {"H_reset_F_zero","H_reset_FM_zero"}:
        out.F=torch.zeros_like(out.F)
    if intervention in {"H_reset_M_zero","H_reset_FM_zero"}:
        out.M=torch.zeros_like(out.M)
    if intervention in {"H_swap","F_swap","M_swap","FM_swap","HFM_swap"}:
        key=intervention.split("_")[0]
        out=swap(out,key)
    if intervention in {"H_reset_F_swap","H_reset_M_swap"}:
        out=swap(out,intervention.split("_")[2])
    return out


def js(p, q):
    m=(p+q)/2
    return 0.5*((p*(p.clamp_min(1e-12)/m.clamp_min(1e-12)).log()).sum(-1)
                +(q*(q.clamp_min(1e-12)/m.clamp_min(1e-12)).log()).sum(-1))


def metrics(state, diag=None):
    ret={"H_norm":float(state.H.norm(dim=(-2,-1)).mean()),
         "F_norm":float(state.F.norm(dim=(-2,-1)).mean()),
         "M_norm":float(state.M.norm(dim=(-2,-1)).mean()),
         "tau":int(state.tau),"external_time":int(state.external_time)}
    for source,name in (("r_F_norm","r_F_norm"),("r_M_norm","r_M_norm"),
                        ("H_delta_norm","H_delta_norm")):
        ret[name]=float(diag[source].mean()) if diag is not None and source in diag else None
    return ret


def behavior(model, state, surfaces, *, clamp="none"):
    p,action_logits,forecast,_,_,diags=probe_policy(model,state,surfaces,read_clamp=clamp)
    p=p.detach(); fp=forecast.softmax(-1).detach()
    return p,fp,diags[-1]


def row(model, payload, state, surfaces, *, section, seed, lifetime_id, N, D=0,
        split="novel", intervention="none", noise=False, revision=0,
        before_hash=None, initial=None, clamp="none", diagnostics=None):
    test=mutate(state,intervention,initial) if intervention!="none" else state
    p,forecasts,diag=behavior(model,test,surfaces,clamp=clamp)
    if not torch.isfinite(p).all():
        raise RuntimeError("nonfinite action probabilities")
    pa=p[:,0].reshape(-1,2)
    bs=float((pa[:,0]-pa[:,1]).mean())
    jsd=float(js(p.reshape(-1,2,2)[:,0,:],p.reshape(-1,2,2)[:,1,:]).mean())
    rng=random.Random(seed*1000003+N*1009+D*17+revision)
    chosen=p.argmax(-1).cpu().tolist()
    realized=[]
    for i,action in enumerate(chosen):
        latent=(i%2) if revision==0 else (1-i%2)
        observed=consequence(rng,latent,action,noise=noise)-4
        realized.append(-math.log(max(float(forecasts[i,action,observed]),1e-12)))
    m=metrics(test,diagnostics or diag)
    return {"run_id":f"{payload['variant']}-{seed}","seed":seed,
            "lifetime_id":lifetime_id,"section":section,
            "history_condition":"noise" if noise else "reversed_A_B" if revision>0 else "paired_A_B",
            "latent_environment":"uninformative_noise" if noise else "paired_1_0" if revision>0 else "paired_0_1",
            "exposure_count":N,"delay":D,
            "probe_type":split,"OOD":split!="seen","revision_count":revision,
            "useful_noise_condition":"noise" if noise else "useful",
            "intervention":intervention if clamp=="none" else f"{intervention}/probe_read_{clamp}",
            "action_probabilities":p.cpu().tolist(),"selected_action":chosen,
            "forecast_probabilities":forecasts.cpu().tolist(),
            "future_loss":sum(realized)/len(realized),
            "BS":bs,"JS":jsd,"PR":None,"GR":None,"RI":None,
            "parameter_hash_before_lifetime":before_hash,
            "parameter_hash_after_lifetime":model.parameter_digest(),**m}


def trace_item(clock, state, diag, phase, *, forecast_logits=None, future_loss=None):
    result={"tau":int(state.tau),"phase":phase,**metrics(state,diag),
            "event_type":phase,"q_F":diag["q_F"].detach().cpu().tolist(),
            "q_M":diag["q_M"].detach().cpu().tolist(),
            "r_F":diag["r_F"].detach().cpu().tolist(),
            "r_M":diag["r_M"].detach().cpu().tolist(),
            "transfer_norm":float(diag["transfer"].norm(dim=(-2,-1)).mean()),
            "write_norm":float(diag["external_update"].norm(dim=(-2,-1)).mean()),
            "intervention_flags":"none","latent_environment":[0,1]*(state.H.shape[0]//2),
            "behavior_logits":None,
            "forecast_logits":forecast_logits.detach().cpu().tolist() if forecast_logits is not None else None,
            "future_loss":future_loss.detach().cpu().tolist() if future_loss is not None else None}
    return result


def evaluate(model, payload, out: Path, *, pairs=4, long=True):
    seed=payload["seed"];device=next(model.parameters()).device
    initial=model.initial_state(2*pairs,device=device)
    frozen=frozen_snapshot(model); before=model.parameter_digest()
    rows=[]; traces=[]; trajectory=[]
    def add(state, **kw):
        r=row(model,payload,state,common_surfaces(seed,pairs,kw.get("split","novel")),
              seed=seed,lifetime_id=f"{seed}-paired-{kw.get('section','')}" ,
              before_hash=before,initial=initial,**kw)
        rows.append(r);return r

    # A: exact matching of all clocks and surfaces; only consequences differ.
    history=[paired(seed+1009*i,32) for i in range(pairs)]
    state=initial
    def record_canonical(kind, current, diag, logits, loss):
        traces.append(trace_item(current.tau,current,diag,kind,
                                 forecast_logits=logits,future_loss=loss))
        trajectory.append({"tau":current.tau,"event_type":kind,
                           "H":current.H.detach().cpu(),"F":current.F.detach().cpu(),
                           "M":current.M.detach().cpu()})
    add(state,section="formation",N=0)
    states={0:state.clone()}
    for t in range(32):
        items=[history[i][2*t+j] for i in range(pairs) for j in range(2)]
        state,loss,logits,diags=play_experience(model,state,items,trace=record_canonical)
        if t+1 in (1,2,4,8,16,32):
            states[t+1]=state.clone()
            r=add(state,section="formation",N=t+1)
            r["future_loss"]=float(loss.mean())
    assert_frozen(model,frozen)

    # B/C/J: fixed N or two-dimensional N x D, fresh matched unrelated stream.
    ns=(1,2,4,8,16) if long else (16,)
    for n in ns:
        ds=((0,10,50,100,500,1000) if n==16 else (0,10,100,500,1000)) if long else (0,10)
        current=states[n].clone();prev=0
        for d in ds:
            if d>prev:
                # Same RNG stream as the full d, advanced by the prefix.
                rng=random.Random(seed+700+n)
                for _ in range(prev):
                    for _pair in range(pairs):rng.choice((20,21,22,23))
                from etrcm.stage2c.world import NUISANCE,tensor_ids,unrelated_token
                for tick in range(prev,d):
                    values=[rng.choice(NUISANCE) for _ in range(pairs)]
                    ids=tensor_ids([v for value in values for v in (value,value)],device)
                    current,diag=model.step(current,unrelated_token(ids))
                    if n==16:
                        traces.append(trace_item(tick,current,diag,"unrelated_delay"))
                        trajectory.append({"tau":current.tau,"event_type":"unrelated_delay",
                                           "H":current.H.detach().cpu(),
                                           "F":current.F.detach().cpu(),"M":current.M.detach().cpu()})
            prev=d
            for split in ("seen","novel","hard_ood") if n==16 else ("novel",):
                r=add(current,section="timescale" if n!=16 else "persistence",N=n,D=d,split=split)
                r["H_over_100"]=r["H_norm"]>100
                r["H_over_1000"]=r["H_norm"]>1000
            if n==16 and d in (0,100,500):
                for intr in ("H_swap","F_swap","M_swap","FM_swap","HFM_swap",
                             "H_reset","H_reset_F_zero","H_reset_M_zero",
                             "H_reset_FM_zero","H_reset_F_swap","H_reset_M_swap"):
                    add(current,section="state_swap",N=n,D=d,intervention=intr)
                if model.variant not in ("no_memory","gru"):
                    for clamp in ("F","M","FM"):
                        add(current,section="probe_read",N=n,D=d,clamp=clamp)
        assert_frozen(model,frozen)

    # D: same number of context/action/consequence ticks, random consequence.
    noise_state=initial.clone()
    noise_hist=[paired(seed+1009*i,16,noise=True) for i in range(pairs)]
    for t in range(16):
        items=[noise_hist[i][2*t+j] for i in range(pairs) for j in range(2)]
        noise_state,_,_,_=play_experience(model,noise_state,items)
    add(noise_state,section="selectivity",N=16,noise=True)

    # E: same new real evidence count, old rule A -> new B and vice versa.
    revised=states[16].clone();base=add(revised,section="revision",N=16,revision=0)
    histories=[paired(seed+1009*i+9191,32) for i in range(pairs)]
    for t in range(32):
        # Swapped condition: first member receives z=B consequences, second z=A.
        items=[]
        for i in range(pairs):
            items.extend((histories[i][2*t+1],histories[i][2*t]))
        revised,_,_,_=play_experience(model,revised,items)
        if t+1 in (1,2,4,8,16,32):
            rr=add(revised,section="revision",N=16,revision=t+1)
            rr["RI"]=base["BS"]-rr["BS"]

    # F/G/H/I: matched formation with window-specific read or external write.
    if model.variant not in ("no_memory","gru"):
        cases=[("none","none","none")]
        for window in ("W1","W2","W3","W4"):
            for clamp in ("F","M","FM"):
                cases.append((window,clamp,"read"))
        for window in ("W1","W2"):
            cases.append((window,"none","write"))
        for window,clamp,kind in cases:
            s=initial.clone()
            for t in range(16):
                items=[history[i][2*t+j] for i in range(pairs) for j in range(2)]
                active=window==("W1" if t<8 else "W2")
                s,_,_,_=play_experience(model,s,items,
                                        read_clamp=clamp if kind=="read" and active else "none",
                                        write_block=kind=="write" and active)
            for _ in range(4):
                s,_=model.step(s,None,read_clamp=clamp if kind=="read" and window=="W2" else "none")
            s=unrelated_delay(model,s,50,seed+1701,
                              read_clamp=clamp if kind=="read" and window=="W3" else "none")
            r=add(s,section="window",N=16,D=50,
                  intervention=f"{kind}_{window}_{clamp}" if kind!="none" else "none",
                  clamp=clamp if kind=="read" and window=="W4" else "none")
            r["window"]=window;r["intervention_kind"]=kind
        assert_frozen(model,frozen)

    # Fill derived summaries only when denominators are numerically meaningful.
    grouped={(r["section"],r["exposure_count"],r["delay"],r["probe_type"]):r
             for r in rows if r["intervention"]=="none" and not r["useful_noise_condition"]=="noise"}
    for r in rows:
        base=grouped.get((r["section"],r["exposure_count"],0,r["probe_type"]))
        if base and abs(base["BS"])>1e-4:r["PR"]=r["BS"]/base["BS"]
        if r["probe_type"]=="novel":
            seen=grouped.get((r["section"],r["exposure_count"],r["delay"],"seen"))
            if seen and abs(seen["BS"])>1e-4:r["GR"]=r["BS"]/seen["BS"]
    assert_frozen(model,frozen)
    after=model.parameter_digest()
    if before!=after:raise AssertionError("parameter digest changed")
    out.mkdir(parents=True,exist_ok=True)
    (out/"rows.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in rows))
    (out/"trajectory.jsonl").write_text("".join(json.dumps(r,allow_nan=False)+"\n" for r in traces))
    torch.save(trajectory,out/"state_tensors.pt")
    manifest={"seed":seed,"variant":payload["variant"],"pairs":pairs,"rows":len(rows),
              "parameter_hash_before":before,"parameter_hash_after":after,
              "checkpoint_sha256":hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--pairs",type=int,default=4)
    p.add_argument("--quick",action="store_true")
    p.add_argument("--posthoc-gamma-zero",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(2)
    model,payload=load_model(args.checkpoint,args.device,
                             posthoc_gamma_zero=args.posthoc_gamma_zero)
    with torch.no_grad():
        result=evaluate(model,payload,args.out,pairs=args.pairs,long=not args.quick)
    print(json.dumps(result),flush=True)
