"""Frozen formal-lifetime evaluation of one Stage 2C.3 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from etrcm.stage1_1.model import LearnedState
from etrcm.stage2c.world import (ABSTRACT,ACTION,OUTCOME,paired_histories,
    context_token,outcome_token,surface,tensor_ids)
from etrcm.stage2c1.diagnostic import forecast_metrics
from etrcm.stage2c3.protocol import MARGINAL,MARGINAL_CE,target_distribution
from etrcm.stage2c3.controls import (MarginalPredictor,ActionOnlyPredictor,
    HistoryOnlyPredictor)
from experiments.stage2c1_lifetime import (LifetimeModel,paired_state,play,bs,delay,
    probe,swap)
from experiments.stage2c2_pathway import fit_oracle_h,parameter_hash
from experiments.stage2c2_probes import collect
from experiments.stage2c2_curve import ridge_accuracy


def load(path,device):
    data=torch.load(path,map_location=device,weights_only=True)
    model=LifetimeModel().to(device)
    model.load_state_dict(data["model"])
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,data


def mean_rows(rows):
    return {key:sum(row[key] for row in rows)/len(rows) for key in rows[0]}


@torch.no_grad()
def clamp_probe(model,state,features,which):
    device=state.H.device
    batch=len(features)
    zero=torch.zeros(batch,model.core.config.value_dim,device=device)
    def tick(s,ids):
        return model.core.step_with_read(s,context_token(ids),
            fast_read_override=zero if "F" in which else None,
            slow_read_override=zero if "M" in which else None)[0]
    fields=(torch.full((batch,),ABSTRACT,device=device,dtype=torch.long),
            tensor_ids([x[0] for x in features],device),
            tensor_ids([x[1] for x in features],device),
            tensor_ids([x[2] for x in features],device))
    context=state.clone()
    for ids in fields:context=tick(context,ids)
    logits=[]
    for a in (0,1):
        ids=torch.full((batch,),ACTION[a],device=device,dtype=torch.long)
        branch=tick(context.clone(),ids)
        logits.append(model.logits(branch,torch.full((batch,),a,device=device,dtype=torch.long)))
    return torch.stack(logits,1).softmax(-1)


@torch.no_grad()
def clamp_metrics(model,state,seed,which):
    rng=random.Random(seed*19+88)
    feature=surface(rng,"novel")
    return forecast_metrics(clamp_probe(model,state,[feature,feature],which))


@torch.no_grad()
def formation_clamped_state(model,seed,n,device,which):
    """Clamp F/M reads at every legal-history tick; leave the final probe native.

    This is an evaluation-only temporal intervention. External evidence and the
    F/M laws still run normally; it tests whether earlier reads helped form H.
    """
    histories=paired_histories(seed,n,split="train")
    state=model.initial_state(2,device)
    zero=torch.zeros(2,model.core.config.value_dim,device=device)
    def tick(event):
        nonlocal state
        state,_=model.core.step_with_read(state,event,
            fast_read_override=zero if "F" in which else None,
            slow_read_override=zero if "M" in which else None)
    for pair in zip(*histories):
        items=list(pair)
        fields=(torch.full((2,),ABSTRACT,device=device,dtype=torch.long),
                tensor_ids([x.color for x in items],device),
                tensor_ids([x.shape for x in items],device),
                tensor_ids([x.nuisance for x in items],device),
                tensor_ids([ACTION[x.action] for x in items],device))
        for ids in fields:tick(context_token(ids))
        tick(outcome_token(tensor_ids([x.outcome for x in items],device)))
    return state


@torch.no_grad()
def formation_zeroed_state(model,seed,n,device,which):
    """Zero selected peripheral state after every legal-history transition."""
    histories=paired_histories(seed,n,split="train")
    state=model.initial_state(2,device)
    def tick(event):
        nonlocal state
        state,_=model.core.step(state,event)
        state=LearnedState(state.H,
            torch.zeros_like(state.F) if "F" in which else state.F,
            torch.zeros_like(state.M) if "M" in which else state.M,
            state.tau,state.external_time)
    for pair in zip(*histories):
        items=list(pair)
        fields=(torch.full((2,),ABSTRACT,device=device,dtype=torch.long),
                tensor_ids([x.color for x in items],device),tensor_ids([x.shape for x in items],device),
                tensor_ids([x.nuisance for x in items],device),
                tensor_ids([ACTION[x.action] for x in items],device))
        for ids in fields:tick(context_token(ids))
        tick(outcome_token(tensor_ids([x.outcome for x in items],device)))
    return state


@torch.no_grad()
def behavior(model,seed,device,bs_threshold=.1):
    records=[]
    for rep in range(4):
        run_seed=seed*101+rep+10_000_000
        exposure={}
        for n in (0,1,2,4,8,16,32):
            state=paired_state(model,run_seed,n,device)
            exposure[str(n)]=bs(model,state,run_seed)["behavioral_separation_entropy"]
        state=paired_state(model,run_seed,16,device)
        final_metric=bs(model,state,run_seed)
        p=torch.tensor(final_metric["prob"],device=device).clamp_min(1e-9)
        z=torch.tensor([[0,0],[1,1]],device=device)
        actions=torch.tensor([[0,1],[0,1]],device=device)
        conditional_ce=float(-(target_distribution(z,actions)*p.log()).sum(-1).mean())
        action_health={"action_TV":sum(final_metric["tv_action"])/2,
                       "history_TV":sum(final_metric["tv_history"])/2,
                       "interaction_y0":final_metric["interaction_y0"],
                       "BS":final_metric["behavioral_separation_entropy"],
                       "conditional_CE":conditional_ce,"CFA":MARGINAL_CE-conditional_ce}
        generalization={split:bs(model,state,run_seed,split)["behavioral_separation_entropy"]
                        for split in ("seen","novel","hard_ood")}
        revision={}
        for n in (0,1,2,4,8,16,32):
            revised=state.clone()
            if n:
                opposing=paired_histories(run_seed+499,n,split="train")
                for pair in zip(*opposing):revised,_=play(model,revised,[pair[1],pair[0]])
            metric=bs(model,revised,run_seed)
            revision[str(n)]={"BS":metric["behavioral_separation_entropy"],
                              "policy_A":metric["entropy_policy"][0][0],
                              "policy_B":metric["entropy_policy"][1][0]}
        meaningful=abs(exposure["16"])>=bs_threshold
        persistence={}
        persistence_ratio={}
        interventions={}
        if meaningful:
            for ticks in (0,10,50,100,500,1000):
                post=delay(model,state,ticks,run_seed+ticks)
                persistence[str(ticks)]=bs(model,post,run_seed)["behavioral_separation_entropy"]
            persistence_ratio={name:value/persistence["0"] for name,value in persistence.items()
                               if abs(persistence["0"])>=bs_threshold}
            base=bs(model,state,run_seed)["behavioral_separation_entropy"]
            interventions["baseline_BS"]=base
            for which in ("H","F","M","FM","HFM"):
                changed=bs(model,swap(state,which),run_seed)["behavioral_separation_entropy"]
                interventions[f"swap_{which}"]={"BS":changed,"change":changed-base}
            for which in ("F","M","FM"):
                changed=clamp_metrics(model,state,run_seed,which)["behavioral_separation_entropy"]
                interventions[f"clamp_{which}"]={"BS":changed,"change":changed-base}
                formed=formation_clamped_state(model,run_seed,16,device,which)
                formed_bs=bs(model,formed,run_seed)["behavioral_separation_entropy"]
                interventions[f"formation_clamp_{which}"]={"BS":formed_bs,
                                                            "change":formed_bs-base}
                zeroed=formation_zeroed_state(model,run_seed,16,device,which)
                zeroed_bs=bs(model,zeroed,run_seed)["behavioral_separation_entropy"]
                interventions[f"formation_zero_{which}"]={"BS":zeroed_bs,
                                                           "change":zeroed_bs-base}
        records.append({"replicate":rep,"exposure":exposure,"generalization":generalization,
                        "final_health":action_health,
                        "revision":revision,"persistence":persistence,
                        "persistence_ratio":persistence_ratio,
                        "interventions":interventions,"meaningful_at_N16":meaningful})
    aggregate={"exposure":{str(n):sum(r["exposure"][str(n)] for r in records)/4
                           for n in (0,1,2,4,8,16,32)},
               "final_health":{key:sum(r["final_health"][key] for r in records)/4
                               for key in records[0]["final_health"]},
               "generalization":{split:sum(r["generalization"][split] for r in records)/4
                                  for split in ("seen","novel","hard_ood")},
               "meaningful_replicates":sum(r["meaningful_at_N16"] for r in records)}
    for ticks in (0,10,50,100,500,1000):
        values=[r["persistence"][str(ticks)] for r in records if str(ticks) in r["persistence"]]
        aggregate.setdefault("persistence",{})[str(ticks)]=sum(values)/len(values) if values else None
        ratios=[r["persistence_ratio"][str(ticks)] for r in records
                if str(ticks) in r["persistence_ratio"]]
        aggregate.setdefault("persistence_ratio",{})[str(ticks)]=sum(ratios)/len(ratios) if ratios else None
    for n in (0,1,2,4,8,16,32):
        aggregate.setdefault("revision",{})[str(n)]=sum(r["revision"][str(n)]["BS"] for r in records)/4
    for which in ("H","F","M","FM","HFM"):
        name=f"swap_{which}"
        values=[r["interventions"][name]["change"] for r in records if name in r["interventions"]]
        aggregate.setdefault("interventions",{})[name]=sum(values)/len(values) if values else None
    for which in ("F","M","FM"):
        name=f"clamp_{which}"
        values=[r["interventions"][name]["change"] for r in records if name in r["interventions"]]
        aggregate["interventions"][name]=sum(values)/len(values) if values else None
        name=f"formation_clamp_{which}"
        values=[r["interventions"][name]["change"] for r in records if name in r["interventions"]]
        aggregate["interventions"][name]=sum(values)/len(values) if values else None
        name=f"formation_zero_{which}"
        values=[r["interventions"][name]["change"] for r in records if name in r["interventions"]]
        aggregate["interventions"][name]=sum(values)/len(values) if values else None
    return {"replicates":records,"aggregate":aggregate}


def latent_probes(model,seed,device):
    train,ytrain=collect(model,seed*7+10_000,"train",64,device)
    test,ytest=collect(model,seed*7+20_000,"novel",64,device)
    return {name:ridge_accuracy(train[name],ytrain,test[name],ytest)
            for name in ("H","F","M","FM")}


def baseline_controls(model,seed,device):
    # Exact balanced-world controls. Both p(y) and p(y|a) are estimated only
    # from training counts, with a separate held-out simulator sample.
    rng=random.Random(seed*13+31)
    observations=[];actions=[]
    for _ in range(20_000):
        z=rng.randrange(2);a=rng.randrange(2)
        y=0 if a==z else rng.randrange(1,4)
        observations.append(y);actions.append(a)
    marginal=MarginalPredictor().fit(torch.tensor(observations)).probability
    action_only=ActionOnlyPredictor().fit(torch.tensor(actions),torch.tensor(observations)).probability
    # True held-out risk integrates exact rule, avoiding sampled evaluation noise.
    targets=target_distribution(torch.tensor([[0,0],[1,1]]),
                                torch.tensor([[0,1],[0,1]])).double()
    marg_ce=float(-(targets*marginal.log()).sum(-1).mean())
    action_ce=float(-(targets*action_only[None].log()).sum(-1).mean())
    # p(y|S) trained on frozen H and sampled next observations, no action input.
    train,labels=collect(model,seed*7+30_000,"train",64,device)
    test,test_labels=collect(model,seed*7+40_000,"novel",64,device)
    generator=torch.Generator(device="cpu").manual_seed(seed*33+8)
    def make_targets(z):
        actions=torch.randint(2,(len(z),),generator=generator).to(device)
        wrong=torch.randint(1,4,(len(z),),generator=generator).to(device)
        return torch.where(actions.eq(z),torch.zeros_like(wrong),wrong)
    ytest=make_targets(test_labels)
    torch.manual_seed(seed+900_000)
    head=HistoryOnlyPredictor(train["H"].shape[-1]).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.01)
    xtrain=train["H"].detach();xtest=test["H"].detach()
    mu=xtrain.mean(0,keepdim=True);scale=xtrain.std(0,unbiased=False,keepdim=True).clamp_min(.01)
    a=((xtrain-mu)/scale).clamp(-10,10);b=((xtest-mu)/scale).clamp(-10,10)
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        # The exact best possible action-blind target is marginal at every H.
        loss=F.cross_entropy(head(a),torch.tensor(MARGINAL,device=device).expand(len(a),4))
        loss.backward();opt.step()
    with torch.no_grad():
        logits=head(b)
        history_sample_ce=float(F.cross_entropy(logits,ytest))
        history_exact_ce=float(F.cross_entropy(
            logits,torch.tensor(MARGINAL,device=device).expand(len(b),4)))
        history_calibration=float((logits.softmax(-1).mean(0)-torch.tensor(MARGINAL,device=device)).abs().sum()/2)
    return {"marginal":{"CE":marg_ce,"entropy":float(-(marginal*marginal.log()).sum()),
                        "TV_calibration":float((marginal-torch.tensor(MARGINAL)).abs().sum()/2)},
            "action_only":{"CE":action_ce,"entropy_by_action":(-(action_only*action_only.log()).sum(-1)).tolist(),
                           "prob":action_only.tolist()},
            "history_only":{"CE":history_exact_ce,"heldout_sample_CE":history_sample_ce,
                            "TV_calibration":history_calibration,
                            "input":"frozen H only; no candidate action"},
            "theoretical_marginal_CE":MARGINAL_CE}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--bs-threshold",type=float,default=.1)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    model,data=load(args.checkpoint,args.device)
    before=parameter_hash(model)
    result={"arm":data["arm"],"seed":args.seed,"checkpoint_step":data["step"],
            "parameter_sha256_before":before,
            "behavior":behavior(model,args.seed,args.device,args.bs_threshold),
            "probes":latent_probes(model,args.seed,args.device),
            "baselines":baseline_controls(model,args.seed,args.device)}
    if data["arm"] in {"A2","A3"}:
        with torch.no_grad():
            norms=[]
            for rep in range(4):
                native=paired_state(model,args.seed*101+99+rep,16,args.device)
                norms.extend(float(v) for v in native.H.norm(dim=(-2,-1)))
            norm=statistics.median(norms)
        oracle=fit_oracle_h(model,args.seed+5_000_000,norm,args.device,steps=500,restarts=2)
        oracle.pop("state_tensors")
        result["oracle_H_final"] = oracle
    after=parameter_hash(model)
    assert before==after
    result["parameter_sha256_after"]=after
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps({"seed":args.seed,"arm":data["arm"],"BS16":result["behavior"]["aggregate"]["exposure"]["16"],
                      "frozen":True}),flush=True)


if __name__=="__main__":main()
