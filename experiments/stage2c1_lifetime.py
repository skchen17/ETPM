"""L3 diagnostic: original ET-RCM lifetime with direct candidate-action head.

This is a new joint-training experiment, not a re-adjudication of Stage 2C.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_5.model import AnatomicalETRCM
from etrcm.stage2c.world import (ABSTRACT, ACTION, OUTCOME, NUISANCE, Experience,
    context_token, outcome_token, unrelated_token, paired_histories, surface,
    consequence, tensor_ids)
from etrcm.stage2c1.diagnostic import ActionHead, core_config, forecast_metrics


class LifetimeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core=AnatomicalETRCM(core_config(),mode="B5_separate")
        self.action_head=ActionHead("late_concat")

    def initial_state(self,batch,device):
        return self.core.initial_state(batch,device=device)

    def logits(self,state,action):
        return self.action_head(self.core._pool(state.H),action)


def play(model,state,items):
    device=state.H.device
    fields=(torch.full((len(items),),ABSTRACT,dtype=torch.long,device=device),
            tensor_ids([x.color for x in items],device),
            tensor_ids([x.shape for x in items],device),
            tensor_ids([x.nuisance for x in items],device),
            tensor_ids([ACTION[x.action] for x in items],device))
    for ids in fields:
        state,_=model.core.step(state,context_token(ids))
    action=tensor_ids([x.action for x in items],device)
    target=tensor_ids([x.outcome-OUTCOME[0] for x in items],device)
    logits=model.logits(state,action)
    loss=F.cross_entropy(logits,target,reduction="none")
    state,_=model.core.step(state,outcome_token(tensor_ids([x.outcome for x in items],device)))
    return state,loss


def probe(model,state,features):
    device=state.H.device
    fields=(torch.full((len(features),),ABSTRACT,dtype=torch.long,device=device),
            tensor_ids([x[0] for x in features],device),
            tensor_ids([x[1] for x in features],device),
            tensor_ids([x[2] for x in features],device))
    context=state.clone()
    for ids in fields:
        context,_=model.core.step(context,context_token(ids))
    logits=[]
    for action in (0,1):
        ids=torch.full((len(features),),ACTION[action],dtype=torch.long,device=device)
        branch,_=model.core.step(context.clone(),context_token(ids))
        logits.append(model.logits(branch,torch.full((len(features),),action,dtype=torch.long,device=device)))
    return torch.stack(logits,1).softmax(-1)


def paired_state(model,seed,n,device):
    histories=paired_histories(seed,n,split="train")
    state=model.initial_state(2,device)
    for pair in zip(*histories):
        state,_=play(model,state,list(pair))
    return state


def bs(model,state,seed,split="novel"):
    rng=random.Random(seed*19+88)
    feature=surface(rng,split)
    probabilities=probe(model,state,[feature,feature])
    metrics=forecast_metrics(probabilities)
    return metrics


def delay(model,state,ticks,seed):
    rng=random.Random(seed)
    for _ in range(ticks):
        token=rng.choice(NUISANCE)
        ids=torch.full((2,),token,dtype=torch.long,device=state.H.device)
        state,_=model.core.step(state,unrelated_token(ids))
    return state


def swap(state,which):
    def maybe(name,tensor):
        return tensor.flip(0).clone() if name in which else tensor.clone()
    return LearnedState(maybe("H",state.H),maybe("F",state.F),
                        maybe("M",state.M),state.tau,state.external_time)


@torch.no_grad()
def evaluate(model,seed,device):
    model.eval(); histories=[]
    for replicate in range(4):
        run_seed=seed*101+replicate
        record={"replicate":replicate,"formation":{},"persistence":{},"generalization":{},
                "revision":{},"swaps":{},"state_norms":{}}
        for n in (0,1,2,4,8,16,32):
            state=paired_state(model,run_seed,n,device)
            record["formation"][str(n)]=bs(model,state,run_seed)["behavioral_separation_entropy"]
        state=paired_state(model,run_seed,16,device)
        record["state_norms"]={name:float(getattr(state,name).norm(dim=tuple(range(1,getattr(state,name).ndim))).mean())
                                for name in ("H","F","M")}
        for split in ("seen","novel","hard_ood"):
            record["generalization"][split]=bs(model,state,run_seed,split)["behavioral_separation_entropy"]
        for ticks in (0,10,100,500):
            post=delay(model,state,ticks,run_seed+ticks)
            record["persistence"][str(ticks)]=bs(model,post,run_seed)["behavioral_separation_entropy"]
        for which in ("F","M","FM"):
            record["swaps"][which]=bs(model,swap(state,which),run_seed)["behavioral_separation_entropy"]
        for n in (0,1,2,4,8,16,32):
            revision=state.clone()
            if n:
                opposing=paired_histories(run_seed+499,n,split="train")
                for pair in zip(*opposing):
                    revision,_=play(model,revision,[pair[1],pair[0]])
            record["revision"][str(n)]=bs(model,revision,run_seed)["behavioral_separation_entropy"]
        record["action_metrics_N16"]=bs(model,state,run_seed)
        histories.append(record)
    return histories


def train(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng=random.Random(args.seed+15)
    model=LifetimeModel().to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=0.001)
    args.out.mkdir(parents=True,exist_ok=True)
    logs=[];start=time.monotonic()
    for step in range(args.steps):
        model.train();opt.zero_grad(set_to_none=True)
        latent=[(i+step)%2 for i in range(16)];rng.shuffle(latent)
        state=model.initial_state(16,args.device)
        losses=[];penalties=[]
        for _ in range(rng.choice((4,8,16))):
            actions=[i%2 for i in range(16)];rng.shuffle(actions)
            items=[]
            for z,a in zip(latent,actions):
                c,s,n=surface(rng,"train")
                items.append(Experience(c,s,n,a,consequence(rng,z,a)))
            state,loss=play(model,state,items)
            losses.append(loss.mean());penalties.append(state.H.square().mean())
        objective=torch.stack(losses).mean()
        total=objective+0.001*torch.stack(penalties).mean()
        if not torch.isfinite(total):raise RuntimeError("nonfinite loss")
        total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if (step+1)%100==0 or step==0:
            row={"step":step+1,"CE":float(objective.detach()),
                 "H":float(state.H.detach().norm(dim=(-2,-1)).mean()),
                 "F":float(state.F.detach().norm(dim=(-2,-1)).mean()),
                 "M":float(state.M.detach().norm(dim=(-2,-1)).mean()),
                 "elapsed_s":time.monotonic()-start}
            logs.append(row);print(json.dumps(row),flush=True)
    before=hashlib.sha256(b"".join(x.detach().cpu().numpy().tobytes() for x in model.parameters())).hexdigest()
    evaluation=evaluate(model,args.seed,args.device)
    after=hashlib.sha256(b"".join(x.detach().cpu().numpy().tobytes() for x in model.parameters())).hexdigest()
    summary={"seed":args.seed,"steps":args.steps,"elapsed_s":time.monotonic()-start,
             "eval_parameter_frozen":before==after,"objective":"Stage2C observed consequence CE + 0.001 H penalty",
             "change_from_Stage2C":"direct candidate-action head only; F/M law unchanged",
             "evaluation":evaluation}
    (args.out/"train_log.json").write_text(json.dumps(logs,indent=2))
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2))
    torch.save({"model":model.state_dict(),"seed":args.seed,"steps":args.steps},args.out/"checkpoint.pt")
    print(json.dumps({"COMPLETED":{"seed":args.seed,"time":summary["elapsed_s"],
                                   "BS16":[x["formation"]["16"] for x in evaluation],
                                   "frozen":before==after}}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--steps",type=int,default=500)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--out",type=Path,required=True)
    train(p.parse_args())
