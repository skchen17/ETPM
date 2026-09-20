"""Outer-lifetime training on observable future consequence only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from etrcm.stage1_4.model import Stage14Config
from etrcm.stage2c.model import BehavioralModel, VARIANTS
from etrcm.stage2c.rollout import play_experience, probe_policy
from etrcm.stage2c.world import SYMBOL_COUNT, consequence, surface, Experience


def config() -> Stage14Config:
    return Stage14Config(hidden_dim=32,latent_slots=1,symbol_count=SYMBOL_COUNT,
                         key_dim=8,value_dim=8,event_type_dim=8,gamma=0.12,
                         rho_fast=0.97,rho_slow=0.9995,eta_external=0.6)


def make_step(rng: random.Random, latent: list[int], tick: int, train_step: int) -> list[Experience]:
    items=[]
    # Independently randomized each tick: clock/index cannot reveal the action.
    # Balanced action counts without a fixed alternating sequence.
    actions=[i%2 for i in range(len(latent))]
    rng.shuffle(actions)
    for i,z in enumerate(latent):
        color,shape,nuisance=surface(rng,"train")
        action=actions[i]
        items.append(Experience(color,shape,nuisance,action,consequence(rng,z,action)))
    return items


@torch.no_grad()
def validate(model: BehavioralModel, seed: int, device: str, lifetimes: int = 16):
    model.eval()
    rng=random.Random(seed+9910)
    latent=[i%2 for i in range(lifetimes)]
    state=model.initial_state(lifetimes,device=device)
    for t in range(16):
        state,_,_,_=play_experience(model,state,make_step(rng,latent,t,7))
    features=[surface(rng,"novel") for _ in latent]
    probs,_,_,_,_,_=probe_policy(model,state,features)
    p=probs.detach().cpu().tolist()
    bs=sum(p[i][0] for i,z in enumerate(latent) if z==0)/(lifetimes/2)-sum(
        p[i][0] for i,z in enumerate(latent) if z==1)/(lifetimes/2)
    correct=sum(p[i][z] for i,z in enumerate(latent))/lifetimes
    return {"behavioral_separation":bs,"mean_correct_action_probability":correct,
            "H_norm":float(state.H.norm(dim=(-2,-1)).mean()),
            "F_norm":float(state.F.norm(dim=(-2,-1)).mean()),
            "M_norm":float(state.M.norm(dim=(-2,-1)).mean())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variant",choices=VARIANTS,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--steps",type=int,default=300)
    p.add_argument("--batch",type=int,default=16)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--log-every",type=int,default=50)
    p.add_argument("--state-penalty",type=float,default=0.0,
                   help="Outer-training H magnitude regularizer; recurrence unchanged")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(args.seed);random.seed(args.seed)
    rng=random.Random(args.seed+15)
    cfg=config()
    model=BehavioralModel(cfg,args.variant).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    train_log=[];start=time.monotonic();events_seen=0;active=None
    for step in range(args.steps):
        model.train();opt.zero_grad(set_to_none=True)
        latent=[(i+step)%2 for i in range(args.batch)]
        rng.shuffle(latent)
        count=rng.choice((4,8,16))
        state=model.initial_state(args.batch,device=args.device)
        losses=[]
        penalties=[]
        for t in range(count):
            items=make_step(rng,latent,t,step)
            state,per_item,_,_=play_experience(model,state,items)
            # Later predictions are where using earlier evidence can reduce CE.
            losses.append(per_item.mean())
            penalties.append(state.H.square().mean())
            events_seen += args.batch*6
        predictive_ce=torch.stack(losses).mean()
        state_cost=torch.stack(penalties).mean()
        loss=predictive_ce+args.state_penalty*state_cost
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss {args.variant} seed={args.seed} step={step}")
        loss.backward()
        if active is None:
            active=sum(x.numel() for x in model.parameters() if x.grad is not None)
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
        opt.step()
        if step%args.log_every==0 or step==args.steps-1:
            row={"step":step+1,"predictive_CE":float(predictive_ce.detach()),
                 "state_penalty":float((args.state_penalty*state_cost).detach()),
                 "total_loss":float(loss.detach()),"grad_preclip":grad,
                 "H":float(state.H.detach().norm(dim=(-2,-1)).mean()),
                 "F":float(state.F.detach().norm(dim=(-2,-1)).mean()),
                 "M":float(state.M.detach().norm(dim=(-2,-1)).mean()),
                 "events_seen":events_seen,"elapsed_s":time.monotonic()-start}
            train_log.append(row);print(json.dumps(row),flush=True)
    validation=validate(model,args.seed,args.device)
    summary={"variant":args.variant,"seed":args.seed,"steps":args.steps,"batch":args.batch,
             "config":asdict(cfg),"nominal_parameters":sum(x.numel() for x in model.parameters()),
             "active_parameters":active,"events_seen":events_seen,"elapsed_s":time.monotonic()-start,
             "validation":validation,
             "objective":"observable consequence CE plus optional H norm regularization; no reward/action/memory label",
             "optimizer":"AdamW lr=0.001; parameter gradient clip=1; no state clip",
             "state_penalty":args.state_penalty}
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2))
    (args.out/"train_log.json").write_text(json.dumps(train_log,indent=2))
    (args.out/"config.json").write_text(json.dumps(asdict(cfg),indent=2))
    torch.save({"model":model.state_dict(),"variant":args.variant,"config":asdict(cfg),
                "seed":args.seed,"steps":args.steps},args.out/"checkpoint.pt")
    hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in args.out.iterdir()
            if x.is_file() and x.name!="hashes.json"}
    (args.out/"hashes.json").write_text(json.dumps(hashes,indent=2))
    print(json.dumps({"COMPLETED":summary}),flush=True)


if __name__=="__main__":main()
