"""Independent Stage 2C.1-L3 replica with preregistered frozen snapshots."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from etrcm.stage2c.world import Experience, consequence, surface
from experiments.stage2c1_lifetime import LifetimeModel, play, paired_state, bs
from experiments.stage2c2_pathway import parameter_hash
from experiments.stage2c2_probes import collect

STEPS=(0,25,50,100,200,300,500)


def gradient_groups(model):
    groups={
        "action_branch":model.action_head.action_embedding.parameters(),
        "consequence_head":model.action_head.head.parameters(),
        "q_M":model.core.q_slow_projection.parameters(),
        "read_gate":list(model.core.two_way_gate.parameters())+list(model.core.slow_read_norm.parameters()),
        "event_encoder":model.core.event_encoder.parameters(),
        "H_core":list(model.core.core_in.parameters())+list(model.core.core_out.parameters())+
                 list(model.core.core_gate.parameters())+list(model.core.norm.parameters()),
    }
    return {name:sum(float(p.grad.detach().square().sum()) for p in params if p.grad is not None)**.5
            for name,params in groups.items()}


def ridge_accuracy(train,y_train,test,y_test):
    mu=train.mean(0,keepdim=True)
    scale=train.std(0,unbiased=False,keepdim=True).clamp_min(.01)
    a=((train-mu)/scale).clamp(-10,10)
    b=((test-mu)/scale).clamp(-10,10)
    a=torch.cat([a,torch.ones(len(a),1,device=a.device)],-1)
    b=torch.cat([b,torch.ones(len(b),1,device=b.device)],-1)
    eye=torch.eye(a.shape[1],device=a.device)
    weight=torch.linalg.solve(a.T@a+eye,a.T@(2*y_train.float()-1))
    return float(((b@weight>0).long()==y_test).float().mean())


@torch.no_grad()
def snapshot(model,seed,step,device):
    before=parameter_hash(model)
    model.eval();records=[]
    for rep in range(4):
        episode_seed=seed*101+rep+step*100_000
        state=paired_state(model,episode_seed,16,device)
        qf,qm=model.core._queries(state.H)
        rm=torch.einsum("bvk,bk->bv",state.M,qm)
        metric=bs(model,state,episode_seed)
        records.append({"TV_action":sum(metric["tv_action"])/2,
                        "TV_history":sum(metric["tv_history"])/2,
                        "interaction_y0":metric["interaction_y0"],
                        "BS":metric["behavioral_separation_entropy"],
                        "r_M_difference":float((rm[0]-rm[1]).norm()),
                        "H_difference":float((state.H[0]-state.H[1]).norm()),
                        "H_norm":float(state.H.norm(dim=(-2,-1)).mean())})
    train,ztrain=collect(model,seed*2+step*10+7,"train",64,device)
    test,ztest=collect(model,seed*2+step*10+7007,"novel",64,device)
    probe=ridge_accuracy(train["M"],ztrain,test["M"],ztest)
    after=parameter_hash(model)
    assert before==after
    return {"step":step,"metrics":{name:sum(x[name] for x in records)/len(records)
                                  for name in records[0]},
            "replicates":records,"M_latent_ridge_test_accuracy":probe,
            "parameter_hash_before":before,"parameter_hash_after":after}


def train(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    rng=random.Random(args.seed+15)
    model=LifetimeModel().to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for step in range(501):
        if step in STEPS:
            snap=snapshot(model,args.seed,step,args.device)
            snap["gradient_preclip_from_preceding_update"]=last_gradient if step else None
            snap["train_CE_from_preceding_update"]=last_ce if step else None
            rows.append(snap)
            torch.save({"seed":args.seed,"step":step,"model":model.state_dict()},
                       args.out/f"checkpoint_{step}.pt")
            print(json.dumps({"seed":args.seed,"step":step,
                              "metrics":snap["metrics"],
                              "M_probe":snap["M_latent_ridge_test_accuracy"]}),flush=True)
        if step==500:break
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
        ce=torch.stack(losses).mean()
        total=ce+.001*torch.stack(penalties).mean()
        if not torch.isfinite(total):raise RuntimeError("nonfinite joint loss")
        total.backward()
        last_gradient=gradient_groups(model)
        last_ce=float(ce.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
    summary={"seed":args.seed,"steps":list(STEPS),"snapshots":rows,
             "training_rule":"Stage2C1 L3 reproduction; unchanged F/M law and CE objective",
             "no_parameter_updates_during_snapshots":True}
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--out",type=Path,required=True)
    train(p.parse_args())
