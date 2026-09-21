"""Matched A0–A3 Stage 2C.3 objective/schedule diagnostics.

Only the existing LifetimeModel is trained. The inherited ET-RCM transition,
external delta write, F/M consolidation, decay, NULL and SELF_OUTPUT rules are
never modified here. Latent z is used by the simulator/loss, never model.step.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from etrcm.stage2c.world import (ABSTRACT, ACTION, OUTCOME, Experience,
    consequence, context_token, outcome_token, paired_histories, surface, tensor_ids)
from etrcm.stage2c1.diagnostic import OracleLatent, balanced_batch, forecast_metrics
from etrcm.stage2c3.protocol import (CHECKPOINTS, MARGINAL_CE, arm_phase,
    evaluator_lr_ratio, target_distribution)
from experiments.stage2c1_lifetime import LifetimeModel, paired_state, probe, bs
from experiments.stage2c2_pathway import parameter_hash


def group_gradients(model: LifetimeModel) -> dict[str, float]:
    core=model.core
    groups={
        "action_embedding":model.action_head.action_embedding.parameters(),
        "consequence_head":model.action_head.head.parameters(),
        "H_core":list(core.core_in.parameters())+list(core.core_out.parameters())+
                 list(core.core_gate.parameters())+list(core.norm.parameters()),
        "event_encoder":core.event_encoder.parameters(),
        "q_F":core.q_fast_projection.parameters(),
        "q_M":core.q_slow_projection.parameters(),
        "read_gate":list(core.two_way_gate.parameters())+
                    list(core.fast_read_norm.parameters())+list(core.slow_read_norm.parameters()),
    }
    return {name:math.sqrt(sum(float(p.grad.detach().square().sum()) for p in ps
                               if p.grad is not None)) for name,ps in groups.items()}


def head_hash(model: LifetimeModel) -> str:
    import hashlib
    digest=hashlib.sha256()
    for name,p in model.action_head.named_parameters():
        digest.update(name.encode());digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def pretrain_evaluator(model: LifetimeModel, seed: int, device: str, steps: int = 1000):
    # Same evaluator pretraining for matched A2/A3 seed; only action head transfers.
    torch.manual_seed(seed+300_000)
    oracle=OracleLatent("late_concat").to(device)
    generator=torch.Generator(device="cpu").manual_seed(seed+300_137)
    opt=torch.optim.AdamW(oracle.parameters(),lr=.003,weight_decay=0.0)
    for _ in range(steps):
        z,a,y=balanced_batch(64,device,generator)
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(oracle(z,a),y)
        loss.backward();opt.step()
    with torch.no_grad():
        z=torch.tensor([0,0,1,1],device=device)
        a=torch.tensor([0,1,0,1],device=device)
        metrics=forecast_metrics(oracle(z,a).softmax(-1).view(2,2,4))
    model.action_head.load_state_dict(oracle.action_head.state_dict())
    return {"steps":steps,"oracle_evaluator_metrics":metrics,
            "transferred_head_hash":head_hash(model),
            "privileged_pretraining":"oracle z enters only pretraining evaluator; no oracle input in lifetime"}


def context_state(model,state,items):
    device=state.H.device
    fields=(torch.full((len(items),),ABSTRACT,dtype=torch.long,device=device),
            tensor_ids([x.color for x in items],device),
            tensor_ids([x.shape for x in items],device),
            tensor_ids([x.nuisance for x in items],device))
    for ids in fields:
        state,_=model.core.step(state,context_token(ids))
    return state


def branch_logits(model,state,action):
    device=state.H.device
    ids=action+ACTION[0]
    next_state,_=model.core.step(state,context_token(ids))
    return next_state,model.logits(next_state,action)


def training_event(model,state,items,latent,arm):
    """A1 branches share exactly the same context state; only outcome is privileged."""
    device=state.H.device
    state=context_state(model,state,items)
    action=tensor_ids([x.action for x in items],device)
    outcome=tensor_ids([x.outcome-OUTCOME[0] for x in items],device)
    if arm=="A1":
        branches=[]
        for a in (0,1):
            a_ids=torch.full_like(action,a)
            _,logits=branch_logits(model,state.clone(),a_ids)
            branches.append(logits)
        logits=torch.stack(branches,1)
        targets=target_distribution(latent[:,None].expand(-1,2),
                                    torch.arange(2,device=device)[None,:].expand(len(items),-1))
        loss=-(targets*F.log_softmax(logits,-1)).sum(-1).mean()
    else:
        loss=F.cross_entropy(branch_logits(model,state.clone(),action)[1],outcome)
    # Actual experienced action/outcome alone updates recurrent state and F/M.
    state,_=branch_logits(model,state,action)
    state,_=model.core.step(state,outcome_token(tensor_ids([x.outcome for x in items],device)))
    return state,loss


def sampled_batch(rng,seed_step,batch,length):
    latent=[(i+seed_step)%2 for i in range(batch)]
    rng.shuffle(latent)
    episodes=[]
    for _ in range(length):
        actions=[i%2 for i in range(batch)];rng.shuffle(actions)
        items=[]
        for z,a in zip(latent,actions):
            c,s,n=surface(rng,"train")
            items.append(Experience(c,s,n,a,consequence(rng,z,a)))
        episodes.append(items)
    return latent,episodes


@torch.no_grad()
def health_snapshot(model,seed,step,device,reps=2):
    before=parameter_hash(model)
    model.eval();records=[]
    for rep in range(reps):
        episode_seed=seed*101+100_000*step+rep
        state=paired_state(model,episode_seed,16,device)
        feature_rng=random.Random(episode_seed*19+88)
        feature=surface(feature_rng,"novel")
        p=probe(model,state,[feature,feature]).clamp_min(1e-9)
        m=forecast_metrics(p)
        z=torch.tensor([0,1],device=device)[:,None].expand(2,2)
        actions=torch.arange(2,device=device)[None,:].expand(2,2)
        target=target_distribution(z,actions)
        conditional_ce=float(-(target*p.log()).sum(-1).mean())
        qf,qm=model.core._queries(state.H)
        rf=torch.einsum("bvk,bk->bv",state.F,qf)
        rm=torch.einsum("bvk,bk->bv",state.M,qm)
        records.append({"action_TV":sum(m["tv_action"])/2,
                        "history_TV":sum(m["tv_history"])/2,
                        "interaction_y0":m["interaction_y0"],
                        "BS":m["behavioral_separation_entropy"],
                        "conditional_CE":conditional_ce,
                        "CFA":MARGINAL_CE-conditional_ce,
                        "H_norm":float(state.H.norm(dim=(-2,-1)).mean()),
                        "F_norm":float(state.F.norm(dim=(-2,-1)).mean()),
                        "M_norm":float(state.M.norm(dim=(-2,-1)).mean()),
                        "q_F_norm":float(qf.norm(dim=-1).mean()),
                        "q_M_norm":float(qm.norm(dim=-1).mean()),
                        "r_F_norm":float(rf.norm(dim=-1).mean()),
                        "r_M_norm":float(rm.norm(dim=-1).mean()),
                        "r_M_difference":float((rm[0]-rm[1]).norm()),
                        "H_difference":float((state.H[0]-state.H[1]).norm())})
    # Direct trace audit on matched simulated evidence, no parameter update.
    state=model.initial_state(2,device)
    token=torch.full((2,),OUTCOME[0],device=device,dtype=torch.long)
    _,trace=model.core.step(state,outcome_token(token))
    trace_row={"consolidation_magnitude":float(trace["transfer"].norm(dim=(-2,-1)).mean()),
               "external_write_magnitude":float(trace["external_update"].norm(dim=(-2,-1)).mean())}
    assert before==parameter_hash(model),"evaluation modified parameters"
    keys=records[0]
    return {"step":step,"metrics":{key:sum(r[key] for r in records)/reps for key in keys},
            "replicates":records,"trace":trace_row,"marginal_CE":MARGINAL_CE,
            "parameter_sha256_before":before,"parameter_sha256_after":parameter_hash(model)}


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model=LifetimeModel().to(args.device)
    initial=parameter_hash(model)
    pretrain=(pretrain_evaluator(model,args.seed,args.device,args.evaluator_pretrain_steps)
              if args.arm in {"A2","A3"} else None)
    protected_head=head_hash(model)
    for p in model.action_head.parameters():p.requires_grad_(args.arm in {"A0","A1"})
    upstream=[p for name,p in model.named_parameters() if not name.startswith("action_head.")]
    downstream=list(model.action_head.parameters())
    opt=torch.optim.AdamW([{"params":upstream,"lr":.001},
                           {"params":downstream,"lr":.001 if args.arm in {"A0","A1"} else 0.0}],
                          weight_decay=.01)
    rng=random.Random(args.seed+15)
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[];start=time.monotonic();last=None
    for step in range(args.steps+1):
        if step in CHECKPOINTS:
            snap=health_snapshot(model,args.seed,step,args.device)
            snap.update({"phase":arm_phase(args.arm,step),"head_hash":head_hash(model),
                         "preceding_update":last,"elapsed_s":time.monotonic()-start})
            if args.arm=="A2":assert snap["head_hash"]==protected_head
            if args.arm=="A3" and step<=500:assert snap["head_hash"]==protected_head
            rows.append(snap)
            torch.save({"seed":args.seed,"step":step,"arm":args.arm,
                        "model":model.state_dict()},args.out/f"checkpoint_{step}.pt")
            print(json.dumps({"arm":args.arm,"seed":args.seed,"step":step,
                              "health":snap["metrics"],"elapsed_s":snap["elapsed_s"]}),flush=True)
        if step==args.steps:break
        model.train()
        ratio=evaluator_lr_ratio(args.arm,step)
        opt.param_groups[1]["lr"]=(.001 if args.arm in {"A0","A1"} else .001*ratio)
        for p in downstream:p.requires_grad_(opt.param_groups[1]["lr"]>0)
        opt.zero_grad(set_to_none=True)
        length=rng.choice((4,8,16))
        z,episodes=sampled_batch(rng,step,16,length)
        latent=torch.tensor(z,device=args.device)
        state=model.initial_state(16,args.device)
        losses=[];penalties=[]
        for items in episodes:
            state,loss=training_event(model,state,items,latent,args.arm)
            losses.append(loss);penalties.append(state.H.square().mean())
        ce=torch.stack(losses).mean()
        total=ce+.001*torch.stack(penalties).mean()
        if not torch.isfinite(total):raise RuntimeError("nonfinite training loss")
        total.backward()
        grads=group_gradients(model)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        opt.step()
        last={"training_CE":float(ce.detach()),"total_loss":float(total.detach()),
              "gradients_preclip":grads,"head_lr_ratio":opt.param_groups[1]["lr"]/.001}
    summary={"arm":args.arm,"seed":args.seed,"steps":args.steps,
             "training_device":args.device,
             "initial_model_sha256":initial,"protected_evaluator_sha256":protected_head,
             "pretrain":pretrain,"snapshots":rows,"elapsed_s":time.monotonic()-start,
             "architecture":"Stage2C1 LifetimeModel unchanged",
             "training_objective":"COUNTERFACTUAL_TRAINING paired target CE" if args.arm=="A1"
                                  else "observed consequence CE",
             "privileged_signal_only_in_training":args.arm in {"A1","A2","A3"},
             "no_policy_or_memory_supervision":True}
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({"COMPLETED":{"arm":args.arm,"seed":args.seed,
                                   "elapsed_s":summary["elapsed_s"]}}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--arm",choices=("A0","A1","A2","A3"),required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--evaluator-pretrain-steps",type=int,default=1000)
    p.add_argument("--out",type=Path,required=True)
    train(p.parse_args())
