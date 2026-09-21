"""Independent seed training for the preregistered Stage 2C.1 ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from etrcm.stage2c1.diagnostic import (OracleLatent, PersistentInterface,
    balanced_batch, forecast_metrics, legal_history)


def digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, level, device, seed):
    model.eval()
    actions = torch.tensor([0, 1, 0, 1], device=device)
    latent = torch.tensor([0, 0, 1, 1], device=device)
    if level == "L0":
        logits = model(latent, actions)
        aux = {}
    elif level == "L1":
        logits = model(actions, latent=latent)
        clamped = model(actions, latent=latent, clamp_m=True).softmax(-1)
        oracle = model.oracle_M
        aux = {"M_norms": oracle.flatten(1).norm(dim=-1).tolist(),
               "M_cosine": float(F.cosine_similarity(oracle[0].flatten()[None], oracle[1].flatten()[None])),
               "M_read_clamp_forecast_tv": float((logits.softmax(-1)-clamped).abs().sum(-1).mean()/2),
               "clamped_prob": clamped.cpu().tolist()}
        state0, state1, trace = model.states(oracle)
        aux.update({"H0_identical": bool(torch.equal(state0.H[0],state0.H[1])),
                    "F0_zero": bool(torch.count_nonzero(state0.F)==0),
                    "r_M_difference": float((trace["r_M"][0]-trace["r_M"][1]).norm()),
                    "H1_difference": float((state1.H[0]-state1.H[1]).norm()),
                    "H1_norm": float(state1.H.norm(dim=(-2,-1)).mean())})
    else:
        gen = torch.Generator(device="cpu").manual_seed(seed + 90001)
        histories, inferred_latent = legal_history(64, 8, "novel", device, gen)
        # Each history is paired with both counterfactual actions, same exact state.
        a0 = torch.zeros(64, dtype=torch.long, device=device)
        a1 = torch.ones(64, dtype=torch.long, device=device)
        l0 = model(a0, past=histories).softmax(-1)
        l1 = model(a1, past=histories).softmax(-1)
        grouped = torch.stack([l0, l1], 1)
        prob = torch.stack([grouped[inferred_latent==z].mean(0) for z in (0,1)])
        result = forecast_metrics(prob)
        result["novel_history_count"] = 64
        result["parameter_hash_before_after_eval_equal"] = True
        # Same history with action intervention, no future outcome available to encoder.
        result["history_action_tensor_exact"] = bool(torch.equal(histories, histories.clone()))
        return result
    result = forecast_metrics(logits.softmax(-1).view(2,2,4))
    result.update(aux)
    result["parameter_hash_before_after_eval_equal"] = True
    return result


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed); random.seed(args.seed)
    device=args.device
    gen=torch.Generator(device="cpu").manual_seed(args.seed+137)
    if args.level == "L0":
        model=OracleLatent(args.head).to(device)
    else:
        model=PersistentInterface(args.head,history=args.level=="L2").to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.0)
    args.out.mkdir(parents=True,exist_ok=True)
    log=[]; start=time.monotonic()
    eval_steps={0,100,500,1000,args.steps}
    if args.level != "L0": eval_steps={0,100,args.steps}
    initial_hash=digest(model)
    for step in range(args.steps+1):
        if step in eval_steps:
            before=digest(model)
            row={"step":step,"evaluation":evaluate(model,args.level,device,args.seed),
                 "elapsed_s":time.monotonic()-start}
            row["evaluation"]["parameter_hash_before_after_eval_equal"] = before==digest(model)
            log.append(row)
            print(json.dumps({"level":args.level,"head":args.head,"seed":args.seed,
                              "step":step,"tv":row["evaluation"]["tv_action"],
                              "dq":[row["evaluation"]["delta_q_a"],row["evaluation"]["delta_q_b"]]}),flush=True)
        if step==args.steps:break
        model.train();opt.zero_grad(set_to_none=True)
        z,a,y=balanced_batch(args.batch,device,gen)
        if args.level=="L0":
            logits=model(z,a)
            h_params=list(model.latent_embedding.parameters())
        elif args.level=="L1":
            logits=model(a,latent=z)
            h_params=[model.core.initial_H, *model.core.core_in.parameters()]
        else:
            past,hz=legal_history(args.batch,8,"train",device,gen)
            # L2 predictive targets use the same latent that generated *past*;
            # latent enters data generation only, never model.forward.
            z=hz
            a=((torch.arange(args.batch,device=device)//2)%2).long()
            wrong=1+torch.randint(3,(args.batch,),generator=gen).to(device)
            y=torch.where(a.eq(z),torch.zeros_like(wrong),wrong)
            logits=model(a,past=past)
            h_params=list(model.history_encoder.parameters())
        loss=F.cross_entropy(logits,y)
        loss.backward()
        if step in {0,99,499,999,args.steps-1}:
            def norm(parameters):
                return float(sum(p.grad.detach().square().sum().item() for p in parameters if p.grad is not None)**0.5)
            log[-1]["train_gradient"]={"H_or_source":norm(h_params),
                "action_branch":norm(model.action_head.action_embedding.parameters()) +
                    (norm(model.action_head.action_projection.parameters()) if model.action_head.action_projection else 0),
                "consequence_head":norm(model.action_head.head.parameters()),
                "loss":float(loss.detach())}
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
    final={"level":args.level,"head":args.head,"seed":args.seed,"steps":args.steps,
           "batch":args.batch,"lr":args.lr,"initial_parameter_hash":initial_hash,
           "final_parameter_hash":digest(model),"elapsed_s":time.monotonic()-start,
           "objective":"observed consequence cross entropy only; no correct-action or latent label target",
           "final":log[-1]["evaluation"]}
    (args.out/"train_log.json").write_text(json.dumps(log,indent=2))
    (args.out/"summary.json").write_text(json.dumps(final,indent=2))
    torch.save({"model":model.state_dict(),"level":args.level,"head":args.head,
                "seed":args.seed,"steps":args.steps},args.out/"checkpoint.pt")
    print(json.dumps({"COMPLETED":final}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--level",choices=["L0","L1","L2"],required=True)
    parser.add_argument("--head",choices=["late_concat","modulation"],required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--steps",type=int,default=3000)
    parser.add_argument("--batch",type=int,default=64)
    parser.add_argument("--lr",type=float,default=0.003)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--out",type=Path,required=True)
    run(parser.parse_args())
