"""Frozen failed-L3 P0/P1 pathway audit; never updates checkpoint weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F

from etrcm.stage1_1.model import LearnedState
from etrcm.stage2c1.diagnostic import forecast_metrics
from experiments.stage2c1_lifetime import LifetimeModel, paired_state, bs


ROOT=Path(__file__).resolve().parents[1]


def parameter_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_l3(seed,device,development=False):
    if development:
        path=ROOT/f"results/stage2c2/frozen_l3/development/{seed}/checkpoint.pt"
    else:
        path=ROOT/f"results/stage2c1/raw/lifetime/{seed}/checkpoint.pt"
    model=LifetimeModel().to(device)
    data=torch.load(path,map_location=device,weights_only=True)
    assert data["seed"]==seed and data["steps"]==500
    model.load_state_dict(data["model"])
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,path


@torch.no_grad()
def native_baseline(model,seed,device):
    rows=[]
    for rep in range(4):
        episode_seed=seed*101+rep
        state=paired_state(model,episode_seed,16,device)
        qf,qm=model.core._queries(state.H)
        rf=torch.einsum("bvk,bk->bv",state.F,qf)
        rm=torch.einsum("bvk,bk->bv",state.M,qm)
        metrics=bs(model,state,episode_seed)
        rows.append({"replicate":rep,"episode_seed":episode_seed,
                     "metrics":metrics,
                     "H_norm":state.H.norm(dim=(-2,-1)).cpu().tolist(),
                     "F_norm":state.F.norm(dim=(-2,-1)).cpu().tolist(),
                     "M_norm":state.M.norm(dim=(-2,-1)).cpu().tolist(),
                     "q_F":qf.cpu().tolist(),"q_M":qm.cpu().tolist(),
                     "r_F":rf.cpu().tolist(),"r_M":rm.cpu().tolist(),
                     "H":state.H.cpu().tolist(),"F":state.F.cpu().tolist(),
                     "M":state.M.cpu().tolist()})
    norms={name:[x for row in rows for x in row[name+"_norm"]] for name in ("H","F","M")}
    norms["r_M"]=[float(torch.tensor(v).norm()) for row in rows for v in row["r_M"]]
    return {"rows":rows,"norms":{name:{"median":statistics.median(v),
                                  "min":min(v),"max":max(v)} for name,v in norms.items()},
            "reproduction":{"TV_action":statistics.mean(statistics.mean(row["metrics"]["tv_action"]) for row in rows),
                            "TV_history":statistics.mean(statistics.mean(row["metrics"]["tv_history"]) for row in rows),
                            "interaction_y0":statistics.mean(row["metrics"]["interaction_y0"] for row in rows),
                            "BS":statistics.mean(row["metrics"]["behavioral_separation_entropy"] for row in rows)}}


def consequence_batch(generator,batch,device):
    # Exactly balanced four z/action cells and only sampled observed outcomes.
    pairs=torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.long).repeat(batch//4,1)
    order=torch.randperm(batch,generator=generator)
    latent,action=pairs[order].unbind(-1)
    wrong=torch.randint(1,4,(batch,),generator=generator)
    target=torch.where(latent.eq(action),torch.zeros_like(wrong),wrong)
    return latent.to(device),action.to(device),target.to(device)


@torch.no_grad()
def heldout_ce(model,states,seed,device):
    generator=torch.Generator().manual_seed(seed*3+920_000)
    z,a,y=consequence_batch(generator,4096,device)
    logits=model.action_head(states[z],a)
    return float(F.cross_entropy(logits,y))


def fit_oracle_h(model,seed,norm,device,steps=1000,restarts=4):
    generator=torch.Generator().manual_seed(seed*3+810_000)
    best=None
    for restart in range(restarts):
        init=torch.randn(2,model.core.config.hidden_dim,generator=generator).to(device)
        raw=torch.nn.Parameter(init)
        opt=torch.optim.Adam([raw],lr=.03)
        trace=[]
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            states=norm*F.normalize(raw,dim=-1)
            z,a,y=consequence_batch(generator,128,device)
            logits=model.action_head(states[z],a)
            loss=F.cross_entropy(logits,y)
            loss.backward();opt.step()
            if step in (0,99,499,steps-1):
                trace.append({"step":step+1,"CE":float(loss.detach())})
        with torch.no_grad():
            states=(norm*F.normalize(raw,dim=-1)).detach()
            fit_ce=heldout_ce(model,states,seed+restart+20_000,device)
        result={"states":states,"fit_ce":fit_ce,"trace":trace,"restart":restart}
        if best is None or fit_ce<best["fit_ce"]:best=result
    states=best["states"]
    with torch.no_grad():
        z=torch.tensor([0,0,1,1],device=device)
        a=torch.tensor([0,1,0,1],device=device)
        prob=model.action_head(states[z],a).softmax(-1).view(2,2,4)
        metrics=forecast_metrics(prob)
        formal_ce=heldout_ce(model,states,seed+40_000,device)
    return {"state_tensors":states.cpu(),"norm_target":norm,
            "state_norms":states.norm(dim=-1).cpu().tolist(),
            "fit_CE":best["fit_ce"],"formal_heldout_CE":formal_ce,
            "selected_restart":best["restart"],"trace":best["trace"],"metrics":metrics}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--development",action="store_true")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    model,source=load_l3(args.seed,args.device,args.development)
    before=parameter_hash(model)
    native=native_baseline(model,args.seed,args.device)
    oracle=fit_oracle_h(model,args.seed,native["norms"]["H"]["median"],args.device,args.steps)
    after=parameter_hash(model)
    assert before==after,"Frozen checkpoint parameters changed"
    args.out.mkdir(parents=True,exist_ok=True)
    torch.save(oracle.pop("state_tensors"),args.out/"oracle_H.pt")
    manifest={"seed":args.seed,"development":args.development,"checkpoint":str(source),
              "checkpoint_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "parameter_sha256_before":before,"parameter_sha256_after":after,
              "weights_frozen":True,"only_fit_variables":"two norm-matched H state tensors",
              "fit_split_seed":args.seed*3+810_000,
              "restart_selection_split_seed_base":args.seed*3+980_000,
              "formal_heldout_split_seed":args.seed*3+1_040_000}
    (args.out/"native.json").write_text(json.dumps(native,indent=2))
    (args.out/"oracle_h.json").write_text(json.dumps(oracle,indent=2))
    (args.out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"seed":args.seed,"native_tv":native["reproduction"]["TV_action"],
                      "native_BS":native["reproduction"]["BS"],
                      "oracle_tv":oracle["metrics"]["tv_action"],
                      "oracle_interaction":oracle["metrics"]["interaction_y0"],
                      "oracle_BS":oracle["metrics"]["behavioral_separation_entropy"],
                      "heldout_CE":oracle["formal_heldout_CE"],"weights_frozen":before==after}),flush=True)


if __name__=="__main__":main()
