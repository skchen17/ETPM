"""Independent sampled-outcome CE audit for every saved training checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from etrcm.stage2c.world import OUTCOME,outcome_token,surface
from etrcm.stage2c3.protocol import CHECKPOINTS,MARGINAL_CE,target_distribution
from experiments.stage2c1_lifetime import paired_state,probe
from experiments.stage2c2_pathway import parameter_hash
from experiments.stage2c3_evaluate import load
from experiments.stage2c3_run_matrix import ARM_DIR


@torch.no_grad()
def audit_one(model,seed,step,device):
    before=parameter_hash(model)
    sampled=[];exact=[];writes=[];transfers=[]
    for rep in range(4):
        episode_seed=seed*157+step*100_000+rep+6_000_000
        state=paired_state(model,episode_seed,16,device)
        feature=surface(random.Random(episode_seed+51),"novel")
        p=probe(model,state,[feature,feature]).clamp_min(1e-9)
        _,trace=model.core.step(state,outcome_token(torch.full((2,),OUTCOME[0],
                                                        device=device,dtype=torch.long)))
        writes.append(float(trace["external_update"].norm(dim=(-2,-1)).mean()))
        transfers.append(float(trace["transfer"].norm(dim=(-2,-1)).mean()))
        z=torch.tensor([[0,0],[1,1]],device=device)
        a=torch.tensor([[0,1],[0,1]],device=device)
        target=target_distribution(z,a)
        exact.append(float(-(target*p.log()).sum(-1).mean()))
        rng=random.Random(episode_seed+73)
        for latent in (0,1):
            for action in (0,1):
                y=torch.tensor([0 if latent==action else rng.randrange(1,4)
                                for _ in range(256)],device=device)
                sampled.append(float(-p[latent,action,y].log().mean()))
    after=parameter_hash(model)
    assert before==after
    return {"step":step,"heldout_observed_CE":sum(sampled)/len(sampled),
            "counterfactual_exact_CE":sum(exact)/len(exact),
            "post_N16_external_write_norm":sum(writes)/len(writes),
            "post_N16_consolidation_norm":sum(transfers)/len(transfers),
            "marginal_CE":MARGINAL_CE,"CFA_heldout":MARGINAL_CE-sum(sampled)/len(sampled),
            "parameter_sha256_before":before,"parameter_sha256_after":after,
            "independent_outcome_samples_per_z_action_replicate":256,
            "paired_state_replicates":4,"surface_split":"novel"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm",choices=tuple(ARM_DIR),required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--split",choices=("development","formal"),required=True)
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args();torch.set_num_threads(1)
    root=Path("results/stage2c3")
    source=root/ARM_DIR[args.arm]/args.split/str(args.seed)
    rows=[]
    for step in CHECKPOINTS:
        model,data=load(source/f"checkpoint_{step}.pt",args.device)
        assert data["seed"]==args.seed and data["arm"]==args.arm and data["step"]==step
        rows.append(audit_one(model,args.seed,step,args.device))
    out=root/"learning_curves"/"heldout_ce"/args.split/args.arm/str(args.seed)/"summary.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"arm":args.arm,"seed":args.seed,"rows":rows},indent=2))
    print(json.dumps({"arm":args.arm,"seed":args.seed,"heldout_CE_1000":rows[-1]["heldout_observed_CE"]}),flush=True)


if __name__=="__main__":main()
