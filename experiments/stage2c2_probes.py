"""Non-causal decodability probes for native frozen L3 H/F/M."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from etrcm.stage2c.world import paired_histories
from experiments.stage2c1_lifetime import play
from experiments.stage2c2_pathway import load_l3, parameter_hash


@torch.no_grad()
def collect(model,seed,split,pairs,device):
    histories=[paired_histories(seed*100_000+i,16,split=split) for i in range(pairs)]
    state=model.initial_state(2*pairs,device)
    for tick in range(16):
        items=[item for pair in histories for item in (pair[0][tick],pair[1][tick])]
        state,_=play(model,state,items)
    h=state.H.flatten(1).detach()
    f=state.F.flatten(1).detach()
    m=state.M.flatten(1).detach()
    labels=torch.tensor([0,1]*pairs,device=device)
    return {"H":h,"F":f,"M":m,"FM":torch.cat([f,m],-1)},labels


def fit_probe(x_train,y_train,x_test,y_test,kind,seed):
    torch.manual_seed(seed)
    mu=x_train.mean(0,keepdim=True)
    std=x_train.std(0,unbiased=False,keepdim=True).clamp_min(.01)
    a=((x_train-mu)/std).clamp(-10,10)
    b=((x_test-mu)/std).clamp(-10,10)
    dim=a.shape[-1]
    net=(nn.Linear(dim,2) if kind=="linear" else
         nn.Sequential(nn.Linear(dim,16),nn.Tanh(),nn.Linear(16,2))).to(a)
    opt=torch.optim.AdamW(net.parameters(),lr=.01,weight_decay=.01)
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        logits=net(a)
        loss=F.cross_entropy(logits,y_train)
        loss.backward();opt.step()
    with torch.no_grad():
        return {"train_accuracy":float((net(a).argmax(-1)==y_train).float().mean()),
                "test_accuracy":float((net(b).argmax(-1)==y_test).float().mean()),
                "test_CE":float(F.cross_entropy(net(b),y_test))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--pairs",type=int,default=128)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    model,path=load_l3(args.seed,args.device)
    before=parameter_hash(model)
    train,ytrain=collect(model,args.seed*2+17,"train",args.pairs,args.device)
    test,ytest=collect(model,args.seed*2+17017,"novel",args.pairs,args.device)
    # Shuffle after extraction; the probes see vectors, never pair position or z.
    g=torch.Generator(device="cpu").manual_seed(args.seed+401)
    tr_order=torch.randperm(len(ytrain),generator=g).to(args.device)
    te_order=torch.randperm(len(ytest),generator=g).to(args.device)
    ytrain=ytrain[tr_order];ytest=ytest[te_order]
    result={}
    for i,name in enumerate(("H","F","M","FM")):
        a=train[name][tr_order].detach();b=test[name][te_order].detach()
        result[name]={kind:fit_probe(a,ytrain,b,ytest,kind,args.seed*37+i*3+j)
                      for j,kind in enumerate(("linear","small_mlp"))}
    after=parameter_hash(model)
    assert before==after
    args.out.mkdir(parents=True,exist_ok=True)
    torch.save({"train":{k:v.cpu() for k,v in train.items()},
                "test":{k:v.cpu() for k,v in test.items()},
                "train_labels":torch.tensor([0,1]*args.pairs),
                "test_labels":torch.tensor([0,1]*args.pairs)},args.out/"states.pt")
    summary={"seed":args.seed,"checkpoint":str(path),"train_pairs":args.pairs,
             "test_pairs":args.pairs,"train_surface_split":"train parity-even",
             "test_surface_split":"novel parity-odd","episode_sets_disjoint":True,
             "latent_labels_probe_only":True,"L3_parameter_sha256_before":before,
             "L3_parameter_sha256_after":after,"results":result}
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({"seed":args.seed,"accuracy":{name:{kind:v["test_accuracy"] for kind,v in d.items()}
                                                   for name,d in result.items()}}),flush=True)


if __name__=="__main__":main()
