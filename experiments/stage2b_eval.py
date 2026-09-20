"""Per-checkpoint Stage 2B behavioral, intervention and stability evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from etrcm.stage1_4.model import Stage14Config
from etrcm.stage2a.language import LanguageETRCM, WordTokenizer, generate, lesion
from etrcm.stage2b.data import COLORS,FAMILIES, Association, counterfactual_pair, make_association
from etrcm.stage2b.interventions import random_norm_matched, replace_memory
from etrcm.stage2b.model import ContextualLanguageETRCM
from stage2b_train import ARMS, build_model


def load(path, device):
    ckpt=torch.load(path,map_location="cpu",weights_only=False)
    config=Stage14Config(**ckpt["config"])
    model=build_model(config,ckpt["arm"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model,WordTokenizer(ckpt["tokenizer"]),ckpt


def split_text(ex, tok):
    before,rest=ex.prefix.rsplit("question :",1)
    return tok.encode(before,bos=True),tok.encode("question :"+rest)


def batch_state(model, sequences, device):
    if len({len(s) for s in sequences})!=1:
        raise ValueError("batch sequences must have equal token length")
    state=model.initial_state(len(sequences),device=device)
    tokens=torch.tensor(sequences,device=device)
    last=None
    for t in range(tokens.shape[1]):
        state,_,last=model.step_token(state,tokens[:,t],source="external")
    return state,last


def run_query(model,state,queries,device,ticks=0):
    if len({len(s) for s in queries})!=1:
        raise ValueError("query sequences must have equal length")
    tokens=torch.tensor(queries,device=device)
    diag=None
    for t in range(tokens.shape[1]):
        state,_,diag=model.step_token(state,tokens[:,t],source="external")
    for _ in range(ticks):
        state,_,diag=model.null_tick(state)
    return state,model.logits(state),diag


def result_rows(examples,model,tok,state,logits,diag,condition):
    rows=[]
    for i,ex in enumerate(examples):
        target=tok.encode(ex.answer)[0]
        candidates=[tok.encode(c)[0] for c in ex.candidates]
        predicted=tok.vocabulary[int(logits[i].argmax())]
        candidate=ex.candidates[int(logits[i,candidates].argmax())]
        rows.append({"family":ex.family,"gap":ex.gap,"split":ex.split,"condition":condition,
                     "answer":ex.answer,"candidate":candidate,"candidate_correct":candidate==ex.answer,
                     "raw_predicted":predicted,"raw_exact":predicted==ex.answer,
                     "answer_ce":float(F.cross_entropy(logits[i:i+1],torch.tensor([target],device=logits.device))),
                     "answer_probability":float(logits[i].softmax(-1)[target]),
                     "H":float(state.H[i].norm()),"F":float(state.F[i].norm()),"M":float(state.M[i].norm()),
                     "r_F":float(diag["r_F"][i].norm()),"r_M":float(diag["r_M"][i].norm()),
                     "read":float(diag["read"][i].norm()),
                     "q_F":diag["q_F"][i].tolist(),"q_M":diag["q_M"][i].tolist(),
                     "r_F_vector":diag["r_F"][i].tolist(),"r_M_vector":diag["r_M"][i].tolist(),
                     "prefix":ex.prefix})
    return rows


@torch.no_grad()
def score_examples(model,tok,examples,device,condition="full",ticks=0):
    groups=defaultdict(list)
    for ex in examples:
        pre,query=split_text(ex,tok)
        groups[(len(pre),len(query))].append((ex,pre,query))
    rows=[]
    for group in groups.values():
        exs=[x[0] for x in group]
        state,_=batch_state(model,[x[1] for x in group],device)
        if condition in {"H","F","M","FM"}:
            state=lesion(state,condition,model)
        elif condition=="zero":
            state=replace_memory(state,F=torch.zeros_like(state.F),M=torch.zeros_like(state.M))
        elif condition=="random":
            state=random_norm_matched(state,torch.Generator(device=device).manual_seed(993))
        elif condition=="shuffle":
            if len(group)<2:
                continue
            order=torch.arange(len(group),device=device).roll(1)
            state=replace_memory(state,F=state.F[order],M=state.M[order])
        next_state,logits,diag=run_query(model,state,[x[2] for x in group],device,ticks)
        rows.extend(result_rows(exs,model,tok,next_state,logits,diag,condition))
    return rows


def aggregate(rows):
    if not rows: return {"n":0}
    n=len(rows)
    return {"n":n,"candidate_accuracy":sum(x["candidate_correct"] for x in rows)/n,
            "raw_accuracy":sum(x["raw_exact"] for x in rows)/n,
            "answer_ce":sum(x["answer_ce"] for x in rows)/n,
            "H":sum(x["H"] for x in rows)/n,"F":sum(x["F"] for x in rows)/n,
            "M":sum(x["M"] for x in rows)/n,"r_F":sum(x["r_F"] for x in rows)/n,
            "r_M":sum(x["r_M"] for x in rows)/n}


@torch.no_grad()
def paired_counterfactual(model,tok,pairs,device):
    output=[]
    for pair_id,pair in enumerate(pairs):
        pre=[split_text(ex,tok)[0] for ex in pair]
        query=[split_text(ex,tok)[1] for ex in pair]
        state,_=batch_state(model,pre,device)
        for condition in ("full","F_swap","M_swap","FM_swap","zero","random","H_reset","H_reset_FM_zero"):
            s=state.clone()
            if condition in {"F_swap","FM_swap"}:
                s=replace_memory(s,F=s.F.flip(0))
            if condition in {"M_swap","FM_swap"}:
                s=replace_memory(s,M=s.M.flip(0))
            if condition in {"zero","H_reset_FM_zero"}:
                s=replace_memory(s,F=torch.zeros_like(s.F),M=torch.zeros_like(s.M))
            if condition=="random":
                s=random_norm_matched(s,torch.Generator(device=device).manual_seed(10000+pair_id))
            if condition in {"H_reset","H_reset_FM_zero"}:
                s=model.reset_active(s)
            final,logits,diag=run_query(model,s,query,device)
            rows=result_rows(pair,model,tok,final,logits,diag,condition)
            output.append({"pair_id":pair_id,"condition":condition,"joint_correct":all(x["candidate_correct"] for x in rows),
                           "mean_answer_ce":sum(x["answer_ce"] for x in rows)/2,"episodes":rows})
    return output


@torch.no_grad()
def generation_eval(model,tok,examples):
    rows=[]
    for ex in examples:
        for decoder in ("greedy","temp_0.7","top_k_8"):
            kwargs=({"greedy":True} if decoder=="greedy" else
                    {"temperature":0.7,"seed":8300+len(rows)} if decoder=="temp_0.7" else
                    {"temperature":1.0,"top_k":8,"seed":8300+len(rows)})
            output,state,trace=generate(model,tok,ex.prefix,max_new_tokens=6,**kwargs)
            semantic=re.findall(r"[A-Za-z]+",output.lower())
            first=semantic[0] if semantic else ""
            compact=" ".join(semantic)
            words=output.split()
            rows.append({"decoder":decoder,"family":ex.family,"gap":ex.gap,"target":ex.answer,
                         "prompt":ex.prefix,
                         "output":output,"first_semantic":first,"semantic_correct":first==ex.answer,
                         "whole_answer_exact":compact==ex.answer,"EOS":len(trace)<6,
                         "punctuation_only":not bool(semantic),
                         "repetition":sum(a==b for a,b in zip(words,words[1:]))/max(1,len(words)-1),
                         "self_output_external_writes":sum(x["write"] for x in trace),
                         "final_tau":state.tau})
    return rows


@torch.no_grad()
def null_sweep(model,tok,examples,device):
    output=[]
    for ex in examples:
        pre,query=split_text(ex,tok)
        state,_=batch_state(model,[pre],device)
        state,_,diag=run_query(model,state,[query],device)
        for tick in range(17):
            if tick in {0,1,2,4,8,16}:
                logits=model.logits(state)
                row=result_rows([ex],model,tok,state,logits,diag,f"K{tick}")[0]
                row["K"]=tick;output.append(row)
            if tick<16:
                state,_,diag=model.null_tick(state)
    return output


@torch.no_grad()
def representation_diagnostic(model,tok,device):
    rng=random.Random(5511); samples=[]
    for family in ("attribute","location"):
        for _ in range(32):
            ex=make_association(rng,family,32,"train")
            state=model.initial_state(1,device=device)
            ids=tok.encode(ex.prefix,bos=True)
            value_id=tok.encode(ex.answer)[0]
            selected=None;last=None
            for token in ids:
                state,_,last=model.step_token(state,torch.tensor([token],device=device))
                if token==value_id and selected is None and "context_key" in last:
                    selected=(last["context_key"][0].clone(),last["context_value"][0].clone())
            if selected is not None:
                samples.append({"family":family,"entity":ex.entity,"value":ex.answer,"key":selected[0].tolist(),
                                "value_vector":selected[1].tolist(),"q":last["q_F"][0].tolist()})
    key_norm=[math.sqrt(sum(y*y for y in x["key"])) for x in samples]
    value_norm=[math.sqrt(sum(y*y for y in x["value_vector"])) for x in samples]
    qk=[sum(a*b for a,b in zip(x["q"],x["key"])) for x in samples]
    similarities=defaultdict(list)
    for i,left in enumerate(samples):
        for right in samples[i+1:]:
            similarity=sum(a*b for a,b in zip(left["key"],right["key"]))
            category=("cross_relation" if left["family"]!=right["family"] else
                      "same_entity_diff_value" if left["entity"]==right["entity"] and left["value"]!=right["value"] else
                      "same_value_diff_entity" if left["value"]==right["value"] and left["entity"]!=right["entity"] else
                      "within_relation_other")
            similarities[category].append(similarity)
    return {"n":len(samples),"key_norm_mean":sum(key_norm)/max(1,len(key_norm)),
            "value_norm_mean":sum(value_norm)/max(1,len(value_norm)),
            "q_dot_k_mean":sum(qk)/max(1,len(qk)),
            "key_similarity":{name:{"n":len(values),"mean":sum(values)/len(values)}
                              for name,values in similarities.items()},"samples":samples}


@torch.no_grad()
def stability(model,tok,device):
    rng=random.Random(2191)
    state=model.initial_state(1,device=device)
    for token in tok.encode("alice has the red key .",bos=True):
        state,_,_=model.step_token(state,torch.tensor([token],device=device),source="external")
    out=[];first100=None;first1000=None;first_bad=None
    for t in range(5000):
        before=state
        if t%5 in (0,1,2):
            text=make_association(rng,"attribute",16,"train").text
            ids=tok.encode(text)
            token=ids[t%len(ids)]
            state,_,diag=model.step_token(state,torch.tensor([token],device=device),source="external")
            kind="external"
        elif t%5==3:
            state,_,diag=model.null_tick(state);kind="null"
        else:
            token=int(model.logits(state)[0].argmax())
            state,_,diag=model.step_token(state,torch.tensor([token],device=device),source="self_output")
            kind="self_output"
        h=float(state.H.norm());finite=bool(all(torch.isfinite(z).all() for z in (state.H,state.F,state.M)))
        if first100 is None and h>100:first100=t+1
        if first1000 is None and h>1000:first1000=t+1
        if first_bad is None and not finite:first_bad=t+1
        out.append({"tick":t+1,"kind":kind,"H":h,"F":float(state.F.norm()),"M":float(state.M.norm()),
                    "read":float(diag["read"].norm()),"H_delta":float((state.H-before.H).norm()),
                    "external_write":bool(diag["external_write_flag"][0]),"finite":finite})
        if finite and t+1 in (100,500,1000,5000):
            probe=state.clone()
            for token in tok.encode("question : what key does alice have ? answer :"):
                probe,_,_=model.step_token(probe,torch.tensor([token],device=device),source="external")
            logits=model.logits(probe)[0]
            target=tok.encode("red")[0]
            candidate_ids=[tok.encode(color)[0] for color in COLORS]
            predicted=COLORS[int(logits[candidate_ids].argmax())]
            out[-1].update({"probe_answer_ce":float(F.cross_entropy(logits[None],torch.tensor([target],device=device))),
                            "probe_candidate":predicted,"probe_correct":predicted=="red"})
        if not finite:break
    snapshots={str(n):out[n-1] if len(out)>=n else None for n in (100,500,1000,5000)}
    return {"first_H_gt_100":first100,"first_H_gt_1000":first1000,"first_nonfinite":first_bad,
            "completed":len(out),"snapshots":snapshots,"trajectory":out}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--evaluation-n",type=int,default=8)
    p.add_argument("--extended",action="store_true")
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    model,tok,checkpoint=load(args.checkpoint,args.device)
    rng=random.Random(7711)
    primary=[];long=[]
    for split in ("train","ood"):
        for gap in (32,64,128):
            for family in FAMILIES:
                primary += [make_association(rng,family,gap,split,entity_count=8) for _ in range(args.evaluation_n)]
        for gap in (256,512,1024):
            for family in ("attribute","interference"):
                long += [make_association(rng,family,gap,split,entity_count=8) for _ in range(args.evaluation_n)]
    all_rows=score_examples(model,tok,primary+long,args.device)
    groups=defaultdict(list)
    for row in all_rows:groups[f"{row['split']}/{row['gap']}/{row['family']}"].append(row)
    summary_by_cell={key:aggregate(rows) for key,rows in groups.items()}
    id_rows=[x for x in all_rows if x["split"]=="train" and x["gap"]<=128]
    ood_rows=[x for x in all_rows if x["split"]=="ood" and x["gap"]<=128]
    long_rows=[x for x in all_rows if x["gap"]>128]
    output={"arm":checkpoint["arm"],"seed":checkpoint["seed"],"steps":checkpoint["steps"],
            "size":checkpoint["config"]["hidden_dim"],"main":{
                "in_distribution":aggregate(id_rows),"lexical_ood":aggregate(ood_rows),
                "long_extrapolation":aggregate(long_rows),"by_cell":summary_by_cell},
            "main_records":all_rows}
    if args.extended:
        pairs=[counterfactual_pair(rng,64) for _ in range(32)]
        paired=paired_counterfactual(model,tok,pairs,args.device)
        pair_groups=defaultdict(list)
        for row in paired:pair_groups[row["condition"]].append(row)
        output["counterfactual"]={key:{"n":len(rows),"joint_accuracy":sum(x["joint_correct"] for x in rows)/len(rows),
                                         "mean_answer_ce":sum(x["mean_answer_ce"] for x in rows)/len(rows),
                                         "candidate_accuracy":sum(sum(z["candidate_correct"] for z in x["episodes"])
                                                                  for x in rows)/(2*len(rows))}
                                   for key,rows in pair_groups.items()}
        output["counterfactual_records"]=paired
        intervention_examples=[make_association(rng,"attribute",64,split,entity_count=8)
                               for split in ("train","ood") for _ in range(12)]
        intervention={name:score_examples(model,tok,intervention_examples,args.device,condition=name)
                      for name in ("full","H","F","M","FM","zero","random","shuffle")}
        output["interventions"]={name:aggregate(rows) for name,rows in intervention.items()}
        output["intervention_records"]=intervention
        output["null_records"]=null_sweep(model,tok,intervention_examples[:8],args.device)
        output["null"]={f"K{k}":aggregate([x for x in output["null_records"] if x["K"]==k])
                        for k in (0,1,2,4,8,16)}
        output["generations"]=generation_eval(model,tok,intervention_examples[:12])
        if checkpoint["arm"] in {"E1","E2"}:
            output["representation"]=representation_diagnostic(model,tok,args.device)
            output["stability"]=stability(model,tok,args.device)
    (args.out/"evaluation.json").write_text(json.dumps(output,indent=2))
    (args.out/"summary.json").write_text(json.dumps({k:v for k,v in output.items() if not k.endswith("records")
                                                      and k not in {"generations","representation","stability"}},indent=2))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()}
    (args.out/"hashes.json").write_text(json.dumps(hashes,indent=2))
    print(json.dumps({"arm":checkpoint["arm"],"seed":checkpoint["seed"],"size":output["size"],
                      "ID":output["main"]["in_distribution"],"OOD":output["main"]["lexical_ood"],
                      "long":output["main"]["long_extrapolation"],
                      "counterfactual":output.get("counterfactual")},default=str),flush=True)


if __name__=="__main__":main()
