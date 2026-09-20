"""Predeclared Stage 2B training: common data, objective and 1000 steps/arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from etrcm.stage1_4.model import Stage14Config
from etrcm.stage2a.data import sample_reasoning
from etrcm.stage2a.language import LanguageETRCM, WordTokenizer
from etrcm.stage2b.data import make_corpus, sample_basic_safe
from etrcm.stage2b.model import ContextualLanguageETRCM


ARMS = {"GRU": "B1_gru", "RNN": "B0_no_memory", "E0": "B5_separate",
        "E1": "E1", "E2": "E2", "E0_late": "E0_late"}


def build_model(config, arm):
    return ContextualLanguageETRCM(config, arm) if arm in {"E1", "E2", "E0_late"} else LanguageETRCM(config, ARMS[arm])


def make_datasets(seed):
    train = make_corpus(seed+1, 4800, "train")
    rng = random.Random(seed+2)
    basic = [sample_basic_safe(rng) for _ in range(800)]
    reasoning = []
    while len(reasoning)<800:
        example=sample_reasoning(rng)
        if example.kind!="binding":
            reasoning.append(example)
    val_basic_rng, val_reason_rng = random.Random(88001), random.Random(88002)
    val_reasoning=[]
    while len(val_reasoning)<100:
        example=sample_reasoning(val_reason_rng)
        if example.kind!="binding":
            val_reasoning.append(example)
    validation = {
        "basic": [sample_basic_safe(val_basic_rng) for _ in range(100)],
        "memory_id": make_corpus(88003, 180, "train"),
        "memory_ood": make_corpus(88004, 180, "ood"),
        "reasoning": val_reasoning,
    }
    return {"association": train, "basic": basic, "reasoning": reasoning}, validation


def make_batch(examples, tok, device):
    sequences = [tok.encode(e.text, bos=True, eos=True) for e in examples]
    width = max(len(ids) for ids in sequences)
    tokens = torch.full((len(examples), width), tok.to_id["<pad>"], dtype=torch.long, device=device)
    answer_positions = []
    for i, (ex, ids) in enumerate(zip(examples, sequences)):
        tokens[i, :len(ids)] = torch.tensor(ids, device=device)
        answer_positions.append(len(tok.encode(ex.prefix, bos=True))-1 if getattr(ex,"answer","") else -1)
    return tokens, torch.tensor(answer_positions, device=device)


def forward(model, tokens, positions, pad_id, answer_weight=4.0):
    state = model.initial_state(tokens.shape[0], device=tokens.device)
    targets = tokens[:, 1:]
    loss_sum = torch.zeros((), device=tokens.device)
    answer_sum = torch.zeros((), device=tokens.device)
    token_count = 0
    answer_count = 0
    final_diag = None
    for t in range(tokens.shape[1]-1):
        state, logits, final_diag = model.step_token(state, tokens[:,t], source="external")
        token_loss = F.cross_entropy(logits, targets[:,t], ignore_index=pad_id, reduction="none")
        loss_sum = loss_sum + token_loss.sum()
        token_count += int(targets[:,t].ne(pad_id).sum())
        selected = positions.eq(t)
        if bool(selected.any()):
            answer_sum = answer_sum + token_loss[selected].sum()
            answer_count += int(selected.sum())
    lm_ce = loss_sum / max(1,token_count)
    answer_ce = answer_sum / max(1,answer_count)
    objective = lm_ce + (answer_weight * answer_ce if answer_count else 0)
    return objective, lm_ce, answer_ce, state, final_diag


@torch.no_grad()
def evaluate(model, examples, tok, device, limit=30):
    model.eval()
    losses=[]; answers=[]
    for ex in examples[:limit]:
        tokens, positions = make_batch([ex], tok, device)
        _, lm_ce, answer_ce, _, _ = forward(model, tokens, positions, tok.to_id["<pad>"], answer_weight=0)
        losses.append(float(lm_ce))
        if positions[0] >= 0:
            answers.append(float(answer_ce))
    return {"CE":sum(losses)/len(losses), "PPL":math.exp(sum(losses)/len(losses)),
            "answer_CE":sum(answers)/len(answers) if answers else None,"n":len(losses)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log-every", type=int, default=100)
    args=p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(args.seed); random.seed(args.seed)
    data, valid = make_datasets(args.seed)
    tok = WordTokenizer.fit([ex.text for part in data.values() for ex in part])
    # The train split alone defines the vocabulary. The lexical OOD split uses
    # unseen combinations, not unseen token IDs.
    config=Stage14Config(hidden_dim=args.hidden_dim, latent_slots=1, symbol_count=len(tok.vocabulary),
                         key_dim=32, value_dim=32, event_type_dim=8, gamma=0.12,
                         rho_fast=0.97, rho_slow=0.9995, eta_external=0.6)
    model=build_model(config,args.arm).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0005)
    nominal=sum(p.numel() for p in model.parameters())
    rng=random.Random(args.seed+5)
    logs=[]; token_total=0; active=None; start=time.monotonic()
    for step in range(args.steps):
        # Homogeneous source/gap batches avoid padding a 16-gap example to 320.
        choice=rng.random()
        if choice<0.15:
            pool=data["basic"]
        elif choice<0.30:
            pool=data["reasoning"]
        else:
            prototype=rng.choice(data["association"])
            pool=[ex for ex in data["association"] if ex.family==prototype.family and ex.gap==prototype.gap]
        examples=rng.choices(pool,k=args.batch_size)
        tokens,positions=make_batch(examples,tok,args.device)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        objective,lm_ce,answer_ce,state,diag=forward(model,tokens,positions,tok.to_id["<pad>"])
        if not torch.isfinite(objective):
            raise RuntimeError(f"nonfinite objective arm={args.arm} seed={args.seed} step={step}")
        objective.backward()
        if active is None:
            active=sum(p.numel() for p in model.parameters() if p.grad is not None)
        gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
        optimizer.step()
        token_total += int(tokens[:,1:].ne(tok.to_id["<pad>"]).sum())
        if step%args.log_every==0 or step==args.steps-1:
            row={"step":step+1,"objective":float(objective.detach()),"train_lm_CE":float(lm_ce.detach()),
                 "train_answer_CE":float(answer_ce.detach()),"gradient_norm_preclip":gradient,
                 "H":float(state.H.detach().norm(dim=(-2,-1)).mean()),
                 "F":float(state.F.detach().norm(dim=(-2,-1)).mean()),
                 "M":float(state.M.detach().norm(dim=(-2,-1)).mean()),
                 "tokens_seen":token_total,"elapsed_s":time.monotonic()-start}
            logs.append(row); print(json.dumps({"arm":args.arm,"seed":args.seed,**row}),flush=True)
    train_elapsed=time.monotonic()-start
    validation={name:evaluate(model,part,tok,args.device) for name,part in valid.items()}
    model_bytes=(config.latent_slots*config.hidden_dim+2*config.value_dim*config.key_dim)*4
    summary={"arm":args.arm,"seed":args.seed,"steps":args.steps,"batch_size":args.batch_size,
             "hidden_dim":args.hidden_dim,"vocab_size":len(tok.vocabulary),"nominal_parameters":nominal,
             "active_parameters":active,"allocated_state_bytes":model_bytes,
             "effective_memory_bytes":0 if args.arm in {"GRU","RNN"} else 2*config.value_dim*config.key_dim*4,
             "train_elapsed_s":train_elapsed,"total_elapsed_s":time.monotonic()-start,"tokens_seen":token_total,
             "compute_seconds_per_token":train_elapsed/max(1,token_total),
             "validation":validation,"train_counts":{k:len(v) for k,v in data.items()},
             "validation_counts":{k:len(v) for k,v in valid.items()},
             "objective":"mean next-token CE + 4*answer-token CE for association/reasoning episodes",
             "optimizer":"AdamW lr=5e-4; parameter gradient clip=1; no state clip"}
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2))
    (args.out/"train_log.json").write_text(json.dumps(logs,indent=2))
    (args.out/"tokenizer.json").write_text(json.dumps(tok.metadata(),indent=2))
    (args.out/"config.json").write_text(json.dumps(vars(config),indent=2))
    torch.save({"model":model.state_dict(),"arm":args.arm,"config":vars(config),
                "tokenizer":tok.vocabulary,"seed":args.seed,"steps":args.steps},args.out/"checkpoint.pt")
    hashes={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in args.out.iterdir() if path.is_file()}
    (args.out/"hashes.json").write_text(json.dumps(hashes,indent=2))
    print(json.dumps({"COMPLETED":summary}),flush=True)


if __name__=="__main__":
    main()
