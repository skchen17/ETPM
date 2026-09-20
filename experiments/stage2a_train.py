"""Train the three small Stage 2A language models on generated English episodes."""

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
from etrcm.stage2a.data import make_corpus
from etrcm.stage2a.language import LanguageETRCM, WordTokenizer, shift_targets


def batch_tensor(examples, tokenizer, max_length, device):
    sequences = [tokenizer.encode(e.text, bos=True, eos=True)[:max_length] for e in examples]
    length = max(len(s) for s in sequences)
    result = torch.full((len(sequences), length), tokenizer.to_id["<pad>"], dtype=torch.long, device=device)
    for i, sequence in enumerate(sequences):
        result[i, :len(sequence)] = torch.tensor(sequence, device=device)
    return result


def forward_loss(model, tokens, pad_id):
    inputs, targets = shift_targets(tokens)
    state = model.initial_state(inputs.shape[0], device=inputs.device)
    loss_sum = torch.zeros((), device=inputs.device)
    count = 0
    for t in range(inputs.shape[1]):
        state, logits, _ = model.step_token(state, inputs[:, t], source="external")
        loss_sum = loss_sum + F.cross_entropy(logits, targets[:, t], ignore_index=pad_id, reduction="sum")
        count += int(targets[:, t].ne(pad_id).sum())
    return loss_sum / max(count, 1), state


@torch.no_grad()
def validate(model, examples, tokenizer, *, batch_size, max_length, device):
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        batch = batch_tensor(examples[start:start + batch_size], tokenizer, max_length, device)
        loss, _ = forward_loss(model, batch, tokenizer.to_id["<pad>"])
        n = int(batch[:, 1:].ne(tokenizer.to_id["<pad>"]).sum())
        total += float(loss) * n
        count += n
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps-per-phase", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=56)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed)
    torch.set_num_threads(4)
    train = make_corpus(args.seed, {"basic": 800, "memory": 800, "reasoning": 800})
    valid = make_corpus(args.seed + 100000, {"basic": 60, "memory": 60, "reasoning": 60})
    tokenizer = WordTokenizer.fit([e.text for part in train.values() for e in part])
    (args.output / "tokenizer.json").write_text(json.dumps(tokenizer.metadata(), indent=2))
    config = Stage14Config(hidden_dim=args.hidden_dim, latent_slots=1, symbol_count=len(tokenizer.vocabulary),
                           key_dim=32, value_dim=32, event_type_dim=8, gamma=0.12,
                           rho_fast=0.97, rho_slow=0.9995, eta_external=0.6)
    details = {"seed": args.seed, "device": args.device, "steps_per_phase": args.steps_per_phase,
               "batch_size": args.batch_size, "max_length": args.max_length,
               "train_counts": {k: len(v) for k, v in train.items()},
               "validation_counts": {k: len(v) for k, v in valid.items()},
               "model_config": vars(config), "optimizer": "AdamW", "lr": 0.0008,
               "gradient_clip": 1.0, "state_clip": False,
               "training_context": "each document resets; no cross-document carry; full BPTT through max_length"}
    (args.output / "config.json").write_text(json.dumps(details, indent=2))
    all_logs = []
    modes = {"gru": "B1_gru", "recurrent": "B0_no_memory", "etrcm": "B5_separate"}
    for label, mode in modes.items():
        torch.manual_seed(args.seed)
        model = LanguageETRCM(config, mode=mode).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008)
        params = sum(p.numel() for p in model.parameters())
        rng = random.Random(args.seed + 913)
        for phase in ("basic", "memory", "reasoning"):
            model.train()
            source = (train["basic"] if phase == "basic" else
                      train["basic"] + train["memory"] if phase == "memory" else
                      train["basic"] + train["memory"] + train["reasoning"])
            start = time.time()
            for step in range(args.steps_per_phase):
                examples = rng.choices(source, k=args.batch_size)
                tokens = batch_tensor(examples, tokenizer, args.max_length, args.device)
                optimizer.zero_grad(set_to_none=True)
                loss, state = forward_loss(model, tokens, tokenizer.to_id["<pad>"])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss: {label}/{phase}/{step}")
                loss.backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                optimizer.step()
                if step % max(1, args.steps_per_phase // 9) == 0 or step == args.steps_per_phase - 1:
                    row = {"model": label, "phase": phase, "step": step,
                           "train_ce": float(loss.detach()), "gradient_norm_preclip": gradient_norm,
                           "H_norm": float(state.H.detach().norm(dim=(-2, -1)).mean()),
                           "F_norm": float(state.F.detach().norm(dim=(-2, -1)).mean()),
                           "M_norm": float(state.M.detach().norm(dim=(-2, -1)).mean()),
                           "elapsed_s": time.time() - start}
                    all_logs.append(row)
            val = {key: validate(model, part, tokenizer, batch_size=args.batch_size,
                                 max_length=args.max_length, device=args.device)
                   for key, part in valid.items()}
            summary = {"model": label, "phase": phase, "parameter_count": params,
                       "validation_ce": val, "validation_ppl": {k: math.exp(v) for k, v in val.items()},
                       "elapsed_phase_s": time.time() - start}
            all_logs.append(summary)
            torch.save({"model": model.state_dict(), "mode": mode, "config": vars(config),
                        "tokenizer": tokenizer.vocabulary, "phase": phase}, args.output / f"{label}_{phase}.pt")
            print(json.dumps(summary), flush=True)
    (args.output / "train_log.json").write_text(json.dumps(all_logs, indent=2))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*.pt")}
    (args.output / "checkpoint_hashes.json").write_text(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
