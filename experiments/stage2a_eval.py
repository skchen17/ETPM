"""Frozen-checkpoint Stage 2A evaluations and a complete, reproducible report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from etrcm.stage1_1.model import LearnedState
from etrcm.stage1_4.model import Stage14Config
from etrcm.stage2a.data import DAYS, NAMES, Example, filler, make_corpus, sample_basic, sample_memory, sample_reasoning
from etrcm.stage2a.language import LanguageETRCM, WordTokenizer, generate, lesion, observe


def load_model(root: Path, label: str, device: str):
    payload = torch.load(root / f"{label}_reasoning.pt", map_location="cpu", weights_only=False)
    config = Stage14Config(**payload["config"])
    model = LanguageETRCM(config, mode=payload["mode"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, WordTokenizer(payload["tokenizer"])


def state_metrics(before: LearnedState, after: LearnedState, diagnostics=None):
    row = {"H": float(after.H.norm()), "F": float(after.F.norm()), "M": float(after.M.norm()),
           "H_delta": float((after.H - before.H).norm()),
           "finite": bool(all(torch.isfinite(x).all() for x in (after.H, after.F, after.M)))}
    if diagnostics is not None:
        row["read"] = float(diagnostics["read"].norm())
        row["r_F"] = float(diagnostics["r_F"].norm())
        row["r_M"] = float(diagnostics["r_M"].norm())
        row["external_write"] = bool(diagnostics["external_write_flag"][0])
    return row


@torch.no_grad()
def evaluate_answer(model, tok, ex: Example, *, intervention="full", ticks=0):
    device = next(model.parameters()).device
    state = model.initial_state(1, device=device)
    prefix = ex.prefix
    before_query, query = prefix.rsplit("question :", 1)
    ids = tok.encode(before_query, bos=True)
    if intervention == "gamma_zero":
        old_mode = model.mode; model.mode = "B6_gamma_zero"
    try:
        state, _ = observe(model, state, ids)
        state = lesion(state, intervention, model) if intervention not in {"gamma_zero", "full"} else state
        before = state.clone()
        state, diag = observe(model, state, tok.encode("question :" + query))
        trajectory = [state_metrics(before, state, diag)]
        for _ in range(ticks):
            prior = state
            state, _, diag = model.null_tick(state)
            trajectory.append(state_metrics(prior, state, diag))
        logits = model.logits(state)[0]
        answer_id = tok.encode(ex.answer)[0]
        ce = float(F.cross_entropy(logits[None], torch.tensor([answer_id], device=device)))
        predicted_id = int(logits.argmax())
        if ex.kind in {"memory", "conversation"}:
            candidates = DAYS if ex.answer in DAYS else NAMES
        elif ex.kind == "binding" or ex.kind == "relation":
            candidates = NAMES
        else:
            candidates = ("yes", "no") if ex.kind == "transitivity" else ("on", "off")
        candidate_ids = [tok.encode(word)[0] for word in candidates]
        candidate_pred = candidates[int(logits[candidate_ids].argmax())]
        return {"kind": ex.kind, "gap": ex.gap_tokens, "answer": ex.answer,
                "predicted_token": tok.vocabulary[predicted_id], "candidate_predicted": candidate_pred,
                "exact": predicted_id == answer_id, "candidate_correct": candidate_pred == ex.answer,
                "answer_ce": ce, "answer_probability": float(logits.softmax(-1)[answer_id]),
                "trajectory": trajectory, "prefix": ex.prefix}
    finally:
        if intervention == "gamma_zero":
            model.mode = old_mode


def aggregate(rows):
    if not rows:
        return {}
    return {"n": len(rows), "exact": sum(r["exact"] for r in rows) / len(rows),
            "candidate_accuracy": sum(r["candidate_correct"] for r in rows) / len(rows),
            "answer_ce": sum(r["answer_ce"] for r in rows) / len(rows),
            "H": sum(r["trajectory"][-1]["H"] for r in rows) / len(rows),
            "F": sum(r["trajectory"][-1]["F"] for r in rows) / len(rows),
            "M": sum(r["trajectory"][-1]["M"] for r in rows) / len(rows),
            "read": sum(r["trajectory"][-1].get("read", 0) for r in rows) / len(rows)}


def repetition(text):
    words = text.split()
    return sum(a == b for a, b in zip(words, words[1:])) / max(len(words) - 1, 1)


@torch.no_grad()
def continuous(model, tok, length, *, seed=77):
    rng = random.Random(seed)
    state = model.initial_state(1, device=next(model.parameters()).device)
    records = []; first_nonfinite = None; first_large = None
    writes_from_self = 0
    for tick in range(length):
        prior = state
        if tick % 5 in (0, 1, 2):
            item = sample_basic(rng)
            token = tok.encode(item.text)[tick % max(len(tok.encode(item.text)), 1)]
            state, _, diag = model.step_token(state, torch.tensor([token], device=state.H.device))
            source = "external"
        elif tick % 5 == 3:
            state, _, diag = model.null_tick(state)
            source = "null"
        else:
            token = int(model.logits(state)[0].argmax())
            state, _, diag = model.step_token(state, torch.tensor([token], device=state.H.device), source="self_output")
            writes_from_self += int(bool(diag["external_write_flag"][0]))
            source = "self_output"
        metrics = state_metrics(prior, state, diag)
        metrics.update({"tick": tick + 1, "source": source, "tau": state.tau, "external_time": state.external_time})
        records.append(metrics)
        if not metrics["finite"]:
            first_nonfinite = tick + 1; break
        if first_large is None and metrics["H"] > 1000:
            first_large = tick + 1
    return {"requested_ticks": length, "completed_ticks": len(records), "first_nonfinite": first_nonfinite,
            "first_H_gt_1000": first_large, "self_output_writes": writes_from_self,
            "max_H": max(r["H"] for r in records), "max_F": max(r["F"] for r in records),
            "max_M": max(r["M"] for r in records), "last": records[-1],
            "trajectory": records}


@torch.no_grad()
def spontaneous(model, tok, count=10):
    rng = random.Random(199)
    outputs = []
    for index in range(count):
        state = model.initial_state(1, device=next(model.parameters()).device)
        events = []; fired = False
        for t in range(5):
            sentence = sample_basic(rng).text
            before = state
            state, _ = observe(model, state, tok.encode(sentence))
            state, _, _ = model.null_tick(state)
            score = float(model.expression(state)["expression_score"][0])
            row = {"time": t, "external_input": sentence, "internal_ticks": 1, "expression_score": score,
                   "state": state_metrics(before, state)}
            if score > 0.45 and not fired:
                before_gen = state
                output, state, trace = generate(model, tok, "", state=state, max_new_tokens=8,
                                                min_new_tokens=3, temperature=0.9, top_k=12,
                                                greedy=False, seed=9000+index)
                row["generated_output"] = output
                row["self_output_trace"] = trace
                row["subsequent_state"] = state_metrics(before_gen, state)
                fired = True
            events.append(row)
        outputs.append({"trajectory": index + 1, "fired": fired, "events": events,
                        "final_tau": state.tau, "final_external_time": state.external_time})
    return outputs


def md_table(rows):
    if not rows:
        return "No rows."
    keys = list(rows[0])
    return "| " + " | ".join(keys) + " |\n|" + "|".join("---" for _ in keys) + "|\n" + "\n".join(
        "| " + " | ".join(str(row.get(k, "")) for k in keys) + " |" for row in rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluation-n", type=int, default=12)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.root.parent.parent / "processed" / args.root.name
    output.mkdir(parents=True, exist_ok=True)
    models = {name: load_model(args.root, name, args.device)[0] for name in ("gru", "recurrent", "etrcm")}
    tok = load_model(args.root, "etrcm", args.device)[1]
    rng = random.Random(52026)
    interventions = ("full", "F", "M", "FM", "H", "gamma_zero")
    memory = defaultdict(list)
    for gap in (32, 128, 512, 1024):
        episodes = [sample_memory(rng, gap) for _ in range(args.evaluation_n)]
        for i, ex in enumerate(episodes):
            for condition in interventions:
                memory[f"{gap}/{condition}"].append(evaluate_answer(models["etrcm"], tok, ex, intervention=condition))
            for baseline in ("gru", "recurrent"):
                memory[f"{gap}/{baseline}"].append(evaluate_answer(models[baseline], tok, ex))
    reasoning = defaultdict(list)
    for _ in range(args.evaluation_n * 6):
        ex = sample_reasoning(rng)
        for label, model in models.items():
            reasoning[label + "/" + ex.kind].append(evaluate_answer(model, tok, ex))
    tick_sweep = defaultdict(list)
    for _ in range(args.evaluation_n):
        ex = sample_reasoning(rng)
        for ticks in (0, 1, 2, 4, 8, 16):
            tick_sweep[f"reasoning/{ticks}"].append(evaluate_answer(models["etrcm"], tok, ex, ticks=ticks))
        ex = sample_memory(rng, 32)
        for ticks in (0, 1, 2, 4, 8, 16):
            tick_sweep[f"memory/{ticks}"].append(evaluate_answer(models["etrcm"], tok, ex, ticks=ticks))
    conversation = defaultdict(list)
    for gap in (8, 32, 128):
        for _ in range(args.evaluation_n):
            answer = rng.choice(DAYS)
            prefix = f"user : the meeting is on {answer} . assistant : yes . {filler(rng,gap)} user : when is the meeting ? assistant : question : answer :"
            ex = Example(prefix + " " + answer + " .", prefix, answer, "conversation", 0.25, gap)
            conversation[str(gap)].append(evaluate_answer(models["etrcm"], tok, ex))
    sample_prompts = ("", "alice walked to the garden .", "user : where is alice ? assistant :",
                      "the meeting is on", "bob has the blue key .", "the light is off .")
    generations = []
    for i in range(20):
        prompt = sample_prompts[i % len(sample_prompts)]
        text, state, trace = generate(models["etrcm"], tok, prompt, seed=1000 + i, max_new_tokens=24,
                                      temperature=0.9, top_k=12, greedy=False)
        generations.append({"type": "seeded", "prompt": prompt, "output": text, "repetition": repetition(text),
                            "eos": len(trace) < 24, "self_output_writes": sum(t["write"] for t in trace),
                            "final_state": state_metrics(state, state)})
    for i in range(20):
        ex = sample_memory(rng, 0) if i % 2 else sample_reasoning(rng)
        text, state, trace = generate(models["etrcm"], tok, ex.prefix, max_new_tokens=12, greedy=True)
        generations.append({"type": "prompt_continuation", "prompt": ex.prefix, "target": ex.answer,
                            "output": text, "repetition": repetition(text), "eos": len(trace) < 12,
                            "self_output_writes": sum(t["write"] for t in trace), "final_state": state_metrics(state, state)})
    continuous_runs = [continuous(models["etrcm"], tok, n) for n in (100, 500, 1000)]
    if all(run["first_nonfinite"] is None and run["first_H_gt_1000"] is None for run in continuous_runs):
        continuous_runs.append(continuous(models["etrcm"], tok, 5000))
    spontaneous_runs = spontaneous(models["etrcm"], tok)
    result = {"memory": {k: aggregate(v) for k, v in memory.items()},
              "reasoning": {k: aggregate(v) for k, v in reasoning.items()},
              "tick_sweep": {k: aggregate(v) for k, v in tick_sweep.items()},
              "conversation": {k: aggregate(v) for k, v in conversation.items()},
              "generations": generations, "continuous": continuous_runs, "spontaneous": spontaneous_runs,
              "memory_examples": {k: v[:2] for k, v in memory.items()},
              "reasoning_examples": {k: v[:2] for k, v in reasoning.items()}}
    (output / "evaluation.json").write_text(json.dumps(result, indent=2))
    train = json.loads((args.root / "train_log.json").read_text())
    conf = json.loads((args.root / "config.json").read_text())
    params = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
    validation = [x for x in train if "validation_ce" in x]
    val_rows = [{"model": x["model"], "phase": x["phase"],
                 **{f"CE_{k}": round(v, 3) for k,v in x["validation_ce"].items()},
                 **{f"PPL_{k}": round(v, 2) for k,v in x["validation_ppl"].items()}}
                for x in validation]
    mem_rows = [{"gap": k.split("/")[0], "condition": k.split("/")[1], "n": v["n"],
                 "candidate_acc": round(v["candidate_accuracy"], 3), "exact": round(v["exact"], 3),
                 "answer_CE": round(v["answer_ce"], 3), "read": round(v["read"], 3)}
                for k,v in result["memory"].items()]
    reasoning_rows = [{"condition_task": k, "n": v["n"], "candidate_acc": round(v["candidate_accuracy"], 3),
                       "exact": round(v["exact"], 3), "answer_CE": round(v["answer_ce"], 3)}
                      for k,v in result["reasoning"].items()]
    tick_rows = [{"task_K": k, "n": v["n"], "candidate_acc": round(v["candidate_accuracy"], 3),
                  "answer_CE": round(v["answer_ce"], 3), "H": round(v["H"], 2),
                  "F": round(v["F"], 2), "M": round(v["M"], 2)}
                 for k,v in result["tick_sweep"].items()]
    conv_rows = [{"gap": k, "n": v["n"], "candidate_acc": round(v["candidate_accuracy"], 3),
                  "exact": round(v["exact"], 3), "answer_CE": round(v["answer_ce"], 3)}
                 for k,v in result["conversation"].items()]
    stage1_instability = any(r["first_nonfinite"] is not None or r["first_H_gt_1000"] is not None for r in continuous_runs)
    report = ["# ET-RCM Stage 2A Natural-Language Prototype Results", "",
      "Exploratory language-domain prototype. All Stage 1.x frozen artifacts are unchanged; no Stage 1 gate is reopened.", "",
      "## 1. Architecture", "",
      "`WordTokenizer → external token event → unmodified Stage 1.5 AnatomicalETRCM (H,F,M) → LM head`. "
      "External tokens use existing delta writes; NULL and SELF_OUTPUT do not write. H update, independent F/M reads, "
      "query-dependent readout-conserving transfer, and decay are inherited without law changes. "
      "The event key/value are the same token id: this is a token-associative proxy, not an explicit entity–value memory encoder.", "",
      "## 2. Model size", "", md_table([{"model": k, "parameters": v} for k,v in params.items()]), "",
      "All three instantiate the same parameter scaffold; no-memory/GRU modes leave memory-path parameters unused. "
      "Therefore nominal parameter equality overstates effective-parameter matching.", "",
      "## 3. Tokenizer", "", f"Lowercase word/punctuation regex, training-only vocabulary size {len(tok.vocabulary)}; unseen tokens map to `<unk>`. "
      "No pretrained tokenizer or corpus.", "",
      "## 4. Training data", "", f"Generated English templates: {conf['train_counts']} training and {conf['validation_counts']} validation examples. "
      "Validation uses a different PRNG seed but shares templates and lexical inventories; results do not establish open-domain generalization. "
      "Basic text includes declaratives, descriptions and dialogue; memory includes meeting-day/ownership questions with 0/8/16/32-token distractors; "
      "reasoning includes variable binding, transitivity, light transitions, spatial relation. No importance labels. "
      "Random train-uniform token baseline CE is `ln(vocab_size)`.", "",
      "## 5. Training configuration", "", "```json", json.dumps(conf, indent=2), "```", "",
      "Teacher-forced next-token prediction; fresh state per document; full BPTT up to max_length. "
      "Gradient clipping is parameter-only (norm 1.0); no hidden-state clipping or stabilization. "
      "One seed, one small model size; no variance claim.", "",
      "## 6. Basic language modeling", "", md_table(val_rows), "",
      "Training-loss, preclip gradient, and final-batch H/F/M norm trajectories are in `train_log.json`; "
      "all checkpoint SHA-256 hashes are in `checkpoint_hashes.json`. "
      f"Uniform-random CE = {math.log(len(tok.vocabulary)):.3f}; PPL = {len(tok.vocabulary)}.", "",
      "## 7. Generated language samples", "", "20 seeded/unconditional and 20 prompt continuations are shown below, including failures. "
      f"Mean adjacent-token repetition = {sum(x['repetition'] for x in generations)/len(generations):.3f}; "
      f"EOS in {sum(x['eos'] for x in generations)}/{len(generations)}; "
      f"SELF_OUTPUT external writes = {sum(x['self_output_writes'] for x in generations)}.", ""]
    for i,g in enumerate(generations,1):
        report += [f"{i}. **{g['type']}** Prompt: `{g['prompt']}`" + (f" Target: `{g['target']}`" if 'target' in g else ""),
                   f"   Output: `{g['output']}`; repetition={g['repetition']:.3f}; EOS={g['eos']}."]
    report += ["", "## 8. Natural-language memory tests", "",
               "Same trained checkpoint, held-out generated episodes, 12 (configurable) episodes/gap. "
               "Gaps 32/128/512/1024 are evaluation-only beyond the training gap range. "
               "Candidate accuracy is among four allowed values; exact is unrestricted next-token argmax, so candidate accuracy may overstate generation quality.", "",
               md_table(mem_rows), "", "## 9. H/F/M intervention results", "",
               "F, M, F+M lesions and H reset are applied after the early statement/distractor but before the question, with the same checkpoint. "
               "Gamma-zero reruns the prefix with transfer disabled, same weights; it is not a newly trained ablation. "
               "For no-memory baselines F/M are unused. Lesions can move logits without proving a uniquely causal slow-memory role.", ""]
    for key in ("32/full", "32/F", "32/M", "32/FM", "32/H", "32/gamma_zero"):
        if result["memory_examples"].get(key):
            ex = result["memory_examples"][key][0]
            report.append(f"- {key}: target `{ex['answer']}`, predicted `{ex['predicted_token']}` "
                          f"(candidate `{ex['candidate_predicted']}`), CE {ex['answer_ce']:.3f}; "
                          f"H/F/M {ex['trajectory'][-1]['H']:.2f}/{ex['trajectory'][-1]['F']:.2f}/{ex['trajectory'][-1]['M']:.2f}.")
    report += ["", "## 10. Natural-language reasoning tests", "", md_table(reasoning_rows), "",
               "Reasoning examples (same held-out prompt family):", ""]
    for key in sorted(result["reasoning_examples"]):
        if key.startswith("etrcm/"):
            for ex in result["reasoning_examples"][key]:
                report.append(f"- `{ex['prefix']}` Target `{ex['answer']}`; free token `{ex['predicted_token']}`; "
                              f"candidate `{ex['candidate_predicted']}`; CE {ex['answer_ce']:.3f}.")
    report += ["", "## 11. NULL-tick sweep", "", md_table(tick_rows), "",
               "NULL ticks follow the query, contain no new token and invoke no external delta write. "
               "The trajectory for each episode, including read and H/F/M norms, is in evaluation.json.", "",
               "## 12. Continuous conversation", "", md_table(conv_rows), "",
               "The state is carried across each dialogue episode without reset; templates and exact transcripts are saved in evaluation.json. "
               "Contradiction and duplicate-response rates cannot be meaningfully estimated from a single-token factual answer protocol; not claimed.", "",
               "## 13. Continuous-running stability", "",
               md_table([{"requested": r["requested_ticks"], "completed": r["completed_ticks"],
                          "first_nonfinite": r["first_nonfinite"], "first_H_gt_1000": r["first_H_gt_1000"],
                          "max_H": round(r["max_H"], 2), "max_F": round(r["max_F"],2),
                          "max_M": round(r["max_M"],2), "self_writes": r["self_output_writes"]}
                         for r in continuous_runs]), "",
               f"Stage 1.5-like strong norm growth in this mixed-stream test: **{stage1_instability}**. "
               "All per-tick H/F/M norms, H increments, read norms, source and clocks are saved. "
               "A finite 1000-tick stream is not proof of arbitrary-duration stability. No state clipping was applied.", "",
               "## 14. Spontaneous-output prototype", "",
               "Exploratory, non-gating interface demo. Expression head was inherited but NOT trained on language emission targets. "
               "Threshold 0.45 is a demonstration threshold, not a calibrated policy. "
               "Demo decoding samples from top-12 at temperature 0.9 and suppresses EOS for the first 3 tokens; "
               "this forces a nonempty utterance and must not be interpreted as learned willingness to speak. "
               "After emission, generated tokens feed SELF_OUTPUT, and the subsequent external event is processed in the same state. "
               "Each of 10 full trajectories follows.", ""]
    for case in spontaneous_runs:
        report.append(f"### Trajectory {case['trajectory']} (fired={case['fired']}; final τ={case['final_tau']}; external t={case['final_external_time']})")
        report.append("")
        for event in case["events"]:
            report.append(f"- time={event['time']}; external=`{event['external_input']}`; NULL={event['internal_ticks']}; "
                          f"score={event['expression_score']:.4f}; generated=`{event.get('generated_output','')}`; "
                          f"subsequent_state={event.get('subsequent_state', event['state'])}.")
        report.append("")
    report += ["## 15. Baseline comparison", "",
               "GRU and no-memory gated recurrent LM are trained on exactly the same generated corpus and schedule. "
               "Validation CE and reasoning/memory candidate accuracy are tabulated above. A baseline doing as well or better "
               "means ET-RCM's added memory is not demonstrated necessary for this language regime.", "",
               "## 16. Failures and negative results", "",
               "See all failed exact matches and per-episode negative cases in evaluation.json. "
               "High candidate accuracy alone is not counted as unconstrained generation. "
               "Long-gap extrapolation, H reset, M lesion and NULL deterioration are interpreted without repairing the frozen Stage 1 gates. "
               "The expression score is untrained, so spontaneous language intent is unproven.", "",
               "## 17. Main bottlenecks", "",
               "Token-level writes use token=key=value rather than compositional entity-value binding; templates and vocabulary are narrow; "
               "training documents reset and are at most 56 tokens; long memory is OOD. "
               "No free-form multi-token answer accuracy or robust contradiction metric is established. "
               "Nominal parameter matching includes inactive branches. Long NULL stability remains separately unresolved.", "",
               "## 18. Scientific interpretation", "",
               "This is prototype evidence for a runnable stateful language loop, not LLM capability, consciousness, autonomous thought, "
               "general reasoning, infinite context, or causal memory. P1–P5 must be interpreted separately using metrics above, "
               "not collapsed into a single success declaration. A changed output under lesion shows intervention sensitivity, "
               "not necessarily a useful persistent memory mechanism.", "",
               "## 19. Recommendation for next stage", "",
               "Do not scale model size on these single-seed synthetic-template results alone. First add a lexical held-out split, "
               "multiple seeds, stronger matched active-parameter baselines, explicit entity-value language encoding, "
               "long-context training, answer-level generation evaluation, and a NULL-stability design study. "
               "Preserve Stage 1.x frozen adjudications and pre-register any new gates.", "",
               "## Reproduction and manifest", "",
               "Run `PYTHONPATH=src .venv/bin/python experiments/stage2a_train.py --output results/stage2a/raw/stage2a_seed42` "
               "then `PYTHONPATH=src .venv/bin/python experiments/stage2a_eval.py --root results/stage2a/raw/stage2a_seed42 "
               "--report reports/STAGE2A_NATURAL_LANGUAGE_PROTOTYPE_RESULTS.md`. "
               "Exact CLI overrides are in config.json; evaluation seeds are fixed in source. "
               "Stage 2A files only are covered by the SHA-256 manifest. Historical frozen file hashes are checked separately.", ""]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report))
    manifest = {}
    for path in [args.report, args.root / "config.json", args.root / "tokenizer.json", args.root / "train_log.json",
                 args.root / "checkpoint_hashes.json", output / "evaluation.json"]:
        manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"report": str(args.report), "report_bytes": args.report.stat().st_size,
                      "vocabulary": len(tok.vocabulary), "params": params,
                      "validation_final": val_rows[-3:], "continuous": [{k:v for k,v in r.items() if k!='trajectory'} for r in continuous_runs]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
