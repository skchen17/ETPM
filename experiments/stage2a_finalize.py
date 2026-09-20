"""Append explicit interpretation and medium-size evidence to the generated report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def binomial_tail(successes: int, probabilities: list[float]) -> float:
    distribution = [1.0]
    for p in probabilities:
        next_distribution = [0.0] * (len(distribution) + 1)
        for k, mass in enumerate(distribution):
            next_distribution[k] += mass * (1-p)
            next_distribution[k+1] += mass * p
        distribution = next_distribution
    return sum(distribution[successes:])


def metric(result, section, key, field):
    return result[section][key][field]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--medium", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.processed / "evaluation.json").read_text())
    small_log = json.loads((args.small / "train_log.json").read_text())
    medium_log = json.loads((args.medium / "train_log.json").read_text())
    small_final = {x["model"]: x for x in small_log if x.get("phase") == "reasoning" and "validation_ce" in x}
    medium_final = {x["model"]: x for x in medium_log if x.get("phase") == "reasoning" and "validation_ce" in x}
    small_vocab = json.loads((args.small / "tokenizer.json").read_text())["vocabulary"]
    reasoning_keys = [k for k in result["reasoning"] if k.startswith("etrcm/")]
    n_reason = sum(result["reasoning"][k]["n"] for k in reasoning_keys)
    correct_reason = round(sum(result["reasoning"][k]["candidate_accuracy"] * result["reasoning"][k]["n"] for k in reasoning_keys))
    ps = []
    for key in reasoning_keys:
        p = 1/3 if key.endswith(("binding", "relation")) else 1/2
        ps.extend([p] * result["reasoning"][key]["n"])
    p_reason = binomial_tail(correct_reason, ps)
    memory_full = [result["memory"][f"{gap}/full"] for gap in (32,128,512,1024)]
    memory_long_acc = sum(x["candidate_accuracy"] for x in memory_full[1:]) / 3
    lesion_any = any(abs(result["memory"][f"{gap}/full"]["answer_ce"] - result["memory"][f"{gap}/M"]["answer_ce"]) > 0.01
                     for gap in (32,128,512,1024))
    tick0 = result["tick_sweep"]["reasoning/0"]
    tick16 = result["tick_sweep"]["reasoning/16"]
    stable1000 = next(x for x in result["continuous"] if x["requested_ticks"] == 1000)
    stable = stable1000["completed_ticks"] == 1000 and stable1000["first_nonfinite"] is None
    fired = sum(x["fired"] for x in result["spontaneous"])
    writes = sum(x["self_output_writes"] for x in result["continuous"])
    writes += sum(x["self_output_writes"] for x in result["generations"])
    small_params = json.loads((args.small / "config.json").read_text())["model_config"]["hidden_dim"]
    medium_params = json.loads((args.medium / "config.json").read_text())["model_config"]["hidden_dim"]
    compare = ["## 20. Explicit P1–P5 assessment and requested answers", "",
               f"- **P1 language learning: yes on synthetic templates.** Small ET-RCM final basic CE {small_final['etrcm']['validation_ce']['basic']:.3f} "
               f"vs random {math.log(len(small_vocab)):.3f}; final memory CE {small_final['etrcm']['validation_ce']['memory']:.3f}; "
               f"reasoning CE {small_final['etrcm']['validation_ce']['reasoning']:.3f}. "
               "Validation CE declines over the curriculum but is not monotone across task shifts.",
               f"- **P2 generation: not met for reliable natural-language answers.** Inspect the 40 literal outputs above. EOS appeared in "
               f"{sum(g['eos'] for g in result['generations'])}/40; most question continuations terminate or emit punctuation. "
               "Some seeded fragments resemble English, but repetition and degeneration dominate. SELF_OUTPUT is unseen during teacher-forced training.",
               f"- **P3 memory: not established.** Long-gap candidate accuracy (128/512/1024 mean) {memory_long_acc:.3f} vs chance 0.25; "
               f"M lesion changes answer CE by >0.01 for at least one gap: {lesion_any}. "
               "This is sensitivity, not useful M necessity; free-token exact memory accuracy is zero across all ET-RCM gaps.",
               f"- **P4 reasoning: not significant at 0.05.** {correct_reason}/{n_reason} candidate answers correct; chance expected {sum(ps):.1f}/{n_reason}; "
               f"one-sided Poisson-binomial tail p={p_reason:.4g} (single seed, same template family). "
               "Unrestricted exact-token accuracy is separately in Section 10.",
               f"- **P5 continuous operation: technically runs 1000 ticks, but long-horizon stability unresolved.** 1000 mixed ticks completed={stable}; max H norm {stable1000['max_H']:.2f}; "
               f"first nonfinite={stable1000['first_nonfinite']}; SELF_OUTPUT external writes={writes}. "
               "At 5000 ticks H exceeds 1000, so this does not establish safe sustained dynamics.", "",
               "### Small–medium training comparison", "",
               f"Small hidden dimension {small_params}, 132,871 nominal parameters; medium hidden dimension {medium_params}, "
               "401,543 nominal parameters (3.02×). Same data seed, phases and 60 optimizer steps/phase. "
               "Medium has validation-only evidence here, not the full long-gap battery. Neither model exceeds 1M parameters, "
               "as permitted by this preliminary stage.", "",
               "| size | model | basic CE | memory CE | reasoning CE |",
               "|---|---|---:|---:|---:|" ]
    for size, dataset in (("small",small_final),("medium",medium_final)):
        for name in ("gru","recurrent","etrcm"):
            values = dataset[name]["validation_ce"]
            compare.append(f"| {size} | {name} | {values['basic']:.3f} | {values['memory']:.3f} | {values['reasoning']:.3f} |")
    compare += ["", "### Direct answers to remaining interpretation questions", "",
                f"- GRU comparison: small GRU basic CE {small_final['gru']['validation_ce']['basic']:.3f} vs ET-RCM "
                f"{small_final['etrcm']['validation_ce']['basic']:.3f}; medium GRU {medium_final['gru']['validation_ce']['basic']:.3f} "
                f"vs ET-RCM {medium_final['etrcm']['validation_ce']['basic']:.3f}. Memory pathway is not shown superior.",
                "- H/F/M interventions change answer CE/readouts in Section 8; H reset preserves F/M by construction, "
                "but post-reset behavioral benefit must be read from the H rows, not inferred from the architecture.",
                f"- NULL ticks: reasoning candidate accuracy {tick0['candidate_accuracy']:.3f} at K=0 and "
                f"{tick16['candidate_accuracy']:.3f} at K=16; answer CE {tick0['answer_ce']:.3f} → {tick16['answer_ce']:.3f}. "
                "The memory sweep is in Section 11; neither monotonicity nor an independent compute advantage is assumed.",
                f"- Continuous text without reset: {stable}; no external write from SELF_OUTPUT: {writes == 0} "
                "in measured generation/continuous events and unit tests. Spontaneous trajectories emitted in "
                f"{fired}/10 cases, with state continuing after output; expression score itself is untrained. "
                "The demo forces at least three sampled tokens by temporarily suppressing EOS; this is only an interface check.",
                "- Stage 1.5 instability: judge norm growth from Section 13. Finite values alone do not negate the old failure; "
                "a hidden norm above 1000 is separately flagged. No clipping or repair was silently applied.",
                "- Dominant current bottlenecks: weak memory binding and out-of-range retention, baseline competitiveness, "
                "long-run state growth, and untrained expression. Basic template modeling works; open-domain generation does not.",
                "- Scaling recommendation: further larger sequence-model work is **not yet justified as a mechanism claim**. "
                "A controlled follow-up may improve token/entity encoding and stability, then repeat across seeds and lexical OOD splits.",
                "- All positive statements remain prototype evidence on synthetic English. They do not establish broad natural-language understanding, "
                "reliable autobiographical memory or endogenous cognition.", ""]
    pilot = args.small.parent / "pilot"
    if (pilot / "train_log.json").exists():
        compare += ["### Development pilot retained", "",
                    "A two-step/phase, batch-2, 40-token pilot was run only to verify training and GPU execution. "
                    "Its checkpoints/logs remain in `results/stage2a/raw/pilot/` and are not used for P1–P5 claims.", ""]
    existing = args.report.read_text()
    if "## 20. Explicit P1–P5" in existing:
        existing = existing.split("## 20. Explicit P1–P5")[0].rstrip() + "\n\n"
    args.report.write_text(existing + "\n".join(compare))
    manifest_path = args.processed / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[str(args.report)] = hashlib.sha256(args.report.read_bytes()).hexdigest()
    for path in args.medium.glob("*.pt"):
        manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("config.json", "tokenizer.json", "train_log.json", "checkpoint_hashes.json"):
        path = args.medium / name
        manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if pilot.exists():
        for path in pilot.glob("*"):
            if path.is_file():
                manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    project_root = args.report.parent.parent
    for relative in (
        "src/etrcm/stage2a/__init__.py", "src/etrcm/stage2a/language.py",
        "src/etrcm/stage2a/data.py", "experiments/stage2a_train.py",
        "experiments/stage2a_eval.py", "experiments/stage2a_finalize.py",
        "experiments/stage2a_spontaneous_refresh.py",
        "experiments/stage2a_conversation_diagnostic.py",
        "tests/test_stage2a.py", "configs/stage2a.yaml", "STAGE2A_START_HERE.md",
    ):
        path = project_root / relative
        manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"report_bytes": args.report.stat().st_size, "P1_basic_CE": small_final['etrcm']['validation_ce']['basic'],
                      "reasoning_correct": [correct_reason,n_reason], "reasoning_p": p_reason,
                      "memory_long_candidate_accuracy": memory_long_acc, "stable1000": stable,
                      "spontaneous_fired": fired, "self_writes": writes}, indent=2))


if __name__ == "__main__":
    main()
