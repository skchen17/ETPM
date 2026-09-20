"""Repeated-question dialogue diagnostic without resetting H/F/M within a stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from etrcm.stage2a.data import DAYS, filler
from etrcm.stage2a.language import generate, observe
from stage2a_eval import load_model, state_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    model, tok = load_model(args.root, "etrcm", args.device)
    rng = random.Random(2611)
    records = []
    for gap in (8, 32, 128):
        for episode in range(12):
            answer = rng.choice(DAYS)
            state = model.initial_state(1, device=args.device)
            statement = f"user : the meeting is on {answer} . assistant : yes ."
            state, _ = observe(model, state, tok.encode(statement, bos=True))
            state, _ = observe(model, state, tok.encode(filler(rng, gap)))
            output1, state, trace1 = generate(model, tok, "user : when is the meeting ? assistant :",
                                             state=state, max_new_tokens=5, greedy=True)
            first_state = state.clone()
            state, _ = observe(model, state, tok.encode(filler(rng, 8)))
            output2, state, trace2 = generate(model, tok, "user : when is the meeting ? assistant :",
                                             state=state, max_new_tokens=5, greedy=True)
            wrong = [day for day in DAYS if day != answer]
            records.append({"gap":gap, "episode":episode, "target":answer, "response1":output1,
                            "response2":output2, "duplicate":output1 == output2,
                            "empty1":not bool(output1), "empty2":not bool(output2),
                            "correct1":answer in output1.split(), "correct2":answer in output2.split(),
                            "contradiction":any(day in (output1+" "+output2).split() for day in wrong),
                            "first_state":state_metrics(first_state,first_state),
                            "final_state":state_metrics(state,state),
                            "tau_increased":state.tau > first_state.tau,
                            "self_output_writes":sum(x["write"] for x in trace1+trace2)})
    output = args.processed / "conversation_diagnostic.json"
    output.write_text(json.dumps(records, indent=2))
    report = args.report.read_text()
    block = ["### Repeated-question stream diagnostic", "",
             "A second, strict continuous-state test asks the same meeting-day question twice, with an "
             "intervening 8-token distractor and no state reset. Responses are unconstrained 5-token greedy "
             "SELF_OUTPUT generations; duplicate/contradiction rates include empty outputs and are interpreted with their coverage.", "",
             "| gap | n | first correct | second correct | empty first | empty second | duplicate | wrong-day contradiction | max H |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for gap in (8,32,128):
        subset = [x for x in records if x["gap"] == gap]
        mean = lambda name: sum(x[name] for x in subset)/len(subset)
        block.append(f"| {gap} | {len(subset)} | {mean('correct1'):.3f} | {mean('correct2'):.3f} | "
                     f"{mean('empty1'):.3f} | {mean('empty2'):.3f} | {mean('duplicate'):.3f} | "
                     f"{mean('contradiction'):.3f} | {max(x['final_state']['H'] for x in subset):.2f} |")
    block += ["", "Examples (all raw episodes are saved in `conversation_diagnostic.json`):", ""]
    for gap in (8,32,128):
        for case in [x for x in records if x["gap"] == gap][:2]:
            block.append(f"- gap={gap}, target `{case['target']}` → first `{case['response1']}`, second `{case['response2']}`; "
                         f"τ increased={case['tau_increased']}; SELF_OUTPUT writes={case['self_output_writes']}.")
    block += ["", "An empty repeated response counts as a duplicate but not as successful recall; low measured contradiction "
              "with high nonanswer rate does not imply consistent factual dialogue.", ""]
    marker = "## 13. Continuous-running stability"
    report = report.replace(marker, "\n".join(block) + "\n" + marker, 1)
    args.report.write_text(report)
    manifest_path = args.processed / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[str(output)] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest[str(args.report)] = hashlib.sha256(args.report.read_bytes()).hexdigest()
    manifest[str(args.report.parent.parent / "experiments/stage2a_conversation_diagnostic.py")] = hashlib.sha256(
        (args.report.parent.parent / "experiments/stage2a_conversation_diagnostic.py").read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"episodes":len(records), "duplicates":sum(x["duplicate"] for x in records),
                      "contradictions":sum(x["contradiction"] for x in records),
                      "empty_first":sum(x["empty1"] for x in records),
                      "empty_second":sum(x["empty2"] for x in records)}))


if __name__ == "__main__":
    main()
