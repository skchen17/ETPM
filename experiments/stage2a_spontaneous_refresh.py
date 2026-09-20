"""Re-run only the exploratory emission demo after documenting minimum-length decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage2a_eval import load_model, spontaneous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    model, tok = load_model(args.root, "etrcm", args.device)
    episodes = spontaneous(model, tok, 10)
    path = args.processed / "evaluation.json"
    record = json.loads(path.read_text())
    if "spontaneous_initial_greedy" not in record:
        record["spontaneous_initial_greedy"] = record["spontaneous"]
    record["spontaneous"] = episodes
    path.write_text(json.dumps(record, indent=2))
    report = args.report.read_text()
    start = report.index("## 14. Spontaneous-output prototype")
    stop = report.index("## 15. Baseline comparison")
    section = ["## 14. Spontaneous-output prototype", "",
               "Exploratory, non-gating interface demo. Inherited expression head was NOT trained on language emission targets; "
               "threshold 0.45 is a demonstration threshold, not a calibrated policy. "
               "**Initial unforced greedy run:** threshold fired in 10/10 trajectories, but every generated string was empty "
               "because EOS was selected immediately. Those original traces remain in `evaluation.json` as "
               "`spontaneous_initial_greedy`; this is a negative result. "
               "The following is a separate decoder-interface diagnostic, not a replacement of that finding. "
               "Decoding samples top-12 at temperature 0.9 and suppresses EOS for the first three tokens. "
               "The minimum length is forced to demonstrate the token/state interface and is not evidence of learned spontaneous intent. "
               "Generated tokens feed SELF_OUTPUT (no external write), and subsequent external events use the same state.", ""]
    for case in episodes:
        section += [f"### Trajectory {case['trajectory']} (fired={case['fired']}; final τ={case['final_tau']}; external t={case['final_external_time']})", ""]
        for event in case["events"]:
            section.append(f"- time={event['time']}; external=`{event['external_input']}`; NULL={event['internal_ticks']}; "
                           f"score={event['expression_score']:.4f}; generated=`{event.get('generated_output','')}`; "
                           f"subsequent_state={event.get('subsequent_state', event['state'])}.")
        section.append("")
    args.report.write_text(report[:start] + "\n".join(section) + "\n" + report[stop:])
    print(json.dumps({"fired": sum(x["fired"] for x in episodes),
                      "nonempty": sum(any(e.get("generated_output") for e in x["events"]) for x in episodes),
                      "outputs": [next((e.get("generated_output") for e in x["events"] if "generated_output" in e), "")
                                  for x in episodes]}))


if __name__ == "__main__":
    main()
