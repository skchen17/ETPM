"""Frozen model-level Stage 2D.7 gates and stopping-rule decisions."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path("results/stage2d7")


def training_classes(folder):
    rows = [json.loads(p.read_text()) for p in sorted(folder.rglob("evaluation.json"))]
    by_config = {}
    for row in rows:
        key = row["arm"]
        by_config.setdefault(key, []).append(row)
    return by_config


def onsets(detail, cohort):
    names = detail[cohort]["healthy_runs"]
    values = [detail[cohort]["onsets"][name] for name in names]
    return {"healthy_count": len(values),
            "D_S_before_or_at_IHA": sum(v and v["D_S"] is not None and
                                      v["behavior"] is not None and
                                      v["D_S"] <= v["behavior"] for v in values),
            "G_proj_before_or_at_IHA": sum(v and v["G_proj"] is not None and
                                         v["behavior"] is not None and
                                         v["G_proj"] <= v["behavior"] for v in values)}


def main(args):
    summary = json.loads((ROOT / "processed/summary.json").read_text())
    groups = summary["cohort_summary"]
    detail = summary["detail"]
    refits = summary["refits"]
    if groups["stage2d7_confirmatory"]["count"] != 24:
        raise AssertionError("new 24-run C0 cohort incomplete")
    selection = json.loads((ROOT / "processed/frozen_development_selection.json").read_text())
    selected_config = selection["selected"]
    formal = training_classes(ROOT / "training_rescue/formal_evaluation")
    if any(len(formal.get(arm, [])) != 8 for arm in ("C0", "C1", "C4", "C5")):
        raise AssertionError("formal selected/control cohort incomplete")
    formal_counts = {arm: dict(Counter(r["class"] for r in rows))
                     for arm, rows in formal.items()}
    selected_arm = selected_config.split("_")[0]
    selected_healthy = formal_counts[selected_arm].get("healthy", 0)
    controls = {arm: formal_counts[arm].get("healthy", 0) for arm in ("C0", "C4", "C5")}
    old, d6, new = (detail[k] for k in (
        "historical_legacy", "stage2d6_confirmatory", "stage2d7_confirmatory"))
    gate = {}
    def setgate(number, status, evidence):
        gate[f"G{number}"] = {"status": status, "evidence": evidence}
    setgate(123, "FAIL", {
        "gain_direction_first8": {"historical": old["projection_gain_direction_first8"],
                                  "stage2d6": d6["projection_gain_direction_first8"],
                                  "stage2d7": new["projection_gain_direction_first8"]},
        "healthy_counts": {k: groups[k]["healthy"]["count"] for k in groups},
        "reason": "no two fully powered independent cohorts and no stable F2 gain deficit"})
    setgate(124, "FAIL", {"top8_direction_first8": {
        "historical": old["high_gain_energy_direction_first8"],
        "stage2d6": d6["high_gain_energy_direction_first8"],
        "stage2d7": new["high_gain_energy_direction_first8"]}})
    rescue_counts = {"historical": old["rescue_count_first8"],
                     "stage2d6": d6["rescue_count_first8"],
                     "stage2d7": new["rescue_count_first8"]}
    setgate(125, "PASS" if all(value >= 6 for value in rescue_counts.values()) else "FAIL",
            {"run_level_pass_first8": rescue_counts})
    destruction_counts = {"historical": old["destruction_count_first8"],
                          "stage2d6": d6["destruction_count_first8"],
                          "stage2d7": new["destruction_count_first8"]}
    enough = (destruction_counts["historical"] >= 6 and
              destruction_counts["stage2d6"] >= 6)
    setgate(126, "PASS_LIMITED" if enough and groups["stage2d7_confirmatory"]["healthy"]["count"] < 8
            else ("PASS" if enough and destruction_counts["stage2d7"] >= 6 else "FAIL"),
            {"run_level_pass_first8": destruction_counts,
             "new_healthy_count": groups["stage2d7_confirmatory"]["healthy"]["count"]})
    setgate(127, "NOT_RUN", "G123 and G124 did not jointly support H-space rotation")
    refit_counts = {k: sum(refits[k]["R1_gate_pass"].values()) for k in refits}
    setgate(128, "PASS" if refit_counts["historical_legacy"] >= 6 and
            refit_counts["stage2d7_confirmatory"] >= 6 else "FAIL",
            {"R1_vs_R0_R2_R3_R4": refit_counts})
    timing = {cohort: onsets(detail, cohort) for cohort in detail}
    setgate(129, "PASS" if timing["historical_legacy"]["D_S_before_or_at_IHA"] >= 6 and
            timing["stage2d6_confirmatory"]["D_S_before_or_at_IHA"] >= 6 and
            timing["stage2d7_confirmatory"]["D_S_before_or_at_IHA"] >= 6 else "FAIL",
            timing)
    authorized = sum(gate[f"G{x}"]["status"].startswith("PASS") for x in (125, 126, 127, 128)) >= 2
    setgate(130, "PASS" if selected_healthy >= 6 and
            all(selected_healthy > count for count in controls.values()) else "FAIL",
            {"selected": selected_config, "selected_healthy": selected_healthy,
             "formal_n": 8, "controls_healthy": controls})
    for number, reason in ((131, "G130 failed; expanded 24-run crossing not authorized"),
                           (132, "G130 failed; no stabilized selected model for mechanism preservation"),
                           (133, "G130/G132 not both passed"),
                           (134, "G133 did not pass; no formal F/M mediation rerun"),
                           (135, "no selected stabilized model for 10k continuous diagnostic")):
        setgate(number, "NOT_RUN", reason)
    payload = {"gates": gate, "training_rescue_authorized": authorized,
               "development_selected": selected_config,
               "formal_class_counts": formal_counts,
               "formal_healthy_rate": {arm: formal_counts[arm].get("healthy", 0)/8
                                       for arm in formal_counts},
               "formal_metric_means": {arm: {
                   "action_TV": statistics.mean(r["health"]["action_TV"] for r in rows),
                   "I_HA": statistics.mean(r["health"]["I_HA"] for r in rows),
                   "BS_abs": statistics.mean(abs(r["health"]["BS"]) for r in rows),
                   "CFA": statistics.mean(r["health"]["CFA"] for r in rows)}
                   for arm, rows in formal.items()},
               "expanded_replication_run": False,
               "dynamics_recheck_run": False,
               "memory_mediation_run": False,
               "continuous_runs_run": False,
               "outcome": "C — state formation dominates; finite state-interface sensitivity is not a replicated frozen-coordinate bottleneck",
               "fusion_redesign_justified": False,
               "memory_law_redesign_justified": False,
               "formal_F_to_M_handoff_reopened": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    deferred = {
        "h_coordinate_rotation": "G123 and G124 failed; conditional H-space experiment not authorized",
        "expanded_replication": "G130 failed at 1/8; expanded crossing not authorized",
        "dynamics_recheck": "G130 and G132 did not both pass",
        "memory_mediation": "G133 did not pass; no F/M formation-window rerun",
        "continuous_runs": "no selected stabilized model after G130 failure",
    }
    for family, reason in deferred.items():
        folder = ROOT / family
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "status.json").write_text(json.dumps({
            "stage": "2D.7", "status": "NOT_RUN", "reason": reason}, indent=2))
    print(json.dumps({"gates": {k: v["status"] for k, v in gate.items()},
                      "formal_healthy": payload["formal_healthy_rate"],
                      "outcome": payload["outcome"]}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=ROOT / "processed/formal_gates.json")
    main(p.parse_args())
