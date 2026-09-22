"""Model-level Stage 2D.7 summaries; no history-pair/neuron pseudoreplication."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


COHORTS = ("historical_legacy", "stage2d6_confirmatory", "stage2d7_confirmatory")


def load(folder):
    return [json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))]


def mean(values):
    return statistics.mean(values) if values else None


def med(values):
    return statistics.median(values) if values else None


def endpoint(row):
    return row["endpoint"]


def metric(row, key):
    return endpoint(row)["visibility"][key]


def group_summary(rows):
    result = {"count": len(rows), "classes": dict(Counter(
        endpoint(r)["class"] for r in rows)), "phenotypes": dict(Counter(
        endpoint(r)["phenotype"] for r in rows))}
    result["integrity"] = {
        "max_hook_error": max((metric(r, "hook_error") for r in rows), default=None),
        "max_per_action_hook_error": max((metric(r, "per_action_hook_error") for r in rows), default=None),
        "max_svd_reconstruction_error": max((metric(r, "svd_reconstruction_error") for r in rows), default=None),
        "matrix_ranks": dict(Counter(metric(r, "matrix_rank") for r in rows)),
        "all_protected_heads_match": all(endpoint(r)["protected_head_matches_checkpoint"] for r in rows),
        "all_model_parameters_unchanged": all(endpoint(r)["parameters_unchanged"] for r in rows),
    }
    for label in ("healthy", "F1_state_formation", "F2_stored_unused", "other_partial"):
        members = [r for r in rows if endpoint(r)["phenotype"] == label]
        result[label] = {"count": len(members)}
        for key in ("D_H", "D_S", "G_proj", "V_H"):
            result[label][key] = mean([metric(r, key) for r in members])
            result[label][key + "_median"] = med([metric(r, key) for r in members])
        for k in (1, 2, 4, 8, 16):
            result[label][f"E_top_{k}"] = mean([metric(r, "energy")[str(k)]["top"] for r in members])
            result[label][f"E_bottom_{k}"] = mean([metric(r, "energy")[str(k)]["bottom"] for r in members])
        for name in ("H", "M"):
            result[label][f"probe_{name}"] = mean([endpoint(r)["health"]["z_probe"][name]
                                                     for r in members])
        result[label]["IHA"] = mean([endpoint(r)["behavior"]["IHA"] for r in members])
        result[label]["BS_abs"] = mean([abs(endpoint(r)["behavior"]["BS"]) for r in members])
    return result


def rescue_pass(row):
    x = endpoint(row)["state_main_rescue"]
    native = x["native"]["behavior"]
    test = x["aligned_state"]["1.0"]
    value = test["behavior"]
    controls = [x[name]["1.0"]["behavior"] for name in (
        "random_state", "low_sensitivity_state", "action_main", "common_shift")]
    return (value["TV"] >= .10 and value["IHA"] >= .10 and abs(value["BS"]) >= .10
            and value["IHA"] - native["IHA"] >= .025
            and abs(value["BS"]) - abs(native["BS"]) >= .025
            and test["fusion_post_I"] - x["native"]["fusion_post_I"] >= .025
            and all(value["IHA"] - c["IHA"] >= .025 and
                    abs(value["BS"]) - abs(c["BS"]) >= .025 for c in controls)
            and test["node_interaction_delta"] < 1e-5
            and test["node_action_delta"] < 1e-5
            and x["cap_ratio"] <= 1.25 + 1e-9)


def destruction_pass(row):
    x = endpoint(row)["healthy_destruction"]
    native = x["native"]["behavior"]
    remove = x["0.0"]
    value = remove["behavior"]
    controls = [x["common_shift"]["behavior"], x["action_main_control"]["behavior"]]
    return (remove["fusion_post_I"] <= .75 * x["native"]["fusion_post_I"]
            and value["IHA"] <= .5 * native["IHA"]
            and abs(value["BS"]) <= .5 * abs(native["BS"])
            and all(c["IHA"] - value["IHA"] >= .025 and
                    abs(c["BS"]) - abs(value["BS"]) >= .025 for c in controls)
            and remove["node_action_delta"] < 1e-5
            and remove["node_interaction_delta"] < 1e-5)


def trajectory_onset(row):
    trace = row.get("trajectory", [])
    if not trace:
        return None
    final = trace[-1]
    base = trace[0]
    thresholds = {}
    for name, key in (("D_H", "D_H"), ("D_S", "D_S"), ("G_proj", "G_proj")):
        first = base["visibility"][key]
        last = final["visibility"][key]
        halfway = first + (last-first)/2
        thresholds[name] = next((x["step"] for x in trace
            if (x["visibility"][key] >= halfway if last >= first else
                x["visibility"][key] <= halfway)), None)
    thresholds["behavior"] = next((x["step"] for x in trace
                                     if x["behavior"]["IHA"] >= .10), None)
    thresholds["probe_H"] = next((x["step"] for x in trace
                                    if x["health"]["z_probe"]["H"] >= .75), None)
    thresholds["projection_precedes"] = bool(thresholds["behavior"] is not None and
                                           thresholds["G_proj"] is not None and
                                           thresholds["G_proj"] <= thresholds["behavior"])
    return thresholds


def temporal_group(rows):
    by_step = {}
    healthy = [r for r in rows if endpoint(r)["class"] == "healthy" and r.get("trajectory")]
    failed = [r for r in rows if endpoint(r)["class"] != "healthy" and r.get("trajectory")]
    for index, step in enumerate((0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500)):
        def values(group, key):
            if key in ("D_H", "D_S", "G_proj"):
                return [r["trajectory"][index]["visibility"][key] for r in group]
            if key == "probe_H":
                return [r["trajectory"][index]["health"]["z_probe"]["H"] for r in group]
            if key == "IHA":
                return [r["trajectory"][index]["behavior"]["IHA"] for r in group]
            if key == "fusion_post_I":
                return [r["trajectory"][index]["fusion_post_I"] for r in group]
            if key == "BS_abs":
                return [abs(r["trajectory"][index]["behavior"]["BS"]) for r in group]
            raise KeyError(key)
        by_step[str(step)] = {group_name: {key: mean(values(group, key))
            for key in ("D_H", "D_S", "G_proj", "probe_H", "fusion_post_I", "IHA", "BS_abs")}
            for group_name, group in (("healthy", healthy), ("failed", failed))}
    return by_step


def refit_pass(row):
    arms = row["arms"]
    r1 = arms["R1"]["heldout_health"]
    if arms["R1"]["heldout_class"] != "healthy":
        return False
    r0 = arms["R0"]["heldout_health"]
    if r1["I_HA"] - r0["I_HA"] < .025 or abs(r1["BS"]) - abs(r0["BS"]) < .025:
        return False
    return all(r1["I_HA"] - arms[name]["heldout_health"]["I_HA"] >= .025 and
               abs(r1["BS"]) - abs(arms[name]["heldout_health"]["BS"]) >= .025
               for name in ("R2", "R3", "R4"))


def passes_behavior(value):
    return value["TV"] >= .10 and value["IHA"] >= .10 and abs(value["BS"]) >= .10


def intervention_summary(rows):
    healthy = [r for r in rows if endpoint(r)["class"] == "healthy"]
    failed = [r for r in rows if endpoint(r)["class"] != "healthy"]
    rescue = {key: {"behavioral_threshold_count": sum(passes_behavior(
                 endpoint(r)["state_main_rescue"][key]["1.0"]["behavior"]) for r in failed),
                    "mean_IHA": mean([endpoint(r)["state_main_rescue"][key]["1.0"]["behavior"]["IHA"]
                                     for r in failed]),
                    "mean_abs_BS": mean([abs(endpoint(r)["state_main_rescue"][key]["1.0"]["behavior"]["BS"])
                                        for r in failed])}
              for key in ("aligned_state", "random_state", "low_sensitivity_state",
                          "action_main", "common_shift")}
    for key in ("native", "fusion_post_positive"):
        rescue[key] = {"behavioral_threshold_count": sum(passes_behavior(
            endpoint(r)["state_main_rescue"][key]["behavior"]) for r in failed),
                       "mean_IHA": mean([endpoint(r)["state_main_rescue"][key]["behavior"]["IHA"]
                                        for r in failed]),
                       "mean_abs_BS": mean([abs(endpoint(r)["state_main_rescue"][key]["behavior"]["BS"])
                                           for r in failed])}
    dose = {}
    for beta in ("0.0", "0.25", "0.5", "0.75", "1.0"):
        dose[beta] = {"post_I": mean([endpoint(r)["healthy_destruction"][beta]["fusion_post_I"]
                                       for r in healthy]),
                      "IHA": mean([endpoint(r)["healthy_destruction"][beta]["behavior"]["IHA"]
                                   for r in healthy]),
                      "BS_abs": mean([abs(endpoint(r)["healthy_destruction"][beta]["behavior"]["BS"])
                                      for r in healthy])}
    return {"failed_count": len(failed), "healthy_count": len(healthy),
            "rescue": rescue, "healthy_beta_dose": dose}


def main(args):
    root = Path("results/stage2d7")
    data = {cohort: load(root / "projection_visibility" / cohort) for cohort in COHORTS}
    summary = {cohort: group_summary(rows) for cohort, rows in data.items()}
    detail = {}
    for cohort, rows in data.items():
        healthy = [r for r in rows if endpoint(r)["class"] == "healthy"]
        failed = [r for r in rows if endpoint(r)["class"] != "healthy"]
        f2 = [r for r in rows if endpoint(r)["phenotype"] == "F2_stored_unused"]
        detail[cohort] = {
            "healthy_runs": [r["run"] for r in healthy],
            "failed_runs": [r["run"] for r in failed],
            "F2_runs": [r["run"] for r in f2],
            "rescue": {r["run"]: rescue_pass(r) for r in failed},
            "destruction": {r["run"]: destruction_pass(r) for r in healthy},
            "onsets": {r["run"]: trajectory_onset(r) for r in rows},
            "temporal_group": temporal_group(rows),
            "intervention_summary": intervention_summary(rows),
        }
        detail[cohort]["rescue_count_first8"] = sum(list(detail[cohort]["rescue"].values())[:8])
        detail[cohort]["destruction_count_first8"] = sum(list(detail[cohort]["destruction"].values())[:8])
        detail[cohort]["projection_gain_direction_first8"] = sum(
            metric(h, "G_proj") > metric(f, "G_proj") for h, f in zip(healthy[:8], failed[:8]))
        detail[cohort]["high_gain_energy_direction_first8"] = sum(
            metric(h, "energy")["8"]["top"] > metric(f, "energy")["8"]["top"]
            for h, f in zip(healthy[:8], failed[:8]))
    refits = {}
    for cohort in COHORTS:
        rows = load(root / "projection_refit_ceiling" / cohort)
        refits[cohort] = {"count": len(rows), "R0_R4_classes":
                         {r["run"]: {k: v["heldout_class"] for k, v in r["arms"].items()}
                          for r in rows},
                         "R1_gate_pass": {r["run"]: refit_pass(r) for r in rows},
                         "means": {arm: {key: mean([
                             abs(r["arms"][arm]["heldout_health"][key]) if key == "BS" else
                             r["arms"][arm]["heldout_health"][key] for r in rows])
                             for key in ("action_TV", "I_HA", "BS", "CFA")}
                             for arm in ("R0", "R1", "R2", "R3", "R4")}}
    payload = {"cohort_summary": summary, "detail": detail, "refits": refits,
               "model_is_independent_unit": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"cohorts": {k: v["count"] for k, v in summary.items()},
                      "summary": {k: {name: {m: value[name][m] for m in
                          ("count", "D_H", "D_S", "G_proj", "E_top_8", "E_bottom_8")}
                          for name in ("healthy", "F1_state_formation", "F2_stored_unused")}
                          for k, value in summary.items()},
                      "counts": {k: {x: detail[k][x] for x in
                          ("rescue_count_first8", "destruction_count_first8",
                           "projection_gain_direction_first8", "high_gain_energy_direction_first8")}
                          for k in detail},
                      "refits": {k: {"count": v["count"], "R1_pass": sum(v["R1_gate_pass"].values())}
                                 for k, v in refits.items()}}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("results/stage2d7/processed/summary.json"))
    main(p.parse_args())
