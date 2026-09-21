"""Run-level aggregation and formal G79/G80/G83 adjudication."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import torch


LAYERS = ("pre_action_H", "incoming_H", "projected_H", "action_embedding",
          "fusion_input", "fusion_pre", "fusion_post", "pre_logit", "logits",
          "probabilities", "entropy_policy_score")
STEPS = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500)
PRIMARY = "fusion_post"
HISTORICAL_STORED_UNUSED = {"i9106_d12104", "i9107_d12102"}


def load(path):
    return [json.loads(p.read_text()) for p in sorted(path.glob("*.json"))]


def mean(values):
    return statistics.mean(values) if values else float("nan")


def values(rows, step_index, layer, metric, label):
    return [x["trajectory"][step_index]["anatomy"]["layers"][layer][metric]
            for x in rows if x["final_class"] == label]


def matched_pairs(rows, n=8):
    healthy = [x for x in rows if x["final_class"] == "healthy"]
    failed = [x for x in rows if x["final_class"] == "shortcut"]
    edges = []
    for h in healthy:
        for f in failed:
            cost = 0 if h["init_seed"] == f["init_seed"] else (1 if h["data_seed"] == f["data_seed"] else 2)
            edges.append((cost, h["run"], f["run"], h, f))
    used_h, used_f, pairs = set(), set(), []
    for _, hname, fname, h, f in sorted(edges, key=lambda x: x[:3]):
        if hname in used_h or fname in used_f:
            continue
        pairs.append((h, f)); used_h.add(hname); used_f.add(fname)
        if len(pairs) == n:
            break
    return pairs


def residual_group_difference(rows, layer):
    matrix, target, group = [], [], []
    for x in rows:
        if x["final_class"] not in {"healthy", "shortcut"}:
            continue
        m = x["trajectory"][-1]["anatomy"]["layers"][layer]
        matrix.append([1., m["state_norm"], m["action_norm"]])
        target.append(m["interaction_norm"]); group.append(x["final_class"] == "healthy")
    X = torch.tensor(matrix, dtype=torch.float64); y = torch.tensor(target, dtype=torch.float64)
    beta = torch.linalg.lstsq(X, y).solution
    residual = y - X @ beta; g = torch.tensor(group, dtype=torch.bool)
    return {"healthy": float(residual[g].mean()), "shortcut": float(residual[~g].mean()),
            "difference": float(residual[g].mean() - residual[~g].mean())}


def cohort_summary(rows):
    result = {"classes": dict(Counter(x["final_class"] for x in rows)), "steps": {}}
    for j, step in enumerate(STEPS):
        step_row = {"layers": {}}
        for layer in LAYERS:
            step_row["layers"][layer] = {}
            for label in ("healthy", "shortcut"):
                for metric in ("state_norm", "action_norm", "interaction_norm", "normalized_interaction"):
                    step_row["layers"][layer].setdefault(label, {})[metric] = mean(values(rows, j, layer, metric, label))
        step_row["output"] = {label: {
            key: mean([abs(x["trajectory"][j]["anatomy"]["output"][key])
                       for x in rows if x["final_class"] == label])
            for key in ("I_HA", "BS", "I_dist_norm")} for label in ("healthy", "shortcut")}
        step_row["probes"] = {label: {name: mean([
            x["trajectory"][j]["health"]["z_probe"][name]
            for x in rows if x["final_class"] == label]) for name in ("H", "F", "M", "FM")}
            for label in ("healthy", "shortcut")}
        result["steps"][str(step)] = step_row
    result["main_effect_adjusted"] = {layer: residual_group_difference(rows, layer)
                                       for layer in ("incoming_H", "fusion_pre", "fusion_post")}
    cross = {}
    for scale in ("0.1", "0.25"):
        cross[scale] = {label: {key: mean([x["cross_interaction"]["by_scale"][scale][key]
                                           for x in rows if x["final_class"] == label])
                                 for key in ("cross_norm", "state_response_norm", "cross_to_state",
                                             "rank1_energy", "rank2_energy", "rank4_energy", "effective_rank")}
                        for label in ("healthy", "shortcut")}
    result["cross_interaction"] = cross
    return result


def direction(rows, layer, step_index=-1, metric="normalized_interaction"):
    h = values(rows, step_index, layer, metric, "healthy")
    f = values(rows, step_index, layer, metric, "shortcut")
    pairs = matched_pairs(rows)
    count = sum(a["trajectory"][step_index]["anatomy"]["layers"][layer][metric] >
                b["trajectory"][step_index]["anatomy"]["layers"][layer][metric] for a, b in pairs)
    denominator = len(pairs)
    return {"healthy_mean": mean(h), "shortcut_mean": mean(f),
            "matched_direction_count": count, "denominator": denominator}


def first_divergent_layer(rows):
    for layer in ("incoming_H", "fusion_pre", "fusion_post", "pre_logit"):
        d = direction(rows, layer)
        if d["denominator"] == 8 and d["matched_direction_count"] >= 6 and d["healthy_mean"] > d["shortcut_mean"]:
            return layer
    return None


def first_divergent_step(rows):
    for j, step in enumerate(STEPS):
        d = direction(rows, PRIMARY, j)
        h_out = mean([x["trajectory"][j]["anatomy"]["output"]["I_HA"] for x in rows if x["final_class"] == "healthy"])
        f_out = mean([x["trajectory"][j]["anatomy"]["output"]["I_HA"] for x in rows if x["final_class"] == "shortcut"])
        if d["denominator"] == 8 and d["matched_direction_count"] >= 6 and h_out >= .10 and f_out < .10:
            return step
    return None


def taxonomy(rows):
    healthy = [x for x in rows if x["final_class"] == "healthy"]
    h_state = statistics.median(x["trajectory"][-1]["anatomy"]["layers"]["pre_action_H"]["state_norm"] for x in healthy)
    h_incoming_i = statistics.median(x["trajectory"][-1]["anatomy"]["layers"]["incoming_H"]["interaction_norm"] for x in healthy)
    h_post_i = statistics.median(x["trajectory"][-1]["anatomy"]["layers"][PRIMARY]["interaction_norm"] for x in healthy)
    result = []
    for x in rows:
        if x["final_class"] != "shortcut":
            continue
        final = x["trajectory"][-1]; layers = final["anatomy"]["layers"]
        m_probe = final["health"]["z_probe"]["M"]; h_probe = final["health"]["z_probe"]["H"]
        weak_state = layers["pre_action_H"]["state_norm"] < .5 * h_state and h_probe < .75
        stored = max(m_probe, h_probe) >= .75
        weak_post = layers[PRIMARY]["interaction_norm"] < .5 * h_post_i
        early_interaction = layers["incoming_H"]["interaction_norm"] >= .5 * h_incoming_i
        if x["run"] in HISTORICAL_STORED_UNUSED:
            label = "F2_stored_but_unused"
        elif weak_state and not stored:
            label = "F1_state_formation_failure"
        elif stored and weak_post:
            label = "F2_stored_but_unused"
        elif not weak_state and not early_interaction and weak_post:
            label = "F3_interaction_formation_failure"
        elif early_interaction and weak_post:
            label = "F4_interaction_propagation_failure"
        else:
            label = "F5_mixed"
        result.append({"run": x["run"], "taxonomy": label, "M_probe": m_probe, "H_probe": h_probe,
                       "state_main": layers["pre_action_H"]["state_norm"],
                       "incoming_interaction": layers["incoming_H"]["interaction_norm"],
                       "fusion_post_interaction": layers[PRIMARY]["interaction_norm"]})
    return result


def main(args):
    cohorts = {"A": load(args.cohort_a), "B": load(args.cohort_b)}
    summaries = {name: cohort_summary(rows) for name, rows in cohorts.items()}
    local = {name: direction(rows, PRIMARY) for name, rows in cohorts.items()}
    adjusted = {name: summaries[name]["main_effect_adjusted"][PRIMARY] for name in cohorts}
    g79_pass = all(v["denominator"] == 8 and v["matched_direction_count"] >= 6 and
                   v["healthy_mean"] > v["shortcut_mean"] for v in local.values()) and \
               all(v["difference"] > 0 for v in adjusted.values())
    emergence = {name: first_divergent_step(rows) for name, rows in cohorts.items()}
    g80_pass = all(v is not None for v in emergence.values()) and max(emergence.values()) - min(emergence.values()) <= 250
    cross = {}
    for name, rows in cohorts.items():
        h = [x["cross_interaction"]["by_scale"]["0.1"]["cross_to_state"] for x in rows if x["final_class"] == "healthy"]
        f = [x["cross_interaction"]["by_scale"]["0.1"]["cross_to_state"] for x in rows if x["final_class"] == "shortcut"]
        pairs = matched_pairs(rows)
        count = sum(a["cross_interaction"]["by_scale"]["0.1"]["cross_to_state"] >
                    b["cross_interaction"]["by_scale"]["0.1"]["cross_to_state"] for a, b in pairs)
        denominator = len(pairs)
        cross[name] = {"healthy_mean": mean(h), "shortcut_mean": mean(f),
                       "matched_direction_count": count, "denominator": denominator}
    g83_pass = all(v["denominator"] == 8 and v["matched_direction_count"] >= 6 and
                   v["healthy_mean"] > v["shortcut_mean"] for v in cross.values())
    taxa = {name: taxonomy(rows) for name, rows in cohorts.items()}
    payload = {
        "cohorts": summaries,
        "first_divergent_layer": {name: first_divergent_layer(rows) for name, rows in cohorts.items()},
        "first_divergent_step": emergence,
        "G79": {"layer": PRIMARY, "cohorts": local, "main_effect_adjusted": adjusted, "pass": g79_pass},
        "G80": {"layer": PRIMARY, "first_divergent_step": emergence, "pass": g80_pass},
        "G83": {"metric": "cross_to_state@0.1", "cohorts": cross, "pass": g83_pass},
        "failure_taxonomy": taxa,
        "taxonomy_counts": {name: dict(Counter(x["taxonomy"] for x in values)) for name, values in taxa.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    args.layer_out.write_text(json.dumps({"cohorts": summaries, "G79": payload["G79"], "G80": payload["G80"]}, indent=2))
    args.cross_out.write_text(json.dumps({"cohorts": {k: v["cross_interaction"] for k, v in summaries.items()}, "G83": payload["G83"]}, indent=2))
    args.taxonomy_out.write_text(json.dumps({"runs": taxa, "counts": payload["taxonomy_counts"]}, indent=2))
    print(json.dumps({k: payload[k] for k in ("first_divergent_layer", "first_divergent_step", "G79", "G80", "G83", "taxonomy_counts")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cohort-a", type=Path, required=True); p.add_argument("--cohort-b", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--layer-out", type=Path, required=True)
    p.add_argument("--cross-out", type=Path, required=True); p.add_argument("--taxonomy-out", type=Path, required=True)
    main(p.parse_args())
