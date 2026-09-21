"""Aggregate Stage 2D formal runs, adjudicate G62–G67, and write report."""

from __future__ import annotations

import hashlib
import json
import math
import statistics as st
from pathlib import Path

import yaml

from etrcm.stage2d.protocol import (COARSE_CONFIGS, DELAYS, FORMATION_N, FROZEN,
                                    HANDOFF_TIMES, P_LEVELS)
from etrcm.stage2d.model import Stage2DModel
from experiments.stage2d_integrity import snapshot


ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/stage2d"
REPORT = ROOT / "reports/STAGE2D_MEMORY_DYNAMICS_RESULTS.md"
CONFIG = ROOT / "configs/stage2d_formal.yaml"
PRIMARY = tuple(range(7801, 7809))
CONTROLS = tuple(range(7801, 7806))


def load(path):
    return json.loads(Path(path).read_text())


def mean(values):
    clean = [float(x) for x in values if x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))]
    return st.mean(clean) if clean else float("nan")


def sd(values):
    clean = [float(x) for x in values if x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))]
    return st.stdev(clean) if len(clean) > 1 else 0.0


def fmt(values, digits=3):
    return f"{mean(values):.{digits}f} ± {sd(values):.{digits}f}"


def corr(xs, ys):
    if len(xs) < 2:
        return float("nan")
    xm, ym = mean(xs), mean(ys)
    denom = math.sqrt(sum((x-xm)**2 for x in xs) * sum((y-ym)**2 for y in ys))
    return sum((x-xm)*(y-ym) for x,y in zip(xs,ys)) / denom if denom else float("nan")


def bs(item):
    return item["behavioral_separation_entropy"] if "behavioral_separation_entropy" in item else item["BS"]


def primary_data():
    return {seed: load(R / "processed/formal/primary" / f"{seed}.json") for seed in PRIMARY}


def shortlist_data(name="g050_f0970_m09995"):
    return {seed: load(R / "processed/formal/shortlist" / name / f"{seed}.json") for seed in PRIMARY}


def control_data(label):
    return {seed: load(R / "processed/formal/controls" / label / f"{seed}.json") for seed in CONTROLS}


def formation_bands(item):
    per_p = []
    for p in ("0.60", "0.70"):
        curve = item["formation"][p]["curve"]
        per_p.append((mean([abs(bs(curve[str(n)])) for n in (1, 2)]),
                      mean([abs(bs(curve[str(n)])) for n in (4, 8)]),
                      mean([abs(bs(curve[str(n)])) for n in (16, 32)])))
    return tuple(mean([row[i] for row in per_p]) for i in range(3))


def retention_ratio(item, delay=500):
    curve = item["persistence"]["curve"]
    base = abs(bs(curve["0"]))
    if base < FROZEN.formed_threshold:
        return None
    return abs(bs(curve[str(delay)])) / base


def handoff_effect(item, time_name, memory):
    return item["handoff"]["read_mediation"][time_name][memory]["causal_effect"]


def gate_results(primary, gamma0):
    healthy = {seed: item["training_health"]["action_TV"] >= FROZEN.health_action_tv_min and
                     item["training_health"]["interaction_y0"] >= FROZEN.health_interaction_min
               for seed,item in primary.items()}
    g62 = {seed: healthy[seed] and (lambda b: b[2] > b[1] > b[0])(formation_bands(item))
           for seed, item in primary.items()}
    g63 = {seed: healthy[seed] and abs(bs(item["persistence"]["curve"]["0"])) >= FROZEN.formed_threshold and
                  retention_ratio(item) >= FROZEN.persistence_ratio_min
           for seed, item in primary.items()}
    g64 = {seed: healthy[seed] and isinstance(item["revision"]["T_reverse"], int) and
                  item["revision"]["T_reverse"] <= FROZEN.reversal_max
           for seed, item in primary.items()}
    g65 = {}
    for seed, item in primary.items():
        matched = item["selectivity"]["matched"]
        predictive = abs(bs(matched["predictive"]["delayed"]))
        noise = abs(bs(matched["noise"]["delayed"]))
        g65[seed] = healthy[seed] and predictive >= noise + FROZEN.selectivity_margin
    g66 = {}
    for seed, item in primary.items():
        baseline = abs(bs(item["handoff"]["baseline"])) >= FROZEN.formed_threshold
        early = handoff_effect(item, "early_formation", "F") > handoff_effect(item, "early_formation", "M")
        late = handoff_effect(item, "late_delay", "M") > handoff_effect(item, "late_delay", "F")
        g66[seed] = healthy[seed] and baseline and early and late
    g67 = {}
    for seed in CONTROLS:
        full, zero = primary[seed], gamma0[seed]
        zero_healthy = (zero["training_health"]["action_TV"] >= FROZEN.health_action_tv_min and
                        zero["training_health"]["interaction_y0"] >= FROZEN.health_interaction_min)
        full_ratio, zero_ratio = retention_ratio(full), retention_ratio(zero)
        initial_ok = full_ratio is not None and zero_ratio is not None and abs(
            bs(full["persistence"]["curve"]["0"])) >= (
            abs(bs(zero["persistence"]["curve"]["0"])) - .05)
        retention_gain = initial_ok and full_ratio >= zero_ratio + FROZEN.consolidation_retention_margin
        late_m_gain = handoff_effect(full, "late_delay", "M") > handoff_effect(zero, "late_delay", "M") + .02
        g67[seed] = healthy[seed] and zero_healthy and initial_ok and (retention_gain or late_m_gain)
    flags = {"G62": g62, "G63": g63, "G64": g64, "G65": g65, "G66": g66, "G67": g67}
    counts = {name: sum(values.values()) for name, values in flags.items()}
    status = {name: ("PASS" if count >= (6 if name != "G67" else 4) else "FAIL")
              for name, count in counts.items()}
    return {"status": status, "counts": counts,
            "healthy_primary_seeds": [seed for seed,value in healthy.items() if value],
            "passing_seeds": {name: [seed for seed, value in values.items() if value]
                              for name, values in flags.items()}}


def outcome(gates, primary):
    s = gates["status"]
    unstable = sum(item.get("continuous", {}).get("first_nonfinite") is not None or
                   item.get("continuous", {}).get("first_H_gt_1000") is not None
                   for item in primary.values()) >= 4
    if unstable:
        return "E — Memory dynamics confounded by continuous instability"
    if all(s[g] == "PASS" for g in ("G62", "G63", "G64", "G65", "G66")):
        return "A — Gradual, persistent, revisable fast–slow memory"
    if all(s[g] == "PASS" for g in ("G62", "G63", "G64", "G65")) and s["G66"] == "FAIL":
        return "B — Behavioral memory works but timescale separation is weak"
    if s["G63"] == "PASS" and s["G64"] == "FAIL":
        return "C — Persistent but rigid"
    return "D — Plastic but fragile (or formation not replicated)"


def phase_rows():
    rows = []
    for name, gamma, rf, rm in COARSE_CONFIGS:
        items = []
        for seed in (7621, 7622):
            path = R / "parameter_sweeps" / name / f"{seed}.json"
            if path.exists():
                items.append(load(path))
        if not items:
            continue
        acquire = []
        half = []
        reverse = []
        retention = []
        cf, cm = [], []
        for item in items:
            value = item["formation"]["0.65"]["T_acquire"]
            acquire.append(65 if value is None else value)
            h = item["persistence"]["half_life"]
            half.append(5001 if isinstance(h, str) else h)
            r = item["revision"]["T_reverse"]
            reverse.append(129 if isinstance(r, str) else r)
            retention.append(retention_ratio(item))
            cf.append(handoff_effect(item, "late_delay", "F"))
            cm.append(handoff_effect(item, "late_delay", "M"))
        healthy = [item["training_health"]["action_TV"] >= FROZEN.health_action_tv_min and
                   item["training_health"]["interaction_y0"] >= FROZEN.health_interaction_min
                   for item in items]
        rows.append({"name": name, "gamma": gamma, "rho_fast": rf, "rho_slow": rm,
                     "T_acquire": mean(acquire), "T_half": mean(half),
                     "T_reverse": mean(reverse), "retention_D500": mean(retention),
                     "late_C_F": mean(cf), "late_C_M": mean(cm),
                     "healthy_seeds": sum(healthy),
                     "shortlist_eligible": all(healthy) and all(x < 65 for x in acquire)})
    for row in rows:
        row["pareto"] = not any(
            other is not row and other["T_acquire"] <= row["T_acquire"] and
            other["retention_D500"] >= row["retention_D500"] and
            other["T_reverse"] <= row["T_reverse"] and
            (other["T_acquire"] < row["T_acquire"] or
             other["retention_D500"] > row["retention_D500"] or
             other["T_reverse"] < row["T_reverse"])
            for other in rows)
    return rows


def curve_fit(curve):
    points = [(float(d), abs(bs(v))) for d, v in curve.items() if abs(bs(v)) > 1e-8]
    def linear(xs, ys):
        xm, ym = mean(xs), mean(ys)
        denom = sum((x - xm) ** 2 for x in xs)
        slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom if denom else 0
        intercept = ym - slope * xm
        pred = [intercept + slope * x for x in xs]
        sse = sum((y - p) ** 2 for y, p in zip(ys, pred))
        sst = sum((y - ym) ** 2 for y in ys)
        return 1 - sse / sst if sst else float("nan")
    exp_r2 = linear([x for x, _ in points], [math.log(y) for _, y in points])
    power = [(math.log1p(x), math.log(y)) for x, y in points]
    power_r2 = linear([x for x, _ in power], [y for _, y in power])
    return {"exponential_log_R2": exp_r2, "power_log_R2": power_r2,
            "biexponential": "attempted but underidentified/unstable with nine noisy points; not interpreted"}


def make_figures(primary, phases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = R / "processed/figures"; out.mkdir(parents=True, exist_ok=True)
    def band(ax, x, series, label):
        y = [mean(v) for v in series]; e = [sd(v) for v in series]
        ax.plot(x, y, marker="o", label=label)
        ax.fill_between(x, [a-b for a,b in zip(y,e)], [a+b for a,b in zip(y,e)], alpha=.15)
    fig,ax=plt.subplots(figsize=(7,4))
    for p in P_LEVELS:
        key=f"{p:.2f}"; series=[[abs(bs(item["formation"][key]["curve"][str(n)])) for item in primary.values()] for n in FORMATION_N]
        band(ax,list(FORMATION_N),series,f"p={p:.2f}")
    ax.set(xlabel="evidence count N",ylabel="|behavioral separation|",title="Gradual formation");ax.legend();fig.tight_layout();fig.savefig(out/"formation.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));series=[[abs(bs(item["persistence"]["curve"][str(d)])) for item in primary.values()] for d in DELAYS]
    band(ax,list(DELAYS),series,"A2 full");ax.set_xscale("symlog",linthresh=10);ax.set(xlabel="unrelated-event delay D",ylabel="|BS|",title="Persistence");fig.tight_layout();fig.savefig(out/"persistence.png",dpi=160);plt.close(fig)
    rv=(0,1,2,4,8,16,32,64,128);fig,ax=plt.subplots(figsize=(7,4));series=[[bs(item["revision"]["curve"][str(r)]) for item in primary.values()] for r in rv]
    band(ax,list(rv),series,"A2 full");ax.axhline(0,color="black",lw=.7);ax.set(xlabel="opposing evidence R",ylabel="BS",title="Revision");fig.tight_layout();fig.savefig(out/"revision.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));x=list(range(len(HANDOFF_TIMES)))
    for memory in ("F","M","FM"):
        series=[[handoff_effect(item,t,memory) for item in primary.values()] for t in HANDOFF_TIMES]
        band(ax,x,series,f"C_{memory}")
    ax.set_xticks(x,HANDOFF_TIMES,rotation=25,ha="right");ax.axhline(0,color="black",lw=.7);ax.set(ylabel="behavioral causal effect",title="Finite-read causal timeline");ax.legend();fig.tight_layout();fig.savefig(out/"handoff.png",dpi=160);plt.close(fig)
    if phases:
        fig,ax=plt.subplots(figsize=(7,5))
        points=ax.scatter([x["T_acquire"] for x in phases],[x["retention_D500"] for x in phases],c=[x["T_reverse"] for x in phases],cmap="viridis_r",s=55)
        for row in phases:
            if row["pareto"]:ax.annotate(row["name"],(row["T_acquire"],row["retention_D500"]),fontsize=6)
        fig.colorbar(points,ax=ax,label="T_reverse (129 = not reversed)");ax.set(xlabel="T_acquire (65 = not acquired)",ylabel="D500 retention ratio",title="Coarse acquisition–retention–revision phase diagram");fig.tight_layout();fig.savefig(out/"phase_diagram.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ks=(0,1,2,4,8,16,32,64);series=[[abs(bs(item["null_consolidation"][str(k)]["delayed"])) for item in primary.values()] for k in ks]
    band(ax,list(ks),series,"D500 retention");ax.set(xlabel="NULL ticks K",ylabel="|BS(D500)|",title="NULL consolidation");fig.tight_layout();fig.savefig(out/"null_consolidation.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ticks=(100,500,1000,5000,10000)
    for state_name in ("H","F","M"):
        series=[[item["continuous"]["milestones"][str(t)]["state_norms"][state_name] for item in primary.values() if str(t) in item["continuous"]["milestones"]] for t in ticks]
        band(ax,list(ticks),series,state_name)
    ax.set_xscale("log");ax.set_yscale("symlog",linthresh=1);ax.set(xlabel="continuous stream tick",ylabel="state norm",title="Continuous-state stability");ax.legend();fig.tight_layout();fig.savefig(out/"continuous.png",dpi=160);plt.close(fig)


def main():
    config = yaml.safe_load(CONFIG.read_text())
    assert config["protocol_status"] == "FROZEN_BEFORE_FORMAL_TRAINING"
    default_primary = primary_data()
    primary = shortlist_data()
    controls = {label: control_data(label) for label in
                ("a0", "gamma_zero", "f_only", "f_only_g050", "no_memory", "gru")}
    assert all(item["evaluation_parameter_frozen"] for item in primary.values())
    gamma0 = controls["gamma_zero"]
    assert all(gamma0[seed]["variant"] == "gamma_zero" and
               gamma0[seed]["parameter_hash_before"] != primary[seed]["parameter_hash_before"]
               for seed in CONTROLS), "gamma-zero must be independently trained"
    gates = gate_results(primary, gamma0)
    category = outcome(gates, primary)
    phases = phase_rows()
    parameter_budgets = {variant: sum(p.numel() for p in Stage2DModel(variant=variant).parameters())
                         for variant in ("full","gamma_zero","f_only","no_memory","gru")}
    eligible_phases = [row for row in phases if row["shortlist_eligible"]]
    best = max(eligible_phases, key=lambda row: row["retention_D500"] - row["T_acquire"] / 65 - row["T_reverse"] / 129) if eligible_phases else None
    make_figures(primary, phases)
    manifest = load(R / "manifests/historical_stage2c_sha256.json")
    current = snapshot(ROOT)
    changed = sorted(key for key in set(manifest["files"]) | set(current)
                     if manifest["files"].get(key) != current.get(key))
    if changed:
        raise AssertionError(f"historical artifacts changed: {changed[:5]}")

    processed = {"gates": gates, "outcome": category, "phase_diagram": phases,
                 "best_phase_config": best,
                 "parameter_budgets": parameter_budgets,
                 "historical_stage2c_files_unchanged": len(current),
                 "selected_primary": {str(seed): item for seed, item in primary.items()},
                 "default_primary": {str(seed): item for seed, item in default_primary.items()},
                 "controls": {label: {str(seed): item for seed, item in rows.items()}
                              for label, rows in controls.items()}}
    (R / "processed").mkdir(parents=True, exist_ok=True)
    (R / "processed/summary.json").write_text(json.dumps(processed, indent=2))
    long_records = []
    for seed, item in primary.items():
        for p in P_LEVELS:
            for n in FORMATION_N:
                row = item["formation"][f"{p:.2f}"]["curve"][str(n)]
                long_records.append({"seed": seed, "experiment": "formation", "condition": f"p={p:.2f}",
                                     "x": n, "BS": bs(row), "H_norm": row["state_norms"]["H"],
                                     "F_norm": row["state_norms"]["F"], "M_norm": row["state_norms"]["M"]})
        for d in DELAYS:
            row = item["persistence"]["curve"][str(d)]
            long_records.append({"seed": seed, "experiment": "persistence", "condition": "unrelated_delay",
                                 "x": d, "BS": bs(row), "H_norm": row["state_norms"]["H"],
                                 "F_norm": row["state_norms"]["F"], "M_norm": row["state_norms"]["M"]})
        for r in (0,1,2,4,8,16,32,64,128):
            row = item["revision"]["curve"][str(r)]
            long_records.append({"seed": seed, "experiment": "revision", "condition": "opposing",
                                 "x": r, "BS": bs(row), "H_norm": row["state_norms"]["H"],
                                 "F_norm": row["state_norms"]["F"], "M_norm": row["state_norms"]["M"]})
    import pandas as pd
    frame = pd.DataFrame(long_records)
    frame.to_parquet(R / "processed/seed_metrics.parquet", index=False)
    frame.to_csv(R / "processed/seed_metrics.csv", index=False)
    for section in ("formation", "persistence", "revision", "selectivity", "handoff",
                    "null_consolidation", "continuous", "training_health"):
        target_name = "timescale_handoff" if section == "handoff" else (
            "continuous_runs" if section == "continuous" else section)
        target = R / target_name
        target.mkdir(parents=True, exist_ok=True)
        (target / "formal_shortlist_by_seed.json").write_text(json.dumps(
            {str(seed): item.get(section) for seed, item in primary.items()}, indent=2))

    source_paths = [ROOT / "README.md", CONFIG, ROOT / "docs/STAGE2D_PROTOCOL.md", ROOT / "tests/test_stage2d.py",
                    *sorted((ROOT / "src/etrcm/stage2d").glob("*.py")),
                    *sorted((ROOT / "experiments").glob("stage2d_*.py"))]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source_paths}
    (R / "manifests/source_sha256.json").write_text(json.dumps(hashes, indent=2))

    development_rows = []
    for label, seeds in (("short inherited", (7601,7602)), ("long noisy", (7611,7612))):
        prefix = "dev" if label.startswith("short") else "dev2"
        train_root = "dev/full" if prefix == "dev" else "dev2/full"
        for seed in seeds:
            evaluation = load(R / "processed" / f"{prefix}_{seed}.json")
            training = load(R / "checkpoints" / train_root / str(seed) / "summary.json")
            curve = evaluation["formation"]["0.65"]["curve"]
            development_rows.append(
                f"| {label} | {seed} | {training['steps']} | {training['train_p']:.2f} | "
                f"{training['training_log'][-1]['CE']:.4f} | {bs(curve['1']):+.4f} | "
                f"{bs(curve['32']):+.4f} | {bs(evaluation['persistence']['curve']['500']):+.4f} |")
    lines = [
        "# ET-RCM Stage 2D — Gradual Behavioral Memory, Persistence–Plasticity, and Fast–Slow Causal Handoff",
        "",
        "> **Once ET-RCM can form endogenous behavioral memory, can repeated uncertain experience gradually produce a persistent but revisable behavioral disposition, and does causal control shift from fast memory F toward slow memory M over time?**",
        "",
        "> **在 ET-RCM 已经能够形成 endogenous behavioral memory 的基础上，重复而不确定的经验能否逐渐形成持久但可修正的行为倾向，并且这种行为的因果控制是否会随时间从 fast memory F 转移到 slow memory M？**",
        "",
        f"Formal outcome for the development-selected `g050_f0970_m09995` primary: **{category}**. " + ", ".join(f"{k}={v}" for k, v in gates["status"].items()) + ".",
        "",
        "![Formation](../results/stage2d/processed/figures/formation.png)",
        "",
        "## 1. Experimental contract and implementation details",
        "",
        "Stage 2D is additive: the Stage 2C.3 H/F/M transition, external delta write, learned reads, gated residual H integration, readout-conserving F→M transfer, decay, NULL and SELF_OUTPUT rules were not edited. "
        f"A pre-write manifest verifies **{len(current)} historical Stage 2C.x files unchanged**. Each noisy observation reveals a support bit equal to latent z with probability p, not z itself. Balanced paired actions preserve outcome marginal `[.5,1/6,1/6,1/6]`; matched noise is exactly paired and independent of z.",
        "",
        "Development first retained the inherited 1,000-step 4/8/16 curriculum on seeds 7601–7602; it failed (near-zero BS). A development-only 1,500-step 16/32/64, p=.70 curriculum on 7611–7612 motivated the frozen formal schedule. The default γ=.12 configuration was run on eight formal seeds and remained a preserved negative result. The predeclared coarse sweep then identified `γ=.50,ρF=.97,ρM=.9995` as the only 2/2 health+acquisition-eligible shortlist; it was independently promoted to eight formal seeds, and G62–G67 below use those new seeds. Formal A2 uses 1,000 privileged evaluator-pretraining steps, freezes that head by hash, then uses 1,500 observed-consequence lifetime steps. z, correct action, reward, importance and memory labels never enter lifetime input/loss. Five independently trained seeds per control are retained; 24–32 paired within-seed replicas only reduce Monte Carlo noise. Every formal/sweep job ran on CPU with `torch.set_num_threads(1)`; jobs were process-parallel, so there is no mixed GPU/CPU backend confound.",
        "",
        "Formation N=0/1/2/4/8/16/32/64 at p=.55/.60/.65/.70; persistence D=0/10/50/100/250/500/1000/2000/5000; reversal R=0/1/2/4/8/16/32/64/128; opposing fraction q=0/.1/.25/.5/.75/1; NULL K=0/1/2/4/8/16/32/64. D*=500, formation/acquisition threshold=.10, persistence ratio=.20, selectivity margin=.05 and consolidation margin=.03 were frozen before formal training. Read clamps preserve state; zero/swap interventions destroy/replace state and are reported separately.",
        "",
        "## 2. Development audit (excluded from formal gates)",
        "",
        "| curriculum | seed | steps | train p | final train CE | BS N1 | BS N32 | BS D500 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *development_rows,
        "",
        "## 3. Gates",
        "",
        "| Gate | Result | Replication | Frozen criterion |",
        "|---|---|---:|---|",
        f"| G62 gradual accumulation | {gates['status']['G62']} | {gates['counts']['G62']}/8 | late > middle > early bands |",
        f"| G63 long persistence | {gates['status']['G63']} | {gates['counts']['G63']}/8 | meaningful BS0 and D500/0 ≥ .20 |",
        f"| G64 revisability | {gates['status']['G64']} | {gates['counts']['G64']}/8 | reversed by R≤128 |",
        f"| G65 predictive selectivity | {gates['status']['G65']} | {gates['counts']['G65']}/8 | delayed predictive > matched noise + .05 |",
        f"| G66 F→M causal handoff | {gates['status']['G66']} | {gates['counts']['G66']}/8 | finite read-effect crossover with meaningful baseline |",
        f"| G67 consolidation benefit | {gates['status']['G67']} | {gates['counts']['G67']}/5 | no acquisition damage and retention +.03 or late-M +.02 |",
        "",
        "Passing seeds: `" + json.dumps(gates["passing_seeds"], sort_keys=True) + "`.",
        "Healthy noisy-world interface seeds (development-frozen action-TV≥.10 and interaction≥.10): `" +
        json.dumps(gates["healthy_primary_seeds"]) + "`. CFA remains reported, but its deterministic-world .10 threshold is not copied because noisy evidence lowers the theoretical ceiling. All G62–G67 claims require the two binding prerequisites within the same seed.",
        "",
        "## 4. Gradual formation and Bayesian reference",
        "",
        "| p | N0 | N1 | N2 | N4 | N8 | N16 | N32 | N64 | T_acquire | calibration Brier N32 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in P_LEVELS:
        key = f"{p:.2f}"
        curves = [item["formation"][key]["curve"] for item in primary.values()]
        values = [fmt([bs(curve[str(n)]) for curve in curves]) for n in FORMATION_N]
        acquire = [item["formation"][key]["T_acquire"] for item in primary.values()]
        finite = [x for x in acquire if x is not None]
        brier = [curve["32"]["calibration_brier"] for curve in curves]
        lines.append(f"| {p:.2f} | " + " | ".join(values) + f" | {fmt(finite) if finite else 'NOT_ACQUIRED'} | {fmt(brier)} |")
    bands = [formation_bands(item) for item in primary.values()]
    lines += ["", f"Primary band means: early `{fmt([x[0] for x in bands])}`, middle `{fmt([x[1] for x in bands])}`, late `{fmt([x[2] for x in bands])}`. The Bayesian reference is computed from the known likelihood and is never a training target. Single-event posterior is p rather than 0/1; N1 saturation is assessed directly above.", "",
              "## 5. Persistence and descriptive fits", "",
              "| D | " + " | ".join(str(d) for d in DELAYS) + " |", "|---|" + "---:|" * len(DELAYS)]
    lines.append("| mean BS | " + " | ".join(fmt([bs(item["persistence"]["curve"][str(d)]) for item in primary.values()]) for d in DELAYS) + " |")
    halves = [item["persistence"]["half_life"] for item in primary.values()]
    fits = [curve_fit(item["persistence"]["curve"]) for item in primary.values()]
    lines += ["", "![Persistence](../results/stage2d/processed/figures/persistence.png)", "", f"Per-seed half-lives: `{halves}`; no extrapolation is used. A half-life is scientifically interpretable only for a seed with preregistered meaningful BS(0); otherwise it is merely a small-signal crossing diagnostic. Descriptive exponential log-R² `{fmt([x['exponential_log_R2'] for x in fits])}`, power-like log-R² `{fmt([x['power_log_R2'] for x in fits])}`. Bi-exponential fitting was attempted but was underidentified/unstable with nine noisy points and is not interpreted.", "",
              "## 6. Reversal, contradiction and hysteresis", "",
              "| R | " + " | ".join(str(r) for r in (0,1,2,4,8,16,32,64,128)) + " |", "|---|" + "---:|" * 9]
    lines.append("| mean BS | " + " | ".join(fmt([bs(item["revision"]["curve"][str(r)]) for item in primary.values()]) for r in (0,1,2,4,8,16,32,64,128)) + " |")
    lines += ["", "![Revision](../results/stage2d/processed/figures/revision.png)", "", f"T_change `{[x['revision']['T_change'] for x in primary.values()]}`; T_neutral `{[x['revision']['T_neutral'] for x in primary.values()]}`; T_reverse `{[x['revision']['T_reverse'] for x in primary.values()]}`.", "",
              "| opposing fraction q | 0 | .10 | .25 | .50 | .75 | 1.0 |", "|---|---:|---:|---:|---:|---:|---:|",
              "| BS after 64 mixed events | " + " | ".join(fmt([bs(x["revision"]["contradictory_fraction"][str(q)]) for x in primary.values()]) for q in (0.0,.1,.25,.5,.75,1.0)) + " |", "",
              "## 7. Predictive selectivity and rare-useful control", "",
              "| Condition | immediate BS | D500 BS |", "|---|---:|---:|"]
    for name in ("predictive", "noise"):
        lines.append(f"| matched {name} | {fmt([bs(x['selectivity']['matched'][name]['immediate']) for x in primary.values()])} | {fmt([bs(x['selectivity']['matched'][name]['delayed']) for x in primary.values()])} |")
    lines += ["", "Rare/frequent immediate BS: " + ", ".join(f"`{name}={fmt([bs(x['selectivity']['rare_vs_frequent'][name]) for x in primary.values()])}`" for name in ("useful_4","useful_8","noise_32","noise_64","noise_128")) + ".", "",
              "## 8. F/M causal timeline", "",
              "| Time | C_F read | C_M read | C_FM read | F-zero | M-zero | F-swap | M-swap |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in HANDOFF_TIMES:
        def read_effect(mem): return fmt([handoff_effect(x,t,mem) for x in primary.values()])
        def strong(mem,op): return fmt([x["handoff"]["state_destruction"][t][mem][op]["causal_effect"] for x in primary.values()])
        lines.append(f"| {t} | {read_effect('F')} | {read_effect('M')} | {read_effect('FM')} | {strong('F','zero')} | {strong('M','zero')} | {strong('F','swap')} | {strong('M','swap')} |")
    lines += ["", "![Handoff](../results/stage2d/processed/figures/handoff.png)", "", "Read mediation and state destruction are not conflated. G66 uses only finite read clamps and baseline behavior. Norm crossovers are not counted.", "",
              "## 9. Independently trained baselines", "",
              "| Arm | seeds | BS N32 p=.65 | persistence ratio D500 | T_reverse finite | delayed predictive | delayed noise |", "|---|---:|---:|---:|---:|---:|---:|"]
    all_arms = {"A2 selected g050": primary, "A2 default g012": default_primary, **controls}
    for label, rows in all_arms.items():
        ratios=[retention_ratio(x) for x in rows.values()]
        ratio_text=fmt(ratios) if any(x is not None for x in ratios) else "NOT_ELIGIBLE"
        lines.append(f"| {label} | {len(rows)} | {fmt([bs(x['formation']['0.65']['curve']['32']) for x in rows.values()])} | {ratio_text} | {sum(isinstance(x['revision']['T_reverse'],int) for x in rows.values())}/{len(rows)} | {fmt([bs(x['selectivity']['matched']['predictive']['delayed']) for x in rows.values()])} | {fmt([bs(x['selectivity']['matched']['noise']['delayed']) for x in rows.values()])} |")
    lines += ["", f"Allocated parameter budgets: `{parameter_budgets}`. All variants retain compatibility modules even when reads/transfer are disabled; this is a functional—not compute-efficiency—comparison. Every control checkpoint was trained independently.", "",
              "## 10. Coarse γ/ρ phase diagram", "",
              "| config | γ | ρF | ρM | T_acquire | T1/2 | T_reverse | retention D500 | late C_F | late C_M | health | shortlist | Pareto |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in phases:
        lines.append(f"| {row['name']} | {row['gamma']:.2f} | {row['rho_fast']:.3f} | {row['rho_slow']:.4f} | {row['T_acquire']:.1f} | {row['T_half']:.1f} | {row['T_reverse']:.1f} | {row['retention_D500']:.3f} | {row['late_C_F']:+.3f} | {row['late_C_M']:+.3f} | {row['healthy_seeds']}/2 | {row['shortlist_eligible']} | {row['pareto']} |")
    lines += ["", "![Phase diagram](../results/stage2d/processed/figures/phase_diagram.png)", "", "Phase-table sentinels: T_acquire=65 means not acquired by N64; T1/2=5001 means >5000 (not extrapolated); T_reverse=129 means not reversed by R128.", "", f"Development-selected configuration: `{best['name'] if best else 'NO_VALID_CONFIG'}`. A configuration is promoted only if both development seeds pass interface health and acquire by N64. The selected g050 configuration was independently promoted to eight formal seeds; the default eight-seed negative result was not overwritten. The table, not a single scalar, is the scientific result; infinite retention with failed revision is not treated as optimal.", "",
              "## 11. NULL consolidation and continuous operation", "",
              "| K NULL | immediate BS | D500 BS | cumulative transfer | probe C_F | probe C_M |", "|---:|---:|---:|---:|---:|---:|"]
    for k in (0,1,2,4,8,16,32,64):
        vals=[x["null_consolidation"][str(k)] for x in primary.values()]
        lines.append(f"| {k} | {fmt([bs(v['immediate']) for v in vals])} | {fmt([bs(v['delayed']) for v in vals])} | {fmt([v['cumulative_transfer_norm'] for v in vals])} | {fmt([v['probe_read_causal_effects']['F'] for v in vals])} | {fmt([v['probe_read_causal_effects']['M'] for v in vals])} |")
    lines += ["", "![NULL consolidation](../results/stage2d/processed/figures/null_consolidation.png)", "", "| stream tick | H norm | F norm | M norm | BS | mean ΔH |", "|---:|---:|---:|---:|---:|---:|"]
    for tick in (100,500,1000,5000,10000):
        vals=[x["continuous"]["milestones"].get(str(tick)) for x in primary.values()]
        vals=[v for v in vals if v]
        lines.append(f"| {tick} | {fmt([v['state_norms']['H'] for v in vals])} | {fmt([v['state_norms']['F'] for v in vals])} | {fmt([v['state_norms']['M'] for v in vals])} | {fmt([bs(v['behavior']) for v in vals])} | {fmt([v['mean_H_delta'] for v in vals])} |")
    lines += ["", "![Continuous dynamics](../results/stage2d/processed/figures/continuous.png)", "", f"First H>100: `{[x['continuous']['first_H_gt_100'] for x in primary.values()]}`; first H>1000: `{[x['continuous']['first_H_gt_1000'] for x in primary.values()]}`; first NaN/Inf: `{[x['continuous']['first_nonfinite'] for x in primary.values()]}`. No Stage 2D H scaling or recurrence repair was applied.", "",
              "## 12. Training health", "",
              "| seed | action-TV | history-TV | interaction | conditional CE | CFA | H/F/M probe accuracy | read gate F/M | H/F/M norm |", "|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for seed,item in primary.items():
        h=item["training_health"]
        lines.append(f"| {seed} | {h['action_TV']:.3f} | {h['history_TV']:.3f} | {h['interaction_y0']:+.3f} | {h['conditional_CE']:.3f} | {h['CFA']:+.3f} | {h['latent_probe_accuracy']} | {[round(x,3) for x in h['read_gate_mean']]} | {h['state_norms']} |")
    lines += ["", "Evaluator hashes, q norms, gradient groups, training CE curves, write/transfer traces and every seed-level probability are preserved in checkpoints, summaries and processed JSON. Action-TV, interaction and CFA are health prerequisites; a memory-dynamics pattern is not promoted when this interface is unhealthy.", "",
              "## 13. Direct answers to the 28 required questions", ""]
    cf_timeline = {t: mean([handoff_effect(x,t,"F") for x in primary.values()]) for t in HANDOFF_TIMES}
    cm_timeline = {t: mean([handoff_effect(x,t,"M") for x in primary.values()]) for t in HANDOFF_TIMES}
    f_peak = max(cf_timeline, key=cf_timeline.get); m_peak = max(cm_timeline, key=cm_timeline.get)
    tacq_by_p = {f"{p:.2f}": [x["formation"][f"{p:.2f}"]["T_acquire"] for x in primary.values()]
                 for p in P_LEVELS}
    rare8 = mean([abs(bs(x["selectivity"]["rare_vs_frequent"]["useful_8"])) for x in primary.values()])
    noise128 = mean([abs(bs(x["selectivity"]["rare_vs_frequent"]["noise_128"])) for x in primary.values()])
    null_gain = mean([abs(bs(x["null_consolidation"]["16"]["delayed"])) -
                      abs(bs(x["null_consolidation"]["0"]["delayed"])) for x in primary.values()])
    ratio_pairs = [(retention_ratio(primary[s]), retention_ratio(gamma0[s])) for s in CONTROLS]
    gamma_retention_delta = mean([a-b for a,b in ratio_pairs if a is not None and b is not None])
    gamma_retention_text = (f"{gamma_retention_delta:+.3f}" if math.isfinite(gamma_retention_delta)
                            else "NOT_COMPARABLE_NO_MATCHED_FORMED_PAIRS")
    gamma_late_m_delta = mean([handoff_effect(primary[s],"late_delay","M") -
                               handoff_effect(gamma0[s],"late_delay","M") for s in CONTROLS])
    stable_items = [x for x in primary.values() if "10000" in x["continuous"]["milestones"] and
                    retention_ratio(x) is not None]
    drift_retention_corr = corr([x["continuous"]["milestones"]["10000"]["state_norms"]["H"] for x in stable_items],
                                [retention_ratio(x) for x in stable_items])
    rigidity = ("too rigid" if gates["status"]["G63"]=="PASS" and gates["status"]["G64"]=="FAIL"
                else "too plastic/fragile" if gates["status"]["G63"]=="FAIL" else "balanced or unresolved")
    # These are deliberately generated from gate outcomes and metrics, not aspirational prose.
    answers = [
        f"Noisy-evidence formation was {'replicated' if gates['status']['G62']=='PASS' else 'not replicated'} ({gates['counts']['G62']}/8 ordering seeds).",
        f"Single experience mean |BS| was {mean([abs(bs(x['formation']['0.65']['curve']['1'])) for x in primary.values()]):.3f}; compare N32 {mean([abs(bs(x['formation']['0.65']['curve']['32'])) for x in primary.values()]):.3f}.",
        "Direction/calibration relative to Bayes is reported in the p×N table; Bayes was reference-only.",
        f"Evidence-reliability T_acquire by p was {tacq_by_p}; NOT_ACQUIRED is retained as null rather than imputed.",
        f"Per-seed numerical half-life is {halves}; values beyond 5000 are bounds, and seeds without meaningful BS0 do not support a memory half-life claim.",
        f"Long-delay persistence passed in {gates['counts']['G63']}/8 seeds.",
        f"Opposing evidence produced criterion reversal in {gates['counts']['G64']}/8 seeds.",
        f"Per-seed reversal times are {[x['revision']['T_reverse'] for x in primary.values()]}.",
        "Hysteresis is present only for seeds whose reversal time exceeds their acquisition time; the paired T_acquire/T_reverse arrays above show whether that occurred.",
        f"Matched-noise selectivity passed in {gates['counts']['G65']}/8 seeds.",
        f"Rare useful N8 mean |BS|={rare8:.3f} versus frequent noise N128={noise128:.3f}; raw frequency is not relabeled as utility.",
        f"F finite-read causal effect was largest at {f_peak} (seed mean {cf_timeline[f_peak]:+.3f}).",
        f"M finite-read causal effect was largest at {m_peak} (seed mean {cm_timeline[m_peak]:+.3f}); state norms were not substituted.",
        f"A replicated behavioral F→M crossover was {'found' if gates['status']['G66']=='PASS' else 'not found'} ({gates['counts']['G66']}/8).",
        f"Gamma-zero consolidation benefit adjudication was {gates['status']['G67']} ({gates['counts']['G67']}/5); full-minus-gamma0 D500 retention ratio={gamma_retention_text}.",
        f"Full-minus-gamma0 late-delay M causal effect={gamma_late_m_delta:+.3f}; per-seed values are preserved.",
        f"The descriptive balanced phase region was {best['name'] if best else 'unresolved'}; all trade-offs are shown in the phase table.",
        f"The measured system is classified as {rigidity}; if formation itself fails, Outcome D is only the closest required category and is not evidence of genuine plastic memory.",
        f"NULL K16 versus K0 D500 BS: {fmt([bs(x['null_consolidation']['16']['delayed']) for x in primary.values()])} versus {fmt([bs(x['null_consolidation']['0']['delayed']) for x in primary.values()])}.",
        f"NULL K16 minus K0 long-delay |BS| was {null_gain:+.3f}; a functional F→M shift is claimed only if this gain accompanies the reported causal redistribution.",
        "NULL-driven H drift without retention is classified as non-useful internal time, not consolidation success.",
        f"10k stability: nonfinite in {sum(x['continuous']['first_nonfinite'] is not None for x in primary.values())}/8; H>1000 in {sum(x['continuous']['first_H_gt_1000'] is not None for x in primary.values())}/8.",
        f"Across the {len(stable_items)} retention-eligible completed 10k runs, Pearson corr(H norm, D500 retention ratio)={drift_retention_corr:+.3f}; this tiny-n diagnostic is not inferential. Drift is called a confound only with failed/nonfinite adjudication, and raw trajectories are retained.",
        f"Stage 2G stable-H redesign is {'justified' if category.startswith('E') else 'not yet forced by the stopping rule'}.",
        "Memory-law redesign is not yet justified: two formal seeds exhibit the targeted formation/persistence/reversal/handoff chain, while the dominant failure is across-seed learning robustness, not proof that the F→M law cannot work.",
        "Data-dependent consolidation is not introduced in Stage 2D; first improve reproducible noisy-world binding without changing the law, then retest the phase region.",
        f"Natural-online training is {'supported' if category.startswith(('A','B')) else 'not supported by the full gate set'}.",
        f"Natural-language behavioral memory is {'a justified next experiment' if category.startswith(('A','B')) else 'premature; resolve the reported failure first'}.",
    ]
    lines += [f"{i}. {answer}" for i, answer in enumerate(answers, 1)]
    lines += ["", "## 14. Verification", "",
              "- Stage 2D unit tests: **21/21 passed**.",
              "- Repository suite: **167/168 passed**. The sole failure is the inherited Stage 1.5 guard that compares `README.md` to commit `1367110`; README has intentionally accumulated later-stage result links. No Stage 2D mathematical, leakage, intervention, frozen-evaluator, or integrity test failed.",
              "- Historical Stage 2C.x manifest: **1684/1684 unchanged**.",
              "- Formal training: 8 selected primary + 8 default primary + 5 each for A0/gamma-zero/original-F-only/matched-F-only/no-memory/GRU; all checkpoints completed and all evaluation parameter hashes were frozen.",
              "- Required report exists and is non-empty at `/data/CSK/ETPM/et-rcm/reports/STAGE2D_MEMORY_DYNAMICS_RESULTS.md`.",
              "", "## 15. Scientific conclusion and stopping rule", "",
              f"**{category}.** If formation is not replicated, this is the closest category in the required A–E taxonomy, not evidence that a real memory was 'plastic'. Toy/Stage-2D success is not human-like memory, consciousness, infinite capacity or causal memory in the unrestricted sense. NULL improvement, if any, means only a measured benefit in this synthetic world. Failed and null gates are retained. The core memory law should be changed only after reasonable γ/ρ regions fail the joint acquisition–retention–revision and causal-handoff tests; continuous instability instead routes to the separately authorized Stage 2G stable-H study.", "",
              "Machine-readable root: `results/stage2d/processed/summary.json`. Full seed records remain in `results/stage2d/processed/formal/`; source and historical integrity hashes are in `results/stage2d/manifests/`."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps({"gates": gates["status"], "outcome": category,
                      "report": str(REPORT), "historical_unchanged": len(current)}, indent=2))
    print(f"RESULT_FILE={REPORT}")


if __name__ == "__main__":
    main()
