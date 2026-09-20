#!/usr/bin/env python3
"""Frozen four-gate analysis with seed replication and explicit null outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/stage1_6.yaml").read_text())
RUN_ID = CONFIG["protocol"]["formal_run_id"]
RAW = ROOT / "results/stage1_6/raw" / RUN_ID
PROCESSED = ROOT / "results/stage1_6/processed" / RUN_ID
REPORTS = ROOT / "reports"
QUESTION = "> **Is ET-RCM failing to use persistent memory because its current integration architecture is incapable of doing so, or because the training process never forces the recurrent core to learn memory-dependent computation?**\n\n> **ET-RCM 当前无法有效利用持久记忆，究竟是因为现有 memory-to-H integration 架构本身做不到，还是因为训练过程从未真正迫使 recurrent core 学会依赖 memory 进行计算？**\n\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_interval(values: np.ndarray, *, seed: int = 1616, replicates: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (replicates, len(values)), replace=True).mean(1)
    return tuple(float(x) for x in np.quantile(draws, [.025, .975]))


def fmt(value: float) -> str:
    return f"{value:.5f}"


def summarize_effect(name: str, values: pd.Series, threshold: float) -> dict:
    array = values.to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    lo, hi = bootstrap_interval(finite) if len(finite) else (None, None)
    return {"effect": name, "mean": float(finite.mean()) if len(finite) else None,
            "ci95": [lo, hi], "invalid_seeds": int(len(array)-len(finite)),
            "positive_seeds": int((array >= threshold).sum()), "threshold": threshold,
            "seed_values": {str(seed): (float(value) if np.isfinite(value) else None)
                            for seed, value in values.items()}}


def main() -> None:
    paths = sorted(RAW.glob("*/interventions.parquet"))
    expected = len(CONFIG["training"]["formal_seeds"]) * len(CONFIG["training"]["arms"])
    if len(paths) != expected:
        raise RuntimeError(f"formal cells incomplete: {len(paths)}/{expected}")
    records = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    expected_seeds = set(CONFIG["training"]["formal_seeds"])
    for arm in CONFIG["training"]["arms"]:
        if set(records.loc[records.training_arm.eq(arm), "seed"]) != expected_seeds:
            raise RuntimeError(f"missing formal seed for {arm}")
    if not np.isfinite(records.future_CE).all():
        raise RuntimeError("nonfinite formal CE")
    paired = records.groupby(["training_arm", "seed", "training_step", "condition"], observed=True).agg(
        CE=("future_CE", "mean"), accuracy=("accuracy", "mean"),
        read_norm=("read_norm", "mean"), gate=("read_gate_slow", "mean"),
        H_norm=("H_norm", "mean"), F_norm=("F_norm", "mean"), M_norm=("M_norm", "mean"),
        candidate_update=("candidate_update_norm", "mean"),
        memory_gradient=("gradient_diagnostic", "mean"),
        core_gradient=("core_gradient_diagnostic", "mean"),
    ).reset_index()
    wide = paired.pivot_table(index=["training_arm", "seed", "training_step"],
                              columns="condition", values="CE").reset_index()
    for name, expression in {
        "D_R": lambda x: x["zero"] - x["learned"],
        "D_M": lambda x: x["M_lesion"] - x["learned"],
        "D_O": lambda x: x["zero"] - x["oracle"],
    }.items():
        wide[name] = expression(wide)
    final_step = CONFIG["training"]["formal_steps"]
    final = wide.loc[wide.training_step.eq(final_step)].set_index(["training_arm", "seed"])
    learned160 = wide.loc[wide.training_arm.eq("learned") & wide.training_step.eq(160)].set_index("seed")
    learned3000 = final.loc["learned"]
    oracle3000 = final.loc["oracle"]
    curriculum3000 = final.loc["curriculum"]
    no_memory3000 = final.loc["no_memory"]
    gate_config = CONFIG["gates"]
    minimum = gate_config["positive_seeds_min"]
    g34_r = summarize_effect("D_R_3000_minus_160", learned3000.D_R-learned160.D_R,
                             gate_config["G34"]["min_D_R_or_D_M_increase_3000_minus_160"])
    g34_m = summarize_effect("D_M_3000_minus_160", learned3000.D_M-learned160.D_M,
                             gate_config["G34"]["min_D_R_or_D_M_increase_3000_minus_160"])
    g35_zero = summarize_effect("oracle_vs_zero", oracle3000.zero-oracle3000.oracle,
                                gate_config["G35"]["min_oracle_vs_zero_CE"])
    g35_random = summarize_effect("oracle_vs_random", oracle3000.random-oracle3000.oracle,
                                  gate_config["G35"]["min_oracle_vs_random_CE"])
    g35_shuffle = summarize_effect("oracle_vs_shuffled", oracle3000.shuffled-oracle3000.oracle,
                                   gate_config["G35"]["min_oracle_vs_shuffled_CE"])
    g36_baseline = summarize_effect("no_memory_vs_full", no_memory3000.learned-learned3000.learned,
                                    gate_config["G36"]["min_full_vs_no_memory_CE"])
    g36_m = summarize_effect("M_lesion_vs_full", learned3000.M_lesion-learned3000.learned,
                             gate_config["G36"]["min_M_lesion_vs_full_CE"])
    denominator = curriculum3000.zero-curriculum3000.oracle
    fraction = (curriculum3000.zero-curriculum3000.learned) / denominator.where(denominator >= gate_config["G37"]["min_oracle_benefit_denominator"])
    g37_fraction = summarize_effect("curriculum_fraction", fraction,
                                    gate_config["G37"]["min_transfer_fraction"])
    g37_abs = summarize_effect("curriculum_learned_vs_zero", curriculum3000.zero-curriculum3000.learned,
                               gate_config["G37"]["min_learned_vs_zero_CE"])
    effects = [g34_r,g34_m,g35_zero,g35_random,g35_shuffle,g36_baseline,g36_m,g37_fraction,g37_abs]
    gates = {
        "G34": "PASS" if max(g34_r["positive_seeds"],g34_m["positive_seeds"])>=minimum else "FAIL",
        "G35": "PASS" if all(x["positive_seeds"]>=minimum for x in (g35_zero,g35_random,g35_shuffle)) else "FAIL",
        "G36": "PASS" if g36_baseline["positive_seeds"]>=minimum and g36_m["positive_seeds"]>=minimum else "FAIL",
        "G37": "PASS" if g37_fraction["positive_seeds"]>=minimum and g37_abs["positive_seeds"]>=minimum else "FAIL",
    }
    PROCESSED.mkdir(parents=True, exist_ok=True)
    records.to_parquet(PROCESSED/"all_interventions.parquet", index=False)
    paired.to_parquet(PROCESSED/"seed_checkpoint_condition.parquet", index=False)
    wide.to_parquet(PROCESSED/"seed_checkpoint_effects.parquet", index=False)
    output = {"run_id":RUN_ID,"cells":len(paths),"episodes":len(records),"gates":gates,
              "effects":effects,"config_sha256":sha256(ROOT/"configs/stage1_6.yaml")}
    (PROCESSED/"gate_summary.json").write_text(json.dumps(output,indent=2,allow_nan=True))
    for report_name, title, arms in [
        ("TRAINING_LENGTH_SCALING_STAGE1_6.md","Training-length scaling",["learned"]),
        ("ORACLE_READ_TRAINING_STAGE1_6.md","Oracle-read training",["learned","oracle","curriculum"]),
        ("MEMORY_NECESSITY_STAGE1_6.md","H-scrub memory necessity",["learned","no_memory","gru","single_memory"]),
        ("MEMORY_DEPENDENCE_LEARNING_CURVE_STAGE1_6.md","Memory-dependence learning curve",["learned","oracle","curriculum"]),
        ("MEMORY_GRADIENT_DIAGNOSTICS_STAGE1_6.md","Gradient and gate diagnostics",["learned","oracle","curriculum"]),
        ("ORACLE_TO_LEARNED_CURRICULUM_STAGE1_6.md","Oracle-to-learned curriculum",["curriculum","oracle","learned"]),
    ]:
        subset=paired.loc[paired.training_arm.isin(arms)]
        table=subset.groupby(["training_arm","training_step","condition"],observed=True)[["CE","accuracy","read_norm","gate","H_norm","F_norm","M_norm","candidate_update","memory_gradient","core_gradient"]].mean().round(5).reset_index()
        extra=""
        if report_name=="MEMORY_GRADIENT_DIAGNOSTICS_STAGE1_6.md":
            diagnostic_path=PROCESSED/"jacobian_gate_diagnostics.parquet"
            if diagnostic_path.exists():
                j=pd.read_parquet(diagnostic_path)
                extra="\n## JVP and recurrent-gate diagnostics\n\n"+j.groupby(["arm","step"]).mean(numeric_only=True).round(5).reset_index().drop(columns=["seed"]).to_markdown(index=False)+"\n"
        REPORTS.joinpath(report_name).write_text(
            f"# {title}\n\nProtocol: `reports/STAGE1_6_PROTOCOL.md`. Eight independent formal seeds, 256 paired held-out episodes/seed/checkpoint, 3000 equal-budget AdamW steps, H scrub after four real B→A exposures, eight non-writing distractors, bridge (B,C), future target (A+C) mod 8. Smaller CE is better. Read interventions happen only at bridge; component lesions begin immediately after H scrub. Gate thresholds were frozen before formal data.\n\n"+
            table.to_markdown(index=False)+"\n\nThese are finite toy-world outcomes; gradients and read norms are diagnostics, not proof of use. Seed-level paired effects and confidence intervals are in `results/stage1_6/processed/stage1_6-formal-v1/gate_summary.json`.\n"+extra)
    final_table=paired.loc[paired.training_step.eq(final_step)].groupby(["training_arm","condition"],observed=True)[["CE","accuracy"]].mean().round(5).reset_index().to_markdown(index=False)
    effect_table=pd.DataFrame([{k:v for k,v in effect.items() if k!="seed_values"} for effect in effects]).to_markdown(index=False)
    gate_table=pd.DataFrame([{"gate":k,"outcome":v} for k,v in gates.items()]).to_markdown(index=False)
    g35=gates["G35"]=="PASS"; g36=gates["G36"]=="PASS"; g37=gates["G37"]=="PASS"
    if not g35:
        adjudication="The oracle-trained current operator did not clear the preregistered benefit gate. A Stage 1.7 operator redesign is authorized as a hypothesis test, not proven necessary; training limits remain possible."
    elif g36 and not g37:
        adjudication="The current operator can use supplied historical memory, but learned retrieval/curriculum transfer remains a blocker. Retain the operator while studying routing training."
    elif g36 and g37:
        adjudication="The current operator can use memory and transfer part of oracle benefit to learned read on this toy. The Stage 1.5 failure was substantially training/optimization-dependent. This is not LM readiness."
    else:
        adjudication="Oracle integration passed, but persistent M necessity did not. Integration capacity and useful persistent-state use remain distinct."
    REPORTS.joinpath("TRAINING_VS_ARCHITECTURE_ADJUDICATION_STAGE1_6.md").write_text(
        "# Training vs architecture adjudication\n\n"+gate_table+"\n\n"+adjudication+"\n\n"+effect_table+"\n")
    answers=[
        "The historical 160-step objective is not directly re-run here; the matched H-scrub scaling tests the same operator on a new memory-necessary objective. Therefore Stage 1.5 exact-objective undertraining remains unresolved.",
        f"G34={gates['G34']}; paired D_R/D_M step-3000 minus step-160 effects above.",
        f"G35={gates['G35']}; supplied historical-read capacity is evaluated causally.",
        "Oracle vs zero/random/shuffled paired CE margins and seed replication are in the effect table.",
        "The observed oracle-training result bounds tested capacity at this scale/training budget; it cannot establish an absolute mathematical ceiling.",
        f"G36={gates['G36']}; lesions start immediately after H scrub, before memory can repopulate H.",
        "No-memory recurrent baseline final CE is shown in the condition table; compare paired per-seed G36 effect.",
        "M-lesion paired CE effect and replication are shown in G36.",
        "The checkpoint-wise intervention table identifies earliest replicated nontrivial dependence; do not infer it from training loss.",
        "D_O vs D_R checkpoint curves in the learning-curve report show whether oracle benefit precedes learned read; descriptive only unless replicated.",
        "Memory-branch gradient norm is reported by checkpoint; near-zero gradients alone are not causal evidence.",
        "Slow read gate and update norms are reported; saturation is a diagnostic, not proof of collapse.",
        "H scrub eliminates a direct H-history shortcut on the new task; it does not retroactively prove the old-world shortcut caused Stage 1.5 failure.",
        f"G37={gates['G37']}; ratio requires a positive oracle-benefit denominator.",
        adjudication,
        "Stage 1.7 operator changes are justified as a future test only if G35 fails; otherwise first address routing or persistent-state use.",
        "Retain the present operator for further toy tests only if oracle integration and persistent M use clear their respective gates.",
        "No. Stage 1.5's G27/G28/G31 failures and the narrow toy scope preclude sequence/LM prototype authorization.",
    ]
    final_report=("# ET-RCM Stage 1.6 Final Report\n\n"+QUESTION+
        f"Formal run `{RUN_ID}`: {len(paths)} training cells, 8 fresh training seeds, {len(records):,} intervention rows. Parent Stage 1.5 reports and frozen results were not modified. No integration architecture or memory-law change.\n\n## Frozen gates\n\n"+gate_table+"\n\n## Paired causal effects\n\n"+effect_table+"\n\n## Final condition outcomes\n\n"+final_table+"\n\n## Experimental details\n\nFour real early B→A exposures, H reset alone, eight identical non-writing distractors, observed bridge B+C and unobserved future class Y=(A-8+C-16) mod 8. Random B/A/C independent. The oracle read is saved from actual F+M at the true historical key after exposure 4; it contains neither C nor Y. During oracle training it replaces only the slow read at the bridge, with the fast read clamped to zero. Learned read uses the existing fast/slow arbitration. Zero/random/shuffled comparators use the same read interface; random is per-episode norm-matched and shuffle is a derangement. F/M lesions occur at the scrub boundary. AdamW, batch 32, selected LR, 3000 draws and evaluation episodes are paired across arms. Checkpoints: 0/50/100/160/300/500/1000/2000/3000. Full raw Parquet, checkpoints, train logs and hashes reside in `results/stage1_6`.\n\n## Answers to the 18 registered questions\n\n")
    final_report += "\n".join(f"{i}. {answer}\n" for i,answer in enumerate(answers,1))
    final_report += "\n## Interpretation and limits\n\n"+adjudication+" The gates are toy-specific and do not imply human-like memory, consciousness, unlimited information capacity or language-model readiness. Missing/negative outcomes are retained.\n"
    REPORTS.joinpath("STAGE1_6_FINAL_REPORT.md").write_text(final_report)
    manifest=[]
    for path in sorted((ROOT/"results/stage1_6").rglob("*")):
        if path.is_file() and path.name != "integrity.json":
            manifest.append({"path":str(path.relative_to(ROOT)),"sha256":sha256(path),"bytes":path.stat().st_size})
    (PROCESSED/"integrity.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"gates":gates,"records":len(records),"manifest_files":len(manifest)}))


if __name__=="__main__":
    main()
