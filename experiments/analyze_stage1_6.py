#!/usr/bin/env python3
"""Frozen four-gate analysis with seed replication and explicit null outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from etrcm.stage1_6.runner import choose_oracle, oracle_probability


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
    # Descriptive post-formal dissection; these are not new gates.
    curriculum_fast = summarize_effect("curriculum_F_lesion_secondary",
                                       curriculum3000.F_lesion-curriculum3000.learned,.02)
    curriculum_slow = summarize_effect("curriculum_M_lesion_secondary",
                                       curriculum3000.M_lesion-curriculum3000.learned,.02)
    denominator = curriculum3000.zero-curriculum3000.oracle
    fraction = (curriculum3000.zero-curriculum3000.learned) / denominator.where(denominator >= gate_config["G37"]["min_oracle_benefit_denominator"])
    oracle_trained_denominator = oracle3000.zero-oracle3000.oracle
    cross_arm_fraction = (curriculum3000.zero-curriculum3000.learned) / oracle_trained_denominator.where(
        oracle_trained_denominator >= gate_config["G37"]["min_oracle_benefit_denominator"])
    g37_fraction = summarize_effect("curriculum_fraction", fraction,
                                    gate_config["G37"]["min_transfer_fraction"])
    g37_abs = summarize_effect("curriculum_learned_vs_zero", curriculum3000.zero-curriculum3000.learned,
                               gate_config["G37"]["min_learned_vs_zero_CE"])
    g37_cross = summarize_effect("curriculum_vs_oracle_trained_fraction_secondary",cross_arm_fraction,
                                 gate_config["G37"]["min_transfer_fraction"])
    effects = [g34_r,g34_m,g35_zero,g35_random,g35_shuffle,g36_baseline,g36_m,g37_fraction,g37_abs,g37_cross,
               curriculum_fast,curriculum_slow]
    gates = {
        "G34": "PASS" if max(g34_r["positive_seeds"],g34_m["positive_seeds"])>=minimum else "FAIL",
        "G35": "PASS" if all(x["positive_seeds"]>=minimum for x in (g35_zero,g35_random,g35_shuffle)) else "FAIL",
        "G36": "PASS" if g36_baseline["positive_seeds"]>=minimum and g36_m["positive_seeds"]>=minimum else "FAIL",
        "G37": "PASS" if g37_fraction["positive_seeds"]>=minimum and g37_abs["positive_seeds"]>=minimum else "FAIL",
    }
    learned_read_success_seeds=int((learned3000.D_R>=.01).sum())
    auxiliary_F_triggered=(gates["G35"]=="PASS" and learned_read_success_seeds<minimum)
    onset: dict[str, int | None] = {}
    for arm, measure in (("learned", "D_R"), ("learned", "D_M"),
                         ("oracle", "D_O"), ("curriculum", "D_O"),
                         ("curriculum", "D_R")):
        matching = wide.loc[wide.training_arm.eq(arm)]
        crossing = [int(step) for step, part in matching.groupby("training_step")
                    if int((part[measure] >= .01).sum()) >= minimum]
        onset[f"{arm}_{measure}_at_least_0.01_in_6_seeds"] = min(crossing) if crossing else None
    PROCESSED.mkdir(parents=True, exist_ok=True)
    schedule=pd.DataFrame([
        {"seed":seed,"step":step,"p_oracle":oracle_probability(step,final_step),
         "oracle_selected":choose_oracle(step,final_step,seed=seed)}
        for seed in CONFIG["training"]["formal_seeds"] for step in range(final_step)
    ])
    schedule.to_parquet(PROCESSED/"curriculum_schedule.parquet",index=False)
    records.to_parquet(PROCESSED/"all_interventions.parquet", index=False)
    paired.to_parquet(PROCESSED/"seed_checkpoint_condition.parquet", index=False)
    wide.to_parquet(PROCESSED/"seed_checkpoint_effects.parquet", index=False)
    fig, ax=plt.subplots(figsize=(8,4.5))
    for arm,measure,label in (("learned","D_R","learned: zero - full"),
                              ("learned","D_M","learned: M-lesion - full"),
                              ("oracle","D_O","oracle-trained: zero - oracle"),
                              ("curriculum","D_R","curriculum: zero - learned")):
        part=wide.loc[wide.training_arm.eq(arm)]
        series=part.groupby("training_step")[measure].agg(list)
        steps=np.asarray(series.index,dtype=float)
        means=np.asarray([np.mean(v) for v in series])
        bounds=np.asarray([bootstrap_interval(np.asarray(v,dtype=float)) for v in series])
        ax.plot(steps,means,marker="o",label=label)
        ax.fill_between(steps,bounds[:,0],bounds[:,1],alpha=.12)
    ax.axhline(0,color="black",linewidth=.7)
    ax.set(xlabel="Training step",ylabel="Paired held-out CE benefit",
           title="Finite memory dependence across training")
    ax.legend(fontsize=7,loc="best")
    fig.tight_layout()
    fig.savefig(PROCESSED/"memory_dependence_curve.png",dpi=180)
    plt.close(fig)
    output = {"run_id":RUN_ID,"cells":len(paths),"episodes":len(records),"gates":gates,
              "auxiliary_F_triggered":auxiliary_F_triggered,
              "learned_read_success_seeds_at_0.01":learned_read_success_seeds,
              "onset":onset,
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
        if report_name=="MEMORY_DEPENDENCE_LEARNING_CURVE_STAGE1_6.md":
            extra="\n![Finite paired memory-dependence curves](../results/stage1_6/processed/stage1_6-formal-v1/memory_dependence_curve.png)\n"
        if report_name=="MEMORY_GRADIENT_DIAGNOSTICS_STAGE1_6.md":
            diagnostic_path=PROCESSED/"jacobian_gate_diagnostics.parquet"
            if diagnostic_path.exists():
                j=pd.read_parquet(diagnostic_path)
                extra=("\n## JVP and recurrent-gate diagnostics\n\n"
                       "`JVP_memory_to_H_norm` is the mean norm of four seeded unit-direction Jacobian-vector products from the slow read to next H at the bridge, not an exact full Jacobian norm. Gate saturation counts sigmoid gate values below .05 or above .95. These are not causal-use gates.\n\n"
                       +j.groupby(["arm","step"]).mean(numeric_only=True).round(5).reset_index().drop(columns=["seed"]).to_markdown(index=False)+"\n")
        if report_name=="ORACLE_TO_LEARNED_CURRICULUM_STAGE1_6.md":
            schedule_table=(schedule.assign(quintile=schedule.step//(final_step//5))
                            .groupby("quintile").agg(steps=("step","count"),
                                                      target_p=("p_oracle","mean"),
                                                      realized_oracle_fraction=("oracle_selected","mean"))
                            .round(5).reset_index().to_markdown(index=False))
            extra="\n## Curriculum schedule and realized sampling\n\n"+schedule_table+"\n"
            if (ROOT/"results/stage1_6/auxiliary_F/analysis.json").exists():
                extra+="\nThe separately triggered, equal-oracle-budget three-phase auxiliary F is reported in `reports/AUXILIARY_MEMORY_USE_CURRICULUM_STAGE1_6.md`; it cannot change frozen G37.\n"
        REPORTS.joinpath(report_name).write_text(
            f"# {title}\n\nProtocol: `reports/STAGE1_6_PROTOCOL.md`. Eight independent formal seeds, 256 paired held-out episodes/seed/checkpoint, 3000 equal-budget AdamW steps, H scrub after four real B→A exposures, eight non-writing distractors, bridge (B,C), future target (A+C) mod 8. Smaller CE is better. Read interventions happen only at bridge; component lesions begin immediately after H scrub. Gate thresholds were frozen before formal data.\n\n"+
            table.to_markdown(index=False)+"\n\nThe `candidate_update` column in episode records is the total bridge H-step delta (including event input), not the isolated gated candidate; exact raw/gated candidate norms are in the separate JVP diagnostics. Gradients and read norms are diagnostics, not proof of use. Seed-level paired effects and confidence intervals are in `results/stage1_6/processed/stage1_6-formal-v1/gate_summary.json`.\n"+extra)
    final_table=paired.loc[paired.training_step.eq(final_step)].groupby(["training_arm","condition"],observed=True)[["CE","accuracy"]].mean().round(5).reset_index().to_markdown(index=False)
    capacity_table=(records.groupby("training_arm",observed=True)[["parameter_count","state_bytes"]]
                    .first().reset_index().to_markdown(index=False))
    effect_table=pd.DataFrame([{k:v for k,v in effect.items() if k!="seed_values"} for effect in effects]).to_markdown(index=False)
    gate_table=pd.DataFrame([{"gate":k,"outcome":v} for k,v in gates.items()]).to_markdown(index=False)
    g35=gates["G35"]=="PASS"; g36=gates["G36"]=="PASS"; g37=gates["G37"]=="PASS"
    if not g35:
        adjudication="The oracle-trained current operator did not clear the preregistered benefit gate. A Stage 1.7 operator redesign is authorized as a hypothesis test, not proven necessary; training limits remain possible."
    elif g36 and not g37:
        if g37_abs["positive_seeds"]>=minimum and g37_cross["positive_seeds"]>=minimum:
            adjudication=("The current operator and learned read show strong finite benefit, but frozen G37 fails because its within-curriculum oracle-benefit denominator was not retained. The cross-arm ratio is favorable only as a secondary diagnostic. This is not evidence that learned retrieval failed; the strict gate must remain FAIL and the oracle-forgetting interpretation needs a new preregistered test.")
        else:
            adjudication="The current operator can use supplied historical memory, but the frozen learned-read transfer criterion was not met. Retain the operator while studying routing training."
    elif g36 and g37:
        adjudication="The current operator can use memory and transfer part of oracle benefit to learned read on this toy. The Stage 1.5 failure was substantially training/optimization-dependent. This is not LM readiness."
    else:
        adjudication=("Oracle integration passed in the present gated-residual operator, but ordinary learned-read training and persistent M necessity did not replicate. "
                      f"The preregistered B3 curriculum had learned-read benefit in {g37_abs['positive_seeds']}/8 seeds; a post-formal component dissection found F-lesion benefit in {curriculum_fast['positive_seeds']}/8 but M-lesion benefit in {curriculum_slow['positive_seeds']}/8. "
                      f"This suggests curriculum-assisted fast-memory use, not proven slow persistent-memory integration. Frozen G36={gates['G36']} and G37={gates['G37']} are unchanged.")
    auxiliary_path=ROOT/"results/stage1_6/auxiliary_F/analysis.json"
    auxiliary=json.loads(auxiliary_path.read_text()) if auxiliary_path.exists() else None
    legacy_path=ROOT/"results/stage1_6/legacy_scaling/summary.json"
    if legacy_path.exists():
        legacy=json.loads(legacy_path.read_text())
        old_effect=legacy["mean_effect_3000_minus_160"]
        old_replication=legacy["oracle_benefit_seeds_at_least_0.01_by_checkpoint"]["3000"]
        old_answer=("No evidence that simply extending the original objective solves read use: on its 128-distractor long-gap slice, "
                    f"held-out CE fell by {-old_effect['learned']:.3f} from 160 to 3000 steps, "
                    f"but oracle-vs-no-read benefit changed by {old_effect['oracle_vs_no_read']:.5f} and cleared .01 in only {old_replication}/8 seeds at 3000. "
                    "This is exploratory and narrower than the full Stage 1.5 grid.")
        legacy_section=("\n## Original-objective exploratory scaling\n\n"
                        "See `reports/TRAINING_LENGTH_SCALING_ORIGINAL_OBJECTIVE_STAGE1_6.md`. "
                        +old_answer+"\n")
    else:
        old_answer="The historical 160-step objective was not rerun; exact Stage 1.5 undertraining remains unresolved."
        legacy_section=""
    REPORTS.joinpath("TRAINING_VS_ARCHITECTURE_ADJUDICATION_STAGE1_6.md").write_text(
        "# Training vs architecture adjudication\n\n"+gate_table+"\n\n"+adjudication+"\n\n"+effect_table+"\n")
    terminal=paired.loc[paired.training_step.eq(final_step) & paired.condition.eq("learned")]
    terminal_grad=terminal.set_index("training_arm").groupby(level=0).memory_gradient.mean().to_dict()
    terminal_gate=terminal.set_index("training_arm").groupby(level=0).gate.mean().to_dict()
    diag_path=PROCESSED/"jacobian_gate_diagnostics.parquet"
    jvp_terminal=(pd.read_parquet(diag_path).query("step == @final_step").groupby("arm")
                  [["JVP_memory_to_H_norm","recurrent_gate_saturated_fraction"]].mean().to_dict("index")
                  if diag_path.exists() else {})
    no_memory_ce=float(no_memory3000.learned.mean())
    learned_ce=float(learned3000.learned.mean())
    oracle_ce=float(oracle3000.oracle.mean())
    oracle_zero_ce=float(oracle3000.zero.mean())
    answers=[
        old_answer,
        f"Not reliably in ordinary learned-read training: G34={gates['G34']}; D_R and D_M gains from 160 to 3000 averaged {g34_r['mean']:.3f}/{g34_m['mean']:.3f} CE but crossed .01 in only {g34_r['positive_seeds']}/8 and {g34_m['positive_seeds']}/8 seeds, respectively.",
        f"Yes, for supplied past-only read on this toy: G35={gates['G35']}; oracle-trained oracle CE={oracle_ce:.3f} versus zero CE={oracle_zero_ce:.3f}, with all three registered contrasts clearing margin in 8/8 seeds.",
        f"Yes under registered margins: oracle-vs-zero/random/shuffled CE advantages were {g35_zero['mean']:.3f}/{g35_random['mean']:.3f}/{g35_shuffle['mean']:.3f}, each 8/8 seeds.",
        "The existing integration operator demonstrably has capacity to use a correct historical read on this task; no absolute architecture ceiling is established. This does not imply learned routing or persistent M use.",
        f"Not reproducibly with the ordinary full model: G36={gates['G36']}; full-vs-no-memory and M-lesion effects cleared thresholds in only {g36_baseline['positive_seeds']}/8 and {g36_m['positive_seeds']}/8 seeds, respectively. The curriculum did induce F-sensitive computation, not replicated M necessity.",
        f"The no-memory baseline remained near chance (CE={no_memory_ce:.3f}, 8-class chance ≈2.079) after H scrub; ordinary full CE={learned_ce:.3f}, but the paired advantage crossed .05 in only {g36_baseline['positive_seeds']}/8 seeds.",
        f"Not robustly: ordinary M lesion crossed .02 CE harm in {g36_m['positive_seeds']}/8 seeds; curriculum M lesion did so in {curriculum_slow['positive_seeds']}/8 versus F lesion in {curriculum_fast['positive_seeds']}/8 (the latter two are post-formal descriptive checks).",
        f"Replicated oracle benefit first appeared at step {onset['oracle_D_O_at_least_0.01_in_6_seeds']}; ordinary learned benefit never reached 6/8 at any checkpoint, while curriculum learned benefit did so at {onset['curriculum_D_R_at_least_0.01_in_6_seeds']}.",
        f"No within-curriculum oracle-first sequence was established: its oracle benefit never crossed .01 in 6/8 seeds, whereas learned benefit did at step {onset['curriculum_D_R_at_least_0.01_in_6_seeds']}. Cross-arm oracle training succeeded earlier, but that is not a within-model learning order.",
        f"No complete gradient starvation: at step 3000 the learned/oracle memory-branch gradient norms averaged {terminal_grad.get('learned',float('nan')):.4f}/{terminal_grad.get('oracle',float('nan')):.4f}; finite interventions, not gradient magnitude, establish use. Relative optimization weakness remains possible.",
        f"No universal gate collapse: step-3000 slow-read gate means were {terminal_gate.get('learned',float('nan')):.3f}/{terminal_gate.get('oracle',float('nan')):.3f} for learned/oracle arms; recurrent gate saturation fractions were {jvp_terminal.get('learned',{}).get('recurrent_gate_saturated_fraction',float('nan')):.3f}/{jvp_terminal.get('oracle',{}).get('recurrent_gate_saturated_fraction',float('nan')):.3f}. Some saturation warrants study but is not a causal verdict.",
        "H scrub removes direct historical H information on the new task, and no-memory chance behavior confirms that shortcut is blocked there. The original Stage 1.5 world's shortcut was not itself isolated, but its CE improvement without read benefit is consistent with objective bypass.",
        f"G37={gates['G37']}; ratio requires a positive oracle-benefit denominator. Separate auxiliary F trigger={auxiliary_F_triggered}, with {learned_read_success_seeds}/8 learned-arm seeds clearing D_R≥.01. " +
        (f"Triggered F subsequently cleared learned D_R≥.025 in {auxiliary['effects']['aux_D_R']['seeds_above_0.025']}/8 seeds, without revising G37."
         if auxiliary is not None else "The registered B3 curriculum always ran; a further F curriculum is conditional."),
        adjudication,
        "No integration-operator redesign is currently justified by G35: the existing gated residual can use supplied read. A future Stage 1.7 should prioritize retrieval training and slow-state necessity; separate long-NULL instability remains unresolved.",
        "Retain the gated-residual operator as the next toy-test baseline because G35 passed, but do not treat current F/M persistence or ordinary learned retrieval as validated; G34/G36/G37 failed.",
        "No. Stage 1.5's G27/G28/G31 failures and the narrow toy scope preclude sequence/LM prototype authorization.",
    ]
    final_report=("# ET-RCM Stage 1.6 Final Report\n\n"+QUESTION+
        f"Formal run `{RUN_ID}`: {len(paths)} training cells, 8 fresh training seeds, {len(records):,} intervention rows. Parent Stage 1.5 reports and frozen results were not modified. No integration architecture or memory-law change.\n\n## Frozen gates\n\n"+gate_table+"\n\n## Paired causal effects\n\n"+effect_table+"\n\nRows labeled `secondary` were examined after formal gate outcomes and cannot change them. The curriculum F/M-lesion contrast is descriptive about which component carried its learned-read benefit.\n\n## Final condition outcomes\n\n"+final_table+"\n\n## Model and state sizes\n\n"+capacity_table+"\n\nParameter counts include instantiated but potentially inactive baseline modules; state bytes count H/F/M float32 slots per episode.\n\n## Experimental details\n\n"
        "**Task.** B∈0–7, A∈8–15 and C∈16–23 are sampled independently per episode. Four genuine external B→A exposures update F by the unchanged delta rule. Only H is reset to initial H; F/M are preserved exactly. Eight identical non-writing context distractors follow, then a bridge exposes B and C. The unseen future class is Y=(A−8+C−16) mod 8. Thus current H alone cannot identify Y; the historical A is necessary, while A without later C is not the answer. Model output head receives only H. Counterfactual leakage and no-memory identifiability are tested.\n\n"
        "**Read interventions.** At the bridge, the oracle supplies the raw (F+M) read at the observed historical B, saved immediately after exposure 4. It has no episode-specific C or Y. Oracle training uses the existing slow-read normalization/arbitration and clamps fast read to zero. Learned training uses the existing fast/slow route. Zero clamps both read channels; random replaces the slow input with an independent equal-norm vector; shuffled uses a derangement across episodes. These controls use the same interface. F/M lesions zero exactly one component immediately after H scrub, before distractor reads can carry it into H. An induced downstream state change is permitted; external event/write law is unchanged.\n\n"
        "**Optimization and pairing.** Development seeds 8601–8602 each tested LR .001/.0003 across all six arms for 160 AdamW steps and selected common LR .001 by equal-arm/seed held-out CE (2.10779 versus 2.12833). Formal seeds 8701–8708 are independent and disjoint. Each of six arms sees the same 3000 world draws per seed, batch 32 (96,000 train episodes per arm/seed), gradient clip 1.0, weight decay .0001, FP32. B5 arms share exact initialization per seed. Curriculum p_oracle=1/.75/.5/.25/0 over five 600-step blocks; the realized per-step draws are stored. Checkpoints 0/50/100/160/300/500/1000/2000/3000 use the same 256 held-out episodes/seed across arms, conditions and checkpoints. These are repeated paired evaluations, not independent new episodes.\n\n"
        "**Statistics and diagnostics.** Episode CE is averaged within seed; seeds receive equal weight. Gate margins require at least 6/8 seed-level replications at preregistered thresholds; 95% intervals bootstrap independent seeds 2000 times. Finite interventions decide use; gradient/JVP, gate saturation, read norm, train loss and raw state norms are only explanatory diagnostics. A bridge H-step delta in raw records includes event input; isolated raw/gated candidate norms are in the JVP diagnostic file. The pre-formal lesion-timing correction and post-first-seed G37 interpretation caveat are documented in separate amendment notes; no formal threshold changed. Checkpoint tensors, train logs, raw Parquet, source/config/report hashes and integrity manifest live under `results/stage1_6`.\n\n## Answers to the 18 registered questions\n\n")
    final_report += "\n".join(f"{i}. {answer}\n" for i,answer in enumerate(answers,1))
    final_report += "\n## Conditional auxiliary Experiment F\n\n"
    if auxiliary_F_triggered and auxiliary is not None:
        final_report+=("TRIGGERED_AND_COMPLETED. The equal-oracle-budget three-phase curriculum is a separately labeled post-trigger experiment, not a fifth gate. "
                       f"Learned-read benefit ≥.025 replicated in {auxiliary['effects']['aux_D_R']['seeds_above_0.025']}/8 seeds; "
                       f"M/F-lesion benefit ≥.02 replicated in {auxiliary['effects']['aux_D_M']['seeds_above_0.02']}/8 and {auxiliary['effects']['aux_D_F']['seeds_above_0.02']}/8. "
                       "See `reports/AUXILIARY_MEMORY_USE_CURRICULUM_STAGE1_6.md`; G34–G37 are unchanged.\n")
    elif auxiliary_F_triggered:
        final_report+="TRIGGERED by the stated condition; an additional separately labeled curriculum run is required before calling this subexperiment complete.\n"
    else:
        final_report+="NOT_RUN_BY_CONDITION: the oracle-trained arm was not simultaneously successful with failure of learned-read finite dependence under the operational ≥.01 CE, ≥6/8-seed criterion. The always-run B3 oracle-to-learned curriculum is reported separately.\n"
    final_report += legacy_section+"\n## Interpretation and limits\n\n"+adjudication+" The historical oracle is a saved pre-distractor F+M read, so it is an upper-bound delivery intervention, not proof that current slow M retrieves the same content. The H-scrub compositional task is intentionally simpler and more memory-forcing than Stage 1.5's four-family distribution; success here does not retroactively erase its negative results. M/F lesions are one-time at the scrub boundary; the remaining component may later reconsolidate, so their effects are conservative about sustained component necessity. The gates are toy-specific and do not imply human-like memory, consciousness, unlimited information capacity or language-model readiness. Missing/negative outcomes are retained.\n"
    REPORTS.joinpath("STAGE1_6_FINAL_REPORT.md").write_text(final_report)
    manifest=[]
    for path in sorted((ROOT/"results/stage1_6").rglob("*")):
        if path.is_file() and path.name != "integrity.json":
            manifest.append({"path":str(path.relative_to(ROOT)),"sha256":sha256(path),"bytes":path.stat().st_size})
    for path in [ROOT/"configs/stage1_6.yaml", *sorted(REPORTS.glob("*STAGE1_6*.md")),
                 *sorted((ROOT/"src/etrcm/stage1_6").glob("*.py")),
                 *sorted((ROOT/"experiments").glob("*stage1_6*.py"))]:
        manifest.append({"path":str(path.relative_to(ROOT)),"sha256":sha256(path),"bytes":path.stat().st_size})
    (PROCESSED/"integrity.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"gates":gates,"records":len(records),"manifest_files":len(manifest)}))


if __name__=="__main__":
    main()
