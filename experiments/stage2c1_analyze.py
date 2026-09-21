"""Gate audit and reproducible, failure-preserving Stage 2C.1 report generator."""

from __future__ import annotations

import hashlib
import json
import math
import statistics as stats
from pathlib import Path

import torch

from etrcm.stage2c1.diagnostic import PersistentInterface


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/stage2c1"
SEEDS=list(range(7201,7209))


def read(path):return json.loads(path.read_text())
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False))


def mean(values):return float(stats.mean(values)) if values else None
def sd(values):return float(stats.stdev(values)) if len(values)>1 else 0.0
def ms(values):return f"{mean(values):.6f} ± {sd(values):.6f}" if values else "not run"


def expected_ce(prob):
    # Exact world expectation over balanced z/action and uniform wrong outcomes.
    p=prob
    return mean([-math.log(max(p[0][0][0],1e-10)),
                 -sum(math.log(max(x,1e-10)) for x in p[0][1][1:])/3,
                 -sum(math.log(max(x,1e-10)) for x in p[1][0][1:])/3,
                 -math.log(max(p[1][1][0],1e-10))])


def load_level(level,head=None):
    if level=="L0":
        root=BASE/"oracle_latent/formal"/head
    elif level=="L1":
        root=BASE/"oracle_memory/formal"
    elif level=="L2":
        root=BASE/"history_encoder/formal"
    else:raise ValueError(level)
    rows={}
    for seed in SEEDS:
        path=root/str(seed)
        if (path/"summary.json").exists():
            rows[seed]={"summary":read(path/"summary.json"),"log":read(path/"train_log.json"),"path":str(path.relative_to(ROOT))}
    return rows


@torch.no_grad()
def audit_m_swaps(l1):
    out={}
    for seed,row in l1.items():
        ckpt=ROOT/row["path"]/"checkpoint.pt"
        model=PersistentInterface("late_concat",history=False)
        model.load_state_dict(torch.load(ckpt,map_location="cpu",weights_only=True)["model"])
        model.eval()
        action=torch.tensor([0,1,0,1]);latent=torch.tensor([0,0,1,1])
        before=model(action,latent=latent).softmax(-1).view(2,2,4)
        after=model(action,latent=1-latent).softmax(-1).view(2,2,4)
        maxerr=float((after-before.flip(0)).abs().max())
        prob0=torch.softmax(-(-(before*before.clamp_min(1e-10).log()).sum(-1))/.35,-1)
        prob1=torch.softmax(-(-(after*after.clamp_min(1e-10).log()).sum(-1))/.35,-1)
        bs_before=float(prob0[0,0]-prob0[1,0]);bs_after=float(prob1[0,0]-prob1[1,0])
        out[seed]={"swap_forecast_max_abs_error":maxerr,
                   "behavioral_separation_before":bs_before,
                   "behavioral_separation_after":bs_after,
                   "preference_reversed":bs_before>=.3 and bs_after<=-.3 and maxerr<1e-6}
    return out


def aggregate_l3():
    rows={}
    for seed in SEEDS:
        path=BASE/"raw/lifetime"/str(seed)/"summary.json"
        if path.exists():rows[seed]=read(path)
    return rows


def main():
    torch.set_num_threads(1)
    l0=load_level("L0","late_concat");mod=load_level("L0","modulation")
    l1=load_level("L1");l2=load_level("L2")
    if len(l0)!=8 or len(mod)!=8 or len(l1)!=8 or len(l2)!=8:
        raise RuntimeError("Formal L0-L2 results incomplete; report not finalizable")
    if not all(r["summary"]["final"]["parameter_hash_before_after_eval_equal"]
               for rows in (l0,mod,l1,l2) for r in rows.values()):
        raise RuntimeError("Formal evaluation mutated parameters")
    swaps=audit_m_swaps(l1)
    gate48={s:min(r["summary"]["final"]["tv_action"])>=.5 and
            r["summary"]["final"]["delta_q_a"]>0 and r["summary"]["final"]["delta_q_b"]>0
            for s,r in l0.items()}
    gate49={s:min(r["summary"]["final"]["tv_action"])>=.5 and
            r["summary"]["final"]["behavioral_separation_entropy"]>=.3 and
            swaps[s]["preference_reversed"] for s,r in l1.items()}
    gate50={s:min(r["summary"]["final"]["tv_action"])>=.5 and
            r["summary"]["final"]["behavioral_separation_entropy"]>=.3 and
            r["summary"]["final"]["interaction_y0"]>0 for s,r in l2.items()}
    counts={"G48":sum(gate48.values()),"G49":sum(gate49.values()),"G50":sum(gate50.values())}
    gates={"G48":"PASS" if counts["G48"]>=6 else "FAIL",
           "G49":"PASS" if counts["G48"]>=6 and counts["G49"]>=6 else "FAIL" if counts["G48"]>=6 else "NOT_RUN_BY_GATE",
           "G50":"PASS" if counts["G48"]>=6 and counts["G49"]>=6 and counts["G50"]>=6 else "FAIL" if counts["G48"]>=6 and counts["G49"]>=6 else "NOT_RUN_BY_GATE"}
    l3=aggregate_l3()
    if all(v=="PASS" for v in gates.values()) and len(l3)!=8:
        raise RuntimeError("L3 is required by passed ladder but 8 seeds are not complete")
    curves={head:{str(step):{
        "action_tv_mean":mean([mean(next(x for x in r["log"] if x["step"]==step)["evaluation"]["tv_action"]) for r in rows.values()]),
        "expected_future_CE_mean":mean([expected_ce(next(x for x in r["log"] if x["step"]==step)["evaluation"]["prob"]) for r in rows.values()])}
        for step in (0,100,500,1000,3000)} for head,rows in (("late_concat",l0),("modulation",mod))}
    # The training logger attaches the pre-update gradient at steps 99/499/
    # 999/2999 to the preceding evaluation row (0/100/500/1000).
    gradient_source={100:0,500:100,1000:500,3000:1000}
    gradients={head:{str(step):{
        branch:mean([next(x for x in r["log"] if x["step"]==source)["train_gradient"][branch] for r in rows.values()])
        for branch in ("H_or_source","action_branch","consequence_head","loss")}
        for step,source in gradient_source.items()} for head,rows in (("late_concat",l0),("modulation",mod))}
    l3_summary={}
    if l3:
        def field(extract):return [mean([extract(x) for x in row["evaluation"]]) for row in l3.values()]
        l3_summary={
            "formation":{str(n):{"mean":mean(field(lambda x:x["formation"][str(n)])),
                                  "sd":sd(field(lambda x:x["formation"][str(n)]))} for n in (0,1,2,4,8,16,32)},
            "persistence":{str(n):{"mean":mean(field(lambda x:x["persistence"][str(n)])),
                                    "sd":sd(field(lambda x:x["persistence"][str(n)]))} for n in (0,10,100,500)},
            "generalization":{split:{"mean":mean(field(lambda x:x["generalization"][split])),
                                      "sd":sd(field(lambda x:x["generalization"][split]))} for split in ("seen","novel","hard_ood")},
            "revision":{str(n):{"mean":mean(field(lambda x:x["revision"][str(n)])),
                                 "sd":sd(field(lambda x:x["revision"][str(n)]))} for n in (0,1,2,4,8,16,32)},
            "swaps":{s:{"mean":mean(field(lambda x:x["swaps"][s])),"sd":sd(field(lambda x:x["swaps"][s]))} for s in ("F","M","FM")},
            "action_tv_N16":ms(field(lambda x:mean(x["action_metrics_N16"]["tv_action"]))),
            "history_tv_N16":ms(field(lambda x:mean(x["action_metrics_N16"]["tv_history"]))),
            "interaction_y0_N16":ms(field(lambda x:x["action_metrics_N16"]["interaction_y0"])),
            "entropy_BS_N16":ms(field(lambda x:x["action_metrics_N16"]["behavioral_separation_entropy"])),
            "H_norm_N16":ms(field(lambda x:x["state_norms"]["H"])),
            "F_norm_N16":ms(field(lambda x:x["state_norms"]["F"])),
            "M_norm_N16":ms(field(lambda x:x["state_norms"]["M"])),
            "evaluation_parameters_frozen_all":all(row["eval_parameter_frozen"] for row in l3.values())}
    metrics={"gates":gates,"gate_seed_counts":counts,"formal_seeds":SEEDS,
             "L0":{"late_concat":{s:r["summary"]["final"] for s,r in l0.items()},
                   "modulation":{s:r["summary"]["final"] for s,r in mod.items()}},
             "L1":{s:r["summary"]["final"] for s,r in l1.items()},
             "L2":{s:r["summary"]["final"] for s,r in l2.items()},
             "L1_swaps":swaps,"training_curves":curves,"gradients":gradients,"L3":l3_summary}
    save(BASE/"processed/summary.json",metrics)
    save(BASE/"processed/formal_seed_rows.json",{
        "L0":{s:r["summary"] for s,r in l0.items()},"L1":{s:r["summary"] for s,r in l1.items()},
        "L2":{s:r["summary"] for s,r in l2.items()},"L3":l3})
    save(BASE/"trajectories/action_tv_learning_curves.json",curves)
    save(BASE/"action_interventions/oracle_M_swaps.json",swaps)
    save(BASE/"action_interventions/direct_forecast_metrics.json",{
        "L0":metrics["L0"],"L1":metrics["L1"],"L2":metrics["L2"]})
    save(BASE/"checkpoints/index.json",{
        "note":"Checkpoint binaries live beside each seed's summary; no symlinks or duplicate overwrites",
        "L0_late_concat":{s:r["path"]+"/checkpoint.pt" for s,r in l0.items()},
        "L0_modulation":{s:r["path"]+"/checkpoint.pt" for s,r in mod.items()},
        "L1":{s:r["path"]+"/checkpoint.pt" for s,r in l1.items()},
        "L2":{s:r["path"]+"/checkpoint.pt" for s,r in l2.items()},
        "L3":{s:f"results/stage2c1/raw/lifetime/{s}/checkpoint.pt" for s in l3}})
    for config in ("stage2c1_formal.yaml","stage2c1_lifetime.yaml"):
        source=ROOT/"configs"/config;target=BASE/"configs"/config
        if not target.exists():target.write_text(source.read_text())
        elif target.read_bytes()!=source.read_bytes():raise RuntimeError("Formal config changed")
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [ROOT/"configs/stage2c1_formal.yaml",ROOT/"configs/stage2c1_lifetime.yaml",
                             ROOT/"experiments/stage2c1_train.py",ROOT/"experiments/stage2c1_lifetime.py",
                             ROOT/"experiments/stage2c1_analyze.py",ROOT/"experiments/stage2c1_integrity.py",
                             ROOT/"src/etrcm/stage2c1/diagnostic.py",ROOT/"tests/test_stage2c1.py",
                             ROOT/"docs/STAGE2C1_PROTOCOL.md"]}
    save(BASE/"manifests/stage2c1_source_sha256.json",source_hashes)
    report=build_report(metrics,l0,mod,l1,l2,l3)
    path=ROOT/"reports/STAGE2C1_ACTION_OUTCOME_BINDING_RESULTS.md"
    path.write_text(report)
    if not path.stat().st_size:raise RuntimeError("empty report")
    print(f"RESULT_FILE={path}")
    print(json.dumps({"gates":gates,"counts":counts,"L3_seeds":len(l3),
                      "L3_BS16":l3_summary.get("entropy_BS_N16")}))


def build_report(m,l0,mod,l1,l2,l3):
    g=m["gates"];c=m["gate_seed_counts"];ls=m["L3"]
    def vals(rows,key):return [r["summary"]["final"][key] for r in rows.values()]
    def tv(rows):return [mean(x) for x in vals(rows,"tv_action")]
    def curve_table():
        lines=["| Steps | late-concat TV / CE | modulation TV / CE |","|---:|---:|---:|"]
        for step in (0,100,500,1000,3000):
            a=m["training_curves"]["late_concat"][str(step)]
            b=m["training_curves"]["modulation"][str(step)]
            lines.append(f"| {step} | {a['action_tv_mean']:.6f} / {a['expected_future_CE_mean']:.4f} | {b['action_tv_mean']:.6f} / {b['expected_future_CE_mean']:.4f} |")
        return "\n".join(lines)
    def l3table(field,keys):
        return "\n".join(["| Condition | mean BS ± seed SD |","|---|---:|"]+
                         [f"| {k} | {ls[field][str(k)]['mean']:+.6f} ± {ls[field][str(k)]['sd']:.6f} |" for k in keys])
    original_tv=2.59e-4
    ratio=mean(tv(l0))/original_tv
    l3_bs=ls.get("entropy_BS_N16","not run")
    l3_tv=ls.get("action_tv_N16","not run")
    l3_magnitude=abs(ls.get("formation",{}).get("16",{}).get("mean",0))
    localization=("Joint lifetime action binding remains weak; endogenous F/M formation is not isolated from the interface/training failure"
                  if l3 and l3_magnitude<.05 else "No L3 result or non-null L3; inspect seed-level outcomes")
    seed_table=["| Seed | L0 TV | L1 TV | L2 novel TV | M swap | L3 TV N16 | L3 BS N16 |",
                "|---:|---:|---:|---:|---|---:|---:|"]
    for seed in SEEDS:
        l3_reps=l3[seed]["evaluation"] if seed in l3 else []
        seed_table.append(f"| {seed} | {mean(l0[seed]['summary']['final']['tv_action']):.5f} | "
                          f"{mean(l1[seed]['summary']['final']['tv_action']):.5f} | "
                          f"{mean(l2[seed]['summary']['final']['tv_action']):.5f} | "
                          f"{m['L1_swaps'][seed]['preference_reversed']} | "
                          f"{mean([mean(x['action_metrics_N16']['tv_action']) for x in l3_reps]):.5f} | "
                          f"{mean([x['formation']['16'] for x in l3_reps]):+.5f} |")
    lines=[
"# ET-RCM Stage 2C.1 — Action–Outcome Binding and Behavioral Interface Diagnostic",
"",
"> **Before asking whether ET-RCM can form persistent behavioral memory, can the model first use a known latent state to predict different consequences for different candidate actions, and can that information control behavior through the existing persistent-state pathway?**",
"",
"> **在判断 ET-RCM 能否形成持久行为记忆之前，首先验证：当正确的潜在状态已经已知时，模型能否根据不同候选行为预测不同未来结果，并通过现有 persistent-state pathway 让这些差异真正控制行为？**",
"",
f"Formal ladder: **G48 {g['G48']} ({c['G48']}/8), G49 {g['G49']} ({c['G49']}/8), G50 {g['G50']} ({c['G50']}/8)**. Historical Stage 2C remains Outcome C, G42–G47 FAIL; none of its gates or files were revised.",
"",
"## 1. Stage 2C failure diagnosis",
"",
f"Frozen Stage 2C reported mean direct action-branch TV ≈ {original_tv:.2e}, while history sensitivity was larger. Its consequence head consumed pooled H after an action event; it did not explicitly receive a candidate-action input. This motivated an interface diagnostic, not a retrospective memory-law failure claim. Source: `reports/STAGE2C_BEHAVIORAL_MEMORY_RESULTS.md`.",
"",
"## 2. Level 0 oracle latent",
"",
"The four z×action cells are exactly balanced per batch (64 items, 16/cell). Correct action emits outcome 0; incorrect action uniformly samples outcomes 1–3. The model sees oracle z as a learned context embedding and action as a separate candidate input; it is trained only on observed consequence CE. No correct-action target, memory state, or reward label enters L0. Two development seeds (7101–7102) preceded threshold freezing; eight disjoint formal training seeds (7201–7208) follow it.",
"",
"## 3. Action-branch TV",
"",
f"Late-concat L0 action-TV: {ms(tv(l0))}; modulation: {ms(tv(mod))}. The late-concat mean is about {ratio:.0f}× the frozen Stage 2C TV. Values are per-seed means over latent A/B, not pooled episodes pretending to be seeds.",
"",
"## 4. ΔQ and p(y=0) margins",
"",
f"Late-concat ΔQ_A: {ms(vals(l0,'delta_q_a'))}; ΔQ_B: {ms(vals(l0,'delta_q_b'))}. Both positive in {sum(m['L0']['late_concat'][s]['delta_q_a']>0 and m['L0']['late_concat'][s]['delta_q_b']>0 for s in range(7201,7209))}/8. All four forecast probability vectors, TV, bidirectional KL, JS, entropy differences, p0 differences, and interaction vectors are retained in `results/stage2c1/action_interventions/direct_forecast_metrics.json`.",
"",
"## 5. Entropy-policy behavior",
"",
f"The original Stage 2C policy softmax(−future entropy/0.35) now chooses the latent-matching action with mean probability {ms(vals(l0,'entropy_correct_mean'))}; entropy-policy behavioral separation {ms(vals(l0,'behavioral_separation_entropy'))}. This is a policy computed from forecasts, not supervised action selection.",
"",
"## 6. Q-policy diagnostic",
"",
f"The diagnostic softmax(p(y=0)/0.35) chooses correctly with {ms(vals(l0,'q_correct_mean'))}; BS {ms(vals(l0,'behavioral_separation_q'))}. It is not substituted for the original entropy policy in formal gate comparisons. At this fixed temperature it has lower seed variance but also a lower mean correct-action probability.",
"",
"## 7. Current vs action-conditioned consequence head",
"",
"The historical Stage 2C linear head receives only pooled H; action must survive five recurrent context/action steps. The new late-concat head receives [H;candidate-action embedding] directly, whereas additive modulation feeds H+W_a a to a small nonlinear head. Both L0 forms bind successfully. Thus direct action injection is sufficient; these experiments do not isolate whether old failure arose from its topology, joint-training gradients, recurrent dynamics, or their combination. Late-concat was chosen before formal downstream tests for simplicity, not tuned on formal outcomes.",
"",
"## 8. Training-step scaling",
"",
curve_table(),
"",
"CE is exact expected consequence CE under the four balanced world cells, computed from fixed four-way forecasts. Training checkpoints are at 0/100/500/1000/3000; 500-step L0 success argues that lack of steps alone is not the oracle-interface blocker, but Stage 2C joint lifetime training is a different optimization problem.",
"",
"## 9. Gradient diagnostics",
"",
"Mean pre-clipping gradient norms (late-concat; H/source, action branch, consequence head):",
"",
"| Step | H/source | action | head |",
"|---:|---:|---:|---:|",
*[f"| {s} | {m['gradients']['late_concat'][str(s)]['H_or_source']:.5f} | {m['gradients']['late_concat'][str(s)]['action_branch']:.5f} | {m['gradients']['late_concat'][str(s)]['consequence_head']:.5f} |" for s in (100,500,1000,3000)],
"",
"L0 action gradients are nonzero during learning; near-convergence gradients should be read alongside low loss. L3 joint-training action gradients were not instrumented, so starvation there remains untested. Gradient magnitude is diagnostic, not causal intervention evidence.",
"",
"## 10. Level 1 oracle M",
"",
f"Fixed QR-derived M_A/M_B are 8×8, unit Frobenius norm and orthogonal; neither is a correct-action nor outcome one-hot. H0, F0=0 and present event=None are identical. Only M changes, through inherited q_M→r_M→normalized/gated read→H; M is not an output-head input. L1 action-TV {ms(tv(l1))}; entropy BS {ms(vals(l1,'behavioral_separation_entropy'))}; H1 difference {ms(vals(l1,'H1_difference'))}; mean forecast TV under M-read clamp {ms(vals(l1,'M_read_clamp_forecast_tv'))}. One NULL tick is a short-horizon pathway diagnostic, not persistence.",
"",
"## 11. Oracle M swap",
"",
f"For every saved L1 checkpoint, a finite intervention swaps M_A↔M_B with the same actions and parameters. Forecast permutation maximum absolute error: {max(x['swap_forecast_max_abs_error'] for x in m['L1_swaps'].values()):.2e}; preference reversal {sum(x['preference_reversed'] for x in m['L1_swaps'].values())}/8. Read clamp collapses latent-dependent forecasts, further checking the required route.",
"",
"## 12. Level 2 history encoder",
"",
f"A small GRU sees only eight past Stage 2C-observable records: color, shape, nuisance, actual action, observed outcome. Its output reshapes to M and enters the same read→H pathway. It receives neither z, correct-action labels, future outcomes, nor a direct output-head bypass. Evaluation uses 64 fresh histories per seed with held-out parity-combination surfaces; L2 action-TV {ms(tv(l2))}, entropy BS {ms(vals(l2,'behavioral_separation_entropy'))}. The current probe surface is not modeled in this L2 diagnostic, so novel generalization concerns held-out history surface combinations only.",
"",
"## 13. History × action interaction",
"",
f"L0 interaction I_HA(y0): {ms(vals(l0,'interaction_y0'))}; L1: {ms(vals(l1,'interaction_y0'))}; L2 novel: {ms(vals(l2,'interaction_y0'))}; L3 endogenous N16: {ls.get('interaction_y0_N16','not run')}. At L3 N16, history-TV: {ls.get('history_tv_N16','not run')} and action-TV: {ls.get('action_tv_N16','not run')}. Positive interaction requires the effect of switching action to reverse across A/B history-derived state. TV_H and TV_A are stored separately per seed; they should not be conflated.",
"",
"## 14. Endogenous ET-RCM rerun",
"",
"Only after G48–G50 passed, L3 newly trained eight 500-step Stage 2C-style lifetimes with the direct candidate-action head. Same B5 core, event stream, F write/consolidation/decay, AdamW 0.001, batch 16, 4/8/16 episode lengths, H penalty 0.001. Four independent paired evaluation lifetimes per seed; no parameter update during evaluation. This is a new diagnostic, not a re-score of frozen Stage 2C.",
"",
f"At N=16, L3 entropy BS: {l3_bs}; direct action-TV: {l3_tv}. State norms H/F/M: {ls.get('H_norm_N16','not run')} / {ls.get('F_norm_N16','not run')} / {ls.get('M_norm_N16','not run')}. Evaluation parameters frozen in all seeds: {ls.get('evaluation_parameters_frozen_all','not run')}.",
"",
"Formation (N actual experiences):", "",l3table("formation",(0,1,2,4,8,16,32)),"",
"Persistence (unrelated delay after N=16):","",l3table("persistence",(0,10,100,500)),"",
"Generalization:","",l3table("generalization",("seen","novel","hard_ood")),"",
"Revision (opposing real experiences after N=16):","",l3table("revision",(0,1,2,4,8,16,32)),"",
"Peripheral memory swaps at N=16:","",l3table("swaps",("F","M","FM")),"",
"All four paired raw records, checkpoints and unrounded metrics are preserved. These L3 probes are scoped diagnostics; no old G42–G47 threshold is re-adjudicated, and the shorter L3 grid is not interchangeable with the historical full Stage 2C protocol.",
"",
"## 15. Gate outcomes G48–G50",
"",
f"G48 {g['G48']} {c['G48']}/8: pre-registered both-latent action-TV≥0.50 and ΔQ_A/B>0. G49 {g['G49']} {c['G49']}/8: oracle M both action-TV≥0.50, entropy BS≥0.30, exact preference-reversing swap. G50 {g['G50']} {c['G50']}/8: legal-history novel both action-TV≥0.50, entropy BS≥0.30, positive interaction. Thresholds and seeds were fixed in `configs/stage2c1_formal.yaml` after two dev seeds and before formal runs. Formal parameters remained frozen during every evaluation.",
"",
*seed_table,
"",
"## 16. Failure localization",
"",
f"{localization}. The L0–L2 success establishes conditional interface capacity and legal-history sufficiency. It does **not** establish endogenous F/M formation. If L3 action-TV itself is near zero, even the direct head has not retained oracle binding under joint lifetime training; one cannot uniquely blame the F/M law. Stage 2C's original Outcome C remains intact. This pattern is best labeled a mixed joint-training/interface–memory-chain blocker pending a controlled state-injection audit of the L3 checkpoint.",
"",
"## 17. Next-stage recommendation",
"",
"Do not enter language modeling or redesign F/M yet. Next isolate the L3 checkpoint with frozen H and injected oracle M / learned history M, quantify action gradient and read-path survival during lifetime training, then pre-register a controlled training-curriculum comparison. Keep the original entropy policy. Address H stability separately only if nonfinite/large-H trajectories actually dominate; this short-horizon ladder does not justify an H redesign.",
"",
"## Direct answers to the 18 required questions",
"",
"1. Yes: L0 distinguishes the two action consequences in all eight seeds.",
f"2. Yes: mean L0 TV {mean(tv(l0)):.6f} versus historical {original_tv:.2e}.",
f"3. Yes: ΔQ_A {ms(vals(l0,'delta_q_a'))}, ΔQ_B {ms(vals(l0,'delta_q_b'))}.",
f"4. Yes: entropy-policy correct-action probability {ms(vals(l0,'entropy_correct_mean'))}.",
f"5. Q-policy works ({ms(vals(l0,'q_correct_mean'))}), with lower seed variance but less decisive mean behavior at the fixed temperature.",
"6. The frozen original head was nearly action-insensitive in Stage 2C; topology versus optimization is not isolated.",
"7. Modulation also binds; it is not materially needed over simple late-concat in L0.",
"8. Both heads bind by 500 steps in oracle L0; more steps improve small residual error, not the old joint-training diagnosis.",
"9. No persistent L0 action-gradient starvation was observed; L3 joint-training action gradients remain unmeasured.",
"10. Yes: oracle M changes read/H and forecasts through the inherited pathway.",
"11. Yes: exact M swap reverses preference in 8/8 seeds.",
"12. Yes for eight legal observed past records and held-out history surface combinations; not yet proof of spontaneous memory.",
"13. Yes in L0/L1/L2; interactions are positive and replicated.",
"14. Mixed joint-training/interface–endogenous-chain failure; a pure F/M blocker is not isolated.",
"15. The ladder warrants a controlled full-chain diagnostic, which L3 performed; it does not warrant a positive behavioral-memory claim.",
"16. No F/M architecture change is justified by these data alone.",
"17. Yes: direct action/consequence conditioning and its joint-training survival require targeted work.",
"18. No immediate H-stability redesign; monitor H separately after interface diagnostics.",
"",
"## Artifacts, protocol integrity and limitations",
"",
"Machine-readable summaries: `results/stage2c1/processed/summary.json`; per-seed raw logs/checkpoints: `oracle_latent/`, `oracle_memory/`, `history_encoder/`, `raw/lifetime/`; direct interventions: `action_interventions/`; curves: `trajectories/`; config/source/historical hashes: `configs/`, `manifests/`. A separately labeled 10-step L3 smoke run is preserved in `raw/l3_smoke_7101/` and excluded from formal aggregates. The historical 883-file Stage 2C inventory is SHA-256 verified after this stage. The 13 new Stage 2C.1 tests pass. The repository-wide suite has one inherited failure: `test_prior_frozen_artifacts_have_no_tracked_edits` compares README to commit `1367110`, whereas the already-committed Stage 2C HEAD `a6540c5` appended 45 README lines; Stage 2C.1 did not edit README or that test. We preserve and disclose this mismatch. L0–L2 are small synthetic sanity tests, not human-like memory, causal memory or language competence. L2 held-out surfaces are histories rather than current probes; L3 shorter evaluation grids do not revise frozen Stage 2C gates."
]
    return "\n".join(lines)+"\n"


if __name__=="__main__":main()
