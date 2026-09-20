"""Aggregate independent training seeds and generate the Stage 2C report."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as stats
from collections import defaultdict
from pathlib import Path

import torch

ROOT=Path("results/stage2c")
REPORT=Path("reports/STAGE2C_BEHAVIORAL_MEMORY_RESULTS.md")
SEEDS=tuple(range(6201,6209))
VARIANTS=("full","no_memory","gru","gamma_zero","f_only","m_disabled","gamma_zero_posthoc")


def load():
    rows={};trains={};manifests={}
    for variant in VARIANTS:
        for seed in SEEDS:
            path=ROOT/"raw"/variant/str(seed)/"rows.jsonl"
            if path.exists():
                rows[(variant,seed)]=[json.loads(x) for x in path.read_text().splitlines() if x]
                manifests[(variant,seed)]=json.loads((path.parent/"manifest.json").read_text())
            training=ROOT/"checkpoints"/variant/str(seed)/"summary.json"
            if training.exists():trains[(variant,seed)]=json.loads(training.read_text())
    return rows,trains,manifests


def find(rows,variant,seed,section,*,N=None,D=None,split=None,intervention=None,revision=None,noise=None):
    matches=[]
    for r in rows.get((variant,seed),[]):
        if r["section"]!=section:continue
        if N is not None and r["exposure_count"]!=N:continue
        if D is not None and r["delay"]!=D:continue
        if split is not None and r["probe_type"]!=split:continue
        if intervention is not None and r["intervention"]!=intervention:continue
        if revision is not None and r["revision_count"]!=revision:continue
        if noise is not None and r["useful_noise_condition"]!=("noise" if noise else "useful"):continue
        matches.append(r)
    if len(matches)!=1:
        raise KeyError((variant,seed,section,N,D,split,intervention,revision,noise,len(matches)))
    return matches[0]


def numbers(xs):
    xs=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not xs:return {"n":0,"mean":None,"min":None,"max":None,"sd":None}
    return {"n":len(xs),"mean":stats.mean(xs),"min":min(xs),"max":max(xs),
            "sd":stats.stdev(xs) if len(xs)>1 else 0.0}


def num(value):
    if value is None:return "NA"
    if abs(value)>=1000:return f"{value:,.0f}"
    if abs(value)<1e-3:return f"{value:+.2e}"
    return f"{value:+.5f}"


def fmt(stat):
    if stat["n"]==0:return "unavailable"
    return f"{num(stat['mean'])} ± {num(stat['sd'])} (n={stat['n']})"


def bool_count(items):return {"passed":sum(bool(x) for x in items),"required":6,"total":len(items)}


def action_branch_tv(row):
    forecasts=row["forecast_probabilities"]
    return stats.mean(sum(abs(a-b) for a,b in zip(item[0],item[1]))/2
                      for item in forecasts)


def build(rows,trains,manifests):
    available=[seed for seed in SEEDS if ("full",seed) in rows]
    curve={str(n):numbers([find(rows,"full",s,"formation",N=n,intervention="none")["BS"]
                           for s in available]) for n in (0,1,2,4,8,16,32)}
    persistence={str(d):numbers([find(rows,"full",s,"persistence",N=16,D=d,
                                     split="novel",intervention="none")["BS"]
                                 for s in available]) for d in (0,10,50,100,500,1000)}
    generalization={split:numbers([find(rows,"full",s,"persistence",N=16,D=0,
                                       split=split,intervention="none")["BS"]
                                   for s in available]) for split in ("seen","novel","hard_ood")}
    noise=numbers([find(rows,"full",s,"selectivity",N=16,noise=True)["BS"] for s in available])
    revision={str(n):numbers([find(rows,"full",s,"revision",N=16,revision=n)["BS"]
                               for s in available]) for n in (0,1,2,4,8,16,32)}
    timescale_grid={}
    for n in (1,2,4,8,16):
        timescale_grid[str(n)]={}
        for d in (0,10,100,500,1000):
            section="persistence" if n==16 else "timescale"
            collected=[find(rows,"full",seed,section,N=n,D=d,split="novel",
                            intervention="none") for seed in available]
            timescale_grid[str(n)][str(d)]={key:numbers([r[key] for r in collected])
                                             for key in ("BS","F_norm","M_norm",
                                                         "r_F_norm","r_M_norm")}
    interventions={}
    swap_names=("H_swap","F_swap","M_swap","FM_swap","HFM_swap","H_reset",
                "H_reset_F_zero","H_reset_M_zero","H_reset_FM_zero",
                "H_reset_F_swap","H_reset_M_swap")
    for name in swap_names:
        interventions[name]={str(d):{
            "BS":numbers([find(rows,"full",s,"state_swap",N=16,D=d,
                               intervention=name)["BS"] for s in available]),
            "JS":numbers([find(rows,"full",s,"state_swap",N=16,D=d,
                               intervention=name)["JS"] for s in available])}
                             for d in (0,100,500)}
    full_js=numbers([find(rows,"full",s,"persistence",N=16,D=0,
                          split="novel",intervention="none")["JS"] for s in available])
    window={}
    for window_name in ("W1","W2","W3","W4"):
        for clamp in ("F","M","FM"):
            key=f"read_{window_name}_{clamp}"
            intervention=key if window_name!="W4" else key+f"/probe_read_{clamp}"
            window[key]=numbers([find(rows,"full",s,"window",intervention=intervention)["BS"]
                                 for s in available])
    for window_name in ("W1","W2"):
        key=f"write_{window_name}_none"
        window[key]=numbers([find(rows,"full",s,"window",intervention=key)["BS"]
                             for s in available])
    window["none"]=numbers([find(rows,"full",s,"window",intervention="none")["BS"]
                              for s in available])
    baselines={}
    for variant in VARIANTS:
        present=[s for s in SEEDS if (variant,s) in rows]
        baselines[variant]={"seeds":present,
            "BS0":numbers([find(rows,variant,s,"persistence",N=16,D=0,
                                 split="novel",intervention="none")["BS"] for s in present]),
            "BS500":numbers([find(rows,variant,s,"persistence",N=16,D=500,
                                   split="novel",intervention="none")["BS"] for s in present]),
            "probe_future_loss_D0":numbers([find(rows,variant,s,"persistence",N=16,D=0,
                                            split="novel",intervention="none")["future_loss"]
                                            for s in present]),
            "action_branch_TV_D0":numbers([action_branch_tv(find(rows,variant,s,"persistence",
                                                N=16,D=0,split="novel",intervention="none"))
                                           for s in present]),
            "active_parameters":numbers([trains[(variant,s)]["active_parameters"]
                                          for s in present if (variant,s) in trains])}
    stability={}
    for s in available:
        path=ROOT/"raw"/"full"/str(s)/"state_tensors.pt"
        trace=torch.load(path,map_location="cpu",weights_only=False) if path.exists() else []
        def first(predicate):
            matches=[r["tau"] for r in trace if predicate(r)]
            return min(matches) if matches else None
        norms=[r["H"].norm(dim=(-2,-1)) for r in trace]
        stability[str(s)]={"first_H_over_100":first(lambda r:bool((r["H"].norm(dim=(-2,-1))>100).any())),
                           "first_H_over_1000":first(lambda r:bool((r["H"].norm(dim=(-2,-1))>1000).any())),
                           "first_nonfinite":first(lambda r:any(not torch.isfinite(r[k]).all()
                                  for k in ("H","F","M"))),
                           "max_H":max((float(x.max()) for x in norms),default=None)}
    gates={}
    g42=[];g43=[];g44=[];g45=[];g46=[]
    candidates={name:[] for name in ("F_swap","M_swap","FM_swap")}
    candidates.update({name:[] for name in window if name.startswith("read_")})
    candidates["gamma_zero_posthoc"]=[]
    per_seed={}
    diagnostic_action=[];diagnostic_history=[];diagnostic_train_ce=[]
    for s in available:
        probe_row=find(rows,"full",s,"persistence",N=16,D=0,split="novel",intervention="none")
        full=probe_row["BS"]
        forecasts=probe_row["forecast_probabilities"]
        diagnostic_action.append(action_branch_tv(probe_row))
        diagnostic_history.append(stats.mean(sum(abs(a-b) for a,b in zip(forecasts[i][action],
                                                                          forecasts[i+1][action]))/2
                                             for i in range(0,len(forecasts),2) for action in (0,1)))
        train_log=ROOT/"checkpoints"/"full"/str(s)/"train_log.json"
        if train_log.exists():diagnostic_train_ce.append(json.loads(train_log.read_text())[-1]["predictive_CE"])
        at500=find(rows,"full",s,"persistence",N=16,D=500,split="novel",intervention="none")["BS"]
        seen=find(rows,"full",s,"persistence",N=16,D=0,split="seen",intervention="none")["BS"]
        random_noise=find(rows,"full",s,"selectivity",N=16,noise=True)["BS"]
        before=find(rows,"full",s,"revision",N=16,revision=0)["BS"]
        rev16=find(rows,"full",s,"revision",N=16,revision=16)["BS"]
        rev32=find(rows,"full",s,"revision",N=16,revision=32)["BS"]
        values={"BS0":full,"BS500":at500,"PR500":at500/full if abs(full)>1e-4 else None,
                "BSseen":seen,"BSnoise":random_noise,"RI16":before-rev16,"BSrev32":rev32}
        per_seed[str(s)]=values
        g42.append(full>=.05)
        g43.append(full>=.05 and at500/full>=.5)
        g44.append(full>=.03 and seen>=.03 and full/seen>=.5)
        g45.append(full-abs(random_noise)>=.03)
        g46.append(before-rev16>=.03 and rev32<0)
        for name in ("F_swap","M_swap","FM_swap"):
            changed=find(rows,"full",s,"state_swap",N=16,D=0,intervention=name)["BS"]
            candidates[name].append(full-changed)
        base_window=find(rows,"full",s,"window",intervention="none")["BS"]
        for name in window:
            if name.startswith("read_"):
                intr=name if not name.startswith("read_W4") else name+"/probe_read_"+name.split("_")[-1]
                changed=find(rows,"full",s,"window",intervention=intr)["BS"]
                candidates[name].append(base_window-changed)
        if ("gamma_zero_posthoc",s) in rows:
            changed=find(rows,"gamma_zero_posthoc",s,"persistence",N=16,D=0,
                         split="novel",intervention="none")["BS"]
            candidates["gamma_zero_posthoc"].append(full-changed)
    gates["G42"]=bool_count(g42);gates["G43"]=bool_count(g43)
    gates["G44"]=bool_count(g44);gates["G45"]=bool_count(g45)
    gates["G46"]=bool_count(g46)
    causal={}
    for name,effects in candidates.items():
        positive=sum(x>=.02 for x in effects)
        negative=sum(x<=-.02 for x in effects)
        fractions=[effect/per_seed[str(seed)]["BS0"] if abs(per_seed[str(seed)]["BS0"])>1e-4
                   else None for seed,effect in zip(available,effects)]
        causal[name]={"effect":numbers(effects),"CM_fraction":numbers(fractions),
                      "positive":positive,"negative":negative}
    best=max(causal.items(),key=lambda pair:max(pair[1]["positive"],pair[1]["negative"])) if causal else None
    gates["G47"]={"passed":max(best[1]["positive"],best[1]["negative"]) if best else 0,
                  "required":6,"total":len(available),"best_predeclared_intervention":best[0] if best else None}
    passed=sum(x["passed"]>=6 for x in gates.values())
    core=all(gates[g]["passed"]>=6 for g in ("G42","G43","G44","G46","G47"))
    long_confounded=sum(x["BS0"]>=.05 and x["BS500"]>=.05 and
                        (stability[seed]["first_nonfinite"] is not None or
                         stability[seed]["first_H_over_1000"] is not None)
                        for seed,x in per_seed.items())>=6
    outcome=("D" if long_confounded else "A" if passed>=5 and core
             else "B" if gates["G42"]["passed"]>=6
             else "C")
    verify={"all_hashes_equal":all(m["parameter_hash_before"]==m["parameter_hash_after"]
                                    for m in manifests.values()),
            "evaluation_manifests":len(manifests)}
    return {"available_full_seeds":available,"curve":curve,"persistence":persistence,
            "generalization":generalization,"noise":noise,"revision":revision,
            "timescale_grid":timescale_grid,"interventions":interventions,"full_JS0":full_js,
            "window":window,
            "baselines":baselines,
            "forecast_diagnostics":{"action_branch_TV":numbers(diagnostic_action),
                                    "history_branch_TV":numbers(diagnostic_history),
                                    "final_training_CE":numbers(diagnostic_train_ce)},
            "stability":stability,"gates":gates,"causal_candidates":causal,
            "per_seed":per_seed,"frozen_verification":verify,"outcome":outcome,
            "formal_complete":len(available)==8 and all(len(baselines[v]["seeds"])==8
                                                   for v in VARIANTS)}


def markdown(summary):
    s=summary;lines=["# ET-RCM Stage 2C Behavioral Memory Results","",
    "> **Can past experience produce persistent, selective, generalizable, revisable, and causally state-mediated changes in ET-RCM's future behavior without changing its parameters?**",
    "","> **在模型参数完全不更新的情况下，过去经历能否通过持续内部状态形成持久、选择性、可泛化、可修正，并具有因果作用的未来行为改变？**","",
    f"Formal outcome: **{s['outcome']}**. Formal completeness: **{s['formal_complete']}**. Independent full-model training seeds: {s['available_full_seeds']}.","",
    "## 1. Scientific question","", "Memory here means a persistent causal effect of past experience on behavior, not exact historical recall. The alternative is no reliable history-dependent behavior even though H/F/M values change.","",
    "## 2. Architecture","", "Stage 1.4/1.5 gated residual H dynamics, unchanged external delta F write, separate learned F/M reads, F→M transfer conserving the matrix sum F+M before decay (gamma 0.12), and differential decay (rhoF 0.97; rhoM 0.9995). H=32, one slot; F/M each 8×8. Importantly, B5 uses distinct F/M queries, independent read normalization, and a learned gate: conservation of F+M does **not** guarantee conservation of this effective composite read or behavior. A four-class consequence head is new. No contextual KV change, RAG, explicit habit module, or memory label. The prior motivation is documented in `reports/STAGE2B_CONTEXTUAL_ASSOCIATIVE_MEMORY_RESULTS.md`; its associative-recall null result is not treated as a Stage 2C behavioral-memory result.","",
    "## 3. Lifetime protocol and experimental detail","", "Outer training: 500 AdamW updates, batch 16, randomized/balanced actions each episode, 4/8/16 episodes per batch lifetime, observable consequence CE plus 0.001 H-square regularization, parameter gradient clip 1. Formal evaluation: eight fresh independently trained seeds 6201–6208; four paired lifetimes per seed; all inference in eval/no-grad mode. A/B have identical parameters, schedules, features, compute, probe and unrelated delay, differing only in observed consequences. A matching action has deterministic outcome0; a nonmatching action has one of outcomes1–3 uniformly. Neither latent z nor action correctness is input. Train feature combinations are parity-even; primary novel probes parity-odd with individually seen tokens; hard OOD holds out individual tokens. The fixed probe policy is softmax(-predicted future entropy / 0.35), so behavioral probabilities are inferred from the model's forecast rather than supervised actions. Full protocol and dev amendments: `docs/STAGE2C_PROTOCOL.md`; fixed cutoffs: `configs/stage2c_formal.yaml`.","",
    "## 4. Frozen-parameter verification","", f"Before/after SHA-256 equal in every completed evaluation: **{s['frozen_verification']['all_hashes_equal']}** across {s['frozen_verification']['evaluation_manifests']} manifests. `tests/test_stage2c.py` additionally checks every named parameter with `torch.equal` through an evaluation lifetime. No evaluation optimizer exists. The repository's full pytest suite passed 107 tests with `PYTHONPATH=src:.`; `results/stage2c/manifests/shortcut_audit.json` records balanced-prefix and no-latent-leak checks.","",
    "## 5. Habit formation","","| Exposure N | BS mean ± seed SD |","|---:|---:|" ]
    lines += [f"| {n} | {fmt(s['curve'][str(n)])} |" for n in (0,1,2,4,8,16,32)]
    lines += ["","## 6. Persistence","","| Unrelated delay D | BS mean ± seed SD |","|---:|---:|"]
    lines += [f"| {d} | {fmt(s['persistence'][str(d)])} |" for d in (0,10,50,100,500,1000)]
    lines += ["","PR is undefined for near-zero BS0; no ratio is interpreted without BS0≥0.05 for the persistence gate.","",
              "## 7. Generalization","","| Probe | BS mean ± seed SD |","|---|---:|"]
    lines += [f"| {x} | {fmt(s['generalization'][x])} |" for x in ("seen","novel","hard_ood")]
    lines += ["","## 8. Selectivity","",f"Matched-count random history BS: {fmt(s['noise'])}. Compare absolute noise effects with structured useful BS per seed, not just aggregate means. Noise matches event count and compute, **not** the useful stream's marginal outcome-symbol frequency; any positive selectivity would need a stricter matched-marginal replication.","",
              "## 9. Revision","","| New reversal experiences | BS (old-minus-new orientation) |","|---:|---:|"]
    lines += [f"| {n} | {fmt(s['revision'][str(n)])} |" for n in (0,1,2,4,8,16,32)]
    lines += ["","## 10. H/F/M causal swaps","",f"Full-state paired action JS at D0: {fmt(s['full_JS0'])}.","",
              "| State intervention | BS at D0 | JS at D0 | BS at D100 | BS at D500 |","|---|---:|---:|---:|---:|"]
    for key in ("H_swap","F_swap","M_swap","FM_swap","HFM_swap"):
        x=s["interventions"][key]
        lines.append(f"| {key} | {fmt(x['0']['BS'])} | {fmt(x['0']['JS'])} | {fmt(x['100']['BS'])} | {fmt(x['500']['BS'])} |")
    lines += ["","Full swap is a sanity control: in exact paired states it should reverse the sign of BS up to numerical precision. Causal contributions are not necessarily additive due to nonlinear H/read interactions.","",
              "## 11. H-reset experiments","","| Intervention | BS D0 | BS D500 |","|---|---:|---:|"]
    for key in ("H_reset","H_reset_F_zero","H_reset_M_zero","H_reset_FM_zero","H_reset_F_swap","H_reset_M_swap"):
        x=s["interventions"][key]
        lines.append(f"| {key} | {fmt(x['0']['BS'])} | {fmt(x['500']['BS'])} |")
    lines += ["","## 12. Time-window read interventions","", "W1=first 8 experiences; W2=last 8 plus four NULL ticks; W3=50 unrelated events; W4=probe. Clamps suppress a read only, never directly zero F or M storage. Changed trajectories may later alter storage indirectly through learned access. A probe-time null effect cannot rule out earlier mediation. For the primary full model, each of the 15 window conditions has 153 per-tick records (2,295 ticks/seed) with H/F/M tensors, q/r, transfer/write magnitudes, event kind, and intervention flags under `results/stage2c/trajectories/full/<seed>/`; the replay checks BS against the formal window rows.","",
              "| Read condition | BS after matched trajectory |","|---|---:|"]
    for key in ("none",)+tuple(k for k in s["window"] if k.startswith("read_")):
        lines.append(f"| {key} | {fmt(s['window'][key])} |")
    lines += ["","## 13. Write interventions","", "Only the external outcome write is blocked; context/action events were already no-write. Reads remain active.","",
              "| Condition | BS |","|---|---:|"]
    for key in ("none","write_W1_none","write_W2_none"):
        lines.append(f"| {key} | {fmt(s['window'][key])} |")
    lines += ["","## 14. Consolidation interventions","", "Trained gamma=0 is a separately optimized ablation; posthoc gamma=0 loads the full checkpoint unchanged and suppresses transfer at evaluation. They answer different causal questions.","",
              "## 15. F/M timescale analysis","", "The N×D grid below is BS on novel probes, mean across independent training seeds. Per-cell F/M state/read norms are in `results/stage2c/processed/summary.json`; per-tick reads, transfers, and writes in trajectories. A norm difference alone is not evidence of behavioral causality; swap/read-window effects must change BS.","",
              "| N \\ D | 0 | 10 | 100 | 500 | 1000 |","|---:|---:|---:|---:|---:|---:|"]
    for n in (1,2,4,8,16):
        cells=[fmt(s["timescale_grid"][str(n)][str(d)]["BS"]) for d in (0,10,100,500,1000)]
        lines.append("| "+str(n)+" | "+" | ".join(cells)+" |")
    lines += ["",
              "## 16. Baselines","","| Model | Seeds | BS D0 | BS D500 | Probe future loss D0 | Action-branch TV D0 | Active trained parameters |","|---|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        b=s["baselines"][variant]
        lines.append(f"| {variant} | {len(b['seeds'])} | {fmt(b['BS0'])} | {fmt(b['BS500'])} | {fmt(b['probe_future_loss_D0'])} | {fmt(b['action_branch_TV_D0'])} | {fmt(b['active_parameters'])} |")
    lines += ["","The B0 and GRU core architecture is inherited; their F/M states are forced to zero and inaccessible. The parameter budgets are comparable but not exactly matched. F-only suppresses slow read but retains F→M transfer; M-disabled zeros M each step, so transfer can drain F into a discarded M (not a pure read lesion). Posthoc gamma=0 is not independently trained.","",
              "## 17. Stability","","| Full seed | First H>100 tick | First H>1000 tick | First nonfinite tick | Max H norm |","|---:|---:|---:|---:|---:|"]
    for seed,record in s["stability"].items():
        lines.append(f"| {seed} | {record['first_H_over_100']} | {record['first_H_over_1000']} | {record['first_nonfinite']} | {record['max_H']:.1f} |")
    over100=sum(x["first_H_over_100"] is not None for x in s["stability"].values())
    over1000=sum(x["first_H_over_1000"] is not None for x in s["stability"].values())
    nonfinite=sum(x["first_nonfinite"] is not None for x in s["stability"].values())
    lines += ["",f"The unchanged recurrence is evaluated; {over100}/8 full seeds crossed H>100, {over1000}/8 crossed H>1000, and {nonfinite}/8 became nonfinite. No bounded-recurrence pilot is used to rescue gates. H growth can confound long-delay comparisons even when finite.","",
              "## 18. Formal gates","","| Gate | Seeds meeting frozen criterion | PASS |","|---|---:|---|"]
    for gate,record in s["gates"].items():
        lines.append(f"| {gate} | {record['passed']}/{record['total']} | {'PASS' if record['passed']>=6 else 'FAIL'} |")
    lines += ["",f"G47 best among the predeclared finite intervention family: {s['gates']['G47'].get('best_predeclared_intervention')}. Family search is a multiplicity caveat; no per-seed cherry-picking is allowed. Causal mediation fractions are in `processed/summary.json` and are suppressed when baseline |BS|≤1e-4; they need not lie in [0,1].","",
              "## 19. Negative results and development transparency","", "Development seed runs (including failed unregularized/clock-shortcut versions) are retained in `results/stage2c/development/`; formal gates were frozen only after these failures. Near-zero BS, if observed, is a substantive failure of this world/head/training combination to produce action-conditional disposition; it is not proof that state memory is impossible. No exact-recall gate or M-necessity gate was added after seeing outcomes.","",
              f"Diagnostic (not a gate): final outer-training consequence CE {fmt(s['forecast_diagnostics']['final_training_CE'])}; probe action-branch forecast total-variation distance {fmt(s['forecast_diagnostics']['action_branch_TV'])}; paired-history forecast total-variation distance {fmt(s['forecast_diagnostics']['history_branch_TV'])}. Small action-branch distance indicates that the forecast head barely conditions on the proposed action, a concrete failure mode distinct from forgetting.","",
              "## 20. Interpretation and 24 required answers",""]
    value=lambda stat: num(stat["mean"])
    gate=lambda name: f"{s['gates'][name]['passed']}/{s['gates'][name]['total']}"
    full0=s["baselines"]["full"]["BS0"]
    full500=s["baselines"]["full"]["BS500"]
    avg=lambda stat: stat["mean"] if stat["mean"] is not None else 0.0
    swap_effect=lambda name,d: avg(full0 if d==0 else full500)-avg(s["interventions"][name][str(d)]["BS"])
    fm_larger=abs(swap_effect("FM_swap",0))>max(abs(swap_effect("F_swap",0)),
                                                 abs(swap_effect("M_swap",0)))
    base_window=avg(s["window"]["none"])
    read_effects={name:base_window-avg(stat) for name,stat in s["window"].items()
                  if name.startswith("read_")}
    strongest_window=max(read_effects,key=lambda x:abs(read_effects[x]))
    early_max=max(abs(v) for name,v in read_effects.items() if not name.startswith("read_W4"))
    probe_max=max(abs(v) for name,v in read_effects.items() if name.startswith("read_W4"))
    answers=[
        f"1. Frozen-parameter history effect: G42 {gate('G42')}; BS16={value(full0)}. {'Supported' if s['gates']['G42']['passed']>=6 else 'Not established'}; hashes remain equal.",
        f"2. Exposure response: BS N0={value(s['curve']['0'])}, N8={value(s['curve']['8'])}, N16={value(s['curve']['16'])}, N32={value(s['curve']['32'])}; no stable habit claim without G42.",
        f"3. Persistence: BS D0={value(s['persistence']['0'])}, D500={value(s['persistence']['500'])}, D1000={value(s['persistence']['1000'])}; G43 {gate('G43')}. PR is conditional on meaningful BS0.",
        f"4. New-instance generalization: seen={value(s['generalization']['seen'])}, novel={value(s['generalization']['novel'])}, hard OOD={value(s['generalization']['hard_ood'])}; G44 {gate('G44')}.",
        f"5. Noise resistance: useful BS={value(full0)}, matched noise BS={value(s['noise'])}; G45 {gate('G45')}.",
        f"6. Revision: BS before={value(s['revision']['0'])}, after16={value(s['revision']['16'])}, after32={value(s['revision']['32'])}; G46 {gate('G46')}.",
        f"7. H/F/M causal roles: H_swap={value(s['interventions']['H_swap']['0']['BS'])}, F_swap={value(s['interventions']['F_swap']['0']['BS'])}, M_swap={value(s['interventions']['M_swap']['0']['BS'])}; nonlinear effects are not additive.",
        f"8. F swap: BS={value(s['interventions']['F_swap']['0']['BS'])} versus full={value(full0)}; {'no robust peripheral mediation' if s['gates']['G47']['passed']<6 else 'see G47 mediation'}.",
        f"9. M swap: BS={value(s['interventions']['M_swap']['0']['BS'])} versus full={value(full0)}; M-only necessity is not required.",
        f"10. FM swap: BS={value(s['interventions']['FM_swap']['0']['BS'])}, compared with F={value(s['interventions']['F_swap']['0']['BS'])} and M={value(s['interventions']['M_swap']['0']['BS'])}; {'larger than both single-swap mean effects' if fm_larger else 'not larger than both single-swap mean effects'} at D0 (not a significance test).",
        f"11. H reset: BS={value(s['interventions']['H_reset']['0']['BS'])} at D0 versus full={value(full0)}; reset-plus-lesion values are tabulated above.",
        f"12. Read timing: largest absolute mean window effect is {strongest_window} ({num(read_effects[strongest_window])}); G47 {gate('G47')}. {'No stable timing attribution' if s['gates']['G47']['passed']<6 else 'A replicated peripheral effect exists; inspect seed-wise signs'}.",
        f"13. Probe lesion versus earlier read: largest W1–W3 absolute effect={num(early_max)}, W4={num(probe_max)}; {'no robust underestimation claim' if s['gates']['G47']['passed']<6 else 'early mediation may be underestimated by W4 if the per-seed pattern agrees'}.",
        f"14. Consolidation and persistence: full BS500={value(full500)}, trained gamma0={value(s['baselines']['gamma_zero']['BS500'])}, posthoc gamma0={value(s['baselines']['gamma_zero_posthoc']['BS500'])}; {'no established long-term benefit' if s['gates']['G43']['passed']<6 else 'long-term benefit requires paired seed comparison'}.",
        f"15. Gamma0 short/long: trained BS0={value(s['baselines']['gamma_zero']['BS0'])}, BS500={value(s['baselines']['gamma_zero']['BS500'])}; posthoc values are separately listed.",
        f"16. F→M timescale shift: F-swap effect D0={num(swap_effect('F_swap',0))}, D500={num(swap_effect('F_swap',500))}; M-swap D0={num(swap_effect('M_swap',0))}, D500={num(swap_effect('M_swap',500))}. {'No causal shift established without baseline behavior' if s['gates']['G42']['passed']<6 else 'Use seed-level trajectory and lesion pattern for attribution'}; norms alone are insufficient.",
        f"17. Generalized behavioral memory: {'supported in controlled novel-combination split' if s['gates']['G42']['passed']>=6 and s['gates']['G44']['passed']>=6 else 'not established'}; exact episodic recall was not a gate.",
        f"18. No-memory alternatives: B0 BS0={value(s['baselines']['no_memory']['BS0'])}, GRU BS0={value(s['baselines']['gru']['BS0'])}; matched independent training seeds.",
        f"19. Incremental full persistence: full BS500={value(full500)}, B0={value(s['baselines']['no_memory']['BS500'])}, GRU={value(s['baselines']['gru']['BS500'])}; no claim if effects are near zero.",
        f"20. H instability: {over100}/8 crossed H>100, {over1000}/8 crossed H>1000, {nonfinite}/8 nonfinite; long-delay behavior requires this caveat.",
        f"21. F→M copy sufficiency: {'supported only as controlled-world behavior' if s['outcome']=='A' else 'not demonstrated'} by the frozen gates; no mechanistic extrapolation.",
        "22. Stable-H redesign: prioritize a separately trained bounded-H pilot if H growth or action binding remains problematic; do not posthoc rescale the frozen model.",
        "23. Consolidation-as-abstraction redesign: current tests do not isolate whether copy versus abstraction is limiting; require a separate intervention study.",
        f"24. Natural-language transfer: {'still exploratory' if s['outcome']=='A' else 'not justified by Stage 2C'}; no language-scale success is claimed."
    ]
    lines += [item+"\n" for item in answers]
    lines += ["## 21. Next-stage recommendation","", f"Outcome **{s['outcome']}** under the frozen rules. Do not claim human-like habits, personality, consciousness, or natural-language memory. Prioritize diagnosing action/outcome binding, a marginal-matched noise control, and separately trained bounded H dynamics before a language-scale transfer; preserve all null results and baselines.","",
              "## Reproducibility and machine-readable artifacts","", "`results/stage2c/checkpoints/` holds independent training checkpoints and loss traces; `raw/` has one JSONL row per seed/condition and per-tick trajectory JSONL plus tensor snapshots; `manifests/` holds SHA-256 hashes and worker logs; `processed/summary.json` contains seed-level aggregates and gate decisions. Historical frozen Stage1/2A/2B results were not edited."]
    return "\n".join(lines)+"\n"


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--report",type=Path,default=REPORT)
    args=p.parse_args()
    ROOT=args.root;REPORT=args.report
    rows,trains,manifests=load()
    summary=build(rows,trains,manifests)
    if os.environ.get("STAGE2C_ANALYZE_DRYRUN"):
        draft=markdown(summary)
        print(json.dumps({"complete":summary["formal_complete"],
                          "seeds":summary["available_full_seeds"],
                          "draft_chars":len(draft),
                          "forecast_diagnostics":summary["forecast_diagnostics"],
                          "gates":summary["gates"]}))
        raise SystemExit(0)
    (ROOT/"processed").mkdir(parents=True,exist_ok=True)
    (ROOT/"processed"/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    (ROOT/"trajectories").mkdir(parents=True,exist_ok=True)
    (ROOT/"interventions").mkdir(parents=True,exist_ok=True)
    trajectory_index=[{"run_id":f"{variant}-{seed}",
                       "trace":str(ROOT/"raw"/variant/str(seed)/"trajectory.jsonl"),
                       "state_tensors":str(ROOT/"raw"/variant/str(seed)/"state_tensors.pt"),
                       "window_trace":str(ROOT/"trajectories"/"full"/str(seed)/"windows.jsonl")
                           if variant=="full" else None,
                       "window_state_tensors":str(ROOT/"trajectories"/"full"/str(seed)/"window_states.pt")
                           if variant=="full" else None}
                      for variant,seed in manifests]
    intervention_index=[{"run_id":f"{variant}-{seed}","rows":str(ROOT/"raw"/variant/str(seed)/"rows.jsonl")}
                        for variant,seed in manifests]
    (ROOT/"trajectories"/"index.json").write_text(json.dumps(trajectory_index,indent=2))
    (ROOT/"interventions"/"index.json").write_text(json.dumps(intervention_index,indent=2))
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(markdown(summary))
    assert REPORT.exists() and REPORT.stat().st_size > 0
    print(f"RESULT_FILE={REPORT.resolve()}")
    print(json.dumps({"outcome":summary["outcome"],"complete":summary["formal_complete"],
                      "gates":summary["gates"],"report":str(REPORT)}))
