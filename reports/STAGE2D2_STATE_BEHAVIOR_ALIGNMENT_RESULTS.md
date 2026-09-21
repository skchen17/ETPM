# ET-RCM Stage 2D.2 — State-to-Behavior Alignment, Controllability, and Initialization-Sensitive Basin Formation

> **Do successful ET-RCM initializations enter the noisy behavioral-memory basin because history- and memory-induced active-state changes align with the downstream directions that can actually control action-conditioned predictions, and can this alignment be causally manipulated to stabilize training?**
>
> **成功的 ET-RCM 初始化是否因为历史和记忆诱导的 H 状态变化更容易对齐到 downstream evaluator 真正能够利用的行为方向，从而进入 noisy behavioral-memory basin？这种 state-to-behavior alignment 是否可以通过有限因果干预和短暂训练引导被稳定建立？**

## Executive verdict

**Outcome D — No Useful Alignment Explanation.** The protected evaluator's finite H response is genuinely low dimensional, and late history-to-H alignment is higher in healthy runs. However, F/M-to-useful-H alignment is not stable across the old and confirmatory cohorts, useful-subspace rotation rescues only 1/8 failed checkpoints, healthy destruction reaches only 5/8, and the development-selected early controllability predictor collapses to AUROC .469 on 24 new runs. The causal and early-prediction prerequisites all fail.

Under the preregistered stopping rule, alignment warmup is therefore forbidden. G76–G78, reduced memory dynamics, peripheral recheck, and selected-model continuous diagnostics are `NOT_RUN_BY_PROTOCOL`, not silently failed or opportunistically redesigned.

## Experimental details

### Frozen architecture and cohorts

No model or memory law was modified. The primary values remain `gamma=.50`, `rho_fast=.97`, and `rho_slow=.9995`; the protected evaluator, learned queries/gate, recurrence, NULL, and SELF_OUTPUT semantics are unchanged.

- Cohort A: all 32 Stage 2D.1 factorial runs (8 initializations × 4 streams).
- Cohort B: 24 new baseline runs (initializations 9401–9408 × streams 12401–12403).
- Evaluation seeds: 15101 for A and 15701 for B, distinct from training seeds.
- Checkpoints: 0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500.
- Confirmatory labels at the frozen Stage 2D.1 resolution: 11/24 healthy, 1/24 partial, 12/24 shortcut.

Basin labels use only final action-TV, |interaction|, and |BS| ≥ .10. Geometry never changes a label.

### Finite response construction

For every run/checkpoint, 128 normalized H directions were tested with central interventions at `.05,.10,.25,.50 × native H scale`. The perturbation was applied as a paired-history contrast. The response vector contained both action-conditioned four-outcome forecasts, their action difference, entropy differences, interaction, and BS. The H-space finite response operator was reconstructed from the direction/response matrix and decomposed by SVD. Gradient/JVP evidence was not used for the formal claim.

The preregistered causal basis was rank 4 at epsilon .10. History alignment projects `H_A-H_B`; F/M/FM alignment projects the actual H proposal removed by a corresponding read clamp. All direction-level values were first reduced to one value per training run.

## A. Useful H response rank

The response is strongly low rank in both independent cohorts:

| Cohort / step | rank-1 energy | rank-2 | rank-4 | rank-8 | effective rank |
|---|---:|---:|---:|---:|---:|
| A / 0 | .902 | .974 | 1.000 | 1.000 | 1.417 |
| A / 500 | .988 | .999 | 1.000 | 1.000 | 1.064 |
| A / 1500 | .978 | .998 | 1.000 | 1.000 | 1.115 |
| B / 0 | .937 | .983 | 1.000 | 1.000 | 1.306 |
| B / 500 | .988 | .998 | 1.000 | 1.000 | 1.071 |
| B / 1500 | .982 | .997 | 1.000 | 1.000 | 1.103 |

Training makes an already compact response nearly rank one. Low evaluator rank by itself does not explain basin membership.

## B–D. History and memory alignment trajectory

At step 500, the first scientifically meaningful behavioral split and a history-alignment difference occur together:

| Cohort | Group | interaction | BS | M probe | A_hist(r=4) | A_M(r=4) |
|---|---|---:|---:|---:|---:|---:|
| A | healthy | .236 | .149 | .641 | .288 | .195 |
| A | failed | .021 | -.007 | .508 | .161 | .156 |
| B | healthy | .248 | .163 | .852 | .270 | .244 |
| B | failed | .028 | .001 | .702 | .182 | .209 |

History alignment remains larger at step 1500 in both cohorts: A `.280/.123`, B `.284/.177` (healthy/failed). Its separation is late rather than an early bootstrap signature.

Memory alignment is inconsistent. At step 1500, A_M is `.194/.148` in A but `.126/.163` in B—the direction reverses. F and FM alignment are also often larger in failed confirmatory runs. Therefore the specific hypothesis “healthy initialization succeeds because memory motion enters useful H directions more strongly” does not replicate.

The earliest old-cohort A_M hint is step 25 (AUROC .660), but confirmatory means are `.120/.107` with AUROC only .517. M decodability separates clearly by step 500 while A_M remains weak. The evidence is most compatible with conditional behavior and history alignment co-emerging, not memory alignment preceding behavior.

### Stored-but-unused cases

The two Stage 2D.1 stored-but-unused candidates had the original `Acc(z|M)=.75` and near-zero BS. Their Stage 2D.2 final A_M values were .133 and .186, versus old-cohort healthy mean .194. One is clearly low and one is near the healthy mean. Stored-but-unused failure is therefore **not consistently equivalent to low M→useful-H alignment**.

## E–F. Frozen rescue and healthy destruction

All interventions preserved parameter hashes and input F/M tensors. Each proposal was row-wise norm matched to the native M-induced H update; no eligible proposal exceeded 1.25× native norm.

For failed checkpoints, useful-only and every lambda rotation passed the full criterion in only 1/8 models. No common intervention reliably improved both interaction and |BS| while outperforming norm-matched random and orthogonal controls. **G73 FAIL (1/8).**

For healthy checkpoints, replacing the native M proposal with its orthogonal component damaged interaction and |BS| beyond the useful-subspace control in 5/8 models. This is suggestive but below the frozen 6/8 threshold. **G74 FAIL (5/8).**

Thus finite causality does not support either robust sufficiency or preregistered necessity.

## G. Initialization controllability confirmation

Eight development candidates covered external/F/M/combined controllability at steps 0 and 25. The frozen winner was step-0 external controllability:

- Cohort A healthy/failed mean: .126/.093.
- Development AUROC: .766, direction healthy > failed.
- Cohort B confirmatory AUROC: **.469**.
- Per-stream AUROC: .600, .375, .417; 0/3 streams reached .70.

**G75 FAIL.** Initialization variance is not explained by this frozen early controllability measure. Feature/sign were not changed after seeing Cohort B.

## H–I. Conditional stages and gates

Because G73, G74, and G75 all failed, Experiment H was not authorized. No alignment, random-subspace, or orthogonal warmup was trained. Consequently there is no selected alignment model and no legitimate G76 success rate.

| Gate | Result | Evidence |
|---|---|---|
| G73 frozen alignment rescue | **FAIL** | 1/8; required ≥6/8 |
| G74 alignment necessity | **FAIL** | 5/8; required ≥6/8 |
| G75 early controllability prediction | **FAIL** | AUROC .469; 0/3 streams replicate |
| G76 training basin stabilization | **NOT RUN BY PROTOCOL** | no diagnostic prerequisite passed |
| G77 dynamics restoration | **NOT RUN BY PROTOCOL** | conditional on G76 PASS |
| G78 peripheral preservation | **NOT RUN BY PROTOCOL** | conditional on G76/G77 |

The reduced formation/persistence/revision/selectivity grid and selected-model 1k/5k/10k continuous run are likewise `NOT_RUN_BY_PROTOCOL`. This avoids treating the already-known Stage 2D.1 baseline continuous result as evidence for an alignment intervention that was never trained.

## Answers to the 26 required questions

1. **Is the useful H response low dimensional?** Yes. Rank 1 explains about 90–99%; rank 2 explains 97–100%.
2. **How does rank change?** Effective rank falls from about 1.3–1.4 at step 0 to about 1.1 at step 1500.
3. **Does history→H alignment differ?** Yes late: healthy A_hist is higher at steps 500/1500 in both cohorts, but not as a reliable early marker.
4. **Does F-induced alignment differ?** Not consistently; confirmatory failed runs often have larger F alignment.
5. **Does M-induced alignment differ?** Not consistently; its final direction reverses across cohorts.
6. **Earliest alignment divergence?** A nonreplicated A_M hint appears at step 25; replicated, scientifically meaningful history alignment appears around step 500.
7. **Before or after behavior?** Approximately simultaneous with meaningful interaction/BS, not demonstrably before.
8. **M information or M→useful-H alignment first?** M information is the clearer/earlier signal; A_M is weak and unstable.
9. **Are stored-but-unused failures high M-probe/low A_M?** One clearly is; the other is not. The pattern is not reliable.
10. **Can rotation rescue failures?** No, only 1/8.
11. **Does random rotation fail selectively?** Useful rotation does not reliably outperform random/orthogonal controls, so specificity is absent.
12. **Does destroying healthy alignment hurt?** In 5/8, below G74.
13. **Does early controllability predict the basin?** No; confirmatory AUROC .469.
14. **Does it explain initialization variance?** No stable fraction can be attributed to the frozen metric.
15. **Did warmup reach ≥6/8?** Not adjudicated; warmup was forbidden by the stopping rule.
16. **Did random warmup lack the same benefit?** Not run because no training intervention was authorized.
17. **Did ability persist after assistance removal?** Not applicable; no assistance was trained.
18. **Was gradual formation restored?** Not run by protocol.
19. **Was persistence restored?** Not run by protocol.
20. **Was revision restored?** Not run by protocol.
21. **Was selectivity restored?** Not run by protocol.
22. **Was peripheral mediation preserved?** Not run by protocol.
23. **Did warmup harm continuous stability?** No selected warmup model exists; this question is not adjudicable.
24. **Best basin description?** Mixed downstream conditional-binding failure plus late history alignment; not a replicated memory→H alignment basin.
25. **Reopen formal F→M handoff?** No. There is no causal alignment stabilization or restored dynamics.
26. **Modify F/M law?** No. The evidence instead points back to event/state representation, protected-evaluator coupling, and recurrent conditional-binding topology.

## Tests, integrity, and claim boundary

Fourteen new tests cover exact finite perturbations, norm matching, projection idempotence, orthogonal reconstruction, matched-rank controls, frozen memory/parameters, absence of latent/correct-action/memory targets, evaluator isolation, exact auxiliary shutoff, ≥80% endogenous updates, formal-evaluation isolation, and historical integrity.

The Stage 2D.2 suite passed 14/14 and the 2,242-file pre-stage manifest had zero changes. The repository-wide run collected 196 tests: 194 passed and the same two inherited frozen-test defects remained (`test_stage1_5` rejects any later README extension; `test_stage2d` includes its own Stage2C-named manifest in the current snapshot although that file was absent from its creation snapshot). Historical tests and artifacts were not rewritten to obtain a green run.

This synthetic diagnostic does not establish causal memory, human-like thought, consciousness, or general controllability. Low response rank is a property of this protected evaluator/task, not a universal property of language models. Null and stopped-by-protocol results are retained without changing thresholds.

## Machine-readable evidence

- `results/stage2d2/behavioral_subspaces/cohort_a|cohort_b/`: finite matrices, SVDs, trajectories, alignments
- `results/stage2d2/processed/alignment_analysis.json`: development/confirmation and rank summaries
- `results/stage2d2/frozen_rescue/` and `healthy_destruction/`: norm-matched causal results
- `results/stage2d2/confirmatory_basin/labels.json`: frozen 24-run basin labels
- `results/stage2d2/processed/formal_gates.json`: G73–G78 adjudication
- `results/stage2d2/manifests/`: pre-stage and verification hashes
