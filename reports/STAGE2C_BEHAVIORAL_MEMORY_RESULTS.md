# ET-RCM Stage 2C Behavioral Memory Results

> **Can past experience produce persistent, selective, generalizable, revisable, and causally state-mediated changes in ET-RCM's future behavior without changing its parameters?**

> **在模型参数完全不更新的情况下，过去经历能否通过持续内部状态形成持久、选择性、可泛化、可修正，并具有因果作用的未来行为改变？**

Formal outcome: **C**. Formal completeness: **True**. Independent full-model training seeds: [6201, 6202, 6203, 6204, 6205, 6206, 6207, 6208].

## 1. Scientific question

Memory here means a persistent causal effect of past experience on behavior, not exact historical recall. The alternative is no reliable history-dependent behavior even though H/F/M values change.

## 2. Architecture

Stage 1.4/1.5 gated residual H dynamics, unchanged external delta F write, separate learned F/M reads, F→M transfer conserving the matrix sum F+M before decay (gamma 0.12), and differential decay (rhoF 0.97; rhoM 0.9995). H=32, one slot; F/M each 8×8. Importantly, B5 uses distinct F/M queries, independent read normalization, and a learned gate: conservation of F+M does **not** guarantee conservation of this effective composite read or behavior. A four-class consequence head is new. No contextual KV change, RAG, explicit habit module, or memory label. The prior motivation is documented in `reports/STAGE2B_CONTEXTUAL_ASSOCIATIVE_MEMORY_RESULTS.md`; its associative-recall null result is not treated as a Stage 2C behavioral-memory result.

## 3. Lifetime protocol and experimental detail

Outer training: 500 AdamW updates, batch 16, randomized/balanced actions each episode, 4/8/16 episodes per batch lifetime, observable consequence CE plus 0.001 H-square regularization, parameter gradient clip 1. Formal evaluation: eight fresh independently trained seeds 6201–6208; four paired lifetimes per seed; all inference in eval/no-grad mode. A/B have identical parameters, schedules, features, compute, probe and unrelated delay, differing only in observed consequences. A matching action has deterministic outcome0; a nonmatching action has one of outcomes1–3 uniformly. Neither latent z nor action correctness is input. Train feature combinations are parity-even; primary novel probes parity-odd with individually seen tokens; hard OOD holds out individual tokens. The fixed probe policy is softmax(-predicted future entropy / 0.35), so behavioral probabilities are inferred from the model's forecast rather than supervised actions. Full protocol and dev amendments: `docs/STAGE2C_PROTOCOL.md`; fixed cutoffs: `configs/stage2c_formal.yaml`.

## 4. Frozen-parameter verification

Before/after SHA-256 equal in every completed evaluation: **True** across 56 manifests. `tests/test_stage2c.py` additionally checks every named parameter with `torch.equal` through an evaluation lifetime. No evaluation optimizer exists. The repository's full pytest suite passed 107 tests with `PYTHONPATH=src:.`; `results/stage2c/manifests/shortcut_audit.json` records balanced-prefix and no-latent-leak checks.

## 5. Habit formation

| Exposure N | BS mean ± seed SD |
|---:|---:|
| 0 | +0.00e+00 ± +0.00e+00 (n=8) |
| 1 | +5.95e-06 ± +4.52e-05 (n=8) |
| 2 | +2.51e-05 ± +5.36e-05 (n=8) |
| 4 | +8.38e-06 ± +1.94e-05 (n=8) |
| 8 | +6.91e-06 ± +1.74e-05 (n=8) |
| 16 | -2.89e-06 ± +8.02e-06 (n=8) |
| 32 | +7.69e-07 ± +3.88e-06 (n=8) |

## 6. Persistence

| Unrelated delay D | BS mean ± seed SD |
|---:|---:|
| 0 | -2.89e-06 ± +8.02e-06 (n=8) |
| 10 | -1.65e-06 ± +6.90e-06 (n=8) |
| 50 | -3.84e-07 ± +3.21e-06 (n=8) |
| 100 | -2.92e-07 ± +1.41e-06 (n=8) |
| 500 | -3.54e-08 ± +9.52e-08 (n=8) |
| 1000 | -1.30e-08 ± +5.66e-08 (n=8) |

PR is undefined for near-zero BS0; no ratio is interpreted without BS0≥0.05 for the persistence gate.

## 7. Generalization

| Probe | BS mean ± seed SD |
|---|---:|
| seen | -2.96e-06 ± +7.87e-06 (n=8) |
| novel | -2.89e-06 ± +8.02e-06 (n=8) |
| hard_ood | -2.93e-06 ± +7.85e-06 (n=8) |

## 8. Selectivity

Matched-count random history BS: -9.86e-06 ± +1.44e-05 (n=8). Compare absolute noise effects with structured useful BS per seed, not just aggregate means. Noise matches event count and compute, **not** the useful stream's marginal outcome-symbol frequency; any positive selectivity would need a stricter matched-marginal replication.

## 9. Revision

| New reversal experiences | BS (old-minus-new orientation) |
|---:|---:|
| 0 | -2.89e-06 ± +8.02e-06 (n=8) |
| 1 | -3.25e-06 ± +6.77e-06 (n=8) |
| 2 | -4.48e-06 ± +8.97e-06 (n=8) |
| 4 | -6.05e-07 ± +9.66e-06 (n=8) |
| 8 | -4.00e-07 ± +4.52e-06 (n=8) |
| 16 | -2.42e-08 ± +3.03e-06 (n=8) |
| 32 | -1.32e-06 ± +2.48e-06 (n=8) |

## 10. H/F/M causal swaps

Full-state paired action JS at D0: +2.57e-09 ± +8.28e-09 (n=8).

| State intervention | BS at D0 | JS at D0 | BS at D100 | BS at D500 |
|---|---:|---:|---:|---:|
| H_swap | +6.03e-07 ± +5.02e-06 (n=8) | +1.04e-08 ± +7.91e-09 (n=8) | +3.91e-07 ± +1.32e-06 (n=8) | +8.94e-08 ± +1.42e-07 (n=8) |
| F_swap | -4.34e-07 ± +4.85e-06 (n=8) | +1.18e-08 ± +1.53e-08 (n=8) | -3.17e-07 ± +1.41e-06 (n=8) | -4.28e-08 ± +1.26e-07 (n=8) |
| M_swap | -3.04e-06 ± +8.69e-06 (n=8) | +1.42e-08 ± +1.48e-08 (n=8) | -3.58e-07 ± +1.31e-06 (n=8) | -2.24e-08 ± +9.49e-08 (n=8) |
| FM_swap | -6.03e-07 ± +5.02e-06 (n=8) | +1.04e-08 ± +7.91e-09 (n=8) | -3.91e-07 ± +1.32e-06 (n=8) | -8.94e-08 ± +1.42e-07 (n=8) |
| HFM_swap | +2.89e-06 ± +8.02e-06 (n=8) | +2.57e-09 ± +8.28e-09 (n=8) | +2.92e-07 ± +1.41e-06 (n=8) | +3.54e-08 ± +9.52e-08 (n=8) |

Full swap is a sanity control: in exact paired states it should reverse the sign of BS up to numerical precision. Causal contributions are not necessarily additive due to nonlinear H/read interactions.

## 11. H-reset experiments

| Intervention | BS D0 | BS D500 |
|---|---:|---:|
| H_reset | -7.22e-05 ± +2.25e-04 (n=8) | -3.91e-06 ± +2.02e-05 (n=8) |
| H_reset_F_zero | -3.16e-05 ± +6.02e-05 (n=8) | -2.04e-07 ± +8.19e-06 (n=8) |
| H_reset_M_zero | +1.59e-06 ± +2.31e-04 (n=8) | -1.10e-05 ± +2.46e-05 (n=8) |
| H_reset_FM_zero | +0.00e+00 ± +0.00e+00 (n=8) | +0.00e+00 ± +0.00e+00 (n=8) |
| H_reset_F_swap | -5.89e-06 ± +2.16e-04 (n=8) | +2.55e-06 ± +2.00e-05 (n=8) |
| H_reset_M_swap | +5.89e-06 ± +2.16e-04 (n=8) | -2.55e-06 ± +2.00e-05 (n=8) |

## 12. Time-window read interventions

W1=first 8 experiences; W2=last 8 plus four NULL ticks; W3=50 unrelated events; W4=probe. Clamps suppress a read only, never directly zero F or M storage. Changed trajectories may later alter storage indirectly through learned access. A probe-time null effect cannot rule out earlier mediation. For the primary full model, each of the 15 window conditions has 153 per-tick records (2,295 ticks/seed) with H/F/M tensors, q/r, transfer/write magnitudes, event kind, and intervention flags under `results/stage2c/trajectories/full/<seed>/`; the replay checks BS against the formal window rows.

| Read condition | BS after matched trajectory |
|---|---:|
| none | -5.01e-07 ± +3.60e-06 (n=8) |
| read_W1_F | -8.94e-08 ± +3.06e-06 (n=8) |
| read_W1_M | +4.28e-08 ± +3.96e-06 (n=8) |
| read_W1_FM | -2.83e-07 ± +2.83e-06 (n=8) |
| read_W2_F | -2.79e-07 ± +2.63e-06 (n=8) |
| read_W2_M | -6.43e-07 ± +2.81e-06 (n=8) |
| read_W2_FM | -4.77e-07 ± +1.46e-06 (n=8) |
| read_W3_F | -1.60e-07 ± +3.57e-06 (n=8) |
| read_W3_M | -7.13e-07 ± +2.78e-06 (n=8) |
| read_W3_FM | -4.15e-07 ± +2.79e-06 (n=8) |
| read_W4_F | -4.82e-07 ± +3.32e-06 (n=8) |
| read_W4_M | -6.22e-07 ± +3.37e-06 (n=8) |
| read_W4_FM | -6.11e-07 ± +3.12e-06 (n=8) |

## 13. Write interventions

Only the external outcome write is blocked; context/action events were already no-write. Reads remain active.

| Condition | BS |
|---|---:|
| none | -5.01e-07 ± +3.60e-06 (n=8) |
| write_W1_none | +6.59e-07 ± +3.03e-06 (n=8) |
| write_W2_none | -2.40e-07 ± +3.79e-06 (n=8) |

## 14. Consolidation interventions

Trained gamma=0 is a separately optimized ablation; posthoc gamma=0 loads the full checkpoint unchanged and suppresses transfer at evaluation. They answer different causal questions.

## 15. F/M timescale analysis

The N×D grid below is BS on novel probes, mean across independent training seeds. Per-cell F/M state/read norms are in `results/stage2c/processed/summary.json`; per-tick reads, transfers, and writes in trajectories. A norm difference alone is not evidence of behavioral causality; swap/read-window effects must change BS.

| N \ D | 0 | 10 | 100 | 500 | 1000 |
|---:|---:|---:|---:|---:|---:|
| 1 | +5.95e-06 ± +4.52e-05 (n=8) | -6.15e-07 ± +1.95e-05 (n=8) | +9.50e-08 ± +1.14e-06 (n=8) | -3.91e-08 ± +1.21e-07 (n=8) | +9.31e-09 ± +6.71e-08 (n=8) |
| 2 | +2.51e-05 ± +5.36e-05 (n=8) | +5.24e-06 ± +1.98e-05 (n=8) | +1.19e-07 ± +8.22e-07 (n=8) | +4.66e-08 ± +8.02e-08 (n=8) | +2.98e-08 ± +7.80e-08 (n=8) |
| 4 | +8.38e-06 ± +1.94e-05 (n=8) | +2.28e-06 ± +1.83e-05 (n=8) | -9.31e-09 ± +1.41e-06 (n=8) | +3.73e-08 ± +9.08e-08 (n=8) | +2.98e-08 ± +3.90e-08 (n=8) |
| 8 | +6.91e-06 ± +1.74e-05 (n=8) | +2.02e-06 ± +8.18e-06 (n=8) | -3.71e-07 ± +1.44e-06 (n=8) | +1.49e-08 ± +9.25e-08 (n=8) | +1.30e-08 ± +3.13e-08 (n=8) |
| 16 | -2.89e-06 ± +8.02e-06 (n=8) | -1.65e-06 ± +6.90e-06 (n=8) | -2.92e-07 ± +1.41e-06 (n=8) | -3.54e-08 ± +9.52e-08 (n=8) | -1.30e-08 ± +5.66e-08 (n=8) |

## 16. Baselines

| Model | Seeds | BS D0 | BS D500 | Probe future loss D0 | Action-branch TV D0 | Active trained parameters |
|---|---:|---:|---:|---:|---:|---:|
| full | 8 | -2.89e-06 ± +8.02e-06 (n=8) | -3.54e-08 ± +9.52e-08 (n=8) | +1.25468 ± +0.03746 (n=8) | +2.59e-04 ± +7.64e-05 (n=8) | 13,495 ± +0.00e+00 (n=8) |
| no_memory | 8 | +3.80e-07 ± +2.27e-06 (n=8) | +2.79e-08 ± +3.23e-08 (n=8) | +1.25148 ± +0.02974 (n=8) | +2.71e-04 ± +7.68e-05 (n=8) | 12,804 ± +0.00e+00 (n=8) |
| gru | 8 | -1.29e-07 ± +2.27e-06 (n=8) | +0.00e+00 ± +0.00e+00 (n=8) | +1.25121 ± +0.01257 (n=8) | +0.00166 ± +6.74e-04 (n=8) | 9,764 ± +0.00e+00 (n=8) |
| gamma_zero | 8 | -5.01e-07 ± +4.00e-06 (n=8) | -1.86e-09 ± +2.69e-08 (n=8) | +1.25247 ± +0.03108 (n=8) | +2.82e-04 ± +8.75e-05 (n=8) | 13,462 ± +0.00e+00 (n=8) |
| f_only | 8 | +2.01e-06 ± +4.96e-06 (n=8) | +2.42e-08 ± +9.91e-08 (n=8) | +1.25178 ± +0.03657 (n=8) | +2.93e-04 ± +9.33e-05 (n=8) | 13,239 ± +0.00e+00 (n=8) |
| m_disabled | 8 | -5.63e-07 ± +1.85e-06 (n=8) | +3.73e-08 ± +9.39e-08 (n=8) | +1.25640 ± +0.03510 (n=8) | +3.07e-04 ± +7.18e-05 (n=8) | 13,239 ± +0.00e+00 (n=8) |
| gamma_zero_posthoc | 8 | +2.62e-06 ± +5.10e-06 (n=8) | +2.24e-08 ± +1.54e-07 (n=8) | +1.25367 ± +0.03931 (n=8) | +2.49e-04 ± +7.43e-05 (n=8) | unavailable |

The B0 and GRU core architecture is inherited; their F/M states are forced to zero and inaccessible. The parameter budgets are comparable but not exactly matched. F-only suppresses slow read but retains F→M transfer; M-disabled zeros M each step, so transfer can drain F into a discarded M (not a pure read lesion). Posthoc gamma=0 is not independently trained.

## 17. Stability

| Full seed | First H>100 tick | First H>1000 tick | First nonfinite tick | Max H norm |
|---:|---:|---:|---:|---:|
| 6201 | 393 | None | None | 297.9 |
| 6202 | 276 | None | None | 480.0 |
| 6203 | 359 | None | None | 364.5 |
| 6204 | 407 | None | None | 307.1 |
| 6205 | 391 | None | None | 319.9 |
| 6206 | 257 | None | None | 606.5 |
| 6207 | 289 | None | None | 502.7 |
| 6208 | 323 | None | None | 422.3 |

The unchanged recurrence is evaluated; 8/8 full seeds crossed H>100, 0/8 crossed H>1000, and 0/8 became nonfinite. No bounded-recurrence pilot is used to rescue gates. H growth can confound long-delay comparisons even when finite.

## 18. Formal gates

| Gate | Seeds meeting frozen criterion | PASS |
|---|---:|---|
| G42 | 0/8 | FAIL |
| G43 | 0/8 | FAIL |
| G44 | 0/8 | FAIL |
| G45 | 0/8 | FAIL |
| G46 | 0/8 | FAIL |
| G47 | 0/8 | FAIL |

G47 best among the predeclared finite intervention family: F_swap. Family search is a multiplicity caveat; no per-seed cherry-picking is allowed. Causal mediation fractions are in `processed/summary.json` and are suppressed when baseline |BS|≤1e-4; they need not lie in [0,1].

## 19. Negative results and development transparency

Development seed runs (including failed unregularized/clock-shortcut versions) are retained in `results/stage2c/development/`; formal gates were frozen only after these failures. Near-zero BS, if observed, is a substantive failure of this world/head/training combination to produce action-conditional disposition; it is not proof that state memory is impossible. No exact-recall gate or M-necessity gate was added after seeing outcomes.

Diagnostic (not a gate): final outer-training consequence CE +1.24905 ± +0.04772 (n=8); probe action-branch forecast total-variation distance +2.59e-04 ± +7.64e-05 (n=8); paired-history forecast total-variation distance +0.00685 ± +0.00194 (n=8). Small action-branch distance indicates that the forecast head barely conditions on the proposed action, a concrete failure mode distinct from forgetting.

## 20. Interpretation and 24 required answers

1. Frozen-parameter history effect: G42 0/8; BS16=-2.89e-06. Not established; hashes remain equal.

2. Exposure response: BS N0=+0.00e+00, N8=+6.91e-06, N16=-2.89e-06, N32=+7.69e-07; no stable habit claim without G42.

3. Persistence: BS D0=-2.89e-06, D500=-3.54e-08, D1000=-1.30e-08; G43 0/8. PR is conditional on meaningful BS0.

4. New-instance generalization: seen=-2.96e-06, novel=-2.89e-06, hard OOD=-2.93e-06; G44 0/8.

5. Noise resistance: useful BS=-2.89e-06, matched noise BS=-9.86e-06; G45 0/8.

6. Revision: BS before=-2.89e-06, after16=-2.42e-08, after32=-1.32e-06; G46 0/8.

7. H/F/M causal roles: H_swap=+6.03e-07, F_swap=-4.34e-07, M_swap=-3.04e-06; nonlinear effects are not additive.

8. F swap: BS=-4.34e-07 versus full=-2.89e-06; no robust peripheral mediation.

9. M swap: BS=-3.04e-06 versus full=-2.89e-06; M-only necessity is not required.

10. FM swap: BS=-6.03e-07, compared with F=-4.34e-07 and M=-3.04e-06; not larger than both single-swap mean effects at D0 (not a significance test).

11. H reset: BS=-7.22e-05 at D0 versus full=-2.89e-06; reset-plus-lesion values are tabulated above.

12. Read timing: largest absolute mean window effect is read_W1_M (-5.44e-07); G47 0/8. No stable timing attribution.

13. Probe lesion versus earlier read: largest W1–W3 absolute effect=+5.44e-07, W4=+1.21e-07; no robust underestimation claim.

14. Consolidation and persistence: full BS500=-3.54e-08, trained gamma0=-1.86e-09, posthoc gamma0=+2.24e-08; no established long-term benefit.

15. Gamma0 short/long: trained BS0=-5.01e-07, BS500=-1.86e-09; posthoc values are separately listed.

16. F→M timescale shift: F-swap effect D0=-2.45e-06, D500=+7.45e-09; M-swap D0=+1.55e-07, D500=-1.30e-08. No causal shift established without baseline behavior; norms alone are insufficient.

17. Generalized behavioral memory: not established; exact episodic recall was not a gate.

18. No-memory alternatives: B0 BS0=+3.80e-07, GRU BS0=-1.29e-07; matched independent training seeds.

19. Incremental full persistence: full BS500=-3.54e-08, B0=+2.79e-08, GRU=+0.00e+00; no claim if effects are near zero.

20. H instability: 8/8 crossed H>100, 0/8 crossed H>1000, 0/8 nonfinite; long-delay behavior requires this caveat.

21. F→M copy sufficiency: not demonstrated by the frozen gates; no mechanistic extrapolation.

22. Stable-H redesign: prioritize a separately trained bounded-H pilot if H growth or action binding remains problematic; do not posthoc rescale the frozen model.

23. Consolidation-as-abstraction redesign: current tests do not isolate whether copy versus abstraction is limiting; require a separate intervention study.

24. Natural-language transfer: not justified by Stage 2C; no language-scale success is claimed.

## 21. Next-stage recommendation

Outcome **C** under the frozen rules. Do not claim human-like habits, personality, consciousness, or natural-language memory. Prioritize diagnosing action/outcome binding, a marginal-matched noise control, and separately trained bounded H dynamics before a language-scale transfer; preserve all null results and baselines.

## Reproducibility and machine-readable artifacts

`results/stage2c/checkpoints/` holds independent training checkpoints and loss traces; `raw/` has one JSONL row per seed/condition and per-tick trajectory JSONL plus tensor snapshots; `manifests/` holds SHA-256 hashes and worker logs; `processed/summary.json` contains seed-level aggregates and gate decisions. Historical frozen Stage1/2A/2B results were not edited.
