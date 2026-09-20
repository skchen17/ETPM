# ET-RCM Stage 1.5 Final Report

> **What dynamical and causal structure does the ET-RCM architecture itself possess, and can a learned state-dependent routing policy select the peripheral information whose finite use actually improves future computation?**

> **ET-RCM 架构自身究竟具有怎样的动力学与因果结构；同时，一个可学习的状态依赖路由机制，能否从外围持续状态中选择那些经真实有限干预验证、确实能够改善未来计算的信息？**

Formal run `stage1_5-formal-v1`: 96 new train cells (8 inherited architectures plus 4 additional capacity variants × 8 seeds), 160 independent evaluation shards, 727,360 machine-readable formal rows. Development: 48 train cells, 2 seeds × 2 LRs for 12 variants. All source/checkpoint/config/record hashes are checked in `results/stage1_5/processed/stage1_5-formal-v1/integrity.json`. Historic Stage 1.4 files remain unchanged. A diagnostic cumulative raw-read-norm table is saved at `results/stage1_5/processed/stage1_5-formal-v1/read_usage.parquet`; it is not a causal-use measure.

## Registered gates

| Gate | Outcome | Criterion |
| --- | --- | --- |
| G27 | FAIL | numerical/finite dynamics |
| G28 | FAIL | nontrivial H/F/M lag hierarchy |
| G29 | PASS | single peripheral predictive causal effect |
| G30 | PASS | explicit read mediation |
| G31 | FAIL | historical oracle ceiling |
| G32 | NOT_RUN_BY_PROTOCOL | finite-benefit learned routing |
| G33 | NOT_RUN_BY_PROTOCOL | routed NULL-time gain |

## Experimental design

FP32 primary. Eight independent seeds 8501–8508, 160 AdamW steps, batch 32, equal four-family world training; two development seeds 8401–8402 chose LR from 0.001/0.0003 using held-out CE before formal training. A1 has 4×8 real-history states per seed and every tick through 1024; A2 uses three independently perturbed components, two finite epsilons and nine checkpoints. A3 probes 768 histories per seed under four decay/transfer variants; A4 has five sparse capacities and six distractor lengths. A5 uses 32 paired states and seven exact swaps at four horizons; A6 uses 64 train + 32 test examples per family. B2 read-clamps F/M independently; B3/B4 use 16 paired episodes per seed/gap and matched controls. The historical oracle bank sees early real external evidence and the observed bridge only, never future targets. Full details are in the 12 topic reports.

## Required trained baselines

Paired long-gap 512/2048 full-condition CE (lower is better):

| Architecture | Mean CE | 95% seed CI |
| --- | --- | --- |
| B0_no_memory | 1.859285 | [1.4959486134466715, 2.2380624161916782] |
| B1_gru | 2.693553 | [2.6006597370840607, 2.7849809173494577] |
| B2_single_memory | 1.810392 | [1.5030970506486483, 2.1124894582200797] |
| B3_joint | 1.825519 | [1.5224607360898517, 2.118707878387067] |
| B4_shared | 1.803064 | [1.515040569473058, 2.088738097343594] |
| B5_separate | 1.691458 | [1.4307998880220112, 1.956677491299342] |
| B6_gamma_zero | 1.696354 | [1.4244567872490734, 2.0791234064323363] |
| B7_random_query | 1.387367 | [1.1294516911322716, 1.5905472795711828] |

Random-q B7 versus learned separate-q B5 CE difference (B5−B7)=0.30409. These are separately trained models, so this contrast is a baseline outcome, not an isolated query intervention.

## Direct answers to the 20 registered questions

1. Long NULL rollout: G27 **FAIL**; max component norm across seeds 2067.4944.

2. Effective H/F/M timescale separation: G28 **FAIL**; F−H area 0.00000, M−F area 0.00000. The preregistered probes showed no positive held-out CE gain at any lag; this does not prove that historical information is absent.

3. Decay versus dynamics/usage: equal-fast, equal-slow and no-transfer curves are reported. Their raw-score ablation differs in 7/8 seeds, but all primary lag-profile areas are zero after clipping negative decoding gains. Consequently no learned-timescale or usage contribution is established.

4. Capacity: at 8192 distractors, trained future CE is 2.150 for (d_H=64,d_M=16), 1.215 for (256,64), and 1.632 for (256,128): larger memory is not monotonically better. The separate 128-item matrix-cell top-1 accuracy falls sharply with interference; neither arm supports unlimited capacity.

5. Independent H/F/M future effects: tick-4 JS means H=0.405584, F=0.000281059, M=0.000817266; G29 **PASS**.

6. F×M, H×F and H×M signed CE interactions: FM=-0.00152575, HF=-0.00174354, HM=0.0305843.

7. H-only held-out future-event CE=0.9827, versus HF=3.3097, HM=1.9939, HFM=4.3855. H-only is best for this frozen linear probe; that is not proof of causal sufficiency.

8. Extra observational F/M information beyond H: HF gain=-2.32692, HM gain=-1.01116, HFM gain=-3.40281 CE; negative gain means the larger frozen linear probe generalized worse, not that the added state lacks all information.

9. Extra causal F/M information is supported only to G29's finite-swap scope: **PASS**; probe gain alone is insufficient.

10. Explicit read mediation: G30 **PASS**; F/M restored-read fractions are 0.892/1.000. The underlying swap JS effects are only about 0.00036/0.00091, so relative mediation does not imply useful retrieval.

11. Correct historical oracle read improves all registered controls: G31 **FAIL**. Its CE advantage over zero, random, shuffled and learned read is −0.00091, −0.00111, −0.00086 and −0.00048 respectively.

12. Oracle minus learned CE advantage=-0.00048 (positive means oracle better).

13. Static CE minus closed-loop CE at K4=-0.00008.

14. Finite RetrievalAdvantage signal: NOT_RUN_BY_PROTOCOL (B5 conditional prerequisite).

15. Learned state-dependent finite-benefit router: NOT_RUN_BY_PROTOCOL.

16. Held-out routing generalization: NOT_RUN_BY_PROTOCOL.

17. Routed NULL-tick prediction benefit: NOT_RUN_BY_PROTOCOL.

18. Independent observed bottlenecks: long-NULL architecture stability; effective multi-timescale evidence; historical-read integration/ceiling. The decision tree stops at the first failed prerequisite; later measured failures remain independently reported.

19. Causal utility should not enter consolidation law in this stage: routing/finite-use prerequisites have not all been met; transfer law was unchanged.

20. Small sequence/language prototype recommendation: **FALSE**; no LM was trained.

## Scientific interpretation and boundaries

Stored information, nonzero read, predictive probe gain, finite causal benefit and long-term retention are different observables. The historical oracle's saved pre-interference bank is an upper-bound intervention, not evidence that current M can retrieve it. Finite toy swap effects do not justify a general causal-memory label. No consciousness, human-like autonomy, unlimited capacity, or spontaneous language content is claimed. All failed, null and conditionally unexecuted outcomes are preserved; no gate was changed after formal results.
