# Stage 1.5 Frozen Analysis Plan

Frozen before any B5 formal evaluation. `configs/stage1_5.yaml` and
`reports/STAGE1_5_PROTOCOL.md` are controlling documents; this file resolves
aggregation details, never changes their margins.

All primary comparisons are paired within episode, averaged within seed,
then equally averaged across eight new seeds. A positive-seed rule is
at least 6/8. Bootstrap intervals resample seeds 2,000 times with seed 1515.
No p-value replaces a registered effect margin.

## Track A

- G27 uses all 1024 NULL ticks from 4 families × 8 histories per seed.
  Each seed must have finite state/entropy/CE/JS and maximum component norm
  <1000, maximum JS from tick 0 <0.5. At tick 128, maximum median full-state
  gain over perturbation component/epsilon/family is <100. All eight seeds
  must satisfy these numerical-safety conditions. Classification of
  fixed-point-like, bounded drift, oscillatory or divergent is descriptive;
  no strict chaos claim follows from finite growth.
- G28 uses held-out ridge-probe constant CE minus fitted CE, clipped to
  nonnegative for profile area. Area is trapezoid over log2(lag), normalized
  to unit log width. Ordering requires F−H >=0.02 and M−F >=0.02 in 6/8
  primary seeds. To support more than imposed decay, late-lag (>=64)
  M profile under primary versus both equal-decay variants must differ by
  >=0.01 in at least 6 seeds; no-transfer is separately reported. If
  probes are poor or the ablation is indistinguishable, G28 fails.
- A4 reports each sparse-grid item-count × distractor cell separately.
  Continuous random-vector associative microbenchmarks are explicitly
  mechanistic and cannot be passed off as trained prediction. Trained
  prediction is a separate paired long-gap curve. No G27–G33 gate is
  redefined from capacity post hoc.
- G29 is assessed at tick 4. FM joint mean JS and at least one of F/M
  single-channel mean JS must each exceed 1e-5 with 6/8 seed replication.
  H effects and signed CE pairwise interactions are reported regardless.
  State distance alone cannot pass.
- A6 uses train-only ridge fits and held-out future H MSE, logits MSE and
  event CE. H-only, HF, HM and HFM are compared. This is observational
  sufficiency; it cannot rescue G29.

## Track B

- G30 considers every F/M single channel meeting the G29 tick-4 causal
  threshold and replication rule. At tick 4 the per-seed descriptive
  mediation fraction is `1−JS(restored)/JS(swapped)` only when the paired
  swap JS exceeds 1e-5. Every relevant channel must have >=0.5 mean
  reduction and >=6/8 positive seeds. Absolute JS and CE are always shown.
  Otherwise label `UNMODELED_PERIPHERAL_PATHWAY`; a near-zero denominator
  yields `UNIDENTIFIABLE_READ_MEDIATION`, never a fictitious ratio.
- G31 uses equal-weight mean of 512 and 2048 distractor gaps at four
  post-bridge ticks. For each seed, compare `CE(comparator)−CE(oracle_static)`
  for zero-M, norm-matched random, cross-episode shuffled, and current
  learned M read. Each comparator requires >=0.01 mean and >=6/8 positive
  seeds. The full no-read control and gap 128 are secondary. B4 contrasts
  closed-loop and static oracle at ticks 1/2/4/8 descriptively.
- G31 is the hard authorization for B5 finite RetrievalAdvantage. If it
  fails, B5/B6/B7 and secondary causal-utility retention are
  `NOT_RUN_BY_PROTOCOL`. If G31 passes, B5 signal is stable only when
  candidate advantage rank agreement across two independent paired future
  batches is positive on >=6/8 development seeds and the best finite
  candidate improves no-read CE by >=0.01. The future target is used only
  to measure finite benefits, never supplied to the router.
- G32 uses held-out worlds and history seeds distinct from routing fit.
  The state-conditioned student must beat zero/random by >=0.01 CE,
  recover >=50% of the oracle improvement, and replicate in >=6/8 seeds.
  G33 compares routed K4−K0 CE to frozen/random NULL controls with >=0.01
  and >=6/8; it cannot be tested if G32 fails.

Precision regression is exploratory. FP32 is the sole primary condition.
Historical frozen artifacts are compared to revision `1367110`; only added
Stage 1.5 files are permitted. The final report must retain all negative,
null, and conditionally unexecuted results.
