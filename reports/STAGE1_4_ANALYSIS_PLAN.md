# Stage 1.4 pre-formal analysis plan

Frozen before formal seed 7401 is trained. This supplements, but does not
change, `STAGE1_4_PROTOCOL.md` and `configs/stage1_4.yaml`.

## Sampling, units, and comparisons

All 8 formal seeds are independently initialized and trained. Each trainable
baseline gets 160 steps, batch 32, the same four world families, same horizon
weights and independently selected development LR (all selected 0.001).
Episode-level CE is first averaged over valid horizons, then equally over
families and seeds. The seed is the unit for replication and bootstrap; rows
from the same trained model are not treated as independent seeds.

Experiment A primary G23 uses B5 K=4 versus K=0, averaged across four
families and valid horizons. Frozen/random NULL controls each use K=4 from
identical pre-NULL state. A secondary equal-compute timing comparison applies
K NULL transitions either before or after the next genuine external event,
then forecasts the subsequent event. It reports post-event latency but is not
substituted for the pre-event G23 endpoint. K=1/2/8/16 trajectories and each
horizon are reported, including non-monotonicity.

Experiment B primary G24 uses B5 h=1 CE after a bridge and four NULL ticks,
at 512 and 2048 distractors equally weighted. M lesion, F lesion,
random-q_M and shuffled-M are **same-checkpoint** interventions before the
bridge; B0/B3/B4/B6/B7 are separately trained baselines. The G24 no-memory
comparison is separately trained B0. Equal world seeds give paired targets.
No q_M alignment or read norm alone counts as autonomous reactivation.

Experiment C primary G25 compares B5 intact against F/M swapped between
episodes from exactly the same H at the pre-NULL start. The registered
endpoint is the mean across internal ticks 1/2/4/8, with H norm difference
and predictive Jensen–Shannon divergence both required. Experiment D reports
the ratio of H-restored to unrestored prediction JS at tick 8; a ratio ≤0.25
is WORKSPACE_MEDIATED, ≥0.75 PERIPHERAL_PERSISTENT, 0.25–0.75 MIXED, and
unstable or subthreshold C effect NOT_IDENTIFIED. This is an architecture
classification, not a gate.

Experiment E/F has 32 item-direction observations per seed. The lesion removes
the rank-one key direction from M only; because learned keys are not
orthogonal, its effect can overlap nearby item directions and is disclosed as
such. Retention is M-only cosine read after a 128-distractor interference
stream with F excluded. Read usage is accumulated effective M contribution
weighted by absolute query–key cosine. Spearman associations and fourfold
held-out rank regression compare exposure+read against exposure+read+CU.
Folds are by episode index; regression is descriptive within a seed, with
seed-level incremental R² used for G26. Development seeds showed negative
incremental R², so Experiment G is irrevocably NOT_RUN_BY_PROTOCOL for this
formal run and G26 cannot pass. Formal CU outcomes cannot retroactively
authorize G.

Normal SELF_OUTPUT writes are exactly zero. The pathological positive control
applies the external delta update to its own repeated content on purpose;
the diagnostic repeat-output propensity is a fixed monotonic transform of
memory readout strength, **not** a trained expression head. The safety audit
is called VALID only if B_bad's memory readout and proxy propensity rise
measurably while the normal model's external write count remains zero.

## Integrity and result boundaries

Every formal training/evaluation shard writes an immutable checkpoint or
Parquet record with SHA256. Analysis verifies 8×8 unique training/evaluation
cells, source/config/selection hashes, finite numeric records and required
field completeness. Any aborted shard is retained under its original ID;
recovery uses a new amended ID, never silent overwrite. All gate thresholds
and ≥6/8 replication requirements remain as in the protocol. Missing or
invalid evidence fails the relevant gate; it is not imputed positive.
