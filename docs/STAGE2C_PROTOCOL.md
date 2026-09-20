# Stage 2C Behavioral Memory protocol

> **Can past experience produce persistent, selective, generalizable, revisable, and causally state-mediated changes in ET-RCM's future behavior without changing its parameters?**

> **在模型参数完全不更新的情况下，过去经历能否通过持续内部状态形成持久、选择性、可泛化、可修正，并具有因果作用的未来行为改变？**

Status: formal gate thresholds, seeds, training budget, and outcome rule frozen **before** formal seeds 6201–6208 were trained. Machine-readable specification: `configs/stage2c_formal.yaml`.

## Distinction from frozen earlier work

Stage 1, 2A, and 2B result files are read-only. Stage 2C uses the existing Stage 1.5 read-intervention interface and Stage 1.4/1.3 memory transition without changing their implementation. It adds a distinct symbolic world, a consequence head, a training script, and lifetime evaluation code. The main gated residual H recurrence is unchanged. A separate bounded-H recurrence pilot was optional and is **not** run in the formal gates.

"Readout-conserving" here refers precisely to the matrix transfer invariant `F+M` before decay. The inherited B5 interface uses separate F/M queries, independent normalization and a learned gate, so its **effective composite read** need not be invariant when content moves from F to M. Stage 2C does not equate matrix conservation with behavioral readout conservation.

## World and objective

Latent z is 0/1, balanced independently of surface. Each observed episode comprises a shared abstract context, color, shape, nuisance, action, and one observable consequence. The matching action yields outcome symbol 0 deterministically; the nonmatching action yields one of three other symbols uniformly. This is a predictability difference, **not** a reward signal or a presented correct-action label. At prediction time, only previous observations and the current context/action are available. The outcome is then presented once as external evidence. Action probing uses the lower predicted consequence entropy of two hypothetical branches. Branches are discarded; they do not create external evidence. Outer training minimizes observed-consequence cross entropy plus 0.001 mean H-square regularization; no action, memory, importance, or habit label enters the loss. The regularizer changes outer optimization, not the state transition. There are 500 optimizer steps, batch 16, and random episode lengths 4/8/16. Actions are randomized and batch balanced each tick. The four train colors and shapes occur only in parity-even combinations. Novel probes use parity-odd combinations of those same tokens; hard OOD uses held-out individual tokens and is exploratory.

The fixed probe policy is `softmax(-predictive_entropy / 0.35)` over the two action branches. It is a diagnostic choice rule, not a trained policy head or a human-like preference model.

The probe never contains z. Paired histories have identical surface/action schedules, event lengths, clocks, parameters, and probe contexts. Only observed consequences differ. Unrelated delay uses matched nuisance streams. Model parameters are compared tensor-by-tensor (`torch.equal`) and by SHA-256 before/after frozen lifetimes. No optimizer is instantiated in evaluation.

Evaluation-log correction after the first two checkpoint evaluations: the initial A/B schedule was globally balanced at N32 but not guaranteed balanced at every shorter prefix. The evaluation generator was corrected to balance N2/4/8/16/32 prefixes, without changing training weights, model code, formal seeds, gate margins, or outcome rules. All affected checkpoint evaluations, including the initial completed ones, are rerun with the corrected generator before final adjudication. The original evaluation artifacts remain only as non-adjudicating development/debug logs, not as formal rows.

Telemetry-only augmentation: the canonical full path now records the intermediate context/color/shape/nuisance/action/outcome transitions, rather than only episode-end outcome ticks. This callback does not change a state transition or training objective. Earlier checkpoints are replayed under the final evaluation source so every formal artifact uses the same telemetry schema.

A separate non-gating telemetry replay for the primary full model logs all 15 window conditions at every event tick (2,295 records per seed) and stores corresponding H/F/M tensors, read vectors, queries, write and transfer magnitudes, forecast losses, and intervention flags. Its final BS is checked against the official evaluation row. Baseline models retain compact rows plus canonical per-tick trajectories; the extra window replay is not a new gate or a baseline advantage claim.

The formal runner refuses to overwrite an existing checkpoint or raw-result directory. Reproduction must use a fresh `--root`. This guard was added after the first formal jobs had started and does not alter those running jobs or their evaluation.

Metadata-only correction: row `latent_environment` and `history_condition` now explicitly identify reversal and noise rather than using the default paired label. The measured BS, gates, and state trajectories are unchanged; all formal checkpoint evaluations are replayed under this final row schema before aggregation.

The final replay archives each earlier compact row/manifest under `results/stage2c/development/pre_final_schema_rows/`, recomputes all 56 frozen-checkpoint evaluations with one source version, and asserts that every condition's BS changed by at most 1e-6. Any mismatch aborts the report rather than silently replacing a result.

## Development-only amendments (retained, not silently replaced)

Two development seeds (5101, 5102) were first trained for 300 steps with a four-symbol high-entropy alternative. Behavioral separation was approximately 0. A first amendment made the nonmatching action's three-symbol future disjoint from the matching action's deterministic outcome; another 300-step development run still showed near-zero separation. Inspection exposed a clock shortcut: training actions had alternated deterministically. A second amendment randomized and balanced actions each tick; a 300-step development run still showed near-zero separation and H norm about 180. A third amendment retained the exact recurrence but added a small H norm penalty in outer training, expanded to 500 steps, and reduced H norm. Behavioral separation remained near zero (approximately 0.000006 and 0.000009 in the two development validations). All versions are retained under `results/stage2c/development/`. No formal training seed was examined during these decisions.

## Evaluation and inference

Exposure N = 0/1/2/4/8/16/32; persistence D = 0/10/50/100/500/1000 unrelated events; novel and hard-OOD probes; useful versus matched-count noise histories; A→B and B→A reversals after 16 experiences; H/F/M/FM/full swaps at N16 and selected delays; H reset alone and combined with zero/swap; read-only F/M/FM clamps in W1–W4; early/late external-write block; gamma=0 both independently trained and same-weights posthoc. The N×D grid is 1/2/4/8/16 by 0/10/100/500/1000. Four paired lifetimes per training seed are averaged **within seed**. The eight independent training seeds, not lifetime replicates, are the statistical units. Exact recall and M necessity are not gates.

BS = P(a0 | history z0, common probe) − P(a0 | history z1, common probe). PR = BS(D)/BS(0), reported only if |BS(0)| > 1e-4; gate G43 additionally requires BS(0) ≥ 0.05. GR = BS(novel)/BS(seen), reported only with nontrivial denominator. Revision index is BS before reversal minus BS after reversal. JS divergence of paired action distributions is also saved. Peripheral intervention effects use change in BS; direct swap may reverse its sign. Norms, reads, queries, transfers, external-write magnitudes, forecast loss, and state tensors are saved in machine-readable files.

## Frozen gates

- G42: BS at N16, D0, novel probe ≥ 0.05 in at least 6/8 seeds.
- G43: BS0 ≥ 0.05 and PR500 ≥ 0.50 in at least 6/8 seeds. Development produced no robust BS0, so a meaningful fraction cannot be estimated from development; 0.50 is a conservative frozen choice, not fitted to formal data.
- G44: novel BS ≥ 0.03, seen BS ≥ 0.03, and novel/seen ≥ 0.50 in at least 6/8 seeds.
- G45: useful BS minus absolute matched-noise BS ≥ 0.03 in at least 6/8 seeds.
- G46: revision lowers BS by ≥ 0.03 at 16 new experiences and reverses its sign by 32 in at least 6/8 seeds.
- G47: the same predeclared F/M/FM swap, W1–W4 read-only clamp, or gamma=0 intervention changes BS by ≥ 0.02 in a consistent direction in at least 6/8 seeds. A posthoc search over an arbitrary intervention per seed cannot pass this gate.

Outcome A requires at least five gates, including G42/G43/G44/G46/G47. B requires G42 but not A. D is reserved for an otherwise meaningful long-delay behavioral effect inseparable from nonfinite or H>1000 instability. Otherwise C. Margins are pragmatic detection thresholds, not physiological or theoretical constants. The failed development runs make a negative formal result likely, but do not change these rules.

## Limitations fixed in advance

The world is deliberately low dimensional. The action policy uses model-predicted entropy, not optimized planning. Raw observed outcome identity, not an action-contextual KV, is passed into the unchanged external delta write. Thus a failure could reflect task/action binding, H dynamics, training horizon, or memory representation; it would not disprove behavioral memory in general. The no-memory and GRU baselines have comparable but not exact active parameter counts; counts are reported. Long-run H growth is measured separately from behavioral adjudication. Novel combination generalization is stronger than replaying identical samples, but not natural-language transfer.

The noise control matches event/episode count and compute, but not the useful stream's marginal outcome-symbol frequency (uniform four-way noise versus a balanced mixture of deterministic outcome0 and three-way alternatives). Accordingly even a positive G45 would require a stricter matched-marginal replication before attributing selectivity solely to predictive structure. This limitation is recorded rather than changing the formal generator after seeing formal outcomes.
