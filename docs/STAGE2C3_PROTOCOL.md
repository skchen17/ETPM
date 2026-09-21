# Stage 2C.3 — Counterfactual Action Training and Behavioral-Memory Chain Recovery

## Scientific question

Does endogenous behavioral memory fail because the model is allowed to minimize future-prediction loss by learning the marginal outcome distribution, rather than learning the history-conditioned consequences of alternative actions?

This stage uses the unchanged Stage 2C.1 `LifetimeModel`, Stage 2C world, ET-RCM external delta write, F/M query/read, readout-conserving transfer, decay, NULL and SELF_OUTPUT semantics. No new memory module, recurrence, policy-label objective, reward classifier, or H/F/M supervision is introduced. Earlier frozen Stage 2C, 2C.1 and 2C.2 files are read-only; their SHA-256 inventory is captured before new experiments.

## World and shortcut controls

Latent `z ∈ {A,B}` is balanced and never supplied to lifetime model inputs. The simulator emits a sampled consequence for the executed action. The matching action has outcome 0; the other action has equiprobable outcomes 1/2/3. All training arms use the same surfaces, actions and sampled outcome stream for each matched seed.

The theoretical no-state/no-action marginal is `[1/2,1/6,1/6,1/6]`, with `CE_marginal = 1.242453324` nats. Its entropy equals this CE. An exact oracle `p(y|z,a)` has average CE `0.5 ln 3 = 0.549306144` nats. Separate fitted marginal and action-only frequency predictors use strictly limited inputs; a post-hoc history-only predictor reads frozen H but no candidate action. To avoid a weak overfit control, history-only is trained against the **exact action-marginal future distribution** at every training H; its exact held-out risk and independently sampled observed CE are both retained. Fitted control results are evaluated on disjoint future observations or exact held-out world risk, as indicated in machine-readable records.

## Arms and compute

`A0`: observed-consequence CE only, same L3 training semantics as Stage 2C.2 scratch; here all arms use the same newly generated matched seed/data schedule. `A1`: `COUNTERFACTUAL_TRAINING`, soft consequence CE over both candidate actions from *the same pre-action context state*. Latent z supplies target distributions only to the outer simulator/loss, never to model inputs. This privileged target is not natural online learning. `A2`: a separately trained oracle-z evaluator head is transferred into the unchanged L3 model; only the downstream action embedding and consequence head transfer. They are then bit-exact frozen during endogenous observed-CE training. The oracle embedding itself never transfers. `A3`: same pretrain and initial frozen phase as A2; after step 500 the evaluator is unfrozen at one tenth upstream LR. No evaluator receives a correct-action, reward, habit or policy-label loss.

Training uses batch 16, lengths 4/8/16, AdamW LR 0.001, H-square penalty 0.001, clip norm 1, 1,000 endogenous steps; evaluator pretraining adds 1,000 steps for A2/A3. Checkpoints at 0/25/50/100/200/300/500/1000. Training steps, initial L3 parameters and simulator data are matched by seed; A1 has extra action-branch computation and A2/A3 have extra evaluator-pretraining computation, so wall-clock/FLOP equality is not claimed. Gradient norms are pre-clipping diagnostics only.

Execution note (no protocol change): remote GPU contention led to a resume-safe switch to CPU workers after five early formal arm-seeds. Initial weights, data streams, objectives and step budgets remain matched, but compute backend is not perfectly paired for A2/A3 seed 7702. Exact per-arm/seed backends are written to `results/stage2c3/manifests/training_backends.json` and treated as a limitation rather than hidden.

Development seeds are 7601–7602. Formal seeds are 7701–7708, separately trained in each primary arm. The seed, arm, phase, parameter hashes, training CE, paired-history forecasts, CFA, state/read/query norms, write/transfer magnitude and gradient groups are saved at every checkpoint. All evaluation is optimizer-free with parameter hashes checked before and after. Development findings cannot be counted as formal replicates.

Gate G57/G58 endpoints use a **separate four-replicate frozen N16 evaluation** per trained seed. The two-replicate checkpoint snapshots are learning-curve diagnostics and never substituted for formal gate observations. Exact conditional CE is computed over both candidate action distributions with balanced z×action cells; a separate sampled-outcome CE audit guards against relying only on analytic targets.

An independent checkpoint audit reloads every saved model, generates disjoint N16 histories on novel surfaces, and computes both exact paired-counterfactual CE and sampled held-out observed CE (256 fresh outcome draws per z×action cell per replicate). This audit does not update parameters and reports its own before/after hashes.

For A2/A3, a separate frozen-checkpoint oracle-H sweep fits two native-median-norm H vectors at every checkpoint, with only the vectors optimized. This tests whether the protected evaluator remains state-controllable across training, not whether native H reaches those directions. Per-checkpoint before/after parameter hashes and head hashes are retained; A2's head must remain bit-exact throughout, and A3's through step 500.

## Formal gates and downstream tests

The numeric thresholds and statuses are frozen in `configs/stage2c3_formal.yaml` after development and before formal training. G57 requires a single anti-shortcut arm to pass action-TV, positive interaction and conditional-CE advantage simultaneously in at least 6/8 formal training seeds. G58 additionally requires N16 entropy-policy separation from frozen parameters. G59 requires exposure dependence, G60 a reproducible F/M/FM swap or read-clamp effect, and G61 requires A2 or A3 to retain action-TV, interaction and BS with *zero oracle input during lifetimes*. A downstream gate is `NOT_RUN_BY_GATE` when its prerequisite phenotype is absent; it is not silently called FAIL.

Each final checkpoint gets N=0/1/2/4/8/16/32 paired histories; seen/novel/hard-OOD surfaces; A→B revision; and H/F/M/FM probes. N16 `|BS|` must first clear the frozen threshold before persistence ratio or peripheral intervention is interpreted. Persistence delays are 0/10/50/100/500/1000 unrelated events. Eligible interventions include H/F/M/FM/HFM swaps, F/M/FM read clamps in the final probe, and F/M/FM read clamps throughout legal-history formation with a native final probe; M alone is not required. A2/A3 receive a frozen-head, native-norm oracle-H ceiling audit. Probes indicate decodability only, never causal behavioral mediation.

Development-stage amendment, frozen before formal training: final-probe F/M null effects can be masked by information already present in H. Formation-window read clamps are therefore listed separately and may satisfy G60 if they yield the *same* directional BS reduction in at least 6/8 formal seeds.

`POST_START_EXPLORATORY`: after the first formal A2 training completed, a stronger formation-window state-zero control was added. It removes F, M, or FM after every legal-history transition while leaving H and model parameters unchanged. It is preserved for diagnosis but **is not eligible for G60 or any formal gate**. Neither kind of formation intervention modifies model weights or F/M laws, and neither alone identifies the exact microscopic site because earlier interventions change later H/query trajectories.

Because one observation with outcome 0 versus 1/2/3 can identify z in this world, a successful model may saturate at N=1. G59 therefore compares the **mean absolute BS** of N16/N32 with the **mean absolute BS** of N0/N1, not the maximum early point. If the effect is already strong at N=1, call it exposure dependence or one-shot acquisition, not a graded multi-exposure improvement; report the full curve without hiding saturation.

Do not redesign F/M write or consolidation unless conditional evaluator capacity survives, oracle H can control it, endogenous behavior still fails, and probes plus finite interventions show missing useful latent information. No full Stage 2C suite or natural-language claim follows from a single development seed.
