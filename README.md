# ET-RCM — Endogenous-Time Readout-Conserving Memory Model

> **Can a model with its own internal time transform transient experience into persistent computational state through repeated internal use, while keeping self-repetition from becoming new evidence?**
>
> **一个具有自身内部时间的模型，能否让短暂经历因为后续内部计算中的反复使用而自然转化为持久计算状态，同时避免把自己的重复思考误当成新的证据？**

ET-RCM is an independent Stage-1 research prototype for a continuously running
state model with active latent state `H`, fast memory `F`, and slow memory `M`.
External events may add content; internal use may only reallocate its lifetime.
The model can advance on a `NULL_EVENT`, but an internal tick is never treated as
a new observation.

```python
state, output = model.step(state=state, event=event)  # event may be all-zero
```

## Claim boundary

### Established prior observations

The read-only reference project `/data/CSK/J-space-project/jstate-closure`
supports narrower observations, not ET-RCM itself:

- instantaneous measured-J alone was insufficient for the tested dynamics
  (`reports/FINAL_REPORT.md` and `reports/CAUSAL_STATE_FIDELITY_V8.md`);
- predictive compact representations were not thereby proven writable causal
  states (`reports/V11_COMPLETE_REPORT.md`, especially its stated strongest
  warranted conclusion);
- variance/PCA and restricted causal directions differ materially
  (`reports/VARIANCE_VS_CAUSAL_GEOMETRY_V12.md`);
- local restricted causal spectra can look low-rank while remaining limited to
  the probed operator (`reports/LOCAL_CAUSAL_JACOBIAN_V12.md` and
  `reports/CAUSAL_TANGENT_GEOMETRY_V12.md`).

These are background constraints. They are not evidence that ET-RCM works.

### Architecture hypotheses

The following are hypotheses tested here, not established facts:

- internal use can move a trace from `F` to `M` without increasing immediate
  readout;
- useful future computation can induce longer memory lifetime without an
  explicit importance label;
- idle internal ticks can improve readiness or consolidation;
- a persistent behavioral state can remain revisable by genuine new evidence.

### Prior-work overlap

Recurrent latent computation, fast weights, test-time training,
multi-timescale/neural memory, internal replay, consolidation, and internal
time all have prior precedents. The narrow combination studied here is:

1. **external evidence creates content**;
2. **internal reuse reallocates lifetime**;
3. **reallocation conserves immediate total-memory readout**;
4. **state may evolve without a new external event**.

This project does not use a vector database, RAG, textual summaries, remember
tokens, a hand-authored importance classifier, or explicit store/retrieve APIs.

## Core transition

For `S_tau = (H_tau, F_tau, M_tau)`, the total associative memory is `A=F+M`.
An external event applies a delta-rule write only once. Internal use with unit
query `q` applies

```text
Delta = gamma (F q) q^T
F' = F - Delta
M' = M + Delta
```

so `F'+M' = F+M` to numerical precision. Fast and slow decay occur afterward.

## Reproduce Stage 1

```bash
python -m pip install -e .
pytest
python experiments/run_toy.py --toy all --config configs/toy_default.yaml
python experiments/analyze_toys.py
```

Raw episode/tick records are written under `results/raw/<run-id>/`;
processed summaries are written under `results/processed/<run-id>/` as Parquet
and JSON. The evidence-level verdict
is in `reports/TOY_VALIDATION_REPORT.md`. A small toy result is never described
as human-like memory, consciousness, infinite capacity, or causal memory.

## Current Stage-1 result (2026-09-17)

The frozen 8-seed run `stage1-20260917` produced 2,104 machine-readable records
and passed all mathematical tests. Repeated exposure, repeated internal use,
idle consolidation, revision, and the unknowable-bit negative control behaved
as intended in these small synthetic protocols. Query-dependent consolidation
outperformed the matched uniform-transfer baseline.

Stage 2 is **not authorized**. The single persistent matrix remained competitive
on the narrow delayed-retention endpoint (N1), and idle graph computation was
exactly equivalent to placing the same transitions at query time (N3). Thus the
current evidence supports an implementation proof and several mechanism checks,
but does not establish that fast/slow state or endogenous idle time is necessary.
See `reports/TOY_VALIDATION_REPORT.md` for exact curves and limitations.

## Stage 1.1 learned-dynamics result (2026-09-17)

Stage 1.1 replaced experimenter-controlled queries/dynamics with a learned
event encoder, learned `q=normalize(W_q RMSNorm(pool(H)))`, one shared gated
recurrent core, learned access strength, and learned prediction heads. B0, B1,
B2, B3, B5 and B6 were trained for three frozen formal seeds. The corrected,
fully disclosed formal run is `stage1_1-formal-v1a1`; the aborted partial v1 run
is preserved and explained in `reports/STAGE1_1_AMENDMENTS.md`.

Frozen outcomes: G8 selective persistence, G12 no-self-evidence and G13
revision passed. G7 learned-query semantics, G9 utility-over-frequency, G10
idle reasoning and G11 matched-compute interleaving failed. In particular,
B6 beat the state-byte-matched B2 at 512 distractors, and unknowable-bit
confidence did not inflate, but NULL ticks added only 0.0026 graph accuracy and
the interleaved schedule margin was 0.0000. The learned query varied with H but
failed the preregistered target-key alignment margin.

Therefore `STAGE2_LANGUAGE_MODEL_AUTHORIZED = FALSE`; no decoder LM or long
stream experiment was run. See `reports/STAGE1_1_FINAL_REPORT.md` and the
topic-specific reports for complete methods, condition tables, negative
results, and all 15 required answers.

Reproduction:

```bash
python experiments/run_stage1_1.py --mode development --device cuda
bash experiments/run_stage1_1_all.sh stage1_1-formal-v1a1 0.001
python experiments/analyze_stage1_1.py --run-id stage1_1-formal-v1a1
```

## Stage 1.2 functional validation (2026-09-18)

Stage 1.2 used eight fresh formal seeds (`3201`–`3208`), a frozen protocol,
equal per-architecture development search budgets, explicit query
interventions, and hard information-path audits. The complete run is
`stage1_2-formal-v1`.

Frozen outcomes:

- **G14 Functional addressing: FAIL.** Removing the target-key projection
  reduced accuracy by only `0.0176` and replicated in `1/8` seeds. Sign flip
  was destructive, so the query matters globally, but target-direction-specific
  functional addressing was not established. Historical G7 remains unchanged.
- **G15 Selective persistence scaling: FAIL.** ET-RCM degraded more slowly than
  the matched single-persistent baseline, but its 2048-distractor accuracy
  margin was only `0.0050`, below the frozen `0.05` threshold.
- **G16 Sequential internal computation: FAIL.** The hard one-query-per-tick
  bottleneck audit passed, but OOD `K8-K1` accuracy gain was only `0.0078` and
  was not systematic.
- **G17 No self-evidence: FAIL as a compound gate.** The cue-free unknowable
  stratum stayed at chance (`0.5005` at K=64) and confidence did not inflate;
  however, the knowable control was already near ceiling and did not achieve
  the preregistered tick benefit.
- **Endogenous-time status: SUPPORTED for consolidation timing.** Moving the
  same NULL compute before rather than after interference improved accuracy by
  `0.2275` in `8/8` seeds. This does not establish sequential cognition.

The exposure×reuse phase diagram showed both genuine exposure and value-free
reuse improving slow retention; the descriptive log-coefficient ratio was
`1.1543` (seed-bootstrap 95% CI `[1.0713, 1.2701]`) and is specific to this toy
distribution. Goal-only autonomous retrieval produced task-dependent A/B/C
query trajectories, while F/M lesions showed that slow memory was behaviorally
used and fast state could become interfering under long streams.

Neither Stage-2 path is authorized. Experiment I was `NOT_RUN_BY_PROTOCOL`, and
no decoder LM training was started. See `reports/STAGE1_2_FINAL_REPORT.md` for
the full A–I methods/results tables and all 17 required answers, and
`reports/NEGATIVE_RESULTS_STAGE1_2.md` for preserved null/negative outcomes.

Reproduction:

```bash
bash experiments/run_stage1_2_development_all.sh
bash experiments/run_stage1_2_all.sh
python experiments/analyze_stage1_2.py --run-id stage1_2-formal-v1
```

## Stage 1.3 continuous dynamics and spontaneous expression (2026-09-18)

Stage 1.3 treated output as a non-halting action in a continuously evolving
`(H,F,M)` state, separated external/NULL/SELF_OUTPUT events, and compared joint,
M-only, F-only, and learned scalar-arbitrated memory reads. The formal run
`stage1_3-formal-v1` contains 64 shards (8 architectures × 8 fresh seeds) and
1,645,312 machine-readable records. All shards use source revision `030bb6a`
and the same frozen selection hash.

Development found no threshold satisfying the registered precision/noise rule
for any architecture. B6 therefore used the disclosed diagnostic fallback
threshold `0.45`; this makes G19 ineligible to pass regardless of formal luck.
The formal outcomes were:

- **G18 Spontaneous evidence integration: FAIL.** B6 correct-emission rate was
  `0.5044`, its margin versus B0 was `-0.0752`, and sufficient-minus-
  insufficient emission was `0.4541` (`0/8` registered seed replications).
- **G19 Silence under noise: FAIL.** The 10,000-tick noise false-emission rate
  was `0.0000`, but formal precision/recall were only `0.5773/0.5000`, and the
  development threshold was not primary-feasible.
- **G20 Memory arbitration: FAIL.** At 2,048 distractors, B6 minus historical
  joint-read accuracy and useful-emission margins were both `-0.0078` (`0/8`).
- **G21 Thought-driven persistence: FAIL.** Mean usage–slow-retention Spearman
  correlation was `0.1445`; only `1/8` seeds reached the registered floor.
- **G22 Self-output is not evidence: FAIL as a compound control gate.** B6 had
  zero external writes and no positive net memory or expression-score increase,
  so the narrower no-amplification safety audit passed. B7 performed nonzero
  self-memory updates but did not reach the registered `+0.10` net-strength
  increase, so the pathological negative control itself was not validated.

Revision was also not healthy under the pre-formal audit: new-value accuracy
rose only from `0.6221` after one new event to `0.6348` after eight. Experiment
J is `NOT_RUN_BY_PROTOCOL`; `STAGE2_LANGUAGE_MODEL_PROTOTYPE_RECOMMENDED =
FALSE`, and no decoder or Stage-2 training was started. A post-formal A7 report
correction separates B6 safety from the compound B7 validation without changing
any gate or authorization result.

See `reports/STAGE1_3_FINAL_REPORT.md` for the complete A–J experimental design,
all 20 required answers, gate measurements and scientific limitations. Topic
reports, Parquet summaries, checkpoints, figures, manifests and verified SHA256
hashes are stored under `reports/`, `results/stage1_3/`, and
`artifacts/stage1_3/`.

Reproduction:

```bash
bash experiments/run_stage1_3_development_all.sh
bash experiments/run_stage1_3_all.sh
python experiments/analyze_stage1_3.py --run-id stage1_3-formal-v1
```

## Stage 2C behavioral memory / habit formation (2026-09-21)

> **Can past experience produce persistent, selective, generalizable, revisable, and causally state-mediated changes in ET-RCM's future behavior without changing its parameters?**

> **在模型参数完全不更新的情况下，过去经历能否通过持续内部状态形成持久、选择性、可泛化、可修正，并具有因果作用的未来行为改变？**

Stage 2C changes the operational definition from exact past-content recall to a
persistent causal effect of past experience on behavior. It uses a controlled
two-rule consequence-prediction world, paired identical present inputs with
different histories, frozen-parameter lifetimes, H/F/M swaps and resets,
read-only time-window clamps, write blocks, gamma-zero controls, an N×D
timescale grid, and six separately trained architectures. Eight independent
formal training seeds are the statistical units. The existing memory law and
main H recurrence are unchanged. There is no direct reward, correct-action,
remember, importance, or memory-supervision label.

**Formal result: Outcome C — state changes but useful generalized behavioral
memory was not established.** All G42–G47 gates failed (`0/8` independent
training seeds each). At 16 experiences the novel-probe behavioral separation
was `-2.89e-6` (seed SD `8.02e-6`); after 500 unrelated events it was
`-3.54e-8`. Predicted consequence distributions for the two proposed actions
were almost identical (mean total variation `2.59e-4`), suggesting an
action-conditioning failure before any long-memory claim. All 56 formal
evaluations had bit-exact frozen parameter hashes; all eight full-model seeds
crossed `||H|| > 100` during long lifetimes, though none crossed 1000 or became
nonfinite. The main recurrence was not changed to improve these results.

The formal protocol and development amendments are in
`docs/STAGE2C_PROTOCOL.md`; fixed thresholds are in
`configs/stage2c_formal.yaml`. The detailed outcome and all 24 scientific
answers are in `reports/STAGE2C_BEHAVIORAL_MEMORY_RESULTS.md`. Compact rows,
checkpoints, seed summaries, and manifests are under `results/stage2c/`.
Dense per-tick tensors and window traces are retained on the research server
under the same path but ignored by Git due to size; the tracked scripts and
checkpoints regenerate them. No Stage 1, 2A, or 2B frozen result was changed.

Reproduction (on a machine with two CUDA devices, from this directory):

```bash
PYTHONPATH=src:. .venv/bin/python experiments/stage2c_formal.py --worker 0 --root results/stage2c_reproduction
PYTHONPATH=src:. .venv/bin/python experiments/stage2c_formal.py --worker 1 --root results/stage2c_reproduction
PYTHONPATH=src:. .venv/bin/python experiments/stage2c_analyze.py --root results/stage2c_reproduction --report reports/STAGE2C_REPRODUCTION.md
PYTHONPATH=src:. .venv/bin/pytest -q
```

## Layout

- `src/etrcm/`: state, memory math, latent dynamics, baselines, and toy suite
- `tests/`: executable mathematical invariants and smoke tests
- `experiments/`: reproducible run and analysis entry points
- `docs/`: architecture, hypotheses, provenance, and Stage-2 gate plan
- `results/`: machine-readable outputs
- `reports/`: Stage-1 through Stage-1.3 protocols, amendments, topic reports,
  negative results, integrity-aware final reports, and Stage-2 go/no-go decisions
# Stage 2D noisy-evidence memory dynamics

Stage 2D asks whether uncertain experience produces a gradual, persistent but
revisable behavioral disposition and whether finite behavioral causal control
passes from F to M. It introduces no new memory law or architecture. The formal
protocol, all negative development runs, independently trained baselines,
gamma/rho sweep, NULL analysis and 10k continuous-state diagnostic are recorded
in [`docs/STAGE2D_PROTOCOL.md`](docs/STAGE2D_PROTOCOL.md) and
[`reports/STAGE2D_MEMORY_DYNAMICS_RESULTS.md`](reports/STAGE2D_MEMORY_DYNAMICS_RESULTS.md).

The Stage 2D report must be read as synthetic-world evidence only. Gradual
behavior, persistence, or NULL-time benefit is not described as human-like
memory, consciousness, unlimited capacity, or unrestricted causal memory.

## Stage 2D.1 training-basin audit (2026-09-21)

> **Why do only a minority of identically specified ET-RCM training runs enter the noisy behavioral-memory regime, and is the successful basin caused by slow-memory routing, by earlier conditional-binding dynamics, or by their interaction?**

Stage 2D.1 froze the Stage 2D architecture and memory laws, crossed eight
initializations with four data streams, audited 14 training checkpoints, and
applied parameter-frozen F/M routing interventions. The baseline produced
16/32 healthy, 1/32 partial, and 15/32 shortcut runs. Initialization and
init×stream/residual effects dominated; the pure stream main effect was small.

High M routing was associated with successful runs but was not sufficient:
the best single frozen intervention improved only 4/8 failed checkpoints.
The development-selected 50-step temporary M-gate floor produced 4/8 healthy
formal runs after the constraint was removed. G68–G72 all failed (4/8, 4/8,
3/8, 2/8, and 5/8 respectively). The outcome is **D — basin not explained**.
No F/M-law redesign or new formal handoff stage is authorized because robust
conditional binding has not yet been guaranteed.

The protocol is in `docs/STAGE2D1_PROTOCOL.md`; the full experimental details,
26 required answers, negative results, and machine-readable evidence map are in
`reports/STAGE2D1_TRAINING_BASIN_RESULTS.md`. This remains synthetic-world
diagnostic evidence, not a claim of human-like or unrestricted causal memory.

## Stage 2D.2 state-to-behavior alignment (2026-09-21)

> **Do successful ET-RCM initializations enter the noisy behavioral-memory basin because history- and memory-induced active-state changes align with downstream behaviorally useful directions?**

Stage 2D.2 froze the architecture and used finite H interventions to estimate
the protected evaluator's useful subspace across the 32 existing runs and a
new 8×3 confirmatory cohort. The response was strongly low rank, and late
history alignment was higher in healthy runs. Memory-induced alignment was not
stable across cohorts, however: frozen useful-subspace rescue passed only 1/8,
healthy destruction 5/8, and a development AUROC .766 early controllability
metric fell to .469 in confirmation.

G73–G75 failed. The preregistered stopping rule therefore prohibited alignment
warmup, so G76–G78 are `NOT_RUN_BY_PROTOCOL`. The outcome is **D — no useful
alignment explanation**. This does not justify changing the F/M law or reopening
formal handoff. See `docs/STAGE2D2_PROTOCOL.md` and
`reports/STAGE2D2_STATE_BEHAVIOR_ALIGNMENT_RESULTS.md`.

## Stage 2D.3 conditional interaction anatomy (2026-09-21)

> **Where does genuine history/state × candidate-action interaction first emerge in successful ET-RCM computations, where does it fail in shortcut runs, and can restoring that specific interaction causally recover behavioral memory?**

Stage 2D.3 keeps the architecture and F/M laws frozen and separates history
main effects, action main effects, and their exact 2×2 factorial interaction at
every protected-evaluator computation boundary. It uses the existing 32-run
factorial cohort, a new 24-run confirmatory cohort, finite H×action probes,
norm-bounded activation restoration/destruction, and—only after the causal
stopping rule permits it—a temporary paired-action observational curriculum.

The complete protocol is in `docs/STAGE2D3_PROTOCOL.md`; all gate decisions,
experimental details, negative controls, failure taxonomy and the 31 required
answers are in `reports/STAGE2D3_CONDITIONAL_INTERACTION_RESULTS.md`.

The layerwise breakpoint and frozen causal intervention replicated (G79–G82
PASS), while generic finite cross-sensitivity did not (G83 FAIL). The selected
50-step paired-action scaffold improved the formal healthy count from 2/8 to
4/8, versus 1/8 shuffled and 2/8 duplicate-compute, but missed the fixed 6/8
gate (G84 FAIL). G85–G87 are therefore `NOT_RUN_BY_PROTOCOL`. The conservative
outcome is **B\***: conditional interaction is behaviorally causal, but the
training basin is not stabilized and no preregistered A–D category is exactly
satisfied. F/M-law redesign and formal handoff remain unauthorized.

## Stage 2D.4 action-conditioned memory access (2026-09-22)

> **When a model considers different candidate actions, should it query the same persistent past differently—and does that bias stabilize conditional behavioral memory?**

Stage 2D.4 adds only an ephemeral action-to-query term during candidate
evaluation. Candidate branches remain read-only: they do not write, consolidate,
decay, advance clocks, or commit H/F/M. The persistent transition and F/M law
remain frozen. Five arms compare legacy access, action-conditioned queries,
action-conditioned gates, an exactly parameter-matched downstream capacity
control, and the prior 50-step paired warmup.

Formal healthy counts were A0 **1/8**, A1 **2/8**, A2 **2/8**, A3 **2/8**,
and A4 **3/8**. A1 learned strong action-specific query/read geometry and
passed the descriptive memory-read interaction gate G90, but query swap and
neutralization changed IHA by only about `0.002` or less and G91 failed.
Accordingly G88, G91–G93 failed; G89 and G94–G96 were not run by their
stopping rules. The outcome is **B — helps but does not stabilize**. A1 is kept
only as an experimental branch, F→M handoff remains closed, and no memory-law
redesign is justified. See `docs/STAGE2D4_PROTOCOL.md` and
`reports/STAGE2D4_ACTION_CONDITIONED_MEMORY_RESULTS.md`.

## Stage 2D.5 conditional interaction transmission (2026-09-22)

Stage 2D.5 kept Stage 2D.4 architecture and the F/M memory law frozen while
tracing 2×2 history×candidate-action interaction through raw/normalized F/M
reads, gate mixing, read→H integration, temporary H, fusion and output.
Sixteen new A1 confirmatory models yielded 2 healthy, 4 partial and 10
shortcut runs; with the original A1 cohort, only 4/24 were healthy.

The read interaction remained largely behaviorally inert. F/M mixing did not
systematically cancel it, and finite residual scaling did not rescue failed
runs. Interaction-only restoration at `fusion_post` rescued 8/8 selected
failed A1 runs; removal there destroyed 8/8 healthy legacy runs. Earlier-node
restoration rescued 0/8. Thus G100/G101 passed, but G97–G99 and G102–G104
failed: the known downstream causal node was reproduced, while no repairable
read→H or fusion transmission edge was localized. G105–G109 were not run by
protocol. The outcome is **D — upstream interaction is mostly epiphenomenal**.
No architecture rescue or F/M-law redesign was authorized, and formal F→M
handoff remains closed. See `docs/STAGE2D5_PROTOCOL.md` and
`reports/STAGE2D5_INTERACTION_TRANSMISSION_RESULTS.md` for the full methods,
controls, negative results and 34 required answers.

## Stage 2D.6 conditional interaction generation (2026-09-22)

Stage 2D.6 diagnoses the frozen `Linear → SiLU → Linear` fusion circuit at
neuron level. It distinguishes pre-existing candidate-conditioned H interaction
from interaction newly generated by SiLU, measures finite behavioral-output
alignment and curvature-weighted state/action overlap, and tests whether a
common preactivation shift alone can rescue failed runs or destroy healthy
behavior. The development cohort is existing legacy C0 checkpoints; an
independent 8-initialization × 3-stream C0 baseline is trained under the
unchanged observed-only objective. No architecture, F/M memory law or training
objective is changed in the diagnostic phase. The protocol and stopping rules
are in `docs/STAGE2D6_PROTOCOL.md`; the complete result and all 35 required
answers are in `reports/STAGE2D6_CONDITIONAL_INTERACTION_GENERATION_RESULTS.md`.

The new 24-run baseline yielded **7 healthy / 17 shortcut**, one healthy run
below the prespecified 8-run confirmation minimum. G110/G111/G113 are therefore
inconclusive, not mechanistic failures. Healthy runs had much larger fusion
state-main and post-SiLU interaction, but curvature-weighted overlap did not
outperform state-main magnitude (G112 FAIL). A development-frozen common
preactivation offset rescued **0/8** failed runs in each cohort (G114 FAIL),
and matched-control-qualified healthy destruction and temporal ordering failed
(G115/G116). F2 rescue also failed (G117). No training intervention was
authorized; G118–G122 are `NOT_RUN_BY_PROTOCOL`. The conservative outcome is
**D — no replicated nonlinear operating-regime explanation**. Fusion/F/M-law
redesign and formal F→M handoff remain unauthorized.

## Stage 2D.7 frozen-evaluator compatibility (2026-09-22)

Stage 2D.7 audited the frozen `H -> W_H H -> fusion` interface in three
independent 24-run C0 cohorts. Healthy H history separation was much larger
than F1/F2, but projection gain and SVD high/low-gain direction signatures
did not replicate. A healthy-scale state-main activation inserted *after*
the frozen state projection rescued all 46 failed models; deleting that
state-main damaged all 26 available healthy models. This is interface
causality, **not** proof that failed H already carries a sufficiently strong
compatible history representation. Frozen-upstream `W_H, b_H` refit restored
0/8 historical and 0/8 new failed models. The development-selected C1
state-projection adaptation reached only **1/8 healthy** formal seeds, equal
to C0, parameter-matched C4 and action-only C5.

G125 passed; G126 passed with a seven-healthy-per-independent-cohort sample
limit. G123/G124/G128/G129/G130 failed. H rotations and expanded/dynamics/
mediation/10k branches were stopped by their prerequisites. The current
interpretation is state formation dominating over a simple frozen-coordinate
mismatch; neither fusion redesign nor F/M-law redesign is justified, and
formal F→M handoff stays closed. Detailed protocol, run-level controls,
all 40 required answers and explicit null results are in
`docs/STAGE2D7_PROTOCOL.md` and
`reports/STAGE2D7_EVALUATOR_COMPATIBILITY_RESULTS.md`.
