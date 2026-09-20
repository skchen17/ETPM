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

## Stage 1.4–1.5 causal-state and routing audits

Stage 1.4's predictive-state study and Stage 1.5's anatomy/routing study are
frozen historical results. Stage 1.5 found finite F/M causal effects and
explicit-read mediation (G29/G30 PASS), but long-NULL stability, tested
timescale advantage, and useful historical-oracle read did not meet their gates
(G27/G28/G31 FAIL). G32/G33 were not run by protocol. These results do not
establish useful autonomous retrieval. See
`reports/STAGE1_4_FINAL_REPORT.md` and `reports/STAGE1_5_FINAL_REPORT.md`;
Stage 1.5 assets are protected by `artifacts/stage1_5_all_assets.sha256`.

## Stage 1.6 training versus integration architecture

> **Is ET-RCM failing to use persistent memory because its current integration architecture is incapable of doing so, or because the training process never forces the recurrent core to learn memory-dependent computation?**

> **ET-RCM 当前无法有效利用持久记忆，究竟是因为现有 memory-to-H integration 架构本身做不到，还是因为训练过程从未真正迫使 recurrent core 学会依赖 memory 进行计算？**

Stage 1.6 preserves the Stage 1.5 gated-residual integration operator and
memory law. It compares training length, learned versus historical-only oracle
read delivery, an oracle-to-learned curriculum, H-scrub memory necessity, and
no-memory/GRU/single-memory baselines. A separate exploratory extension scales
the *original* Stage 1.5 four-family objective; it is not one of the four new
gates. Formal thresholds were frozen in `reports/STAGE1_6_PROTOCOL.md` before
the eight fresh training seeds. Formal results: G34 FAIL (2/8 replicated
training-length memory-benefit gains), G35 PASS (8/8 oracle-read advantages
over zero/random/shuffled), G36 FAIL (ordinary H-scrub full-vs-no-memory and
M-lesion effects each replicated in only 2/8), G37 FAIL (only 3/8 valid
within-curriculum oracle-benefit denominators). A triggered, separately
labeled three-phase curriculum recovered learned-read benefit in 6/8 seeds but
did not establish slow-M necessity. The original-objective extension lowered
held-out CE substantially without a replicated historical-oracle advantage.
Thus the current integration operator is capable of using supplied history on
this toy; training/routing and persistent-M use remain unresolved. Full methods,
seed-level effects, null results and limitations are in
`reports/STAGE1_6_FINAL_REPORT.md`. Sequence/LM training remains unauthorized.

Reproduction from the server project root after `pip install -e .`:

```bash
PYTHONPATH=src python experiments/run_stage1_6_grid.py --phase development
PYTHONPATH=src python experiments/select_stage1_6.py
PYTHONPATH=src python experiments/run_stage1_6_grid.py --phase formal --learning-rate 0.001
PYTHONPATH=src python experiments/run_stage1_6_legacy_grid.py --cpu-workers 8
PYTHONPATH=src python experiments/analyze_stage1_6_legacy.py
PYTHONPATH=src python experiments/diagnose_stage1_6.py
PYTHONPATH=src python experiments/analyze_stage1_6.py
PYTHONPATH=src python experiments/verify_stage1_6.py
```

Completed cells are skipped by the grid launchers; an independent rerun needs
an isolated checkout and a new run identifier, without overwriting frozen data.

## Layout

- `src/etrcm/`: state, memory math, latent dynamics, baselines, and toy suite
- `tests/`: executable mathematical invariants and smoke tests
- `experiments/`: reproducible run and analysis entry points
- `docs/`: architecture, hypotheses, provenance, and Stage-2 gate plan
- `results/`: machine-readable outputs
- `reports/`: Stage-1 through Stage-1.3 protocols, amendments, topic reports,
  negative results, integrity-aware final reports, and Stage-2 go/no-go decisions
