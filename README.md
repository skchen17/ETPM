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

## Layout

- `src/etrcm/`: state, memory math, latent dynamics, baselines, and toy suite
- `tests/`: executable mathematical invariants and smoke tests
- `experiments/`: reproducible run and analysis entry points
- `docs/`: architecture, hypotheses, provenance, and Stage-2 gate plan
- `results/`: machine-readable outputs
- `reports/`: Stage-1, Stage-1.1 and Stage-1.2 protocols, topic reports, negative results,
  integrity-aware final report, and Stage-2 go/no-go decision
