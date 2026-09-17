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

Raw episode/tick records are written to `results/raw/toy_records.parquet`;
processed summaries are written as Parquet and JSON. The evidence-level verdict
is in `reports/TOY_VALIDATION_REPORT.md`. A small toy result is never described
as human-like memory, consciousness, infinite capacity, or causal memory.

## Layout

- `src/etrcm/`: state, memory math, latent dynamics, baselines, and toy suite
- `tests/`: executable mathematical invariants and smoke tests
- `experiments/`: reproducible run and analysis entry points
- `docs/`: architecture, hypotheses, provenance, and Stage-2 gate plan
- `results/`: machine-readable outputs
- `reports/`: human-readable evidence report

