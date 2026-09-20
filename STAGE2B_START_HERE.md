# Stage 2B — contextual associative memory

The Stage 2B predeclared protocol is `configs/stage2b.yaml`. E0 is the
unchanged Stage 2A token-identity implementation. E1/E2 differ only in the
language-to-memory write representation and the necessary read-before-write
ordering. E0_late is available as an ordering-control diagnostic, not a formal
substitute for E0.

The authoritative report is
`reports/STAGE2B_CONTEXTUAL_ASSOCIATIVE_MEMORY_RESULTS.md`. Frozen Stage 1.x
and Stage 2A files remain untouched, tested against the Stage 1.5 1088-asset
manifest and Stage 2A commit `d458abd`.

Reproduce a formal arm (repeat for seeds 2401–2405 and arms GRU/RNN/E0/E1/E2):

```bash
PYTHONPATH=src .venv/bin/python experiments/stage2b_train.py \
  --arm E1 --seed 2401 --steps 1000 --batch-size 16 --hidden-dim 64 \
  --out results/stage2b/formal/small/seed2401/E1 --device cuda:0

PYTHONPATH=src .venv/bin/python experiments/stage2b_eval.py \
  --checkpoint results/stage2b/formal/small/seed2401/E1/checkpoint.pt \
  --out results/stage2b/processed/small/seed2401/E1 \
  --evaluation-n 8 --extended --device cpu

PYTHONPATH=src .venv/bin/python experiments/stage2b_report.py \
  --root results/stage2b \
  --report reports/STAGE2B_CONTEXTUAL_ASSOCIATIVE_MEMORY_RESULTS.md

.venv/bin/python -m pytest -q
sha256sum -c artifacts/stage1_5_all_assets.sha256
```

Medium uses hidden dimension 128 and seed 2501; `E0_late` is an optional
ordering diagnostic stored separately under `results/stage2b/ordering_control/`.
GRU-76 is an optional active-parameter-matching baseline under
`results/stage2b/active_parameter_control/`.
The post-training residual-scale diagnostic is separate from formal gates.
