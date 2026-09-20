# Stage 2A — minimal natural-language prototype

This is an exploratory successor to Stage 1.6. Historical Stage 1.x reports,
protocols, results, and the frozen README are not re-adjudicated or edited.

The authoritative result is
`reports/STAGE2A_NATURAL_LANGUAGE_PROTOTYPE_RESULTS.md`. It contains the
actual generations, answer tests, interventions, NULL sweeps, stability traces,
and negative findings. Machine-readable records are under
`results/stage2a/{raw,processed}/`.

The interface lives in `src/etrcm/stage2a/`. It inherits the existing
external delta write, readout-conserving F→M transfer, independent learned
reads, and gated-residual H update. Its lexical event mapping is deliberately
minimal: external token id is both key and value id; generated tokens have
`SELF_OUTPUT` kind and are never external evidence.

Reproduction (from repository root):

```bash
PYTHONPATH=src .venv/bin/python experiments/stage2a_train.py \
  --output results/stage2a/raw/stage2a_seed42 \
  --steps-per-phase 60 --batch-size 6 --max-length 56 --hidden-dim 64 --seed 42

PYTHONPATH=src .venv/bin/python experiments/stage2a_train.py \
  --output results/stage2a/raw/stage2a_medium_seed42 \
  --steps-per-phase 60 --batch-size 6 --max-length 56 --hidden-dim 128 --seed 42

PYTHONPATH=src .venv/bin/python experiments/stage2a_eval.py \
  --root results/stage2a/raw/stage2a_seed42 \
  --report reports/STAGE2A_NATURAL_LANGUAGE_PROTOTYPE_RESULTS.md \
  --evaluation-n 12

PYTHONPATH=src .venv/bin/python experiments/stage2a_spontaneous_refresh.py \
  --root results/stage2a/raw/stage2a_seed42 \
  --processed results/stage2a/processed/stage2a_seed42 \
  --report reports/STAGE2A_NATURAL_LANGUAGE_PROTOTYPE_RESULTS.md

PYTHONPATH=src .venv/bin/python experiments/stage2a_finalize.py \
  --small results/stage2a/raw/stage2a_seed42 \
  --medium results/stage2a/raw/stage2a_medium_seed42 \
  --processed results/stage2a/processed/stage2a_seed42 \
  --report reports/STAGE2A_NATURAL_LANGUAGE_PROTOTYPE_RESULTS.md

PYTHONPATH=src .venv/bin/python experiments/stage2a_conversation_diagnostic.py \
  --root results/stage2a/raw/stage2a_seed42 \
  --processed results/stage2a/processed/stage2a_seed42 \
  --report reports/STAGE2A_NATURAL_LANGUAGE_PROTOTYPE_RESULTS.md

.venv/bin/python -m pytest -q
sha256sum -c artifacts/stage1_5_all_assets.sha256
```

The two training commands use independent GPUs in the recorded run; set
`--device` explicitly where needed. The evaluation is costly because 1024-token
gaps are replayed under six interventions plus two baselines.
