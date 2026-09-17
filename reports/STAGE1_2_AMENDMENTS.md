# Stage 1.2 amendments

No amendments at protocol freeze. Corrections must be appended with timestamp,
affected run IDs, whether outcomes were visible, and old/new implementation
revisions. Prior outputs must remain available.

## A1 — 2026-09-18, development-budget extension before formal execution

The equal-budget 180-step development sweep completed before any formal seed
was run. B2, B3 and B6 were exactly tied at near-chance validation accuracy for
all three learning rates; all endogenous-difficulty candidates were also far
below the frozen 40–80% target band. Selecting formal hyperparameters from
these ties would be uninformative.

Every architecture therefore receives the same additive 600-step development
budget with the same candidate LRs and seeds. Selection is maximum mean
validation accuracy, then lower mean loss for exact accuracy ties, then lower
LR. The first sweep is preserved under `stage1_2-development-v1`. The extended
sweep uses `stage1_2-development-v1a1`. No formal outcomes were visible; gates,
formal seeds, formal budgets and model laws are unchanged. Exact settings are
in `configs/stage1_2_development_amendment1.yaml`.
