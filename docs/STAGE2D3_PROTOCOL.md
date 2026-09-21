# Stage 2D.3 — Conditional Interaction Anatomy and Minimal Binding Rescue

> **Where does genuine history/state × candidate-action interaction first emerge in successful ET-RCM computations, where does it fail in shortcut runs, and can restoring that specific interaction causally recover behavioral memory?**

> **成功 ET-RCM 中，“历史形成的内部状态 × 当前候选行为”这一条件计算究竟在哪一层首次形成？失败模型在哪一层断掉？如果只恢复这个具体 interaction，能否因果地恢复 behavioral memory？**

## Frozen scope

The Stage 2D architecture, protected late-concat evaluator, external write,
readout-conserving F→M transfer, decay, NULL and SELF_OUTPUT semantics are
unchanged. The formal labels remain the endpoint Stage 2D labels and never use
an intermediate activation metric.

The independent-run cohort is:

- Cohort A: the 32 existing Stage 2D.1 factorial baseline runs.
- Cohort B: 24 newly trained C0 runs (8 new initializations × 3 new streams).

## Factorial estimand

For every hooked representation `z`, the four matched cells are decomposed as

`I = z(H_A,a_A)-z(H_A,a_B)-z(H_B,a_A)+z(H_B,a_B)`.

State and action main effects are measured separately. Replicate histories,
directions and layers are within-run measurements; all formal comparisons are
first aggregated to one value per independently trained run.

The computation order is: pre-action H, H entering the evaluator, pooled H,
action embedding, fusion input, fusion preactivation, fusion postactivation,
pre-logit representation, logits, probabilities and entropy-policy score.

## Causal registration

The first nonlinear evaluator fusion representation (`fusion_post`) is fixed
before confirmatory and causal evaluation. Cross-model templates are never
raw-hidden swaps: healthy logit interactions are mapped into a failed model's
post-fusion coordinates through that failed model's protected output matrix.
Intervention cell norms are bounded by the matched healthy interaction scale.

Controls are history-only, action-only, norm-matched random factorial and
shuffled factorial signs. Healthy necessity removes or reverses the native
post-fusion factorial component while leaving the algebraic state/action main
effects unchanged.

## Stopping rule

Temporary paired-action training is permitted only if G81 or G82 passes. It
uses two legally simulated observed-action consequences from the same state;
z, correct-action labels, memory labels and oracle states never enter the
model. The auxiliary loss is exactly zero after the temporary window and in
all formal evaluation. If neither causal gate passes, G84–G87 and all dependent
dynamics/continuous experiments are recorded as `NOT_RUN_BY_PROTOCOL`.
