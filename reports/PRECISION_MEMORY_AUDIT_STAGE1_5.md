# Small-Update Precision Audit — Stage 1.5

Formal run `stage1_5-formal-v1`. Protocol: `reports/STAGE1_5_PROTOCOL.md`; analysis plan: `reports/STAGE1_5_ANALYSIS_PLAN.md`. Raw Parquet: `results/stage1_5/stage1_5-formal-v1/evaluation/`.

## Methods

Exploratory scalar rank-one-direction reduction: 1000 repeated consolidations with γ=1e-5 and no decay/external events. FP32 storage, BF16 storage, and FP32 shadow accumulation with BF16 consumption are compared to `(1−γ)^K`. This does not enter gates.

## Results

At update 1000:

| Storage/consumption | F | M | F+M | Analytic F error |
| --- | --- | --- | --- | --- |
| FP32 | 0.990051 | 0.009950 | 1.000001 | 0.000001 |
| BF16 | 1.000000 | 0.003906 | 1.000000 | 0.009950 |
| FP32_shadow_BF16_consumption | 0.990051 | 0.009950 | 1.000001 | 0.000001 |

## Scope and limitations

This is a scalar precision regression, not a BF16 full-model benchmark. FP32 remains the formal scientific condition.
