# ET-RCM Stage 2C.3 — Counterfactual Action Training Results

> **Does endogenous behavioral memory fail because the model is allowed to minimize future-prediction loss by learning the marginal outcome distribution, rather than learning the history-conditioned consequences of alternative actions?**

> **endogenous behavioral memory 的失败，是否主要因为训练目标允许模型通过学习总体结果分布来降低预测损失，从而绕过‘历史状态 × 候选行为 → 不同未来结果’这一真正需要的条件计算？**

Formal classification: **A — shortcut corrected with replicated behavioral, exposure and peripheral evidence**. Gates: G57=PASS, G58=PASS, G59=PASS, G60=PASS, G61=PASS.

## 1. Frozen protocol, data and controls

Development seeds [7601, 7602]; formal matched seeds [7701, 7702, 7703, 7704, 7705, 7706, 7707, 7708], four arms each, 1,000 endogenous steps plus 1,000 privileged evaluator-pretrain steps for A2/A3. Architecture and F/M law are unchanged. Stage 2C–2C.2 prior hashes: 1072 files unchanged. Training checkpoint steps: [0, 25, 50, 100, 200, 300, 500, 1000]. No policy, correct-action, reward, habit or memory label loss. A1 uses `COUNTERFACTUAL_TRAINING` targets from the simulator; A2/A3 have oracle-z evaluator pretraining but **zero oracle input** in lifetime training/evaluation. A1 branch compute and A2/A3 extra pretraining mean FLOPs are not equal despite matched endogenous steps and per-seed data. Gate endpoints use a separate four-replicate frozen N16 evaluation; two-replicate checkpoint metrics are descriptive learning curves, not substituted as gate observations. Initial weights and simulator streams are seed-matched. Due remote GPU contention and a resume-safe acceleration, five early arm-seeds used CUDA and the remainder CPU; exact backends are in `manifests/training_backends.json`. This creates a disclosed backend difference (especially A2/A3 seed 7702) even though architectures, data and step budgets match.

Theoretical `p_marginal=[1/2,1/6,1/6,1/6]`, `CE_marginal=1.242453` nats, exact conditional oracle CE=0.549306. Formal empirical controls are fitted from restricted inputs: p(y), p(y|a), and frozen-state p(y|H) without candidate action.

| Arm | fitted marginal CE | action-only CE | history-only held-out CE | final conditional CE | final CFA |
|---|---:|---:|---:|---:|---:|
| A0 | 1.2426 ± 0.0001 | 1.2426 ± 0.0001 | 1.2425 ± 0.0000 | 1.1638 ± 0.2241 | 0.0786 ± 0.2241 |
| A1 | 1.2426 ± 0.0001 | 1.2426 ± 0.0001 | 1.2426 ± 0.0000 | 0.5497 ± 0.0004 | 0.6927 ± 0.0004 |
| A2 | 1.2426 ± 0.0001 | 1.2426 ± 0.0001 | 1.2427 ± 0.0001 | 0.5503 ± 0.0004 | 0.6921 ± 0.0004 |
| A3 | 1.2426 ± 0.0001 | 1.2426 ± 0.0001 | 1.2426 ± 0.0001 | 0.5512 ± 0.0020 | 0.6913 ± 0.0020 |

Restricted-control calibration and entropy (seed means):

| Arm | marginal TV calibration | marginal entropy | action-only entropy | history-only TV calibration |
|---|---:|---:|---:|---:|
| A0 | 0.00496 | 1.24268 | 1.24266 | 0.00044 |
| A1 | 0.00496 | 1.24268 | 1.24266 | 0.00094 |
| A2 | 0.00496 | 1.24268 | 1.24266 | 0.00128 |
| A3 | 0.00496 | 1.24268 | 1.24266 | 0.00110 |

Paired conditional-CE advantages over restricted controls (positive favors state+action; exact one-sided sign test is descriptive, not a gate):

| Arm | vs marginal ΔCE (wins/8, sign p) | vs action-only | vs history-only |
|---|---|---|---|
| A0 | 0.0787 ± 0.2241 (4/8, p=0.6367) | 0.0788 ± 0.2241 (4/8, p=0.6367) | 0.0787 ± 0.2241 (4/8, p=0.6367) |
| A1 | 0.6928 ± 0.0003 (8/8, p=0.0039) | 0.6929 ± 0.0004 (8/8, p=0.0039) | 0.6929 ± 0.0004 (8/8, p=0.0039) |
| A2 | 0.6922 ± 0.0004 (8/8, p=0.0039) | 0.6923 ± 0.0004 (8/8, p=0.0039) | 0.6923 ± 0.0005 (8/8, p=0.0039) |
| A3 | 0.6914 ± 0.0020 (8/8, p=0.0039) | 0.6914 ± 0.0020 (8/8, p=0.0039) | 0.6914 ± 0.0021 (8/8, p=0.0039) |

Development-only checkpoint-1000 observations used to freeze the numerical thresholds before formal seeds; none count toward G57–G61:

| Arm | development seed | TV_A | interaction | BS | CFA |
|---|---:|---:|---:|---:|---:|
| A0 | 7601 | 0.0055 | +0.0012 | +0.0007 | -0.0012 |
| A0 | 7602 | 0.0119 | +0.0002 | +0.0001 | -0.0002 |
| A1 | 7601 | 0.9997 | +1.9994 | +0.9170 | +0.6930 |
| A1 | 7602 | 0.8337 | +1.3231 | +0.5592 | +0.4121 |
| A2 | 7601 | 0.9995 | +1.9990 | +0.9172 | +0.6925 |
| A2 | 7602 | 0.9996 | +1.9992 | +0.9167 | +0.6923 |
| A3 | 7601 | 0.9944 | +1.9888 | +0.9166 | +0.6902 |
| A3 | 7602 | 0.9998 | +1.9995 | +0.9167 | +0.6922 |

## 2. Final seed-matched arm comparison

| Arm | action-TV | history-TV | interaction y0 | entropy BS | CE advantage | G57 seeds | G58 seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.1181 ± 0.3115 | 0.1160 ± 0.3124 | 0.2227 ± 0.6284 | 0.1113 ± 0.3137 | 0.0786 ± 0.2241 | 1/8 | 1/8 |
| A1 | 0.9993 ± 0.0006 | 0.9993 ± 0.0006 | 1.9987 ± 0.0013 | 0.9169 ± 0.0001 | 0.6927 ± 0.0004 | 8/8 | 8/8 |
| A2 | 0.9989 ± 0.0008 | 0.9989 ± 0.0008 | 1.9978 ± 0.0016 | 0.9167 ± 0.0008 | 0.6921 ± 0.0004 | 8/8 | 8/8 |
| A3 | 0.9974 ± 0.0040 | 0.9974 ± 0.0040 | 1.9947 ± 0.0081 | 0.9155 ± 0.0016 | 0.6913 ± 0.0020 | 8/8 | 8/8 |

Formal unit is the training seed, not a checkpoint or replicate. Both TV and interaction must coexist with positive conditional CE advantage within the *same* seed. A2/A3 evaluator-head protection is verified by hashes; a high TV alone is not endogenous memory.

Per-seed frozen endpoints (the rare A0 success is retained):

| Arm | Seed | TV_A | interaction | BS N16 | CFA | G57 | G58 |
|---|---:|---:|---:|---:|---:|---|---|
| A0 | 7701 | 0.0084 | +0.0028 | +0.0020 | +0.0005 | False | False |
| A0 | 7702 | 0.0145 | -0.0008 | -0.0006 | -0.0026 | False | False |
| A0 | 7703 | 0.8890 | +1.7780 | +0.8876 | +0.6333 | True | True |
| A0 | 7704 | 0.0124 | -0.0001 | +0.0002 | -0.0014 | False | False |
| A0 | 7705 | 0.0053 | -0.0008 | -0.0006 | -0.0007 | False | False |
| A0 | 7706 | 0.0046 | +0.0015 | +0.0013 | +0.0005 | False | False |
| A0 | 7707 | 0.0044 | +0.0006 | +0.0005 | -0.0006 | False | False |
| A0 | 7708 | 0.0058 | +0.0006 | +0.0004 | +0.0001 | False | False |
| A1 | 7701 | 0.9987 | +1.9974 | +0.9167 | +0.6923 | True | True |
| A1 | 7702 | 0.9996 | +1.9992 | +0.9170 | +0.6929 | True | True |
| A1 | 7703 | 0.9997 | +1.9994 | +0.9168 | +0.6930 | True | True |
| A1 | 7704 | 0.9996 | +1.9992 | +0.9168 | +0.6929 | True | True |
| A1 | 7705 | 0.9998 | +1.9996 | +0.9169 | +0.6930 | True | True |
| A1 | 7706 | 0.9980 | +1.9961 | +0.9171 | +0.6920 | True | True |
| A1 | 7707 | 0.9996 | +1.9992 | +0.9167 | +0.6929 | True | True |
| A1 | 7708 | 0.9997 | +1.9994 | +0.9169 | +0.6929 | True | True |
| A2 | 7701 | 0.9994 | +1.9988 | +0.9169 | +0.6923 | True | True |
| A2 | 7702 | 0.9992 | +1.9983 | +0.9174 | +0.6923 | True | True |
| A2 | 7703 | 0.9988 | +1.9975 | +0.9169 | +0.6923 | True | True |
| A2 | 7704 | 0.9978 | +1.9957 | +0.9164 | +0.6918 | True | True |
| A2 | 7705 | 0.9993 | +1.9986 | +0.9154 | +0.6917 | True | True |
| A2 | 7706 | 0.9976 | +1.9952 | +0.9182 | +0.6914 | True | True |
| A2 | 7707 | 0.9995 | +1.9991 | +0.9163 | +0.6924 | True | True |
| A2 | 7708 | 0.9996 | +1.9993 | +0.9163 | +0.6928 | True | True |
| A3 | 7701 | 0.9996 | +1.9992 | +0.9167 | +0.6923 | True | True |
| A3 | 7702 | 0.9987 | +1.9975 | +0.9179 | +0.6919 | True | True |
| A3 | 7703 | 0.9966 | +1.9932 | +0.9146 | +0.6913 | True | True |
| A3 | 7704 | 0.9996 | +1.9993 | +0.9166 | +0.6926 | True | True |
| A3 | 7705 | 0.9984 | +1.9967 | +0.9140 | +0.6915 | True | True |
| A3 | 7706 | 0.9877 | +1.9753 | +0.9134 | +0.6864 | True | True |
| A3 | 7707 | 0.9985 | +1.9971 | +0.9148 | +0.6913 | True | True |
| A3 | 7708 | 0.9997 | +1.9995 | +0.9165 | +0.6928 | True | True |

Per-arm replication counts:

| Arm | G57 | G58 | G59 | best G60 intervention |
|---|---:|---:|---:|---|
| A0 | 1/8 | 1/8 | 1/8 | formation_clamp_M 1/8 |
| A1 | 8/8 | 8/8 | 8/8 | formation_clamp_FM 6/8 |
| A2 | 8/8 | 8/8 | 8/8 | formation_clamp_FM 8/8 |
| A3 | 8/8 | 8/8 | 8/8 | formation_clamp_FM 7/8 |

## 3. Learning-curve and gradient audit

| Arm | Step | train CE | held-out observed CE | CF exact CE | CFA | TV_A | TV_H | interaction | BS | H norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0 | nan | 1.3902 | 1.3901 | -0.1468 | 0.0422 | 0.0027 | +0.0000 | -0.0001 | 80.52 |
| A0 | 25 | 1.2500 | 1.2453 | 1.2454 | -0.0031 | 0.0200 | 0.0066 | -0.0000 | -0.0000 | 38.83 |
| A0 | 50 | 1.2451 | 1.2446 | 1.2448 | -0.0026 | 0.0129 | 0.0041 | -0.0003 | -0.0003 | 32.07 |
| A0 | 100 | 1.2578 | 1.2481 | 1.2482 | -0.0061 | 0.0141 | 0.0066 | +0.0003 | +0.0003 | 27.99 |
| A0 | 200 | 1.2608 | 1.2458 | 1.2460 | -0.0037 | 0.0150 | 0.0048 | +0.0001 | +0.0000 | 20.20 |
| A0 | 300 | 1.2256 | 1.2438 | 1.2435 | -0.0013 | 0.0092 | 0.0042 | +0.0000 | +0.0000 | 17.53 |
| A0 | 500 | 1.2315 | 1.2454 | 1.2456 | -0.0034 | 0.0130 | 0.0091 | -0.0002 | -0.0001 | 13.78 |
| A0 | 1000 | 1.1948 | 1.1637 | 1.1635 | +0.0788 | 0.1180 | 0.1150 | +0.2231 | +0.1116 | 12.55 |
| A1 | 0 | nan | 1.3902 | 1.3901 | -0.1468 | 0.0422 | 0.0027 | +0.0000 | -0.0001 | 80.52 |
| A1 | 25 | 1.2446 | 1.2442 | 1.2442 | -0.0022 | 0.0135 | 0.0066 | -0.0001 | -0.0001 | 31.11 |
| A1 | 50 | 1.2433 | 1.2436 | 1.2437 | -0.0005 | 0.0092 | 0.0085 | +0.0011 | +0.0008 | 20.23 |
| A1 | 100 | 1.2424 | 1.2436 | 1.2435 | -0.0011 | 0.0079 | 0.0094 | -0.0005 | -0.0004 | 11.91 |
| A1 | 200 | 1.1653 | 1.1402 | 1.1403 | +0.1061 | 0.1422 | 0.1401 | +0.2702 | +0.1774 | 12.89 |
| A1 | 300 | 0.9385 | 0.9015 | 0.9016 | +0.3447 | 0.5015 | 0.4942 | +0.9784 | +0.4684 | 14.48 |
| A1 | 500 | 0.7128 | 0.6380 | 0.6381 | +0.6048 | 0.8746 | 0.8722 | +1.7432 | +0.8017 | 7.05 |
| A1 | 1000 | 0.5985 | 0.5497 | 0.5497 | +0.6928 | 0.9994 | 0.9994 | +1.9987 | +0.9169 | 3.81 |
| A2 | 0 | nan | 1.8721 | 1.8730 | -0.6150 | 0.5599 | 0.0133 | +0.0109 | +0.0094 | 80.52 |
| A2 | 25 | 1.2578 | 1.2641 | 1.2640 | +0.0082 | 0.0656 | 0.0214 | +0.0294 | +0.0198 | 77.66 |
| A2 | 50 | 1.2395 | 1.2333 | 1.2332 | +0.0162 | 0.0896 | 0.0411 | +0.0537 | +0.0374 | 54.97 |
| A2 | 100 | 1.1908 | 1.1795 | 1.1795 | +0.0528 | 0.2304 | 0.1489 | +0.2229 | +0.0971 | 42.85 |
| A2 | 200 | 1.0373 | 1.0075 | 1.0075 | +0.2080 | 0.4232 | 0.3344 | +0.6459 | +0.3116 | 35.86 |
| A2 | 300 | 0.7852 | 0.8713 | 0.8709 | +0.5125 | 0.7647 | 0.7487 | +1.4844 | +0.6801 | 26.77 |
| A2 | 500 | 0.7107 | 0.6385 | 0.6385 | +0.6039 | 0.8788 | 0.8745 | +1.7446 | +0.8005 | 17.35 |
| A2 | 1000 | 0.6265 | 0.5503 | 0.5503 | +0.6922 | 0.9991 | 0.9991 | +1.9983 | +0.9167 | 11.62 |
| A3 | 0 | nan | 1.8721 | 1.8730 | -0.6150 | 0.5599 | 0.0133 | +0.0109 | +0.0094 | 80.52 |
| A3 | 25 | 1.2578 | 1.2641 | 1.2640 | +0.0082 | 0.0658 | 0.0214 | +0.0294 | +0.0198 | 77.66 |
| A3 | 50 | 1.2395 | 1.2344 | 1.2343 | +0.0144 | 0.0910 | 0.0428 | +0.0502 | +0.0338 | 54.91 |
| A3 | 100 | 1.1927 | 1.1778 | 1.1777 | +0.0554 | 0.2148 | 0.1482 | +0.2241 | +0.0979 | 42.79 |
| A3 | 200 | 1.0375 | 1.0045 | 1.0044 | +0.2039 | 0.4171 | 0.3366 | +0.6379 | +0.3066 | 34.74 |
| A3 | 300 | 0.7846 | 0.8733 | 0.8728 | +0.5137 | 0.7838 | 0.7474 | +1.4899 | +0.6842 | 25.99 |
| A3 | 500 | 0.7122 | 0.6415 | 0.6415 | +0.6027 | 0.8966 | 0.8737 | +1.7464 | +0.8019 | 17.60 |
| A3 | 1000 | 0.6263 | 0.5507 | 0.5508 | +0.6918 | 0.9985 | 0.9985 | +1.9970 | +0.9157 | 10.31 |

Mean pre-clipping gradient norms on the update preceding step 1000 (descriptive, not causal):

| Arm | action embed | consequence head | H core | event encoder | q_F | q_M | read gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.00657 | 0.15324 | 0.06736 | 0.06161 | 0.01178 | 0.00746 | 0.00449 |
| A1 | 0.00049 | 0.01609 | 0.02997 | 0.01933 | 0.00147 | 0.00065 | 0.00054 |
| A2 | 0.00000 | 0.00000 | 0.09876 | 0.07650 | 0.00697 | 0.00461 | 0.00597 |
| A3 | 0.00384 | 0.27490 | 0.14677 | 0.21830 | 0.02826 | 0.02250 | 0.00323 |

Post-N16 matched evidence event, frozen-model external write and F→M transfer norms at step 1000:

| Arm | external write norm | consolidation norm |
|---|---:|---:|
| A0 | 0.13330 ± 0.02165 | 0.00993 ± 0.00504 |
| A1 | 0.13101 ± 0.01630 | 0.01021 ± 0.00257 |
| A2 | 0.13560 ± 0.01219 | 0.01036 ± 0.00198 |
| A3 | 0.13465 ± 0.01291 | 0.01050 ± 0.00207 |

Each machine-readable checkpoint also records H/F/M and q/r norms, write and transfer magnitudes, and all seven gradient groups. Frozen A2 downstream gradients are expected to be zero by design. Nonzero or small gradient is not proof of causal starvation. No H recurrence or memory law was changed.

## 4. Exposure, persistence, generalization and revision

| Arm | BS N0 | N1 | N2 | N4 | N8 | N16 | N32 | G59 seeds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 0.0000 ± 0.0000 | 0.1108 ± 0.3102 | 0.1079 ± 0.3087 | 0.1106 ± 0.3137 | 0.1113 ± 0.3129 | 0.1113 ± 0.3137 | 0.1115 ± 0.3134 | 1/8 |
| A1 | 0.0000 ± 0.0000 | 0.9167 ± 0.0003 | 0.9168 ± 0.0002 | 0.9169 ± 0.0002 | 0.9169 ± 0.0001 | 0.9169 ± 0.0001 | 0.9166 ± 0.0005 | 8/8 |
| A2 | 0.0000 ± 0.0000 | 0.9166 ± 0.0023 | 0.9166 ± 0.0007 | 0.9165 ± 0.0007 | 0.9166 ± 0.0008 | 0.9167 ± 0.0008 | 0.9171 ± 0.0014 | 8/8 |
| A3 | 0.0000 ± 0.0000 | 0.9163 ± 0.0009 | 0.9162 ± 0.0008 | 0.9163 ± 0.0010 | 0.9158 ± 0.0019 | 0.9155 ± 0.0016 | 0.9158 ± 0.0017 | 8/8 |

G59 compares mean absolute BS at N16/N32 with mean absolute BS at N0/N1. One event may fully identify z in this deterministic-vs-nonzero world; N1 saturation therefore establishes exposure dependence, **not** graded improvement across repeated exposures. Persistence ratio is interpreted only where an individual replicate has preregistered meaningful N16 BS. `null` means not eligible, not failure. Exact D=0/10/50/100/500/1000 values and A→B revision probabilities are in each `behavior/formal/*/*/evaluation.json`.

| Arm | seen BS | novel BS | hard-OOD BS | persistence D0 / D10 / D100 / D1000 (eligible only) | revision BS at 0 / 32 |
|---|---:|---:|---:|---:|---:|
| A0 | 0.1114 ± 0.3137 | 0.1113 ± 0.3137 | 0.1114 ± 0.3136 | 0.8876 ± 0.0000 / 0.8907 ± 0.0000 / 0.8993 ± 0.0000 / 0.8967 ± 0.0000 | 0.1113 ± 0.3137 / 0.1108 ± 0.3131 |
| A1 | 0.9169 ± 0.0001 | 0.9169 ± 0.0001 | 0.9169 ± 0.0001 | 0.9169 ± 0.0001 / 0.8952 ± 0.0616 / 0.4446 ± 0.3410 / 0.0177 ± 0.1185 | 0.9169 ± 0.0001 / 0.3447 ± 0.6829 |
| A2 | 0.9167 ± 0.0008 | 0.9167 ± 0.0008 | 0.9167 ± 0.0008 | 0.9167 ± 0.0008 / 0.9171 ± 0.0017 / 0.6249 ± 0.3732 / 0.1355 ± 0.3040 | 0.9167 ± 0.0008 / 0.8477 ± 0.1960 |
| A3 | 0.9156 ± 0.0015 | 0.9155 ± 0.0016 | 0.9155 ± 0.0016 | 0.9155 ± 0.0016 / 0.9163 ± 0.0041 / 0.4817 ± 0.4112 / 0.1467 ± 0.3138 | 0.9155 ± 0.0016 / 0.7958 ± 0.2400 |

Eligible persistence ratios (BS(D)/BS(0); only individual replicates with meaningful BS(0) enter):

| Arm | D10/D0 | D50/D0 | D100/D0 | D500/D0 | D1000/D0 |
|---|---:|---:|---:|---:|---:|
| A0 | 1.0035 ± 0.0000 | 1.0100 ± 0.0000 | 1.0132 ± 0.0000 | 1.0101 ± 0.0000 | 1.0103 ± 0.0000 |
| A1 | 0.9764 ± 0.0672 | 0.7116 ± 0.3365 | 0.4850 ± 0.3720 | 0.2427 ± 0.4691 | 0.0194 ± 0.1293 |
| A2 | 1.0005 ± 0.0013 | 0.8771 ± 0.2624 | 0.6817 ± 0.4072 | 0.1981 ± 0.3409 | 0.1479 ± 0.3321 |
| A3 | 1.0008 ± 0.0034 | 0.9720 ± 0.0809 | 0.5264 ± 0.4495 | 0.2016 ± 0.3513 | 0.1605 ± 0.3433 |

A→B revision, mean entropy-policy probability of choosing a_A after each paired environment's opposing evidence:

| Arm | opposing exposures | P(a_A | former A history) | P(a_A | former B history) |
|---|---:|---:|---:|
| A0 | 0 | 0.5549 ± 0.1489 | 0.4436 ± 0.1493 |
| A0 | 1 | 0.5550 ± 0.1490 | 0.4436 ± 0.1487 |
| A0 | 2 | 0.5547 ± 0.1487 | 0.4435 ± 0.1492 |
| A0 | 4 | 0.5547 ± 0.1486 | 0.4435 ± 0.1494 |
| A0 | 8 | 0.5547 ± 0.1484 | 0.4437 ± 0.1495 |
| A0 | 16 | 0.5547 ± 0.1485 | 0.4438 ± 0.1496 |
| A0 | 32 | 0.5545 ± 0.1479 | 0.4437 ± 0.1498 |
| A1 | 0 | 0.9586 ± 0.0003 | 0.0417 ± 0.0003 |
| A1 | 1 | 0.9582 ± 0.0008 | 0.0417 ± 0.0004 |
| A1 | 2 | 0.9580 ± 0.0020 | 0.0418 ± 0.0002 |
| A1 | 4 | 0.9121 ± 0.1824 | 0.0420 ± 0.0017 |
| A1 | 8 | 0.8120 ± 0.3346 | 0.0982 ± 0.2203 |
| A1 | 16 | 0.6695 ± 0.4303 | 0.1567 ± 0.3077 |
| A1 | 32 | 0.5106 ± 0.4590 | 0.1659 ± 0.3070 |
| A2 | 0 | 0.9583 ± 0.0006 | 0.0416 ± 0.0005 |
| A2 | 1 | 0.9583 ± 0.0006 | 0.0416 ± 0.0006 |
| A2 | 2 | 0.9583 ± 0.0006 | 0.0416 ± 0.0006 |
| A2 | 4 | 0.9583 ± 0.0006 | 0.0415 ± 0.0007 |
| A2 | 8 | 0.9583 ± 0.0006 | 0.0414 ± 0.0009 |
| A2 | 16 | 0.9583 ± 0.0006 | 0.0441 ± 0.0161 |
| A2 | 32 | 0.9583 ± 0.0007 | 0.1106 ± 0.2275 |
| A3 | 0 | 0.9578 ± 0.0011 | 0.0423 ± 0.0027 |
| A3 | 1 | 0.9578 ± 0.0011 | 0.0443 ± 0.0133 |
| A3 | 2 | 0.9577 ± 0.0014 | 0.0465 ± 0.0267 |
| A3 | 4 | 0.9576 ± 0.0015 | 0.0668 ± 0.1413 |
| A3 | 8 | 0.9573 ± 0.0025 | 0.0704 ± 0.1620 |
| A3 | 16 | 0.9562 ± 0.0069 | 0.0724 ± 0.1622 |
| A3 | 32 | 0.9510 ± 0.0259 | 0.1551 ± 0.3049 |

## 5. Native-state decodability and finite interventions

| Arm | H probe | F probe | M probe | FM probe | G60 best peripheral intervention (passing seeds) |
|---|---:|---:|---:|---:|---|
| A0 | 0.6562 ± 0.1610 | 0.6133 ± 0.1635 | 0.8730 ± 0.0894 | 0.8213 ± 0.1320 | formation_clamp_M (1/8) |
| A1 | 1.0000 ± 0.0000 | 0.9980 ± 0.0055 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | formation_clamp_FM (6/8) |
| A2 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9990 ± 0.0028 | 1.0000 ± 0.0000 | formation_clamp_FM (8/8) |
| A3 | 0.9990 ± 0.0028 | 0.9980 ± 0.0055 | 0.9990 ± 0.0028 | 0.9990 ± 0.0028 | formation_clamp_FM (7/8) |

Mean change in entropy-policy BS under finite state swaps/read clamps, among eligible N16 replicates only (negative means loss/reversal of A-vs-B separation):

| Arm | eligible seeds | H swap | F swap | M swap | FM swap | HFM swap | F clamp | M clamp | FM clamp | formation F clamp | formation M clamp | formation FM clamp | formation F zero | formation M zero | formation FM zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 1/8 | -1.7751 | -0.0009 | +0.0009 | -0.0001 | -1.7752 | +0.0011 | +0.0008 | +0.0020 | -0.0041 | -1.1112 | -0.8847 | -0.0046 | -0.8888 | -0.8865 |
| A1 | 8/8 | -1.4661 | -0.0000 | -0.3740 | -0.3677 | -1.8337 | -0.0001 | -0.0611 | -0.0571 | -0.3168 | -0.4500 | -0.6377 | -0.0293 | -0.6616 | -0.7995 |
| A2 | 8/8 | -1.8336 | -0.0000 | +0.0001 | +0.0001 | -1.8334 | -0.0000 | +0.0001 | +0.0001 | -0.1438 | -0.2918 | -0.5684 | -0.1082 | -0.5255 | -0.7925 |
| A3 | 8/8 | -1.8196 | +0.0000 | -0.0138 | -0.0115 | -1.8311 | -0.0005 | -0.0008 | -0.0011 | -0.2005 | -0.3405 | -0.4694 | -0.0895 | -0.6502 | -0.7116 |

Probes are non-causal and can support only a *stored-but-unused candidate*. Peripheral mediation requires a replicated finite intervention in an arm with meaningful BS. H/HFM swaps are recorded separately and cannot by themselves satisfy G60. Final-probe read clamps and preregistered formation-window read clamps both act within the native read→H computation, never at the output head. Formation-window F/M state-zero interventions are stronger controls added after the first formal A2 training completed; they are reported as `POST_START_EXPLORATORY` and are **excluded from G60**. A null final-state intervention alone does not rule out earlier involvement. Formation interventions establish involvement but not a unique microscopic site because they alter later H/query trajectories.

## 6. Frozen evaluator and state-to-head transfer

A2: pretraining oracle-state evaluator TV=0.9992 ± 0.0001 (minimum 0.9991); evaluator head hash matched pretrain at step 500 in 8/8 and step 1000 in 8/8 seeds; frozen-head native-norm oracle-H mean action-TV=0.9993 ± 0.0002, interaction=1.9986 ± 0.0005. A2 is protected through step 1000.
A3: pretraining oracle-state evaluator TV=0.9992 ± 0.0001 (minimum 0.9991); evaluator head hash matched pretrain at step 500 in 8/8 and step 1000 in 0/8 seeds; frozen-head native-norm oracle-H mean action-TV=0.9993 ± 0.0003, interaction=1.9986 ± 0.0006. A3 is expected to alter the head after step 500.

Native-norm frozen oracle-H action-TV ceiling at every checkpoint (mean across formal seeds):

| Step | A2 oracle-H TV | A2 interaction | A3 oracle-H TV | A3 interaction |
|---:|---:|---:|---:|---:|
| 0 | 0.9983 | 1.9966 | 0.9983 | 1.9966 |
| 25 | 0.9982 | 1.9964 | 0.9982 | 1.9964 |
| 50 | 0.9970 | 1.9940 | 0.9970 | 1.9940 |
| 100 | 0.9986 | 1.9973 | 0.9986 | 1.9973 |
| 200 | 0.9993 | 1.9985 | 0.9993 | 1.9986 |
| 300 | 0.9991 | 1.9981 | 0.9991 | 1.9981 |
| 500 | 0.9995 | 1.9991 | 0.9995 | 1.9991 |
| 1000 | 0.9991 | 1.9983 | 0.9991 | 1.9982 |

## 7. Gate adjudication and failure localization

| Gate | Formal status | Decision rule |
|---|---|---|
| G57 | PASS | ≥6/8 in one anti-shortcut arm: TV≥0.2, I≥0.3, CFA≥0.1 |
| G58 | PASS | ≥6/8 same arm: |BS(N16)|≥0.1 with G57 phenotype |
| G59 | PASS | ≥6/8 same arm: mean|BS|(N16,N32) minus mean|BS|(N0,N1)≥0.1 |
| G60 | PASS | ≥6/8 same F/M/FM swap, final read clamp, or formation read clamp: directional BS reduction≥0.05 in G58-eligible seeds |
| G61 | PASS | A2 or A3: ≥6/8 TV+interaction+BS, zero oracle lifetime input |

**Final category: A — shortcut corrected with replicated behavioral, exposure and peripheral evidence.** A0 passed the joint behavioral criteria in only 1/8 seeds, so the original objective is possible but unstable, not absolutely incapable. A1 shows that privileged counterfactual targets eliminate the shortcut; it is not natural online counterfactual learning. A2 shows that protecting a strong evaluator lets observed-history state formation drive behavior; A3 adds no clear endpoint benefit over A2. G60 is specifically supported by preregistered formation-window read clamps; final-probe F/M swaps/clamps are mostly near null, consistent with history information already residing in H at decision time. This stage does **not** establish uniformly long persistence or reliable revision: mean D1000 persistence ratios are small and A2/A3 resist opposing evidence. No F/M-law or consolidation redesign is justified by these comparisons. A preregistered full Stage 2C rerun is warranted for the successful training schedules, but a natural-language claim still requires natural-online, non-privileged replication.

## 8. Direct answers to the 24 required questions

1. A0 is marginal-like in 7/8 seeds and behaviorally succeeds in only 1/8; the rare success lowers mean CE to 1.1638 and raises mean TV/BS to 0.1181/+0.1113.
2. A0 mean CFA is +0.0786 nats but only 1/8 jointly pass TV/interaction/CFA, so the mean must not be read as stable advantage over marginal.
3. A1 final TV=0.9993; G57 joint-criterion seeds=8/8.
4. A1 mean interaction=+1.9987; per-seed coincidence with TV/CFA is required.
5. A1 mean BS(N16)=+0.9169; G58 seeds=8/8.
6. A2 final TV=0.9989; head hash stayed fixed in 8/8.
7. A2 BS(N16)=+0.9167; G58 seeds=8/8.
8. A3 does not improve on A2: TV 0.9974 vs 0.9989; BS +0.9155 vs +0.9167; both are 8/8.
9. A1 has the highest mean N16 BS (0.9169) and all anti-shortcut arms are 8/8; A2 has the strongest preregistered FM-formation mediation (8/8).
10. A1/A2/A3 beat marginal, action-only and history-only in 8/8 paired seeds (descriptive one-sided sign p=0.0039); CFA is interpreted jointly with TV/interaction.
11. G59=PASS with 8/8 in every anti-shortcut arm; acquisition saturates after one informative event, so this is one-shot exposure dependence, not graded repetition benefit.
12. Persistence is finite and heterogeneous: mean D100 ratios are A1 0.485, A2 0.682, A3 0.526; by D1000 they are 0.019/0.148/0.161. Robust long-delay persistence is not established.
13. A1/A2/A3 retain BS≈0.915–0.917 on seen, novel and tested hard-OOD surfaces; this supports tested combinatorial generalization only.
14. Revision is incomplete: A1 changes substantially but does not reliably reverse by 32 opposing experiences; A2/A3 mostly retain the old disposition (BS 0.848/0.796 at 32).
15. Anti-shortcut H/F/M/FM probes are approximately 0.998–1.000; in A0, M is highest on average at 0.873. Probes remain non-causal.
16. In anti-shortcut arms z is behaviorally used and formation-window mediation is replicated; seven A0 seeds have decodable M but no behavior, a stored-but-unused candidate pattern.
17. G60=PASS; A2 formation_clamp_FM passes 8/8 (A1 6/8, A3 7/8), while final-probe F/M swaps/clamps are mostly near zero.
18. M-probe≥0.70 with |BS| below threshold occurred in 7 formal arm-seeds: [('A0', 7701), ('A0', 7702), ('A0', 7704), ('A0', 7705), ('A0', 7706), ('A0', 7707), ('A0', 7708)]; these are stored-but-unused *candidates* only.
19. Primary failure is a marginal/action-neglect shortcut plus joint optimization instability: A0 is 1/8, while three anti-shortcut schedules are 8/8. This does not imply every A0 run must fail.
20. The training schedule/objective must protect conditional evaluation; no new evaluator architecture is presently required because A1/A2/A3 and oracle-H ceilings are replicated.
21. F/M write redesign: no. The unchanged law supports replicated formation-window mediation once training is corrected.
22. Consolidation redesign: no evidence from this stage. Long-delay/revision weaknesses motivate targeted diagnosis before changing the law.
23. Full Stage 2C suite: yes, a preregistered independent-seed rerun is warranted for A2 and an A1 diagnostic arm, preserving A0 and all persistence/revision negatives.
24. Natural-language behavioral-memory study: not yet. A1 uses privileged counterfactual supervision and A2/A3 privileged evaluator pretraining; natural-online learning, long persistence and revision remain unresolved.

## 9. Reproducibility and limitations

All raw checkpoints and per-seed JSON records are under `results/stage2c3/`. Explicit counterfactual targets provide privileged outer supervision; the resulting learned state, if any, still receives only legal observed events at evaluation. A2/A3 oracle evaluator pretraining also uses latent z before lifetime training; no oracle state is injected during final lifetimes. Paired histories use common surfaces/actions and distinct observed consequences. Formal parameter hashes are unchanged during evaluation. Historical Stage 2C–2C.2 artifacts are checked against their pre-stage SHA-256 baseline. All 15 Stage 2C.3 tests pass; the 147-test repository suite has 146 passes and one inherited Stage 1.5 README immutability failure caused by a pre-Stage-2C.3 committed README change. Finite seeds, one synthetic world, short training and a narrow evaluator family limit generalization. Report all nulls and avoid consciousness/human-like-memory claims.
