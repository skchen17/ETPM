# ET-RCM Stage 2B — Contextual Associative Memory Results

> **Can contextual language associations be compressed into fixed-size recurrent F/M memory and later retrieved behaviorally?**

> **由语言上下文形成的关联信息，能否不依赖保存所有历史 KV，而被压缩进固定规模的 F/M persistent state，并在未来通过 learned query 被重新取回并影响行为？**

**Formal conclusion: Outcome C — contextual KV does not materially improve memory behavior.** This adjudication uses five independent small-model training seeds. A single medium-size seed is a scale check, not a replication. No M-necessity gate was imposed.

## 1. Protocol, lineage and architecture

Stage 1.x and 2A were treated as frozen. E0 reuses the unchanged Stage 2A raw-token KV implementation, including its write-before-read order. E1 derives normalized K and unconstrained V from contextual H; E2 concatenates the current token embedding. E1/E2 read old F/M, update H, derive KV, then external delta-write, consolidate and decay. This necessary ordering difference is a confound in E0 comparisons and prevents attributing any gain solely to representation without an E0-late ordering control. That optional ordering-matched raw-token control is reported separately when available; it never substitutes for E0 gates. The integration operator, query, F/M reads, consolidation law, decay, NULL and SELF_OUTPUT rules were not modified. Early auxiliary-data OOD leakage was found before formal completion, interrupted, and corrected; see `reports/STAGE2B_PROTOCOL_AMENDMENT.md`. The two 100-step development seeds are not adjudicative.

## 2. Data, lexical OOD and shortcut checks

Synthetic English has entity-attribute, location, revision, transfer relation, temporal revision and 8/16/32-entity interference families. Training gaps are 16/32/64/128; evaluation 32/64/128 is in-distribution and 256/512/1024 extrapolates. Every lexical item appears in training; the parity of name+value indexes makes answer-bearing train and OOD combinations disjoint for attribute/location/revision/interference. Those four families define the strict lexical-OOD headline; all-family OOD cells, including temporal and transfer relation, remain visible separately. Auxiliary basic text never co-occurs a name with a color/place, and auxiliary reasoning excludes name-color binding. Counterfactual pairs have exactly the same token bag, different swapped relations, shared distractor/question, and randomized statement position. Name, answer, order and gap are randomized but not perfectly balanced, especially across 32 owner names. Relation-transfer and last-update temporal tasks still admit recency/template shortcuts; causal memory interventions, not surface accuracy, determine memory claims. Training answer-label audit (seed 2402; min/max counts): {'attribute': {'distinct_labels': 8, 'minimum': 93, 'maximum': 116, 'max_min_ratio': 1.2473118279569892}, 'location': {'distinct_labels': 8, 'minimum': 79, 'maximum': 116, 'max_min_ratio': 1.4683544303797469}, 'revision': {'distinct_labels': 8, 'minimum': 86, 'maximum': 117, 'max_min_ratio': 1.3604651162790697}, 'relation': {'distinct_labels': 32, 'minimum': 18, 'maximum': 34, 'max_min_ratio': 1.8888888888888888}, 'temporal': {'distinct_labels': 8, 'minimum': 84, 'maximum': 112, 'max_min_ratio': 1.3333333333333333}, 'interference': {'distinct_labels': 8, 'minimum': 87, 'maximum': 121, 'max_min_ratio': 1.3908045977011494}}. Unit tests verify the lexical split, paired token inventory and a bounded attribute-label imbalance; the relation-name imbalance remains a limitation.

## 3. Training scale, parameters and language modeling

All formal runs use 1000 optimizer steps, batch 16, AdamW 5e-4, parameter-gradient clip 1, no state clip; objective = mean next-token CE + 4×answer-token CE. The answer loss is ordinary task supervision, not a memory key, importance or entity label. Train-only lowercase word/punctuation vocabulary, OOV `<unk>`. Nominal parameters include inactive inherited branches; active parameters are counted from non-null gradients on a training batch. Compute/token is measured wall time; not hardware-normalized FLOPs. Allocated state includes H/F/M even in no-memory baselines; effective F/M bytes are separately zero there. Training logs include loss, preclip gradient and H/F/M norms. Validation sets share templates but use separate RNG streams.

| size | arm | n_seeds | nominal | active | state_B | effective_FM_B | ms_per_token | basic_CE | basic_PPL | memory_ID_CE | memory_ID_PPL | memory_OOD_CE | memory_OOD_PPL | reasoning_CE | reasoning_PPL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small | GRU | 5 | 150721 | 55980 | 8448 | 0 | 0.276 | 0.575 | 1.778 | 1.390 | 4.034 | 1.519 | 4.596 | 2.761 | 16.164 |
| small | RNN | 5 | 150721 | 68204 | 8448 | 0 | 0.323 | 0.848 | 2.340 | 4.470 | 88.792 | 4.440 | 85.921 | 3.743 | 43.787 |
| small | E0 | 5 | 150721 | 72687 | 8448 | 8192 | 0.646 | 0.795 | 2.217 | 4.588 | 99.648 | 4.539 | 94.622 | 3.387 | 29.797 |
| small | E1 | 5 | 154817 | 76783 | 8448 | 8192 | 0.680 | 0.842 | 2.349 | 4.546 | 95.570 | 4.448 | 86.843 | 3.716 | 42.618 |
| small | E2 | 5 | 156865 | 78831 | 8448 | 8192 | 0.659 | 0.823 | 2.291 | 4.400 | 83.087 | 4.330 | 77.031 | 3.603 | 37.559 |
| medium | GRU | 1 | 435393 | 164652 | 8704 | 0 | 0.259 | 0.546 | 1.727 | 0.896 | 2.451 | 1.077 | 2.934 | 1.307 | 3.695 |
| medium | RNN | 1 | 435393 | 213676 | 8704 | 0 | 0.311 | 0.616 | 1.851 | 3.968 | 52.856 | 3.998 | 54.470 | 2.909 | 18.337 |
| medium | E0 | 1 | 435393 | 222575 | 8704 | 8192 | 0.622 | 0.568 | 1.765 | 3.802 | 44.778 | 3.782 | 43.883 | 2.526 | 12.505 |
| medium | E1 | 1 | 443585 | 230767 | 8704 | 8192 | 0.626 | 0.579 | 1.785 | 3.768 | 43.293 | 3.778 | 43.736 | 2.720 | 15.181 |
| medium | E2 | 1 | 445633 | 232815 | 8704 | 8192 | 0.644 | 0.579 | 1.785 | 3.682 | 39.745 | 3.736 | 41.918 | 2.767 | 15.910 |

## 4. Associative recall — E0/E1/E2, GRU and RNN

Candidate accuracy is constrained to each family's answer set; unrestricted next-token accuracy and answer CE are separate. Pooled candidate chance is not uniform across families (transfer-owner has 32 names); paired arm differences are the primary comparison. Long extrapolation uses attribute/interference only.

| size | arm | split | candidate_acc | seed_SD | raw_exact | answer_CE |
|---|---|---|---|---|---|---|
| small | GRU | ID | 0.175 | 0.046 | 0.172 | 2.313 |
| small | GRU | strict lexical OOD | 0.033 | 0.038 | 0.033 | 2.777 |
| small | GRU | all OOD cells | 0.039 | 0.035 | 0.039 | 2.863 |
| small | GRU | 256+ | 0.156 | 0.047 | 0.156 | 2.351 |
| small | RNN | ID | 0.161 | 0.049 | 0.158 | 2.631 |
| small | RNN | strict lexical OOD | 0.167 | 0.049 | 0.167 | 2.280 |
| small | RNN | all OOD cells | 0.132 | 0.037 | 0.128 | 2.702 |
| small | RNN | 256+ | 0.104 | 0.045 | 0.092 | 2.748 |
| small | E0 | ID | 0.433 | 0.107 | 0.432 | 1.892 |
| small | E0 | strict lexical OOD | 0.508 | 0.088 | 0.508 | 1.544 |
| small | E0 | all OOD cells | 0.417 | 0.104 | 0.415 | 2.030 |
| small | E0 | 256+ | 0.315 | 0.134 | 0.308 | 2.252 |
| small | E1 | ID | 0.172 | 0.032 | 0.168 | 2.513 |
| small | E1 | strict lexical OOD | 0.165 | 0.055 | 0.163 | 2.288 |
| small | E1 | all OOD cells | 0.128 | 0.039 | 0.126 | 2.619 |
| small | E1 | 256+ | 0.181 | 0.033 | 0.175 | 2.412 |
| small | E2 | ID | 0.217 | 0.122 | 0.217 | 2.440 |
| small | E2 | strict lexical OOD | 0.246 | 0.204 | 0.246 | 2.278 |
| small | E2 | all OOD cells | 0.188 | 0.139 | 0.188 | 2.692 |
| small | E2 | 256+ | 0.148 | 0.072 | 0.146 | 2.666 |
| medium | GRU | ID | 0.257 | 0.000 | 0.257 | 1.939 |
| medium | GRU | strict lexical OOD | 0.000 | 0.000 | 0.000 | 5.395 |
| medium | GRU | all OOD cells | 0.007 | 0.000 | 0.007 | 4.590 |
| medium | GRU | 256+ | 0.146 | 0.000 | 0.146 | 3.627 |
| medium | RNN | ID | 0.181 | 0.000 | 0.181 | 2.317 |
| medium | RNN | strict lexical OOD | 0.219 | 0.000 | 0.219 | 2.055 |
| medium | RNN | all OOD cells | 0.174 | 0.000 | 0.174 | 2.408 |
| medium | RNN | 256+ | 0.219 | 0.000 | 0.219 | 2.376 |
| medium | E0 | ID | 0.576 | 0.000 | 0.576 | 1.498 |
| medium | E0 | strict lexical OOD | 0.615 | 0.000 | 0.615 | 1.185 |
| medium | E0 | all OOD cells | 0.535 | 0.000 | 0.535 | 1.602 |
| medium | E0 | 256+ | 0.344 | 0.000 | 0.344 | 2.168 |
| medium | E1 | ID | 0.201 | 0.000 | 0.201 | 2.315 |
| medium | E1 | strict lexical OOD | 0.188 | 0.000 | 0.188 | 2.024 |
| medium | E1 | all OOD cells | 0.146 | 0.000 | 0.146 | 2.398 |
| medium | E1 | 256+ | 0.094 | 0.000 | 0.094 | 2.624 |
| medium | E2 | ID | 0.188 | 0.000 | 0.188 | 2.402 |
| medium | E2 | strict lexical OOD | 0.198 | 0.000 | 0.198 | 2.113 |
| medium | E2 | all OOD cells | 0.153 | 0.000 | 0.153 | 2.457 |
| medium | E2 | 256+ | 0.240 | 0.000 | 0.198 | 2.488 |

### Gap-by-gap recall, including 256/512/1024 extrapolation

For strict OOD gaps 32–128, only the four train/OOD-disjoint answer-pair families are included; long gaps use attribute/interference, exactly as generated. Each cell gives examples per seed.

| size | arm | split | gap | families | n_per_seed | candidate_acc | raw_exact | answer_CE |
|---|---|---|---|---|---|---|---|---|
| small | GRU | train | 32 | all 6 | 48 | 0.158 | 0.154 | 2.326 |
| small | GRU | train | 64 | all 6 | 48 | 0.188 | 0.183 | 2.288 |
| small | GRU | train | 128 | all 6 | 48 | 0.179 | 0.179 | 2.326 |
| small | GRU | train | 256 | long 2 | 16 | 0.263 | 0.263 | 1.889 |
| small | GRU | train | 512 | long 2 | 16 | 0.237 | 0.237 | 1.807 |
| small | GRU | train | 1024 | long 2 | 16 | 0.338 | 0.338 | 1.935 |
| small | GRU | ood | 32 | strict 4 | 32 | 0.025 | 0.025 | 2.801 |
| small | GRU | ood | 64 | strict 4 | 32 | 0.056 | 0.056 | 2.742 |
| small | GRU | ood | 128 | strict 4 | 32 | 0.019 | 0.019 | 2.789 |
| small | GRU | ood | 256 | long 2 | 16 | 0.025 | 0.025 | 2.796 |
| small | GRU | ood | 512 | long 2 | 16 | 0.037 | 0.037 | 2.800 |
| small | GRU | ood | 1024 | long 2 | 16 | 0.037 | 0.037 | 2.881 |
| small | RNN | train | 32 | all 6 | 48 | 0.171 | 0.171 | 2.592 |
| small | RNN | train | 64 | all 6 | 48 | 0.171 | 0.167 | 2.614 |
| small | RNN | train | 128 | all 6 | 48 | 0.142 | 0.138 | 2.687 |
| small | RNN | train | 256 | long 2 | 16 | 0.138 | 0.125 | 2.432 |
| small | RNN | train | 512 | long 2 | 16 | 0.138 | 0.125 | 2.695 |
| small | RNN | train | 1024 | long 2 | 16 | 0.050 | 0.050 | 3.083 |
| small | RNN | ood | 32 | strict 4 | 32 | 0.181 | 0.181 | 2.254 |
| small | RNN | ood | 64 | strict 4 | 32 | 0.181 | 0.181 | 2.259 |
| small | RNN | ood | 128 | strict 4 | 32 | 0.138 | 0.138 | 2.328 |
| small | RNN | ood | 256 | long 2 | 16 | 0.138 | 0.138 | 2.462 |
| small | RNN | ood | 512 | long 2 | 16 | 0.087 | 0.075 | 2.759 |
| small | RNN | ood | 1024 | long 2 | 16 | 0.075 | 0.037 | 3.057 |
| small | E0 | train | 32 | all 6 | 48 | 0.446 | 0.442 | 1.865 |
| small | E0 | train | 64 | all 6 | 48 | 0.446 | 0.446 | 1.862 |
| small | E0 | train | 128 | all 6 | 48 | 0.408 | 0.408 | 1.948 |
| small | E0 | train | 256 | long 2 | 16 | 0.287 | 0.287 | 2.252 |
| small | E0 | train | 512 | long 2 | 16 | 0.312 | 0.312 | 2.212 |
| small | E0 | train | 1024 | long 2 | 16 | 0.325 | 0.300 | 2.309 |
| small | E0 | ood | 32 | strict 4 | 32 | 0.544 | 0.544 | 1.471 |
| small | E0 | ood | 64 | strict 4 | 32 | 0.475 | 0.475 | 1.565 |
| small | E0 | ood | 128 | strict 4 | 32 | 0.506 | 0.506 | 1.596 |
| small | E0 | ood | 256 | long 2 | 16 | 0.325 | 0.325 | 2.056 |
| small | E0 | ood | 512 | long 2 | 16 | 0.412 | 0.412 | 1.988 |
| small | E0 | ood | 1024 | long 2 | 16 | 0.225 | 0.212 | 2.698 |
| small | E1 | train | 32 | all 6 | 48 | 0.163 | 0.163 | 2.497 |
| small | E1 | train | 64 | all 6 | 48 | 0.175 | 0.171 | 2.518 |
| small | E1 | train | 128 | all 6 | 48 | 0.179 | 0.171 | 2.524 |
| small | E1 | train | 256 | long 2 | 16 | 0.150 | 0.150 | 2.443 |
| small | E1 | train | 512 | long 2 | 16 | 0.175 | 0.150 | 2.378 |
| small | E1 | train | 1024 | long 2 | 16 | 0.287 | 0.275 | 2.462 |
| small | E1 | ood | 32 | strict 4 | 32 | 0.169 | 0.169 | 2.270 |
| small | E1 | ood | 64 | strict 4 | 32 | 0.150 | 0.150 | 2.275 |
| small | E1 | ood | 128 | strict 4 | 32 | 0.175 | 0.169 | 2.319 |
| small | E1 | ood | 256 | long 2 | 16 | 0.100 | 0.100 | 2.445 |
| small | E1 | ood | 512 | long 2 | 16 | 0.188 | 0.188 | 2.299 |
| small | E1 | ood | 1024 | long 2 | 16 | 0.188 | 0.188 | 2.445 |
| small | E2 | train | 32 | all 6 | 48 | 0.221 | 0.221 | 2.395 |
| small | E2 | train | 64 | all 6 | 48 | 0.237 | 0.237 | 2.454 |
| small | E2 | train | 128 | all 6 | 48 | 0.192 | 0.192 | 2.470 |
| small | E2 | train | 256 | long 2 | 16 | 0.075 | 0.075 | 2.826 |
| small | E2 | train | 512 | long 2 | 16 | 0.188 | 0.175 | 2.740 |
| small | E2 | train | 1024 | long 2 | 16 | 0.212 | 0.212 | 2.243 |
| small | E2 | ood | 32 | strict 4 | 32 | 0.294 | 0.294 | 2.161 |
| small | E2 | ood | 64 | strict 4 | 32 | 0.225 | 0.225 | 2.273 |
| small | E2 | ood | 128 | strict 4 | 32 | 0.219 | 0.219 | 2.401 |
| small | E2 | ood | 256 | long 2 | 16 | 0.125 | 0.125 | 2.749 |
| small | E2 | ood | 512 | long 2 | 16 | 0.175 | 0.175 | 2.548 |
| small | E2 | ood | 1024 | long 2 | 16 | 0.113 | 0.113 | 2.888 |
| medium | GRU | train | 32 | all 6 | 48 | 0.312 | 0.312 | 1.876 |
| medium | GRU | train | 64 | all 6 | 48 | 0.271 | 0.271 | 1.927 |
| medium | GRU | train | 128 | all 6 | 48 | 0.188 | 0.188 | 2.013 |
| medium | GRU | train | 256 | long 2 | 16 | 0.500 | 0.500 | 1.345 |
| medium | GRU | train | 512 | long 2 | 16 | 0.250 | 0.250 | 1.443 |
| medium | GRU | train | 1024 | long 2 | 16 | 0.125 | 0.125 | 1.545 |
| medium | GRU | ood | 32 | strict 4 | 32 | 0.000 | 0.000 | 5.451 |
| medium | GRU | ood | 64 | strict 4 | 32 | 0.000 | 0.000 | 5.352 |
| medium | GRU | ood | 128 | strict 4 | 32 | 0.000 | 0.000 | 5.383 |
| medium | GRU | ood | 256 | long 2 | 16 | 0.000 | 0.000 | 5.761 |
| medium | GRU | ood | 512 | long 2 | 16 | 0.000 | 0.000 | 5.872 |
| medium | GRU | ood | 1024 | long 2 | 16 | 0.000 | 0.000 | 5.798 |
| medium | RNN | train | 32 | all 6 | 48 | 0.167 | 0.167 | 2.276 |
| medium | RNN | train | 64 | all 6 | 48 | 0.229 | 0.229 | 2.276 |
| medium | RNN | train | 128 | all 6 | 48 | 0.146 | 0.146 | 2.399 |
| medium | RNN | train | 256 | long 2 | 16 | 0.500 | 0.500 | 1.842 |
| medium | RNN | train | 512 | long 2 | 16 | 0.188 | 0.188 | 2.410 |
| medium | RNN | train | 1024 | long 2 | 16 | 0.062 | 0.062 | 2.825 |
| medium | RNN | ood | 32 | strict 4 | 32 | 0.219 | 0.219 | 2.022 |
| medium | RNN | ood | 64 | strict 4 | 32 | 0.188 | 0.188 | 2.029 |
| medium | RNN | ood | 128 | strict 4 | 32 | 0.250 | 0.250 | 2.113 |
| medium | RNN | ood | 256 | long 2 | 16 | 0.125 | 0.125 | 2.454 |
| medium | RNN | ood | 512 | long 2 | 16 | 0.250 | 0.250 | 2.349 |
| medium | RNN | ood | 1024 | long 2 | 16 | 0.188 | 0.188 | 2.375 |
| medium | E0 | train | 32 | all 6 | 48 | 0.583 | 0.583 | 1.428 |
| medium | E0 | train | 64 | all 6 | 48 | 0.542 | 0.542 | 1.540 |
| medium | E0 | train | 128 | all 6 | 48 | 0.604 | 0.604 | 1.524 |
| medium | E0 | train | 256 | long 2 | 16 | 0.375 | 0.375 | 2.395 |
| medium | E0 | train | 512 | long 2 | 16 | 0.312 | 0.312 | 2.113 |
| medium | E0 | train | 1024 | long 2 | 16 | 0.312 | 0.312 | 2.256 |
| medium | E0 | ood | 32 | strict 4 | 32 | 0.625 | 0.625 | 1.091 |
| medium | E0 | ood | 64 | strict 4 | 32 | 0.562 | 0.562 | 1.281 |
| medium | E0 | ood | 128 | strict 4 | 32 | 0.656 | 0.656 | 1.182 |
| medium | E0 | ood | 256 | long 2 | 16 | 0.250 | 0.250 | 2.139 |
| medium | E0 | ood | 512 | long 2 | 16 | 0.562 | 0.562 | 1.561 |
| medium | E0 | ood | 1024 | long 2 | 16 | 0.250 | 0.250 | 2.547 |
| medium | E1 | train | 32 | all 6 | 48 | 0.146 | 0.146 | 2.248 |
| medium | E1 | train | 64 | all 6 | 48 | 0.208 | 0.208 | 2.277 |
| medium | E1 | train | 128 | all 6 | 48 | 0.250 | 0.250 | 2.420 |
| medium | E1 | train | 256 | long 2 | 16 | 0.375 | 0.375 | 1.719 |
| medium | E1 | train | 512 | long 2 | 16 | 0.000 | 0.000 | 2.419 |
| medium | E1 | train | 1024 | long 2 | 16 | 0.000 | 0.000 | 3.573 |
| medium | E1 | ood | 32 | strict 4 | 32 | 0.156 | 0.156 | 2.009 |
| medium | E1 | ood | 64 | strict 4 | 32 | 0.188 | 0.188 | 1.985 |
| medium | E1 | ood | 128 | strict 4 | 32 | 0.219 | 0.219 | 2.076 |
| medium | E1 | ood | 256 | long 2 | 16 | 0.062 | 0.062 | 2.497 |
| medium | E1 | ood | 512 | long 2 | 16 | 0.000 | 0.000 | 2.706 |
| medium | E1 | ood | 1024 | long 2 | 16 | 0.125 | 0.125 | 2.831 |
| medium | E2 | train | 32 | all 6 | 48 | 0.167 | 0.167 | 2.336 |
| medium | E2 | train | 64 | all 6 | 48 | 0.167 | 0.167 | 2.386 |
| medium | E2 | train | 128 | all 6 | 48 | 0.229 | 0.229 | 2.484 |
| medium | E2 | train | 256 | long 2 | 16 | 0.500 | 0.500 | 1.888 |
| medium | E2 | train | 512 | long 2 | 16 | 0.188 | 0.188 | 2.616 |
| medium | E2 | train | 1024 | long 2 | 16 | 0.125 | 0.000 | 3.084 |
| medium | E2 | ood | 32 | strict 4 | 32 | 0.156 | 0.156 | 2.178 |
| medium | E2 | ood | 64 | strict 4 | 32 | 0.188 | 0.188 | 2.059 |
| medium | E2 | ood | 128 | strict 4 | 32 | 0.250 | 0.250 | 2.101 |
| medium | E2 | ood | 256 | long 2 | 16 | 0.062 | 0.062 | 2.613 |
| medium | E2 | ood | 512 | long 2 | 16 | 0.250 | 0.188 | 2.276 |
| medium | E2 | ood | 1024 | long 2 | 16 | 0.312 | 0.250 | 2.453 |

### Family-by-family recall (small model, five seeds)

Only OOD rows marked `primary_OOD=yes` contribute to the strict lexical-OOD headline. The other OOD family rows are exploratory and retained to expose template/recency shortcuts.

| arm | family | split | primary_OOD | n_per_seed | candidate_acc | raw_exact | answer_CE |
|---|---|---|---|---|---|---|---|
| GRU | attribute | train | no | 24 | 0.233 | 0.233 | 1.900 |
| GRU | attribute | ood | yes | 24 | 0.033 | 0.033 | 2.724 |
| GRU | location | train | no | 24 | 0.200 | 0.183 | 2.167 |
| GRU | location | ood | yes | 24 | 0.017 | 0.017 | 2.841 |
| GRU | revision | train | no | 24 | 0.200 | 0.200 | 1.874 |
| GRU | revision | ood | yes | 24 | 0.033 | 0.033 | 2.817 |
| GRU | relation | train | no | 24 | 0.025 | 0.025 | 3.780 |
| GRU | relation | ood | no | 24 | 0.017 | 0.017 | 3.796 |
| GRU | temporal | train | no | 24 | 0.125 | 0.125 | 2.259 |
| GRU | temporal | ood | no | 24 | 0.083 | 0.083 | 2.273 |
| GRU | interference | train | no | 24 | 0.267 | 0.267 | 1.900 |
| GRU | interference | ood | yes | 24 | 0.050 | 0.050 | 2.726 |
| RNN | attribute | train | no | 24 | 0.192 | 0.192 | 2.251 |
| RNN | attribute | ood | yes | 24 | 0.192 | 0.192 | 2.355 |
| RNN | location | train | no | 24 | 0.150 | 0.150 | 2.422 |
| RNN | location | ood | yes | 24 | 0.150 | 0.150 | 2.430 |
| RNN | revision | train | no | 24 | 0.333 | 0.333 | 1.844 |
| RNN | revision | ood | yes | 24 | 0.192 | 0.192 | 2.043 |
| RNN | relation | train | no | 24 | 0.033 | 0.025 | 4.384 |
| RNN | relation | ood | no | 24 | 0.025 | 0.008 | 4.333 |
| RNN | temporal | train | no | 24 | 0.125 | 0.117 | 2.593 |
| RNN | temporal | ood | no | 24 | 0.100 | 0.092 | 2.757 |
| RNN | interference | train | no | 24 | 0.133 | 0.133 | 2.294 |
| RNN | interference | ood | yes | 24 | 0.133 | 0.133 | 2.293 |
| E0 | attribute | train | no | 24 | 0.583 | 0.583 | 1.259 |
| E0 | attribute | ood | yes | 24 | 0.467 | 0.467 | 1.554 |
| E0 | location | train | no | 24 | 0.442 | 0.442 | 1.773 |
| E0 | location | ood | yes | 24 | 0.500 | 0.500 | 1.654 |
| E0 | revision | train | no | 24 | 0.858 | 0.858 | 0.732 |
| E0 | revision | ood | yes | 24 | 0.767 | 0.767 | 0.896 |
| E0 | relation | train | no | 24 | 0.150 | 0.142 | 3.525 |
| E0 | relation | ood | no | 24 | 0.125 | 0.117 | 3.686 |
| E0 | temporal | train | no | 24 | 0.383 | 0.383 | 1.780 |
| E0 | temporal | ood | no | 24 | 0.342 | 0.342 | 2.316 |
| E0 | interference | train | no | 24 | 0.183 | 0.183 | 2.281 |
| E0 | interference | ood | yes | 24 | 0.300 | 0.300 | 2.072 |
| E1 | attribute | train | no | 24 | 0.200 | 0.175 | 2.401 |
| E1 | attribute | ood | yes | 24 | 0.183 | 0.175 | 2.404 |
| E1 | location | train | no | 24 | 0.133 | 0.133 | 2.316 |
| E1 | location | ood | yes | 24 | 0.133 | 0.133 | 2.347 |
| E1 | revision | train | no | 24 | 0.308 | 0.308 | 1.825 |
| E1 | revision | ood | yes | 24 | 0.225 | 0.225 | 2.080 |
| E1 | relation | train | no | 24 | 0.058 | 0.058 | 3.895 |
| E1 | relation | ood | no | 24 | 0.025 | 0.025 | 3.991 |
| E1 | temporal | train | no | 24 | 0.175 | 0.175 | 2.345 |
| E1 | temporal | ood | no | 24 | 0.083 | 0.083 | 2.570 |
| E1 | interference | train | no | 24 | 0.158 | 0.158 | 2.294 |
| E1 | interference | ood | yes | 24 | 0.117 | 0.117 | 2.321 |
| E2 | attribute | train | no | 24 | 0.275 | 0.275 | 2.230 |
| E2 | attribute | ood | yes | 24 | 0.250 | 0.250 | 2.461 |
| E2 | location | train | no | 24 | 0.200 | 0.200 | 2.304 |
| E2 | location | ood | yes | 24 | 0.192 | 0.192 | 2.337 |
| E2 | revision | train | no | 24 | 0.408 | 0.408 | 1.555 |
| E2 | revision | ood | yes | 24 | 0.383 | 0.383 | 1.945 |
| E2 | relation | train | no | 24 | 0.067 | 0.067 | 3.913 |
| E2 | relation | ood | no | 24 | 0.017 | 0.017 | 4.142 |
| E2 | temporal | train | no | 24 | 0.192 | 0.192 | 2.213 |
| E2 | temporal | ood | no | 24 | 0.125 | 0.125 | 2.897 |
| E2 | interference | train | no | 24 | 0.158 | 0.158 | 2.424 |
| E2 | interference | ood | yes | 24 | 0.158 | 0.158 | 2.369 |

### Independent formal seed records

| seed | arm | ID_acc | OOD_acc | long_acc | ID_CE |
|---|---|---|---|---|---|
| 2401 | GRU | 0.097 | 0.083 | 0.135 | 2.481 |
| 2401 | RNN | 0.194 | 0.167 | 0.062 | 2.692 |
| 2401 | E0 | 0.285 | 0.354 | 0.094 | 2.261 |
| 2401 | E1 | 0.181 | 0.250 | 0.146 | 2.508 |
| 2401 | E2 | 0.403 | 0.594 | 0.260 | 2.239 |
| 2402 | GRU | 0.181 | 0.021 | 0.177 | 2.312 |
| 2402 | RNN | 0.160 | 0.104 | 0.073 | 2.641 |
| 2402 | E0 | 0.444 | 0.521 | 0.406 | 1.960 |
| 2402 | E1 | 0.118 | 0.125 | 0.177 | 2.734 |
| 2402 | E2 | 0.118 | 0.104 | 0.177 | 2.714 |
| 2403 | GRU | 0.208 | 0.062 | 0.188 | 2.382 |
| 2403 | RNN | 0.188 | 0.146 | 0.177 | 2.526 |
| 2403 | E0 | 0.583 | 0.562 | 0.417 | 1.615 |
| 2403 | E1 | 0.174 | 0.115 | 0.198 | 2.382 |
| 2403 | E2 | 0.132 | 0.104 | 0.115 | 2.550 |
| 2404 | GRU | 0.208 | 0.000 | 0.083 | 2.224 |
| 2404 | RNN | 0.188 | 0.240 | 0.104 | 2.413 |
| 2404 | E0 | 0.451 | 0.552 | 0.375 | 1.790 |
| 2404 | E1 | 0.194 | 0.146 | 0.156 | 2.455 |
| 2404 | E2 | 0.278 | 0.250 | 0.094 | 2.209 |
| 2405 | GRU | 0.181 | 0.000 | 0.198 | 2.168 |
| 2405 | RNN | 0.076 | 0.177 | 0.104 | 2.885 |
| 2405 | E0 | 0.403 | 0.552 | 0.281 | 1.832 |
| 2405 | E1 | 0.194 | 0.188 | 0.229 | 2.485 |
| 2405 | E2 | 0.153 | 0.177 | 0.094 | 2.486 |

### Optional E0-late ordering control (not part of G38–G41)

| seed | arm | ID_acc | OOD_acc | ID_answer_CE | pair_joint | Hreset_gain | mem_zero_benefit | active_parameters |
|---|---|---|---|---|---|---|---|---|
| 2401 | E0_late | 0.424 | 0.521 | 2.002 | 0.281 | 0.000 | -0.100 | 72687 |
| 2402 | E0_late | 0.458 | 0.469 | 1.914 | 0.156 | 0.000 | -0.015 | 72687 |
| 2403 | E0_late | 0.556 | 0.573 | 1.696 | 0.375 | 0.000 | -0.076 | 72687 |
| 2404 | E0_late | 0.479 | 0.521 | 1.783 | 0.312 | -0.031 | -0.002 | 72687 |
| 2405 | E0_late | 0.472 | 0.531 | 1.755 | 0.344 | 0.047 | 0.017 | 72687 |

Paired contextual-minus-ordering-control comparisons:

| arm_vs_E0_late | mean_ID_candidate_gain | positive_ID_seeds | mean_pair_joint_gain | positive_pair_seeds | mean_answer_CE_benefit | positive_CE_seeds |
|---|---|---|---|---|---|---|
| E1 | -0.306 | 0 | -0.200 | 0 | -0.683 | 0 |
| E2 | -0.261 | 0 | -0.206 | 0 | -0.610 | 0 |

The optional controls were trained on CPU while mandatory formal arms used GPU. This is a numerical/hardware caveat for close differences; data, seeds and optimizer schedule are otherwise matched.

### Optional GRU-76 active-parameter control (not part of G38–G41)

| seed | hidden | active_parameters | ID_acc | OOD_acc | ID_answer_CE |
|---|---|---|---|---|---|
| 2401 | 76 | 71988 | 0.188 | 0.021 | 2.191 |
| 2402 | 76 | 71988 | 0.215 | 0.000 | 2.153 |
| 2403 | 76 | 71988 | 0.222 | 0.000 | 2.039 |
| 2404 | 76 | 71988 | 0.181 | 0.000 | 2.082 |
| 2405 | 76 | 71988 | 0.104 | 0.000 | 2.298 |

## 5. Counterfactual same-token/different-relation binding

Each pair swaps the two colors while preserving the complete token inventory; the query relation changes its correct answer. Candidate choices are the two colors, so pairwise chance joint accuracy is 0.25. F/M swaps preserve the receiver episode's H and the other memory component. Answer CE and downstream candidate decisions are intervention outcomes, not geometric proxies.

| size | arm | condition | pairs_per_seed | joint_acc | candidate_acc | answer_CE |
|---|---|---|---|---|---|---|
| small | E0 | full | 32 | 0.212 | 0.419 | 1.670 |
| small | E0 | F_swap | 32 | 0.212 | 0.419 | 1.671 |
| small | E0 | M_swap | 32 | 0.200 | 0.441 | 1.584 |
| small | E0 | FM_swap | 32 | 0.200 | 0.441 | 1.585 |
| small | E0 | zero | 32 | 0.212 | 0.425 | 1.666 |
| small | E0 | random | 32 | 0.206 | 0.425 | 1.663 |
| small | E0 | H_reset | 32 | 0.050 | 0.500 | 4.184 |
| small | E0 | H_reset_FM_zero | 32 | 0.000 | 0.500 | 4.844 |
| small | E1 | full | 32 | 0.094 | 0.537 | 2.279 |
| small | E1 | F_swap | 32 | 0.094 | 0.537 | 2.280 |
| small | E1 | M_swap | 32 | 0.081 | 0.531 | 2.280 |
| small | E1 | FM_swap | 32 | 0.081 | 0.528 | 2.280 |
| small | E1 | zero | 32 | 0.106 | 0.531 | 2.274 |
| small | E1 | random | 32 | 0.100 | 0.525 | 2.284 |
| small | E1 | H_reset | 32 | 0.031 | 0.512 | 3.488 |
| small | E1 | H_reset_FM_zero | 32 | 0.000 | 0.500 | 3.814 |
| small | E2 | full | 32 | 0.087 | 0.481 | 2.361 |
| small | E2 | F_swap | 32 | 0.087 | 0.484 | 2.368 |
| small | E2 | M_swap | 32 | 0.081 | 0.481 | 2.346 |
| small | E2 | FM_swap | 32 | 0.081 | 0.487 | 2.354 |
| small | E2 | zero | 32 | 0.075 | 0.475 | 2.363 |
| small | E2 | random | 32 | 0.075 | 0.469 | 2.453 |
| small | E2 | H_reset | 32 | 0.025 | 0.503 | 3.327 |
| small | E2 | H_reset_FM_zero | 32 | 0.000 | 0.500 | 3.732 |
| medium | E0 | full | 32 | 0.344 | 0.438 | 1.797 |
| medium | E0 | F_swap | 32 | 0.344 | 0.438 | 1.797 |
| medium | E0 | M_swap | 32 | 0.281 | 0.438 | 1.576 |
| medium | E0 | FM_swap | 32 | 0.281 | 0.438 | 1.576 |
| medium | E0 | zero | 32 | 0.344 | 0.422 | 1.814 |
| medium | E0 | random | 32 | 0.312 | 0.422 | 1.822 |
| medium | E0 | H_reset | 32 | 0.094 | 0.484 | 3.840 |
| medium | E0 | H_reset_FM_zero | 32 | 0.000 | 0.500 | 4.404 |
| medium | E1 | full | 32 | 0.031 | 0.516 | 2.249 |
| medium | E1 | F_swap | 32 | 0.031 | 0.516 | 2.250 |
| medium | E1 | M_swap | 32 | 0.031 | 0.516 | 2.261 |
| medium | E1 | FM_swap | 32 | 0.031 | 0.516 | 2.265 |
| medium | E1 | zero | 32 | 0.031 | 0.516 | 2.393 |
| medium | E1 | random | 32 | 0.031 | 0.500 | 2.461 |
| medium | E1 | H_reset | 32 | 0.000 | 0.500 | 2.946 |
| medium | E1 | H_reset_FM_zero | 32 | 0.000 | 0.500 | 4.274 |
| medium | E2 | full | 32 | 0.062 | 0.516 | 2.408 |
| medium | E2 | F_swap | 32 | 0.062 | 0.516 | 2.408 |
| medium | E2 | M_swap | 32 | 0.062 | 0.516 | 2.415 |
| medium | E2 | FM_swap | 32 | 0.062 | 0.516 | 2.415 |
| medium | E2 | zero | 32 | 0.062 | 0.516 | 2.385 |
| medium | E2 | random | 32 | 0.062 | 0.531 | 2.568 |
| medium | E2 | H_reset | 32 | 0.000 | 0.469 | 6.153 |
| medium | E2 | H_reset_FM_zero | 32 | 0.000 | 0.500 | 4.782 |

## 6. Functional memory interventions and H reset

Lesions, zero, norm-matched random and batch-deranged shuffled memory are applied after the premise/distractor and before the question. Correct-memory benefit is `CE_control−CE_full`; a positive value is required. H reset restores only active H and leaves F/M bit-exact, then the question is replayed. Shuffled controls are performed only in groups with at least two examples and use a cyclic derangement. F/M swap and H-reset/FM-zero results above are the stronger relation-specific interventions.

| size | arm | condition | n_per_seed | candidate_acc | answer_CE |
|---|---|---|---|---|---|
| small | E0 | full | 24 | 0.442 | 1.611 |
| small | E0 | H | 24 | 0.133 | 4.269 |
| small | E0 | F | 24 | 0.442 | 1.612 |
| small | E0 | M | 24 | 0.442 | 1.610 |
| small | E0 | FM | 24 | 0.433 | 1.610 |
| small | E0 | zero | 24 | 0.433 | 1.610 |
| small | E0 | random | 24 | 0.442 | 1.618 |
| small | E0 | shuffle | 24 | 0.450 | 1.602 |
| small | E1 | full | 24 | 0.158 | 2.257 |
| small | E1 | H | 24 | 0.125 | 3.458 |
| small | E1 | F | 24 | 0.150 | 2.249 |
| small | E1 | M | 24 | 0.175 | 2.268 |
| small | E1 | FM | 24 | 0.167 | 2.257 |
| small | E1 | zero | 24 | 0.167 | 2.257 |
| small | E1 | random | 24 | 0.200 | 2.258 |
| small | E1 | shuffle | 24 | 0.150 | 2.260 |
| small | E2 | full | 24 | 0.233 | 2.410 |
| small | E2 | H | 24 | 0.050 | 3.332 |
| small | E2 | F | 24 | 0.233 | 2.411 |
| small | E2 | M | 24 | 0.208 | 2.392 |
| small | E2 | FM | 24 | 0.208 | 2.401 |
| small | E2 | zero | 24 | 0.208 | 2.401 |
| small | E2 | random | 24 | 0.217 | 2.495 |
| small | E2 | shuffle | 24 | 0.225 | 2.448 |
| medium | E0 | full | 24 | 0.417 | 1.999 |
| medium | E0 | H | 24 | 0.292 | 3.707 |
| medium | E0 | F | 24 | 0.417 | 1.999 |
| medium | E0 | M | 24 | 0.417 | 1.968 |
| medium | E0 | FM | 24 | 0.417 | 1.962 |
| medium | E0 | zero | 24 | 0.417 | 1.962 |
| medium | E0 | random | 24 | 0.417 | 2.068 |
| medium | E0 | shuffle | 24 | 0.375 | 1.943 |
| medium | E1 | full | 24 | 0.333 | 1.950 |
| medium | E1 | H | 24 | 0.125 | 2.853 |
| medium | E1 | F | 24 | 0.375 | 1.984 |
| medium | E1 | M | 24 | 0.375 | 2.056 |
| medium | E1 | FM | 24 | 0.375 | 2.085 |
| medium | E1 | zero | 24 | 0.375 | 2.085 |
| medium | E1 | random | 24 | 0.417 | 2.139 |
| medium | E1 | shuffle | 24 | 0.375 | 2.019 |
| medium | E2 | full | 24 | 0.167 | 2.155 |
| medium | E2 | H | 24 | 0.083 | 6.113 |
| medium | E2 | F | 24 | 0.167 | 2.153 |
| medium | E2 | M | 24 | 0.167 | 2.144 |
| medium | E2 | FM | 24 | 0.167 | 2.140 |
| medium | E2 | zero | 24 | 0.167 | 2.140 |
| medium | E2 | random | 24 | 0.083 | 2.294 |
| medium | E2 | shuffle | 24 | 0.083 | 2.194 |

### Seed-level contextual gains and gate inputs

- E1: ID candidate gain vs E0 = [-0.10416666666666666, -0.32638888888888884, -0.40972222222222227, -0.2569444444444444, -0.20833333333333334] (mean -0.261, bootstrap 95% CI (-0.3527777777777778, -0.16944444444444443)); ID answer-CE benefit = [-0.2465150066547923, -0.7735149178446994, -0.7673480058502817, -0.6655035628419783, -0.6533981790352197] (CI (-0.7494458820463882, -0.4320582409699757)); OOD candidate gain = [-0.10416666666666669, -0.39583333333333337, -0.4479166666666667, -0.40625, -0.36458333333333337]; pair-joint gain = [0.03125, -0.0625, -0.125, -0.28125, -0.15625]; H-reset peripheral gain = [0.0, -0.015625, 0.015625, 0.0, 0.0625]; correct-memory CE benefits vs zero/random/shuffle = {'zero': [-0.008590683341026306, -0.03628257910410593, 0.0745914628108344, -0.016619771718978882, -0.012119640906651963], 'random': [-0.013184661666552078, -0.025965819756190278, 0.08214926719665527, -0.04116964836915349, 0.0004014720519383008], 'shuffle': [0.002589702606201172, -0.02900707721710205, 0.013281558950742234, 0.0071328431367874146, 0.020303880174954658]}.
- E2: ID candidate gain vs E0 = [0.11805555555555558, -0.32638888888888884, -0.45138888888888895, -0.1736111111111111, -0.25] (mean -0.217, bootstrap 95% CI (-0.37083333333333335, -0.029166666666666653)); ID answer-CE benefit = [0.02185007060567523, -0.7530797567839422, -0.9354325329057045, -0.4191689078385632, -0.6547057882158294] (CI (-0.8063460735190245, -0.24276131646086782)); OOD candidate gain = [0.23958333333333331, -0.4166666666666667, -0.4583333333333333, -0.30208333333333337, -0.375]; pair-joint gain = [0.25, -0.0625, -0.375, -0.21875, -0.21875]; H-reset peripheral gain = [0.03125, 0.0, 0.015625, -0.015625, -0.015625]; correct-memory CE benefits vs zero/random/shuffle = {'zero': [-0.0826060386995473, -0.13632858792940805, 0.14249101777871465, -0.058190633853276275, 0.09432429075241089], 'random': [0.054516763736804474, -0.08251445988814066, 0.3710088084141412, -0.01029684642950679, 0.09487553437550877], 'shuffle': [0.10471391926209117, 0.017811516920725357, 0.06128680209318782, -0.007826648652553558, 0.018815308809280396]}.

## 7. Representation diagnostics and semantic alignment

K/V norms and q·k are descriptive only; they cannot establish semantic storage by themselves. The functional memory interventions above are the causal behavioral check. Per-token vectors and episode labels are preserved in raw evaluation JSON.

| arm | seed | key_norm | value_norm | q_dot_k | same_entity_diff_value | same_value_diff_entity | cross_relation |
|---|---|---|---|---|---|---|---|
| E1 | 2401 | 1.000 | 4.003 | 0.035 | 0.922 | 0.925 | 0.376 |
| E1 | 2402 | 1.000 | 4.302 | -0.032 | 0.609 | 0.709 | 0.179 |
| E1 | 2403 | 1.000 | 4.143 | 0.225 | 0.881 | 0.903 | -0.180 |
| E1 | 2404 | 1.000 | 4.563 | 0.453 | 0.694 | 0.739 | 0.025 |
| E1 | 2405 | 1.000 | 5.040 | 0.319 | 0.803 | 0.834 | 0.199 |
| E2 | 2401 | 1.000 | 2.479 | 0.048 | 0.769 | 0.865 | 0.367 |
| E2 | 2402 | 1.000 | 3.386 | 0.192 | 0.723 | 0.832 | 0.322 |
| E2 | 2403 | 1.000 | 3.743 | -0.012 | 0.859 | 0.884 | -0.087 |
| E2 | 2404 | 1.000 | 3.462 | 0.097 | 0.459 | 0.596 | 0.416 |
| E2 | 2405 | 1.000 | 3.961 | 0.408 | 0.895 | 0.923 | -0.220 |

## 8. Generation and answer extraction

Unrestricted 6-token generation uses greedy, temperature 0.7 or top-k 8. The first alphabetic token after `answer:` is scored as the semantic answer; whole-answer exact requires that the entire alphabetic output be the single target. EOS, punctuation-only, repetition and SELF_OUTPUT write violations are recorded. These measures are not silently replaced by candidate accuracy. The next 48 literal samples include failures.

| size | arm | decoder | n | semantic_acc | whole_exact | EOS | punct_only | repetition | self_writes |
|---|---|---|---|---|---|---|---|---|---|
| small | GRU | greedy | 60 | 0.250 | 0.083 | 0.600 | 0.000 | 0.020 | 0 |
| small | GRU | temp_0.7 | 60 | 0.183 | 0.033 | 0.383 | 0.000 | 0.016 | 0 |
| small | GRU | top_k_8 | 60 | 0.167 | 0.000 | 0.317 | 0.000 | 0.014 | 0 |
| small | RNN | greedy | 60 | 0.250 | 0.000 | 0.000 | 0.000 | 1.000 | 0 |
| small | RNN | temp_0.7 | 60 | 0.183 | 0.000 | 0.000 | 0.000 | 0.180 | 0 |
| small | RNN | top_k_8 | 60 | 0.167 | 0.000 | 0.000 | 0.000 | 0.130 | 0 |
| small | E0 | greedy | 60 | 0.417 | 0.000 | 0.000 | 0.000 | 0.993 | 0 |
| small | E0 | temp_0.7 | 60 | 0.383 | 0.000 | 0.000 | 0.000 | 0.730 | 0 |
| small | E0 | top_k_8 | 60 | 0.350 | 0.000 | 0.000 | 0.000 | 0.499 | 0 |
| small | E1 | greedy | 60 | 0.183 | 0.000 | 0.000 | 0.017 | 0.980 | 0 |
| small | E1 | temp_0.7 | 60 | 0.183 | 0.000 | 0.000 | 0.000 | 0.207 | 0 |
| small | E1 | top_k_8 | 60 | 0.167 | 0.000 | 0.000 | 0.000 | 0.142 | 0 |
| small | E2 | greedy | 60 | 0.233 | 0.000 | 0.000 | 0.000 | 0.987 | 0 |
| small | E2 | temp_0.7 | 60 | 0.250 | 0.000 | 0.000 | 0.000 | 0.323 | 0 |
| small | E2 | top_k_8 | 60 | 0.217 | 0.000 | 0.000 | 0.000 | 0.236 | 0 |
| medium | GRU | greedy | 12 | 0.417 | 0.000 | 1.000 | 0.000 | 0.333 | 0 |
| medium | GRU | temp_0.7 | 12 | 0.250 | 0.000 | 0.917 | 0.000 | 0.083 | 0 |
| medium | GRU | top_k_8 | 12 | 0.333 | 0.000 | 1.000 | 0.000 | 0.069 | 0 |
| medium | RNN | greedy | 12 | 0.250 | 0.000 | 0.000 | 0.000 | 1.000 | 0 |
| medium | RNN | temp_0.7 | 12 | 0.083 | 0.000 | 0.000 | 0.000 | 0.233 | 0 |
| medium | RNN | top_k_8 | 12 | 0.083 | 0.000 | 0.000 | 0.000 | 0.117 | 0 |
| medium | E0 | greedy | 12 | 0.333 | 0.000 | 0.000 | 0.000 | 1.000 | 0 |
| medium | E0 | temp_0.7 | 12 | 0.333 | 0.000 | 0.000 | 0.000 | 0.883 | 0 |
| medium | E0 | top_k_8 | 12 | 0.250 | 0.000 | 0.000 | 0.000 | 0.783 | 0 |
| medium | E1 | greedy | 12 | 0.333 | 0.000 | 0.000 | 0.000 | 1.000 | 0 |
| medium | E1 | temp_0.7 | 12 | 0.167 | 0.000 | 0.000 | 0.000 | 0.283 | 0 |
| medium | E1 | top_k_8 | 12 | 0.167 | 0.000 | 0.000 | 0.000 | 0.188 | 0 |
| medium | E2 | greedy | 12 | 0.167 | 0.000 | 0.000 | 0.000 | 1.000 | 0 |
| medium | E2 | temp_0.7 | 12 | 0.167 | 0.000 | 0.000 | 0.000 | 0.275 | 0 |
| medium | E2 | top_k_8 | 12 | 0.167 | 0.000 | 0.000 | 0.000 | 0.117 | 0 |

1. E0 greedy: target `yellow`, prompt `zack has the yellow key . david has the blue key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . the room had a table and a chair . they walked along the road after lunch . a bird waited by the window . the room had a table and a question : what key does zack have ? answer :` (gap 64) → `yellow yellow yellow yellow yellow yellow`; semantic `yellow`; correct=True; EOS=False.
2. E0 temp_0.7: target `yellow`, prompt `zack has the yellow key . david has the blue key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . the room had a table and a chair . they walked along the road after lunch . a bird waited by the window . the room had a table and a question : what key does zack have ? answer :` (gap 64) → `yellow yellow yellow black black yellow`; semantic `yellow`; correct=True; EOS=False.
3. E0 top_k_8: target `yellow`, prompt `zack has the yellow key . david has the blue key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . the room had a table and a chair . they walked along the road after lunch . a bird waited by the window . the room had a table and a question : what key does zack have ? answer :` (gap 64) → `yellow yellow yellow yellow orange yellow`; semantic `yellow`; correct=True; EOS=False.
4. E0 greedy: target `green`, prompt `yasmin has the green key . kate has the green key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . they walked along the road after lunch . a bird waited by the window . a small train passed the old bridge . the room had a table and a chair question : what key does yasmin have ? answer :` (gap 64) → `green green green green green green`; semantic `green`; correct=True; EOS=False.
5. E0 temp_0.7: target `green`, prompt `yasmin has the green key . kate has the green key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . they walked along the road after lunch . a bird waited by the window . a small train passed the old bridge . the room had a table and a chair question : what key does yasmin have ? answer :` (gap 64) → `white purple purple blue purple green`; semantic `white`; correct=False; EOS=False.
6. E0 top_k_8: target `green`, prompt `yasmin has the green key . kate has the green key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . they walked along the road after lunch . a bird waited by the window . a small train passed the old bridge . the room had a table and a chair question : what key does yasmin have ? answer :` (gap 64) → `red purple green orange orange blue`; semantic `red`; correct=False; EOS=False.
7. E0 greedy: target `white`, prompt `quinn has the white key . noah has the yellow key . the room had a table and a chair . the room had a table and a chair . the wind was quiet near the door . a bird waited by the window . they walked along the road after lunch . a small train passed the old bridge . a small train passed the old bridge . they walked along the road after lunch question : what key does quinn have ? answer :` (gap 64) → `yellow yellow yellow yellow yellow yellow`; semantic `yellow`; correct=False; EOS=False.
8. E0 temp_0.7: target `white`, prompt `quinn has the white key . noah has the yellow key . the room had a table and a chair . the room had a table and a chair . the wind was quiet near the door . a bird waited by the window . they walked along the road after lunch . a small train passed the old bridge . a small train passed the old bridge . they walked along the road after lunch question : what key does quinn have ? answer :` (gap 64) → `purple black yellow yellow orange kitchen`; semantic `purple`; correct=False; EOS=False.
9. E0 top_k_8: target `white`, prompt `quinn has the white key . noah has the yellow key . the room had a table and a chair . the room had a table and a chair . the wind was quiet near the door . a bird waited by the window . they walked along the road after lunch . a small train passed the old bridge . a small train passed the old bridge . they walked along the road after lunch question : what key does quinn have ? answer :` (gap 64) → `purple orange purple white yellow black`; semantic `purple`; correct=False; EOS=False.
10. E0 greedy: target `white`, prompt `wendy has the white key . kate has the purple key . the room had a table and a chair . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . the room had a table and a chair . a bird waited question : what key does wendy have ? answer :` (gap 64) → `purple purple purple purple purple purple`; semantic `purple`; correct=False; EOS=False.
11. E0 temp_0.7: target `white`, prompt `wendy has the white key . kate has the purple key . the room had a table and a chair . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . the room had a table and a chair . a bird waited question : what key does wendy have ? answer :` (gap 64) → `purple purple purple purple purple yellow`; semantic `purple`; correct=False; EOS=False.
12. E0 top_k_8: target `white`, prompt `wendy has the white key . kate has the purple key . the room had a table and a chair . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . the room had a table and a chair . a bird waited question : what key does wendy have ? answer :` (gap 64) → `purple white green green white purple`; semantic `purple`; correct=False; EOS=False.
13. E1 greedy: target `yellow`, prompt `zack has the yellow key . david has the blue key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . the room had a table and a chair . they walked along the road after lunch . a bird waited by the window . the room had a table and a question : what key does zack have ? answer :` (gap 64) → `blue blue blue blue blue blue`; semantic `blue`; correct=False; EOS=False.
14. E1 temp_0.7: target `yellow`, prompt `zack has the yellow key . david has the blue key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . the room had a table and a chair . they walked along the road after lunch . a bird waited by the window . the room had a table and a question : what key does zack have ? answer :` (gap 64) → `yellow yellow yellow black black orange`; semantic `yellow`; correct=True; EOS=False.
15. E1 top_k_8: target `yellow`, prompt `zack has the yellow key . david has the blue key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . the room had a table and a chair . they walked along the road after lunch . a bird waited by the window . the room had a table and a question : what key does zack have ? answer :` (gap 64) → `green white yellow blue orange yellow`; semantic `green`; correct=False; EOS=False.
16. E1 greedy: target `green`, prompt `yasmin has the green key . kate has the green key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . they walked along the road after lunch . a bird waited by the window . a small train passed the old bridge . the room had a table and a chair question : what key does yasmin have ? answer :` (gap 64) → `purple purple purple purple purple purple`; semantic `purple`; correct=False; EOS=False.
17. E1 temp_0.7: target `green`, prompt `yasmin has the green key . kate has the green key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . they walked along the road after lunch . a bird waited by the window . a small train passed the old bridge . the room had a table and a chair question : what key does yasmin have ? answer :` (gap 64) → `white purple purple white purple red`; semantic `white`; correct=False; EOS=False.
18. E1 top_k_8: target `green`, prompt `yasmin has the green key . kate has the green key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . they walked along the road after lunch . a bird waited by the window . a small train passed the old bridge . the room had a table and a chair question : what key does yasmin have ? answer :` (gap 64) → `red purple green red orange blue`; semantic `red`; correct=False; EOS=False.
19. E1 greedy: target `white`, prompt `quinn has the white key . noah has the yellow key . the room had a table and a chair . the room had a table and a chair . the wind was quiet near the door . a bird waited by the window . they walked along the road after lunch . a small train passed the old bridge . a small train passed the old bridge . they walked along the road after lunch question : what key does quinn have ? answer :` (gap 64) → `blue blue blue blue blue blue`; semantic `blue`; correct=False; EOS=False.
20. E1 temp_0.7: target `white`, prompt `quinn has the white key . noah has the yellow key . the room had a table and a chair . the room had a table and a chair . the wind was quiet near the door . a bird waited by the window . they walked along the road after lunch . a small train passed the old bridge . a small train passed the old bridge . they walked along the road after lunch question : what key does quinn have ? answer :` (gap 64) → `purple black purple green orange blue`; semantic `purple`; correct=False; EOS=False.
21. E1 top_k_8: target `white`, prompt `quinn has the white key . noah has the yellow key . the room had a table and a chair . the room had a table and a chair . the wind was quiet near the door . a bird waited by the window . they walked along the road after lunch . a small train passed the old bridge . a small train passed the old bridge . they walked along the road after lunch question : what key does quinn have ? answer :` (gap 64) → `blue orange purple white orange black`; semantic `blue`; correct=False; EOS=False.
22. E1 greedy: target `white`, prompt `wendy has the white key . kate has the purple key . the room had a table and a chair . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . the room had a table and a chair . a bird waited question : what key does wendy have ? answer :` (gap 64) → `purple purple purple purple purple purple`; semantic `purple`; correct=False; EOS=False.
23. E1 temp_0.7: target `white`, prompt `wendy has the white key . kate has the purple key . the room had a table and a chair . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . the room had a table and a chair . a bird waited question : what key does wendy have ? answer :` (gap 64) → `purple white purple green purple yellow`; semantic `purple`; correct=False; EOS=False.
24. E1 top_k_8: target `white`, prompt `wendy has the white key . kate has the purple key . the room had a table and a chair . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . the room had a table and a chair . a bird waited question : what key does wendy have ? answer :` (gap 64) → `purple white red green white green`; semantic `purple`; correct=False; EOS=False.
25. E2 greedy: target `yellow`, prompt `zack has the yellow key . david has the blue key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . the room had a table and a chair . they walked along the road after lunch . a bird waited by the window . the room had a table and a question : what key does zack have ? answer :` (gap 64) → `blue blue blue blue blue blue`; semantic `blue`; correct=False; EOS=False.
26. E2 temp_0.7: target `yellow`, prompt `zack has the yellow key . david has the blue key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . the room had a table and a chair . they walked along the road after lunch . a bird waited by the window . the room had a table and a question : what key does zack have ? answer :` (gap 64) → `purple blue blue black blue blue`; semantic `purple`; correct=False; EOS=False.
27. E2 top_k_8: target `yellow`, prompt `zack has the yellow key . david has the blue key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . the room had a table and a chair . they walked along the road after lunch . a bird waited by the window . the room had a table and a question : what key does zack have ? answer :` (gap 64) → `green white yellow blue. blue`; semantic `green`; correct=False; EOS=False.
28. E2 greedy: target `green`, prompt `yasmin has the green key . kate has the green key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . they walked along the road after lunch . a bird waited by the window . a small train passed the old bridge . the room had a table and a chair question : what key does yasmin have ? answer :` (gap 64) → `green green green green green green`; semantic `green`; correct=True; EOS=False.
29. E2 temp_0.7: target `green`, prompt `yasmin has the green key . kate has the green key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . they walked along the road after lunch . a bird waited by the window . a small train passed the old bridge . the room had a table and a chair question : what key does yasmin have ? answer :` (gap 64) → `green green green green green green`; semantic `green`; correct=True; EOS=False.
30. E2 top_k_8: target `green`, prompt `yasmin has the green key . kate has the green key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . they walked along the road after lunch . a bird waited by the window . a small train passed the old bridge . the room had a table and a chair question : what key does yasmin have ? answer :` (gap 64) → `green green green green orange blue`; semantic `green`; correct=True; EOS=False.
31. E2 greedy: target `white`, prompt `quinn has the white key . noah has the yellow key . the room had a table and a chair . the room had a table and a chair . the wind was quiet near the door . a bird waited by the window . they walked along the road after lunch . a small train passed the old bridge . a small train passed the old bridge . they walked along the road after lunch question : what key does quinn have ? answer :` (gap 64) → `yellow yellow yellow yellow yellow yellow`; semantic `yellow`; correct=False; EOS=False.
32. E2 temp_0.7: target `white`, prompt `quinn has the white key . noah has the yellow key . the room had a table and a chair . the room had a table and a chair . the wind was quiet near the door . a bird waited by the window . they walked along the road after lunch . a small train passed the old bridge . a small train passed the old bridge . they walked along the road after lunch question : what key does quinn have ? answer :` (gap 64) → `yellow orange yellow green orange orange`; semantic `yellow`; correct=False; EOS=False.
33. E2 top_k_8: target `white`, prompt `quinn has the white key . noah has the yellow key . the room had a table and a chair . the room had a table and a chair . the wind was quiet near the door . a bird waited by the window . they walked along the road after lunch . a small train passed the old bridge . a small train passed the old bridge . they walked along the road after lunch question : what key does quinn have ? answer :` (gap 64) → `orange orange yellow green yellow orange`; semantic `orange`; correct=False; EOS=False.
34. E2 greedy: target `white`, prompt `wendy has the white key . kate has the purple key . the room had a table and a chair . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . the room had a table and a chair . a bird waited question : what key does wendy have ? answer :` (gap 64) → `purple purple purple purple purple purple`; semantic `purple`; correct=False; EOS=False.
35. E2 temp_0.7: target `white`, prompt `wendy has the white key . kate has the purple key . the room had a table and a chair . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . the room had a table and a chair . a bird waited question : what key does wendy have ? answer :` (gap 64) → `purple purple purple purple purple yellow`; semantic `purple`; correct=False; EOS=False.
36. E2 top_k_8: target `white`, prompt `wendy has the white key . kate has the purple key . the room had a table and a chair . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . the room had a table and a chair . a bird waited question : what key does wendy have ? answer :` (gap 64) → `purple purple red green purple purple`; semantic `purple`; correct=False; EOS=False.
37. GRU greedy: target `yellow`, prompt `zack has the yellow key . david has the blue key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . the room had a table and a chair . they walked along the road after lunch . a bird waited by the window . the room had a table and a question : what key does zack have ? answer :` (gap 64) → `green. the red the key`; semantic `green`; correct=False; EOS=False.
38. GRU temp_0.7: target `yellow`, prompt `zack has the yellow key . david has the blue key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . the room had a table and a chair . they walked along the road after lunch . a bird waited by the window . the room had a table and a question : what key does zack have ? answer :` (gap 64) → `purple the key the key the`; semantic `purple`; correct=False; EOS=False.
39. GRU top_k_8: target `yellow`, prompt `zack has the yellow key . david has the blue key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . the room had a table and a chair . they walked along the road after lunch . a bird waited by the window . the room had a table and a question : what key does zack have ? answer :` (gap 64) → `green the green key. the`; semantic `green`; correct=False; EOS=False.
40. GRU greedy: target `green`, prompt `yasmin has the green key . kate has the green key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . they walked along the road after lunch . a bird waited by the window . a small train passed the old bridge . the room had a table and a chair question : what key does yasmin have ? answer :` (gap 64) → `red the red key the on`; semantic `red`; correct=False; EOS=False.
41. GRU temp_0.7: target `green`, prompt `yasmin has the green key . kate has the green key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . they walked along the road after lunch . a bird waited by the window . a small train passed the old bridge . the room had a table and a chair question : what key does yasmin have ? answer :` (gap 64) → `white. the on the table`; semantic `white`; correct=False; EOS=False.
42. GRU top_k_8: target `green`, prompt `yasmin has the green key . kate has the green key . the room had a table and a chair . the room had a table and a chair . a bird waited by the window . they walked along the road after lunch . they walked along the road after lunch . a bird waited by the window . a small train passed the old bridge . the room had a table and a chair question : what key does yasmin have ? answer :` (gap 64) → `red purple. the red the`; semantic `red`; correct=False; EOS=False.
43. GRU greedy: target `white`, prompt `quinn has the white key . noah has the yellow key . the room had a table and a chair . the room had a table and a chair . the wind was quiet near the door . a bird waited by the window . they walked along the road after lunch . a small train passed the old bridge . a small train passed the old bridge . they walked along the road after lunch question : what key does quinn have ? answer :` (gap 64) → `red the red key the on`; semantic `red`; correct=False; EOS=False.
44. GRU temp_0.7: target `white`, prompt `quinn has the white key . noah has the yellow key . the room had a table and a chair . the room had a table and a chair . the wind was quiet near the door . a bird waited by the window . they walked along the road after lunch . a small train passed the old bridge . a small train passed the old bridge . they walked along the road after lunch question : what key does quinn have ? answer :` (gap 64) → `purple key the green key the`; semantic `purple`; correct=False; EOS=False.
45. GRU top_k_8: target `white`, prompt `quinn has the white key . noah has the yellow key . the room had a table and a chair . the room had a table and a chair . the wind was quiet near the door . a bird waited by the window . they walked along the road after lunch . a small train passed the old bridge . a small train passed the old bridge . they walked along the road after lunch question : what key does quinn have ? answer :` (gap 64) → `purple the purple key walked.`; semantic `purple`; correct=False; EOS=False.
46. GRU greedy: target `white`, prompt `wendy has the white key . kate has the purple key . the room had a table and a chair . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . the room had a table and a chair . a bird waited question : what key does wendy have ? answer :` (gap 64) → `red the red key the on`; semantic `red`; correct=False; EOS=False.
47. GRU temp_0.7: target `white`, prompt `wendy has the white key . kate has the purple key . the room had a table and a chair . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . the room had a table and a chair . a bird waited question : what key does wendy have ? answer :` (gap 64) → `green table. the on the`; semantic `green`; correct=False; EOS=False.
48. GRU top_k_8: target `white`, prompt `wendy has the white key . kate has the purple key . the room had a table and a chair . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . a small train passed the old bridge . the room had a table and a chair . the room had a table and a chair . a bird waited question : what key does wendy have ? answer :` (gap 64) → `red off the off the on`; semantic `red`; correct=False; EOS=False.

## 9. NULL ticks and stability

NULL ticks have no external token or delta write. The table shows CE, candidate accuracy and H/F/M/read norms on identical questions at K=0/1/2/4/8/16. It does not assume monotone improvement.

| size | arm | K | candidate_acc | answer_CE | H | F | M | r_F | r_M |
|---|---|---|---|---|---|---|---|---|---|
| small | E0 | 0 | 0.325 | 1.987 | 61.858 | 2.246 | 0.372 | 0.203 | 0.084 |
| small | E0 | 1 | 0.325 | 1.990 | 62.527 | 2.178 | 0.374 | 0.192 | 0.084 |
| small | E0 | 2 | 0.325 | 1.993 | 63.188 | 2.112 | 0.377 | 0.181 | 0.084 |
| small | E0 | 4 | 0.325 | 1.999 | 64.487 | 1.987 | 0.381 | 0.161 | 0.084 |
| small | E0 | 8 | 0.325 | 2.011 | 67.010 | 1.758 | 0.388 | 0.128 | 0.084 |
| small | E0 | 16 | 0.350 | 2.038 | 71.885 | 1.376 | 0.399 | 0.084 | 0.084 |
| small | E1 | 0 | 0.200 | 2.146 | 64.514 | 3.112 | 2.876 | 0.333 | 0.629 |
| small | E1 | 1 | 0.200 | 2.145 | 65.286 | 3.016 | 2.878 | 0.338 | 0.625 |
| small | E1 | 2 | 0.200 | 2.144 | 66.053 | 2.923 | 2.880 | 0.310 | 0.623 |
| small | E1 | 4 | 0.200 | 2.142 | 67.569 | 2.747 | 2.883 | 0.261 | 0.618 |
| small | E1 | 8 | 0.250 | 2.142 | 70.552 | 2.427 | 2.886 | 0.187 | 0.611 |
| small | E1 | 16 | 0.250 | 2.147 | 76.406 | 1.898 | 2.885 | 0.099 | 0.604 |
| small | E2 | 0 | 0.225 | 2.465 | 58.105 | 3.203 | 1.471 | 0.346 | 0.322 |
| small | E2 | 1 | 0.200 | 2.456 | 59.289 | 3.103 | 1.465 | 0.346 | 0.319 |
| small | E2 | 2 | 0.150 | 2.449 | 60.450 | 3.006 | 1.460 | 0.307 | 0.317 |
| small | E2 | 4 | 0.150 | 2.439 | 62.699 | 2.823 | 1.454 | 0.246 | 0.314 |
| small | E2 | 8 | 0.125 | 2.427 | 66.952 | 2.494 | 1.447 | 0.162 | 0.315 |
| small | E2 | 16 | 0.150 | 2.426 | 74.063 | 1.952 | 1.443 | 0.081 | 0.323 |
| medium | E0 | 0 | 0.125 | 2.425 | 93.758 | 2.250 | 0.269 | 0.257 | 0.056 |
| medium | E0 | 1 | 0.125 | 2.410 | 94.777 | 2.182 | 0.269 | 0.255 | 0.051 |
| medium | E0 | 2 | 0.125 | 2.397 | 95.795 | 2.116 | 0.270 | 0.245 | 0.048 |
| medium | E0 | 4 | 0.125 | 2.372 | 97.810 | 1.990 | 0.271 | 0.227 | 0.043 |
| medium | E0 | 8 | 0.125 | 2.321 | 101.826 | 1.761 | 0.273 | 0.195 | 0.036 |
| medium | E0 | 16 | 0.125 | 2.235 | 109.147 | 1.379 | 0.278 | 0.145 | 0.028 |
| medium | E1 | 0 | 0.375 | 2.124 | 88.998 | 1.956 | 6.652 | 0.763 | 3.580 |
| medium | E1 | 1 | 0.375 | 2.126 | 91.402 | 1.870 | 6.612 | 0.870 | 3.507 |
| medium | E1 | 2 | 0.375 | 2.129 | 93.891 | 1.792 | 6.576 | 0.780 | 3.486 |
| medium | E1 | 4 | 0.375 | 2.137 | 99.107 | 1.653 | 6.517 | 0.626 | 3.445 |
| medium | E1 | 8 | 0.375 | 2.156 | 110.371 | 1.426 | 6.433 | 0.403 | 3.368 |
| medium | E1 | 16 | 0.250 | 2.195 | 135.378 | 1.094 | 6.339 | 0.167 | 3.236 |
| medium | E2 | 0 | 0.250 | 2.273 | 82.336 | 3.784 | 6.182 | 0.864 | 4.140 |
| medium | E2 | 1 | 0.250 | 2.268 | 83.081 | 3.653 | 6.186 | 0.947 | 4.111 |
| medium | E2 | 2 | 0.250 | 2.265 | 83.864 | 3.529 | 6.190 | 0.860 | 4.099 |
| medium | E2 | 4 | 0.250 | 2.261 | 85.544 | 3.299 | 6.197 | 0.709 | 4.072 |
| medium | E2 | 8 | 0.250 | 2.263 | 89.311 | 2.893 | 6.211 | 0.483 | 4.010 |
| medium | E2 | 16 | 0.250 | 2.302 | 98.183 | 2.246 | 6.222 | 0.225 | 3.851 |

Mixed-stream stability uses 60% external, 20% NULL and 20% SELF_OUTPUT ticks without reset or state clipping. Every trajectory is saved with first H>100, H>1000, NaN/Inf, read and H-step norms; finite 5000 ticks do not prove asymptotic stability. An in-distribution `alice has the red key` fact is injected; at 100/500/1000/5000 ticks a cloned, off-path state answers its question, recording CE/candidate without altering the continuing stream.

| size | seed | arm | completed | first_H_100 | first_H_1000 | first_bad | H_100 | H_1000 | H_5000 | probe_CE_1000 | probe_correct_1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| small | 2401 | E1 | 5000 | 139 | 881 | None | 67.150 | 1161.674 | 6824.875 | 1.882 | FAIL |
| small | 2402 | E1 | 5000 | 92 | 974 | None | 108.733 | 1029.200 | 5262.299 | 3.418 | FAIL |
| small | 2403 | E1 | 5000 | 101 | 679 | None | 98.994 | 1450.441 | 5157.491 | 2.218 | FAIL |
| small | 2404 | E1 | 5000 | 114 | 1083 | None | 87.730 | 918.483 | 4880.680 | 2.290 | FAIL |
| small | 2405 | E1 | 5000 | 101 | 701 | None | 98.858 | 1431.184 | 3781.801 | 2.107 | FAIL |
| small | 2401 | E2 | 5000 | 148 | 1062 | None | 60.725 | 941.307 | 4372.761 | 1.426 | FAIL |
| small | 2402 | E2 | 5000 | 91 | 656 | None | 112.493 | 1571.951 | 7844.135 | 3.113 | FAIL |
| small | 2403 | E2 | 5000 | 158 | 756 | None | 54.579 | 1308.555 | 6072.481 | 3.049 | FAIL |
| small | 2404 | E2 | 5000 | 88 | 801 | None | 121.015 | 1201.191 | 5099.764 | 2.215 | FAIL |
| small | 2405 | E2 | 5000 | 129 | 1208 | None | 73.028 | 830.855 | 4488.531 | 2.478 | FAIL |
| medium | 2501 | E1 | 5000 | 74 | 478 | None | 162.484 | 1900.958 | 7677.726 | 1.381 | PASS |
| medium | 2501 | E2 | 5000 | 65 | 491 | None | 167.632 | 2142.502 | 11330.604 | 1.322 | FAIL |

### Separate post-training residual-scale diagnostic

A forward hook scales the frozen `core_out` proposal by alpha in one E1/E2 seed. This is an exploratory evaluation-only perturbation, not retraining, and is excluded from G38–G41. Any norm reduction must be read alongside answer accuracy/CE; a changed operator cannot be credited to contextual KV.

| arm | alpha | recall_candidate | answer_CE | H_1000 | H_5000 | first_H_1000 | first_nonfinite |
|---|---|---|---|---|---|---|---|
| E1 | 1.000 | 0.375 | 1.961 | 1161.674 | 6824.875 | 881 | None |
| E1 | 0.500 | 0.375 | 2.290 | 810.326 | 4242.079 | 1221 | None |
| E1 | 0.250 | 0.188 | 3.167 | 517.023 | 2809.586 | 1850 | None |
| E1 | 0.100 | 0.125 | 3.442 | 378.964 | 1893.793 | 2641 | None |
| E2 | 1.000 | 0.375 | 1.980 | 941.307 | 4372.761 | 1062 | None |
| E2 | 0.500 | 0.438 | 2.015 | 591.504 | 2973.783 | 1642 | None |
| E2 | 0.250 | 0.188 | 2.596 | 379.381 | 1998.627 | 2546 | None |
| E2 | 0.100 | 0.188 | 3.592 | 332.748 | 1650.949 | 3002 | None |

## 10. Formal gates and outcome

| gate | PASS | E1 | E2 |
|---|---|---|---|
| G38 | FAIL | FAIL | FAIL |
| G39 | FAIL | FAIL | FAIL |
| G40 | FAIL | FAIL | FAIL |
| G41 | FAIL | FAIL | FAIL |

Selected outcome: **C — contextual KV does not materially improve memory behavior**. G38 requires >=4/5 same-direction seeds and a positive paired bootstrap CI for either ID candidate accuracy or answer CE. G39 requires pair-joint accuracy above 0.25 and >=0.05 gain vs E0 in >=4/5 seeds. G40 requires >=0.03 H-reset peripheral gain in >=4/5. G41 requires positive correct-memory CE benefit against zero/random/shuffled in >=4/5. No gate was altered after inspecting formal outcomes. Outcome A additionally requires a contextual advantage over the E0-late ordering control; observed check=False. M necessity was explicitly excluded.

## 11. Explicit answers and negative results

### Direct 22-question answer matrix

| question | measured answer |
|---|---|
| 1. Contextual KV improves basic LM? | No in this protocol: E0/E1/E2 mean CE 0.795/0.842/0.823; lower is better. |
| 2. Improves associative recall? | No: ID candidate E0/E1/E2 0.433/0.172/0.217; G38=False. |
| 3. E1 or E2 more stable? | E1 has lower five-seed mean H5000 (5181.4); E1/E2 means 5181.4/5575.5, H>1000 in 5/5 and 5/5. Relative advantage does not establish stability. |
| 4. Raw token KV the main bottleneck? | Not established: G38=False, ordering-specific=False; E1/E2 mean ID candidate gain vs E0-late -0.306/-0.261. |
| 5. Counterfactual binding established? | Pair joint E0/E1/E2 0.212/0.094/0.087 vs chance 0.25; G39=False. E0-late mean=0.294 when available. |
| 6. Same tokens, swapped relations distinguished? | Not reliably: correct paired joint accuracy best contextual 0.087; both answers must switch, not just one. |
| 7. Correct memory reduces answer CE? | No aggregate benefit for best arm E2: full CE 2.410, zero CE 2.401. |
| 8. Zero/random/shuffled worse? | Not consistently: best arm E2 CE controls 2.401/2.495/2.448 vs correct 2.410; G41=False. |
| 9. H-reset peripheral benefit? | No reproducible benefit: best arm H-reset candidate 0.503 vs H-reset+FM0 0.500; G40=False. |
| 10. Separate F/M lesions? | Best arm full/F0/M0/FM0 answer CE 2.410/2.411/2.392/2.401; M necessity was not a gate. |
| 11. Lexical OOD benefit? | No: strict OOD candidate E0/E1/E2 0.508/0.165/0.246. |
| 12. GRU still stronger? | Basic LM stronger=True; recall stronger=False. GRU/E0/E2 basic CE 0.575/0.795/0.823; ID candidate 0.175/0.433/0.217. |
| 13. Long-gap advantage? | No contextual advantage: 256+ candidate GRU/E0/E2 0.156/0.315/0.148; this is extrapolation. |
| 14. Semantic organization in KV? | Not established functionally. See same-entity/different-value, same-value/different-entity and cross-relation cosine diagnostics; geometry alone is descriptive. |
| 15. Causal intervention support? | G41=False; best arm E2 pair full/FM-swap CE 2.361/2.354 and unrelated-example full/zero CE 2.410/2.401. Geometry alone is insufficient. |
| 16. Better free generation? | No vs E0: greedy semantic-answer accuracy E0/E1/E2 0.417/0.183/0.233; different data/objective prevent a controlled Stage 2A cross-stage estimate. |
| 17. EOS/punctuation collapse? | Not predominant for E2, but no whole-answer success: greedy EOS 0.000, punctuation-only 0.000, whole-answer exact 0.000. |
| 18. NULL ticks harmful? | No uniform short-horizon deterioration: E2 answer CE K0→K16 2.465→2.426; H still rises 58.11→74.06. |
| 19. H norm growth remains? | H>1000 occurred in E1 5/5 and E2 5/5 mixed streams; mean H5000 5181.4/5575.5. Finite trajectories are not boundedness proofs. |
| 20. Modify H dynamics? | Yes: separately prioritize stability redesign, while preserving the contextual-KV comparison. The exploratory alpha diagnostic reports norm and recall jointly. |
| 21. Priority next? | Given Outcome C, prioritize functional routing/encoding discrimination, generation feedback and H stability before capacity scaling. |
| 22. Larger language prototype justified? | No on this controlled synthetic evidence alone; current gate vector is {'G38': False, 'G39': False, 'G40': False, 'G41': False}. Robust out-of-template binding, useful memory interventions, free generation and long-run stability need replication. |

1. Contextual KV basic LM: compare CE/PPL in Section 3; GRU remains the training-strength reference.
2. Associative recall: best ID candidate gain vs E0 is E2 -0.217; see OOD and CE separately, not only a pooled headline.
3. E1/E2 stability: compare all 5-seed CE/gain and trajectory tables; size-128 has only one seed.
4. Raw-token encoding as Stage 2A's *main* bottleneck is not established without both robust functional gain and an ordering-matched raw control; E0 differs in step order.
5–6. Same-token counterfactual binding and swapped answer identity are judged by pair-joint accuracy, not token frequency or q·k cosine.
7–10. Correct/zero/random/shuffled/F-swap/M-swap effects and H-reset peripheral recall are in Sections 5–6; zero effect or negative benefit is retained rather than dismissed.
11. Lexical OOD effects are in Section 4 and the seed-level gain list; all vocabulary tokens are known.
12–13. GRU and long-gap comparisons are in Section 4. Cross-stage Stage 2A percentages are not directly comparable because data, objective and lengths changed.
14–15. K/V organization is descriptive; finite memory interventions determine whether it is behaviorally useful.
16–17. Generation quality, punctuation/EOS collapse and whole-answer exact are in Section 8; teacher-forced CE is not substituted for successful free answering.
18–20. NULL effects and H drift are in Section 9. Any H stabilization should be a separately matched experiment, not folded into the contextual-KV comparison.
21–22. Prioritize whichever failure is measured: lexical writing if G38 fails, routing/integration if memory interventions are null, long-run H stability if trajectories drift, and SELF_OUTPUT-aware decoding if free generation collapses. Do not scale to a larger language prototype on template-only, single-size evidence.

## 12. Reproduction, artifacts and scientific boundaries

All 1000-step checkpoints, tokenizer metadata, configs, training logs, evaluation records, seed summaries, generation examples, intervention vectors and 5000-tick trajectories are under `results/stage2b/`. The SHA-256 manifest is `results/stage2b/manifest.json`; the Stage 1.5 frozen manifest and Stage 2A commit-byte tests must pass. This controlled synthetic experiment cannot establish LLM capability, human-like thought, general memory, autonomous intelligence, consciousness or infinite context. A negative gate remains negative even if an individual sample looks compelling.
