# Functional Query Addressing — Stage 1.2

Signed cosine is descriptive only; all conditions clone the same state.

| condition | accuracy | loss | target_retrieval_score | downstream_hidden_change | target_projection | absolute_cosine | squared_projection | nontarget_projection |
|---|---|---|---|---|---|---|---|---|
| original | 0.1445 | 3.6319 | 0.0778 | 2.8814 | 0.2188 | 0.8412 | 0.7090 | 0.5917 |
| parallel | 0.1211 | 3.9055 | 0.0625 | 2.8470 | 0.2188 | 0.8412 | 0.7090 | 0.5917 |
| perpendicular | 0.1270 | 3.5856 | 0.0524 | 2.7730 | 0.2188 | 0.8412 | 0.7090 | 0.5917 |
| random | 0.0605 | 4.3566 | -0.0029 | 2.8576 | 0.2188 | 0.8412 | 0.7090 | 0.5917 |
| signflip | 0.0049 | 5.6551 | -0.0778 | 2.8826 | 0.2188 | 0.8412 | 0.7090 | 0.5917 |
| strongest_nontarget_zero | 0.1465 | 3.6205 | 0.0788 | 2.8690 | 0.2188 | 0.8412 | 0.7090 | 0.5917 |
| target_zero | 0.1270 | 3.5856 | 0.0524 | 2.7730 | 0.2188 | 0.8412 | 0.7090 | 0.5917 |

G14: **FAIL**. Target-removal accuracy drop=0.0176; specificity=0.0195. Historical G7 is unchanged.
