# Conditional auxiliary F: phase-wise oracle-to-learned curriculum

This is a **post-trigger exploratory experiment**, not a fifth gate. It was triggered because formal G35 passed while ordinary learned-read training had finite D_R≥.01 in only 2/8 seeds. The three-phase schedule, total oracle budget, optimizer, seeds, data and controls were fixed in `reports/STAGE1_6_AUX_F_PROTOCOL.md` before training. It uses no query/path labels. The formal B3 five-block curriculum has the same expected 1500 oracle deliveries and is seed-paired here.

## Final condition outcomes

| condition   |      CE |   accuracy |
|:------------|--------:|-----------:|
| F_lesion    | 2.72686 |    0.3999  |
| M_lesion    | 0.67064 |    0.70996 |
| learned     | 0.60737 |    0.72754 |
| oracle      | 0.88223 |    0.65918 |
| random      | 1.92926 |    0.51416 |
| shuffled    | 2.59411 |    0.48486 |
| zero        | 1.53433 |    0.52051 |

## Seed-paired effects

| effect                         |       mean | 95% seed CI                                  |   valid seeds |
|:-------------------------------|-----------:|:---------------------------------------------|--------------:|
| aux_D_R                        |  0.92696   | [0.29697802103025595, 1.7049942494323511]    |             8 |
| aux_D_O                        |  0.652099  | [0.18676061213118517, 1.2921393922490125]    |             8 |
| aux_D_M                        |  0.0632663 | [6.149324380793076e-05, 0.16407101692795506] |             8 |
| aux_D_F                        |  2.11949   | [0.5008935810566101, 3.7078427983965554]     |             8 |
| aux_minus_formal_B3_D_R        | -0.704112  | [-1.5033057727561627, 0.026333539990370806]  |             8 |
| aux_oracle_fraction_valid_only |  1.79606   | [1.1004910783121344, 2.97727504693704]       |             5 |

Auxiliary learned read clears ≥.025 CE benefit in 6/8 seeds. Scrub-boundary M and F lesions clear ≥.02 in 2/8 and 4/8 seeds. The auxiliary-versus-formal B3 D_R difference is descriptive and paired by seed; the frozen G37 outcome is unchanged. An oracle-reintroduction denominator below .05 is reported as missing, never as a pass.

Raw episodes, phase checkpoints, logs, hashes and seed-checkpoint effects are under `results/stage1_6/auxiliary_F/`. This toy result does not authorize a language model or establish persistent M necessity.
