# Training vs architecture adjudication

| gate   | outcome   |
|:-------|:----------|
| G34    | FAIL      |
| G35    | PASS      |
| G36    | FAIL      |
| G37    | FAIL      |

Oracle integration passed in the present gated-residual operator, but ordinary learned-read training and persistent M necessity did not replicate. The preregistered B3 curriculum had learned-read benefit in 6/8 seeds; a post-formal component dissection found F-lesion benefit in 6/8 but M-lesion benefit in 0/8. This suggests curriculum-assisted fast-memory use, not proven slow persistent-memory integration. Frozen G36=FAIL and G37=FAIL are unchanged.

| effect                                          |        mean | ci95                                             |   invalid_seeds |   positive_seeds |   threshold |
|:------------------------------------------------|------------:|:-------------------------------------------------|----------------:|-----------------:|------------:|
| D_R_3000_minus_160                              | 0.614565    | [4.522779490798712e-05, 1.7387914092485175]      |               0 |                2 |       0.01  |
| D_M_3000_minus_160                              | 0.280345    | [-0.00011986367753706873, 0.6815422703842927]    |               0 |                2 |       0.01  |
| oracle_vs_zero                                  | 2.81491     | [2.0779377373114998, 3.5661207667105086]         |               0 |                8 |       0.05  |
| oracle_vs_random                                | 3.63797     | [2.8793404879515188, 4.468620885442243]          |               0 |                8 |       0.02  |
| oracle_vs_shuffled                              | 4.98985     | [4.128236408240672, 5.805042752040317]           |               0 |                8 |       0.02  |
| no_memory_vs_full                               | 0.436838    | [-4.374398558866232e-05, 1.0397187691125964]     |               0 |                2 |       0.05  |
| M_lesion_vs_full                                | 0.280365    | [4.970747977495194e-06, 0.6814563841453349]      |               0 |                2 |       0.02  |
| curriculum_fraction                             | 1.03255     | [1.0001770284552625, 1.0971926644838594]         |               5 |                3 |       0.5   |
| curriculum_learned_vs_zero                      | 1.63107     | [0.4596438921686929, 2.8540544090032633]         |               0 |                6 |       0.025 |
| curriculum_vs_oracle_trained_fraction_secondary | 0.503372    | [0.12672949486452134, 0.910045675874835]         |               0 |                3 |       0.5   |
| curriculum_F_lesion_secondary                   | 4.26258     | [1.7101605419311596, 6.803371011067173]          |               0 |                6 |       0.02  |
| curriculum_M_lesion_secondary                   | 0.000170435 | [2.5843104012324195e-07, 0.00039943971581664047] |               0 |                0 |       0.02  |
