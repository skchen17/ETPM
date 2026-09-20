# H/F/M Causal Anatomy — Stage 1.5

Formal run `stage1_5-formal-v1`. Protocol: `reports/STAGE1_5_PROTOCOL.md`; analysis plan: `reports/STAGE1_5_ANALYSIS_PLAN.md`. Raw Parquet: `results/stage1_5/stage1_5-formal-v1/evaluation/`.

## Methods

At identical real external prefix, donor episode `ep−1` supplies exactly the specified H/F/M components. H, F, M, HF, HM, FM, HFM swaps are rolled for 1/2/4/8 NULL ticks. Target CE and prediction JS are paired against intact state; pairwise signed CE interactions are E_AB−E_A−E_B.

## Results

G29: **PASS**. Tick-4 finite effects:

| Swap | Prediction JS | JS 95% CI | Signed CE effect | JS-positive seeds |
| --- | --- | --- | --- | --- |
| F | 0.000281 | [0.0001933045307074721, 0.0003941456427156709] | 0.001789 | 8 |
| FM | 0.001019 | [0.0007899384780500895, 0.001241762079473574] | -0.004451 | 8 |
| H | 0.405584 | [0.37321800835959495, 0.43175217954021716] | 3.441059 | 8 |
| HF | 0.405270 | [0.37262811988246086, 0.43139394105562584] | 3.441104 | 8 |
| HFM | 0.406633 | [0.37426675000483556, 0.43225348403866515] | 3.467070 | 8 |
| HM | 0.406899 | [0.3745059969610566, 0.432681575753395] | 3.466930 | 8 |
| M | 0.000817 | [0.0006163377885428645, 0.001042034862302543] | -0.004714 | 8 |

Signed interactions:

| Interaction | Mean CE | 95% seed CI |
| --- | --- | --- |
| FM | -0.001526 | [-0.002802959045220632, -0.00019411388784647032] |
| HF | -0.001744 | [-0.009987833148625214, 0.006297200353583317] |
| HFM | 0.001621 | [-0.0013335279072634876, 0.004680814454331994] |
| HM | 0.030584 | [0.01106118893949315, 0.048738418071297925] |

## Scope and limitations

Swap effects are finite causal interventions for this checkpoint/world, not a general causal-memory property. Prediction JS, not H distance alone, determines G29.
