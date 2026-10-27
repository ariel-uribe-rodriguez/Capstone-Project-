# Results dashboard

This dashboard reports the **12 weekly evaluations** for eight black-box functions. Source: `weekly_data.xlsx`. The recorded objective values and predictions are exported to `results/weekly_observations.csv`.

**Scope:** The best-observed values are maxima over the **12 weekly observations only**; they do not include the initial experimental design. Leaderboard ranks were supplied separately and are not computed from the workbook.

## Observed convergence

![Convergence by function](figures/convergence.png)

Each function has a separate vertical scale. The line is the running maximum of the observed weekly values, while crosses indicate individual evaluations.

| Function | Dimensions | Best week | Best observed weekly objective | Rank / 39 |
|---|---:|---:|---:|---:|
| F1 | 2 | 7 | 8.6068e-33 | 33 |
| F2 | 2 | 12 | 0.697246 | 13 |
| F3 | 3 | 4 | -0.0342692 | 35 |
| F4 | 4 | 12 | 0.591293 | 19 |
| F5 | 4 | 10 | 6136.6515 | 23 |
| F6 | 5 | 9 | -0.111471 | 9 |
| F7 | 6 | 10 | 2.606740 | 19 |
| F8 | 8 | 12 | 9.966392 | 24 |

## Surrogate diagnostics

![Prediction diagnostics](figures/prediction_diagnostics.png)

The line shows recorded predictive means; the band is mean ± 1.96 times the recorded predictive standard deviation; crosses show actual observations. These are **nominal**, not independently calibrated, intervals. For F5, the workbook's standard-deviation column is much smaller than the observed prediction discrepancies; its scaling and interpretation warrant verification before formal coverage claims.

## Interpretation and limitations

- F2, F4 and F8 achieved their highest **weekly** objective in week 12; F3 peaked in week 4.
- F1 values are extremely close to zero, making percentage prediction errors unsuitable.
- Function objective scales differ substantially; do not compare raw objective magnitudes across functions.
- These plots describe the historical experiment; they do not establish causality or superiority over alternative acquisition strategies.
- To produce complete best-so-far trajectories, incorporate the initial-design observations and validate their provenance.

## Reproducing the figures

From this folder, install `numpy`, `matplotlib` and `artifact_tool`, then run:

```bash
python build_dashboard.py
```

This regenerates both PNG figures and the CSV summaries from `weekly_data.xlsx`.
