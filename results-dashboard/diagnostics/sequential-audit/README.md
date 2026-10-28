# Phase 4 — Leakage-free sequential GP audit

This package reconstructs each function's GP *at the historical evaluated point* for Weeks 1–12, fitting only the initial design plus earlier weekly observations. It uses the uploaded eight scripts' literal `x_new`, `y_new` arrays and their matching initial `.npy` arrays.

## Run

```bash
pip install numpy scikit-learn
python sequential_gp_audit.py --functions 5 7 --restarts 10
python sequential_gp_audit.py --restarts 10
```

The included output CSVs are a **fast exploratory reconstruction with 2 optimiser restarts**, not the full original settings. Use `--restarts 10` for most functions; the original F2 script uses 15, F8 uses 5. To match those exactly, run separate commands with `--functions 2 --restarts 15` and `--functions 8 --restarts 5` (note each run overwrites the result CSVs). Kernel fits may vary with restart count. Do not claim exact historical predictions were recovered unless the original contemporaneous model logs agree.

## Data and model interpretation

- F1 uses a signed-log transform and standardisation. Prediction intervals are calculated in the modelled transformed scale, then inverse-transformed. `point_prediction` is the inverse-transformed GP mean, not generally the expectation on the original scale.
- F5 uses a natural-log target, an RBF kernel, and `normalize_y=True`. The 95% interval is computed in log space and exponentiated; `point_prediction` is the lognormal median, while `expected_y` is the lognormal mean under the GP's Gaussian predictive distribution.
- F2–F4 and F6–F8 use standardised targets and Matérn 5/2 kernels. Predictive means and standard deviations are rescaled to objective units.
- The standardised residual (`zscore`) is computed on the *modelled* scale. The interval coverage is checked on the original scale after correct inverse transformation.
- These scripts do not reconstruct the candidate search, historical UCB selection, or the actual model state logged before each experiment. They assess sequential predictions at observed points, assuming chronological ordering of the arrays.
- Original `.npy` arrays have overlapping filenames. They are mapped by dimensions and content, not by filename alone.
- `ConvergenceWarning` messages are counted, not suppressed; investigate kernel hyperparameters at their bounds.
- Only 12 weekly evaluations per function are available; coverage estimates have high sampling uncertainty.

## Exploratory results (2 restarts)

| Function | MAE | RMSE | 95% interval coverage |
|---|---:|---:|---:|
| F1 | 1.88e-7 | 6.05e-7 | 12/12 |
| F2 | 0.1016 | 0.1329 | 10/12 |
| F3 | 0.0481 | 0.0618 | 11/12 |
| F4 | 0.4946 | 0.7885 | 10/12 |
| F5 | 1369.37 | 1896.18 | 12/12 |
| F6 | 0.2260 | 0.2903 | 9/12 |
| F7 | 0.3809 | 0.4822 | 4/12 |
| F8 | 0.0922 | 0.1303 | 11/12 |

F5's earlier reported 0% coverage is not reproduced when prediction intervals are correctly constructed in log space and transformed back. F7's poor coverage persists in this reconstruction. This does not prove that the original weekly forecasts were identical to these reconstructions.

## Suggested GitHub destination

`results-dashboard/diagnostics/sequential-audit/`

The package includes the original uploaded scripts and initial data for reproducibility. Review before publishing.
