"""Regenerate F7 figure from the existing Phase 4 CSV; no GP refitting."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
root = Path(__file__).resolve().parent
src = root.parent / 'sequential-audit' / 'results' / 'sequential_predictions.csv'
d = pd.read_csv(src).query('function == 7').copy()
L = np.array([[float(v.strip()) for v in re.search(r'length_scale=\[([^]]+)\]', s).group(1).split(',')] for s in d.kernel])
fig, ax = plt.subplots(3, 1, figsize=(10, 10), layout='constrained')
ax[0].plot(d.week, d.observed, 'o-', label='Observed')
ax[0].plot(d.week, d.point_prediction, 's--', label='Pre-evaluation GP mean')
ax[0].fill_between(d.week, d.lo, d.hi, alpha=.2, label='Nominal 95% interval')
ax[0].legend(ncol=3, loc='upper left'); ax[0].set_ylabel('Objective')
ax[0].set_title('F7 | Sequential predictions and observations')
ax[1].axhline(0, color='gray')
for v in (-1.96, 1.96): ax[1].axhline(v, color='gray', ls='--')
ax[1].scatter(d.week, d.zscore); ax[1].set_ylabel('Standardised residual')
ax[1].set_title('F7 | 4 of 12 intervals contain observations')
for j in range(6): ax[2].plot(d.week, L[:,j], marker='.', label=f'x{j+1}')
ax[2].axhline(5, ls='--', color='gray', alpha=.5)
ax[2].set_ylabel('Fitted length scale'); ax[2].set_xlabel('Week')
ax[2].legend(ncol=6, loc='upper right')
ax[2].set_title('F7 | Matérn 5/2 ARD length scales (upper bound = 5)')
fig.savefig(root / 'f7_sequential_diagnostics.png', dpi=170)
