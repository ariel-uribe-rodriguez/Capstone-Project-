from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'weekly_observations.csv'
if not DATA.exists(): DATA=HERE/'weekly_observations.csv'
with DATA.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
metrics=[]
fig,axes=plt.subplots(4,2,figsize=(12,13))
for k,ax in enumerate(axes.flat,1):
    rr=sorted([r for r in rows if r['function']==f'F{k}'],key=lambda r:int(r['week']))
    assert len(rr)==12
    t=np.array([int(r['week']) for r in rr]); y=np.array([float(r['observed_value']) for r in rr]);mu=np.array([float(r['predicted_mean']) for r in rr]);sd=np.array([float(r['predicted_std']) for r in rr]);e=y-mu
    valid=np.isfinite(sd)&(sd>0)&np.isfinite(e)
    z=np.full(len(y),np.nan);z[valid]=e[valid]/sd[valid]
    cover=np.full(len(y),False);cover[valid]=np.abs(z[valid])<=1.96
    metrics.append(dict(function=f'F{k}',n=len(y),n_valid_std=int(valid.sum()),MAE=float(np.mean(np.abs(e))),RMSE=float(np.sqrt(np.mean(e**2))),coverage_95=float(cover[valid].mean()) if valid.any() else '',mean_interval_width_95=float(np.mean(3.92*sd[valid])) if valid.any() else '',mean_abs_z=float(np.mean(np.abs(z[valid]))) if valid.any() else '',max_abs_z=float(np.max(np.abs(z[valid]))) if valid.any() else ''))
    ax.axhline(0,color='black',lw=.8);ax.axhline(1.96,color='gray',ls='--',lw=.8);ax.axhline(-1.96,color='gray',ls='--',lw=.8)
    ax.scatter(t,z,marker='o',s=24);ax.set_title(f'F{k}: standardised prediction errors',loc='left',fontweight='bold');ax.set_xticks([1,3,5,7,9,12]);ax.set_xlabel('Week');ax.set_ylabel('(Observed − predicted) / std');ax.grid(alpha=.15)
fig.suptitle('Recorded GP uncertainty calibration · 12 weekly evaluations',fontsize=15,fontweight='bold');fig.tight_layout(rect=[0,0,1,.965]);fig.savefig(HERE/'figures'/'standardised_residuals.png',dpi=160);plt.close(fig)
with (HERE/'validation_metrics.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
print('Saved',HERE/'validation_metrics.csv')
for m in metrics:print(m['function'],'MAE',f"{m['MAE']:.4g}",'RMSE',f"{m['RMSE']:.4g}",'coverage',f"{m['coverage_95']:.1%}")
