"""Rebuild publication-ready dashboard from weekly_observations.csv.
Usage: python build_dashboard.py (requires numpy, matplotlib; no Excel dependency).
Running maxima intentionally exclude initial design observations.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
with (ROOT/'weekly_observations.csv').open(newline='',encoding='utf-8-sig') as f:
    records=list(csv.DictReader(f))
by={f'F{i}':sorted((r for r in records if r['function']==f'F{i}'),key=lambda r:int(r['week'])) for i in range(1,9)}
assert all(len(rr)==12 for rr in by.values()), 'Expected 12 observations per function'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
fig,axs=plt.subplots(4,2,figsize=(13,14))
for j,ax in enumerate(axs.flat,1):
    rr=by[f'F{j}']; x=np.array([int(r['week']) for r in rr]); y=np.array([float(r['observed_value']) for r in rr]); best=np.maximum.accumulate(y)
    ax.plot(x,best,marker='o',markersize=3.5,linewidth=2)
    ax.scatter(x,y,marker='x',s=28,alpha=.65)
    ax.set_title(f'F{j}  |  {rr[0]["dimensions"]} variables  |  rank {rr[0]["rank"]}/39',loc='left',fontweight='bold',pad=9)
    ax.set_xlabel('Week');ax.set_ylabel('Objective value');ax.set_xticks([1,3,5,7,9,12]);ax.grid(alpha=.18)
    if j==1: ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
fig.subplots_adjust(top=.905,bottom=.065,left=.09,right=.97,hspace=.58,wspace=.27)
fig.suptitle('Observed convergence across 12 weeks',y=.982,fontsize=18,fontweight='bold')
fig.legend(handles=[Line2D([0],[0],color='C0',marker='o',lw=2,label='Best weekly value so far'),Line2D([0],[0],color='C0',marker='x',lw=0,label='Weekly observation')],loc='upper center',bbox_to_anchor=(.5,.951),ncol=2,frameon=False)
fig.savefig(OUT/'convergence.png',dpi=175);plt.close(fig)

fig,axs=plt.subplots(4,2,figsize=(13,14))
for j,ax in enumerate(axs.flat,1):
    rr=by[f'F{j}']; x=np.array([int(r['week']) for r in rr]);mu=np.array([float(r['predicted_mean']) for r in rr]);sd=np.array([float(r['predicted_std']) for r in rr]);obs=np.array([float(r['observed_value']) for r in rr])
    ax.plot(x,mu,marker='o',linewidth=1.5)
    ax.fill_between(x,mu-1.96*sd,mu+1.96*sd,alpha=.18)
    ax.scatter(x,obs,marker='x',s=33,zorder=3)
    ax.set_title(f'F{j}  |  recorded surrogate predictions',loc='left',fontweight='bold',pad=9)
    ax.set_xlabel('Week');ax.set_ylabel('Objective value');ax.set_xticks([1,3,5,7,9,12]);ax.grid(alpha=.18)
    if j==1:ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
fig.subplots_adjust(top=.905,bottom=.065,left=.09,right=.97,hspace=.58,wspace=.27)
fig.suptitle('Prediction versus observation',y=.982,fontsize=18,fontweight='bold')
fig.legend(handles=[Line2D([0],[0],color='C0',marker='o',lw=1.5,label='Recorded predictive mean'),Line2D([0],[0],color='C0',lw=9,alpha=.18,label='Nominal 95% interval'),Line2D([0],[0],color='C0',marker='x',lw=0,label='Observed value')],loc='upper center',bbox_to_anchor=(.5,.951),ncol=3,frameon=False)
fig.savefig(OUT/'prediction_diagnostics.png',dpi=175);plt.close(fig)

# F1 is near zero: show sign and order of magnitude without inventing a continuous scale.
rr=by['F1'];x=np.array([int(r['week']) for r in rr]);y=np.array([float(r['observed_value']) for r in rr]);
fig,axs=plt.subplots(1,2,figsize=(12,4.6))
ax=axs[0];ax.scatter(x,y,marker='o');ax.axhline(0,color='gray',lw=.8);ax.set_yscale('symlog',linthresh=1e-180,linscale=1);ax.set_xticks(range(1,13));ax.set_xlabel('Week');ax.set_ylabel('Observed objective (symmetric-log scale)');ax.set_title('F1: signed observations',loc='left',fontweight='bold');ax.grid(alpha=.2)
ax=axs[1];magnitudes=np.log10(np.abs(y));ax.scatter(x,magnitudes,marker='o');ax.set_xticks(range(1,13));ax.set_xlabel('Week');ax.set_ylabel(r'$\log_{10}(|f(x)|)$');ax.set_title('F1: orders of magnitude',loc='left',fontweight='bold');ax.grid(alpha=.2)
for xi,yi,mag in zip(x,y,magnitudes):
    if yi<0:ax.annotate('−',(xi,mag),xytext=(0,6),textcoords='offset points',ha='center',fontsize=10)
fig.suptitle('Function 1  |  near-zero objective diagnostics',fontweight='bold',fontsize=15,y=.99)
fig.subplots_adjust(top=.81,bottom=.18,left=.10,right=.98,wspace=.33)
fig.savefig(OUT/'function_1_near_zero.png',dpi=175);plt.close(fig)
print('Created 3 figures from',len(records),'weekly observations')
