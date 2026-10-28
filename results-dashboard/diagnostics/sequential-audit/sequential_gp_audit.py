"""Leakage-free sequential GP reconstruction for BBO capstone F1-F8.
Run: python sequential_gp_audit.py --functions 5 7  (or omit --functions for all).
The audit predicts at the ACTUAL evaluated x_t, not necessarily the acquisition's
historical recommendation; historical selection cannot be verified from these files.
"""
import argparse
import ast
import csv
from pathlib import Path
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parent
FILES = {1:'BO_F1-V3.py',2:'BO_F2_V2.py',3:'BO_F3_V2.py',4:'BO_F4_V2.py',5:'BO_F5_V2.py',6:'BO_F6_V2.py',7:'BO_F7_V2.py',8:'BO_F8_V2.py'}
INITIAL = {1:'initial',2:'initial(1)',3:'initial(2)',4:'initial(3)',5:'initial(4)',6:'initial(5)',7:'initial(6)',8:'initial(7)'}

def extract_array(path, name):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            call=node.value
            if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute) or call.func.attr!='array':
                raise ValueError(f'{name} is not a literal np.array in {path}')
            return np.asarray(ast.literal_eval(call.args[0]),dtype=float)
    raise ValueError(f'{name} not found in {path}')

def signed_log(y,eps=1e-8):
    return np.sign(y)*np.log1p(np.abs(y)/eps)/np.log(10)

def inverse_signed_log(z,eps=1e-8):
    return np.sign(z)*eps*np.expm1(np.log(10)*np.abs(z))

def gp_for(f,X,y,restarts):
    d=X.shape[1]
    if f==1:
        target=signed_log(y)
        scaler=StandardScaler().fit(target.reshape(-1,1))
        model_y=scaler.transform(target.reshape(-1,1)).ravel()
        kernel=ConstantKernel(1,(1e-3,1e3))*Matern([.2]*d,(1e-3,50),nu=2.5)+WhiteKernel(1e-8,(1e-12,1e-2))
        alpha=1e-10; normalize=False
    elif f==5:
        if np.any(y<=0): raise ValueError('F5 log requires positive observations')
        model_y=np.log(y); scaler=None
        kernel=ConstantKernel(1,(1e-3,1e3))*RBF([.25]*d,(1e-2,5))+WhiteKernel(1e-6,(1e-10,1e-1))
        alpha=1e-10; normalize=True # sklearn default alpha in original F5
    else:
        scaler=StandardScaler().fit(y.reshape(-1,1))
        model_y=scaler.transform(y.reshape(-1,1)).ravel()
        kernel=ConstantKernel(1,(1e-3,1e3))*Matern([.2]*d,(1e-2,5),nu=2.5)+WhiteKernel(1e-3,(1e-8,1))
        alpha=1e-8; normalize=False
    gp=GaussianProcessRegressor(kernel=kernel,alpha=alpha,normalize_y=normalize,n_restarts_optimizer=restarts,random_state=4)
    gp.fit(X,model_y)
    return gp,scaler

def forecast(f,gp,scaler,x):
    mu,sd=gp.predict(x.reshape(1,-1),return_std=True)
    mu=float(mu[0]); sd=float(sd[0]); z=1.959963984540054
    if f==5:
        # log-normal median and mean; quantile interval is asymmetric in original units
        return dict(mu_model=mu,sd_model=sd,point_prediction=float(np.exp(mu)),expected_y=float(np.exp(mu+.5*sd*sd)),lo=float(np.exp(mu-z*sd)),hi=float(np.exp(mu+z*sd)))
    if f==1:
        mu_z=mu*scaler.scale_[0]+scaler.mean_[0]; sd_z=sd*scaler.scale_[0]
        return dict(mu_model=mu,sd_model=sd,point_prediction=float(inverse_signed_log(mu_z)),expected_y=np.nan,lo=float(inverse_signed_log(mu_z-z*sd_z)),hi=float(inverse_signed_log(mu_z+z*sd_z)))
    mu_y=mu*scaler.scale_[0]+scaler.mean_[0]; sd_y=sd*scaler.scale_[0]
    return dict(mu_model=mu,sd_model=sd,point_prediction=float(mu_y),expected_y=float(mu_y),lo=float(mu_y-z*sd_y),hi=float(mu_y+z*sd_y))

def run_function(f,restarts):
    script=ROOT/'data'/FILES[f]
    xnew=extract_array(script,'x_new'); ynew=extract_array(script,'y_new').ravel()
    stem=INITIAL[f]
    Xi=np.load(ROOT/'data'/(stem+'_inputs.npy')) if stem=='initial' else np.load(ROOT/'data'/('initial_inputs'+stem[len('initial'):]+'.npy'))
    yi=np.load(ROOT/'data'/(stem+'_outputs.npy')) if stem=='initial' else np.load(ROOT/'data'/('initial_outputs'+stem[len('initial'):]+'.npy'))
    yi=yi.ravel()
    assert len(xnew)==len(ynew)==12 and len(Xi)==len(yi) and xnew.shape[1]==Xi.shape[1]
    rows=[]
    for t in range(12):
        X=np.vstack([Xi,xnew[:t]])
        y=np.r_[yi,ynew[:t]]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always',ConvergenceWarning)
            gp,scaler=gp_for(f,X,y,restarts)
        p=forecast(f,gp,scaler,xnew[t]); obs=float(ynew[t]); error=obs-p['point_prediction']
        # standardized errors are in the GP's modeled space
        if f==1: target=float(signed_log(np.array([obs]))[0]); target=(target-scaler.mean_[0])/scaler.scale_[0]
        elif f==5: target=float(np.log(obs))
        else: target=(obs-scaler.mean_[0])/scaler.scale_[0]
        zscore=(target-p['mu_model'])/p['sd_model'] if p['sd_model']>0 else np.nan
        rows.append(dict(function=f,week=t+1,n_train=len(X),observed=obs,error=error,zscore=zscore,covered_95=int(p['lo']<=obs<=p['hi']),kernel=str(gp.kernel_),warnings=len(w),**p))
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--functions',type=int,nargs='+',choices=range(1,9),default=list(range(1,9)));parser.add_argument('--restarts',type=int,default=10);args=parser.parse_args()
    rows=[]
    for f in args.functions:
        print(f'Reconstructing F{f} ...',flush=True)
        rows.extend(run_function(f,args.restarts))
    out=ROOT/'results';out.mkdir(exist_ok=True)
    with (out/'sequential_predictions.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    with (out/'sequential_metrics.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.writer(fh);w.writerow(['function','n','MAE','RMSE','coverage_95','mean_abs_z','warning_count'])
        for f in args.functions:
            r=[q for q in rows if q['function']==f]; e=np.array([q['error'] for q in r]); z=np.array([q['zscore'] for q in r]);
            w.writerow([f,len(r),float(np.mean(abs(e))),float(np.sqrt(np.mean(e**2))),float(np.mean([q['covered_95'] for q in r])),float(np.nanmean(abs(z))),sum(q['warnings'] for q in r)])
    print('Wrote results/sequential_predictions.csv and results/sequential_metrics.csv')
if __name__=='__main__': main()
