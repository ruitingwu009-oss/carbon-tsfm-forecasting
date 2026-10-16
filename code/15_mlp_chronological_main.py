# Refits the MLP with chronological early-stopping validation (used in Table 2 and Table A1) and writes
# results5/forecasts_<market>.pkl, in which 'MLP' is the chronological-validation MLP and 'MLP_random_val' the original one.
# chronological-validation MLP as main benchmark: full forecast matrices, metrics, DM, Holm
import numpy as np, pandas as pd, pickle, copy, warnings, os
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
warnings.filterwarnings('ignore')
exec(open('common.py').read())
def feats(win):
    lw=np.log(win); r=np.diff(lw)
    return np.concatenate([r[-20:],[lw[-1]-lw[-5:].mean(),lw[-1]-lw[-20:].mean(),lw[-1]-lw[-60:].mean(),r[-20:].std()]])
D='results3/'; O='results5/'; os.makedirs(O,exist_ok=True)
for k,df in S.items():
    y=df.close.values.astype(float);N=len(y);split=int(N*.8)
    tr=np.arange(L,split-HMAX)
    Xtr=np.array([feats(y[t-L+1:t+1]) for t in tr]); Ytr=np.array([[np.log(y[t+h]/y[t]) for h in H] for t in tr])*100
    nv=int(len(tr)*0.1); vi=np.arange(len(tr)-nv,len(tr)); ti=np.arange(0,len(tr)-nv-HMAX)
    sc2=StandardScaler().fit(Xtr[ti])
    m=MLPRegressor(hidden_layer_sizes=(64,32),alpha=1e-3,learning_rate_init=1e-3,random_state=0)
    best=np.inf;bad=0;bm=None
    for ep in range(500):
        m.partial_fit(sc2.transform(Xtr[ti]),Ytr[ti])
        v=-m.score(sc2.transform(Xtr[vi]),Ytr[vi])
        if v<best-1e-4: best=v;bad=0;bm=copy.deepcopy(m)
        else:
            bad+=1
            if bad>=10: break
    s=pickle.load(open(D+f'forecasts_{k}.pkl','rb')); o=s['origins']
    X=np.array([feats(y[t-L+1:t+1]) for t in o]); Pr=bm.predict(sc2.transform(X))/100
    M=np.full((len(o),60),np.nan)
    for j,h in enumerate(H): M[:,h-1]=y[o]*np.exp(Pr[:,j])
    s['P']['MLP_random_val']=s['P']['MLP']; s['P']['MLP']=M
    pickle.dump(s,open(O+f'forecasts_{k}.pkl','wb')); print(k,'epochs',ep+1,flush=True)
