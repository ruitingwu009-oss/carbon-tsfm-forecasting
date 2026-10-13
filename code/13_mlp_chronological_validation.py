import numpy as np, pandas as pd, pickle, copy, warnings
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
exec(open('common.py').read())
def feats(win):
    lw = np.log(win); r = np.diff(lw)
    return np.concatenate([r[-20:], [lw[-1] - lw[-5:].mean(), lw[-1] - lw[-20:].mean(), lw[-1] - lw[-60:].mean(), r[-20:].std()]])
D='results3/'
rows=[]
for k,df in S.items():
    y=df.close.values.astype(float);N=len(y);split=int(N*.8)
    tr=np.arange(L,split-HMAX)
    Xtr=np.array([feats(y[t-L+1:t+1]) for t in tr]); Ytr=np.array([[np.log(y[t+h]/y[t]) for h in H] for t in tr])*100
    # original (random 10% validation, sklearn default) — refit to confirm reproduction
    sc=StandardScaler().fit(Xtr)
    orig=MLPRegressor(hidden_layer_sizes=(64,32),alpha=1e-3,learning_rate_init=1e-3,max_iter=500,early_stopping=True,random_state=0).fit(sc.transform(Xtr),Ytr)
    # chronological: last 10% of training origins as validation, with a 60-origin gap so that no training target overlaps validation
    nv=int(len(tr)*0.1); vi=np.arange(len(tr)-nv,len(tr)); ti=np.arange(0,len(tr)-nv-HMAX)
    sc2=StandardScaler().fit(Xtr[ti])
    m=MLPRegressor(hidden_layer_sizes=(64,32),alpha=1e-3,learning_rate_init=1e-3,random_state=0)
    best=np.inf;bad=0;bm=None
    for ep in range(500):
        m.partial_fit(sc2.transform(Xtr[ti]),Ytr[ti])
        sc_v=-m.score(sc2.transform(Xtr[vi]),Ytr[vi])
        if sc_v<best-1e-4: best=sc_v;bad=0;bm=copy.deepcopy(m)
        else:
            bad+=1
            if bad>=10: break
    s=pickle.load(open(D+f'forecasts_{k}.pkl','rb')); o=s['origins']
    X=np.array([feats(y[t-L+1:t+1]) for t in o])
    P1=orig.predict(sc.transform(X))/100; P2=bm.predict(sc2.transform(X))/100
    for j,h in enumerate(H):
        ok=o+h<N; yt=y[o[ok]+h]
        for nm,P in [('random_val',P1),('chrono_val',P2)]:
            f=y[o[ok]]*np.exp(P[ok,j]); rows.append(dict(Market=k,Variant=nm,h=h,MAPE=np.mean(np.abs(yt-f)/yt)*100))
        f0=s['P']['MLP'][ok,h-1]; rows.append(dict(Market=k,Variant='paper',h=h,MAPE=np.mean(np.abs(yt-f0)/yt)*100))
    print(k,'epochs chrono',ep+1,flush=True)
R=pd.DataFrame(rows); R.to_csv('mlp_chrono.csv',index=False)
print(R.groupby(['Market','Variant']).MAPE.mean().unstack().round(2))
