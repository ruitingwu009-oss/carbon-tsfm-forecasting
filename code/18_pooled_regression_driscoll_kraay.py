# Pooled jump regressions (Table 7, Table A5) with Newey-West SEs computed within markets and Driscoll-Kraay SEs.
import numpy as np, pandas as pd
from scipy import stats
exec(open('04_jump_regressions.py').read().split("if __name__")[0])
MK=['EUA','CEA','Hubei','Guangdong']
def pooled(h,model,**kw):
    Ys=[];Xs=[];FE=[];DT=[]
    for j,m in enumerate(MK):
        d,X=build(m,h,model,**kw); s=st[m]; o=s['origins']; ok=o+h<len(s['y'])
        Ys.append(d); Xs.append(np.column_stack(list(X.values()))); FE.append(np.full(len(d),j)); DT.append(s['dates'][o[ok]]); names=list(X.keys())
    Y=np.concatenate(Ys); Xm=np.vstack(Xs); fe=np.concatenate(FE); dt=np.concatenate(DT)
    Xf=np.column_stack([(fe==j).astype(float) for j in range(4)]+[Xm]); lag=max(h,5)
    b=np.linalg.lstsq(Xf,Y,rcond=None)[0]; e=Y-Xf@b; XtXi=np.linalg.inv(Xf.T@Xf); U=Xf*e[:,None]
    def nw(Z):
        S0=Z.T@Z
        for l in range(1,lag+1):
            w=1-l/(lag+1); G=Z[l:].T@Z[:-l]; S0+=w*(G+G.T)
        return S0
    out={}
    # (a) original: concatenated
    V={'concat':XtXi@nw(U)@XtXi}
    # (b) market-block HAC (no lags across market boundaries; markets independent)
    S=sum(nw(U[fe==j]) for j in range(4)); V['block']=XtXi@S@XtXi
    # (c) Driscoll-Kraay: sum scores by calendar date (union calendar), HAC over dates
    ud,inv=np.unique(dt,return_inverse=True); Z=np.zeros((len(ud),U.shape[1])); np.add.at(Z,inv,U)
    V['dk']=XtXi@nw(Z)@XtXi
    for i,nm in enumerate(names):
        k=4+i; out[nm]={'coef':b[k],**{f'se_{t}':np.sqrt(v[k,k]) for t,v in V.items()},**{f'p_{t}':2*stats.norm.sf(abs(b[k]/np.sqrt(v[k,k]))) for t,v in V.items()}}
    return out,len(Y)
rows=[]
for model in ['Chronos-2','TimesFM-2.5']:
    for h in [10,20]:
        for spec,kw in [('base',{}),('split',{'split_persist':True})]:
            r,n=pooled(h,model,**kw)
            for nm,v in r.items():
                if nm=='Flat20' and spec=='split': continue
                rows.append(dict(Model=model,h=h,spec=spec,var=nm,N=n,**v))
        for thr in [0.02,0.03,0.04,0.05]:
            for win in [3,5,10]:
                r,n=pooled(h,model,thr=thr,win=win); rows.append(dict(Model=model,h=h,spec=f'sens_{int(thr*100)}%_w{win}',var='Jump',N=n,**r['Jump']))
        for k in [2.5,3.0]:
            r,n=pooled(h,model,win=5,volscale=k); rows.append(dict(Model=model,h=h,spec=f'sens_{k}sd_w5',var='Jump',N=n,**r['Jump']))
R=pd.DataFrame(rows); R.to_csv('pooled_dk.csv',index=False)
pd.set_option('display.width',250)
print(R[R.spec.isin(['base','split'])].round(3).to_string())
s=R[R.spec.str.startswith('sens')]
print(s.pivot_table(index=['spec'],columns=['Model','h'],values='p_dk').round(3).to_string())
print(s.pivot_table(index=['spec'],columns=['Model','h'],values='p_concat').round(3).to_string())
