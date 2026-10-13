import pandas as pd, numpy as np, pickle
from scipy import stats
D='results3/'
st={m:pickle.load(open(D+f'forecasts_{m}.pkl','rb')) for m in ['EUA','CEA','Hubei','Guangdong']}
def nw_ols(Y,Xm,lag):
    b=np.linalg.lstsq(Xm,Y,rcond=None)[0]; e=Y-Xm@b
    XtXi=np.linalg.inv(Xm.T@Xm); S0=(Xm*e[:,None]).T@(Xm*e[:,None])
    for l in range(1,lag+1):
        w=1-l/(lag+1); G=(Xm[l:]*e[l:,None]).T@(Xm[:-l]*e[:-l,None]); S0+=w*(G+G.T)
    V=XtXi@S0@XtXi; return b,np.sqrt(np.diag(V))
def build(m,h,model,thr=0.03,win=5,split_persist=False,volscale=None):
    s=st[m]; o=s['origins'];y=s['y'];P=s['P'];N=len(y); r=np.diff(np.log(y))
    ok=o+h<N; oo=o[ok]; yt=y[oo+h]
    d=(np.abs(yt-P[model][ok,h-1])-np.abs(yt-P['Naive'][ok,h-1]))/yt*100
    if volscale:  # threshold = k * rolling 60-day sd of returns ending before window
        J=[float(np.any(np.abs(r[t-win:t])>volscale*np.std(r[t-win-60:t-win]))) for t in oo]
    else:
        J=[float(np.any(np.abs(r[t-win:t])>thr)) for t in oo]
    F=[np.mean(np.abs(r[t-20:t])<1e-9) for t in oo]
    X={'Jump':np.array(J),'Flat20':np.array(F)}
    if split_persist:
        # most recent jump in the window, observed at least one day before the origin, classified by its observed next-day move
        Rv=[];Ps=[]
        for t in oo:
            idx=[i for i in range(t-win,t-1) if abs(r[i])>thr]   # r[i] observed, r[i+1] observed (i+1<=t-1)
            if not idx: Rv.append(0.);Ps.append(0.);continue
            i=idx[-1]; rr=-r[i+1]/r[i]
            Rv.append(float(rr>=0.5)); Ps.append(float(rr<0.5))
        X={'JumpReversed':np.array(Rv),'JumpPersistent':np.array(Ps),'Flat20':np.array(F)}
    return d,X
def reg(markets,h,model,**kw):
    Ys=[];Xs=[];FE=[]
    for j,m in enumerate(markets):
        d,X=build(m,h,model,**kw); Ys.append(d); Xs.append(np.column_stack(list(X.values()))); FE.append(np.full(len(d),j)); names=list(X.keys())
    Y=np.concatenate(Ys); Xm=np.vstack(Xs); fe=np.concatenate(FE)
    dummies=[(fe==j).astype(float) for j in range(len(markets))]
    Xf=np.column_stack(dummies+[Xm])
    b,se=nw_ols(Y,Xf,max(h,5)); k=len(markets)
    res={}
    for i,nm in enumerate(names):
        bb,s=b[k+i],se[k+i]; res[nm]=(bb,s,2*stats.norm.sf(abs(bb/s)))
    return res,len(Y)
if __name__=='__main__':
    rows=[]
    MK=['EUA','CEA','Hubei','Guangdong']
    for model in ['Chronos-2','TimesFM-2.5']:
        for h in [1,10,20]:
            for scope in MK+['Pooled']:
                mk=MK if scope=='Pooled' else [scope]
                res,n=reg(mk,h,model)
                for nm,(b,s,p) in res.items(): rows.append(dict(Model=model,h=h,Scope=scope,Var=nm,coef=b,se=s,p=p,N=n,Spec='baseline'))
                res,n=reg(mk,h,model,split_persist=True)
                for nm,(b,s,p) in res.items(): rows.append(dict(Model=model,h=h,Scope=scope,Var=nm,coef=b,se=s,p=p,N=n,Spec='persistence'))
    R=pd.DataFrame(rows); R.to_csv('mech_table.csv',index=False)
    q=R[(R.Spec=='baseline')&(R.Var=='Jump')]
    print(q.pivot_table(index=['Model','Scope'],columns='h',values=['coef','p']).round(3).to_string())
    q=R[(R.Spec=='persistence')&(R.Var!='Flat20')]
    print(q.pivot_table(index=['Model','Scope','Var'],columns='h',values=['coef','p']).round(3).to_string())
