import pandas as pd, numpy as np, pickle
D='results3/'
def vr(r,q):
    rq=pd.Series(r).rolling(q).sum().dropna(); return rq.var()/(q*np.var(r))
rows=[]
for m in ['EUA','CEA','Hubei','Guangdong']:
    s=pickle.load(open(D+f'forecasts_{m}.pkl','rb')); o=s['origins'];y=s['y'];r=np.diff(np.log(y));dt=s['dates']
    half=len(o)//2
    for nm,idx in [('first',o[:half]),('second',o[half:])]:
        rr=r[idx[0]-1:idx[-1]]; yy=y[idx[0]:idx[-1]+1]
        j=np.where(np.abs(rr[:-1])>0.03)[0]; rev=-rr[j+1]/rr[j]
        tr20=np.array([np.log(y[t]/y[t-20]) for t in idx])
        rows.append(dict(Market=m,half=nm,start=str(dt[idx[0]])[:10],end=str(dt[idx[-1]])[:10],
            total_change_pct=(yy[-1]/yy[0]-1)*100,sd=rr.std()*100,AC1=pd.Series(rr).autocorr(1),VR20=vr(rr,20),
            mean_abs_trend20=np.mean(np.abs(tr20))*100,share_same_sign_trend=np.mean(np.sign(tr20)==np.sign(np.median(tr20))),
            jumps=len(j),reversal=np.mean(rev) if len(j) else np.nan,zero_share=np.mean(np.abs(rr)<1e-9)*100))
G=pd.DataFrame(rows); pd.set_option('display.width',250); print(G.round(3).to_string()); G.to_csv('subperiod_dynamics.csv',index=False)
