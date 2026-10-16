# Table 2, Table A1 and Holm-adjusted DM results with rectangular (main) and Bartlett weights (Section 4.2).
import numpy as np, pandas as pd, pickle
from scipy import stats
O='results5/'; MK=['EUA','CEA','Hubei','Guangdong']; H=[1,5,10,20,40,60]
st={m:pickle.load(open(O+f'forecasts_{m}.pkl','rb')) for m in MK}
def dm(d,h,kernel='rect'):
    N=len(d);db=d.mean();g=[np.sum((d[k:]-db)*(d[:N-k]-db))/N for k in range(h)]
    fb=False
    if kernel=='rect':
        v=(g[0]+2*sum(g[1:]))/N
        if v<=0: fb=True
    if kernel=='bart' or fb: v=(g[0]+2*sum((1-k/h)*g[k] for k in range(1,h)))/N
    s=db/np.sqrt(v)*np.sqrt((N+1-2*h+h*(h-1)/N)/N); return s,2*stats.t.sf(abs(s),N-1),fb
def holm(p):
    o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for k,i in enumerate(o): run=max(run,(m-k)*p[i]); adj[i]=min(run,1)
    return adj
met=[];dms=[]
for m,s in st.items():
    o=s['origins'];y=s['y'];P=s['P'];N=len(y)
    for mo in P:
        for h in H:
            ok=o+h<N; yt=y[o[ok]+h]; f=P[mo][ok,h-1]; e=yt-f
            met.append(dict(Market=m,Model=mo,h=h,MAPE=np.mean(np.abs(e)/yt)*100,RMSE=np.sqrt(np.mean(e**2))))
            if mo=='Naive' or mo=='MLP_random_val': continue
            e0=yt-P['Naive'][ok,h-1]; d=e**2-e0**2
            z,p,fb=dm(d,h,'rect'); zb,pb,_=dm(d,h,'bart')
            dms.append(dict(Market=m,Model=mo,h=h,DM=z,p=p,fallback=fb,DM_bart=zb,p_bart=pb))
M=pd.DataFrame(met); Dm=pd.DataFrame(dms)
Dm['p_holm']=Dm.groupby(['Market','Model']).p.transform(lambda s:holm(s.values))
Dm['p_bart_holm']=Dm.groupby(['Market','Model']).p_bart.transform(lambda s:holm(s.values))
M.to_csv('metrics_v5.csv',index=False); Dm.to_csv('dm_v5.csv',index=False)
avg=M.groupby(['Market','Model'])[['MAPE','RMSE']].mean().round(2)
print(avg.unstack(0).to_string())
print('fallbacks',Dm.fallback.sum(),Dm[Dm.fallback][['Market','Model','h']].values.tolist())
old=pd.read_csv('results3/dm_holm.csv') if __import__('os').path.exists('results3/dm_holm.csv') else Dm.assign(DM_old=Dm.DM)
chk=old.merge(Dm,on=['Market','Model','h'],suffixes=('_old',''))
print('non-MLP max diff DM', (chk[chk.Model!='MLP'].DM_old-chk[chk.Model!='MLP'].DM).abs().max())
print(Dm[Dm.Model=='MLP'].pivot_table(index='Market',columns='h',values=['DM','p']).round(3).to_string())
for col,ph in [('p','p_holm'),('p_bart','p_bart_holm')]:
    print(col,'raw<.05',(Dm[col]<.05).sum(),'holm<.05',(Dm[ph]<.05).sum(),'of',len(Dm))
f=Dm[Dm.Model.isin(['Chronos-2','TimesFM-2.5'])]
print(f[(f.p_holm<.05)|(f.p_bart_holm<.05)|(f.p<.05)|(f.p_bart<.05)][['Market','Model','h','DM','p','p_holm','DM_bart','p_bart','p_bart_holm']].round(3).to_string())
print(Dm[(Dm.p_holm<.05)|(Dm.p_bart_holm<.05)][['Market','Model','h','DM','p_holm','DM_bart','p_bart_holm']].round(3).to_string())
