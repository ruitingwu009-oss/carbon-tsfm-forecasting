import pandas as pd, numpy as np, pickle
from scipy import stats
D='results3/'; Q='results4/'
MK=['EUA','CEA','Hubei','Guangdong']
st={m:pickle.load(open(D+f'forecasts_{m}.pkl','rb')) for m in MK}
exec(open('06b_dm_bartlett.py').read().split("H=[1,5")[0])
H=[1,5,10,20,40,60]
# 1) post-release from 31 Oct 2025
A=np.datetime64('2025-10-31'); rows=[];dms=[]
for m,s in st.items():
    o=s['origins'];y=s['y'];P=s['P'];N=len(y);dt=s['dates'][o]; sel=dt>=A; oo=o[sel]; ids=np.where(sel)[0]
    for mo in ['Naive','ARIMA','Chronos-2','TimesFM-2.5']:
        for h in H:
            ok=oo+h<N; yt=y[oo[ok]+h]; idx=ids[ok]; f=P[mo][idx,h-1]
            rows.append(dict(Market=m,Model=mo,h=h,MAPE=np.mean(np.abs(yt-f)/yt)*100,n=ok.sum()))
            if mo in ('Chronos-2','TimesFM-2.5'):
                z,p=dm_b((yt-f)**2-(yt-P['Naive'][idx,h-1])**2,h); dms.append(dict(Market=m,Model=mo,h=h,DM=z,p=p))
R=pd.DataFrame(rows); R.to_csv('postrelease31.csv',index=False); Dd=pd.DataFrame(dms); Dd.to_csv('postrelease31_dm.csv',index=False)
print(R.groupby(['Market','Model']).MAPE.mean().unstack().round(2)); print(R[R.h==1].groupby('Market').n.max())
print(Dd.pivot_table(index=['Market','Model'],columns='h',values='p').round(3))
# 2) Holm within market-model family (main squared-loss tests, original estimator)
old=pd.read_csv(D+'dm_all_horizons_v2.csv'); old=old[old.vs=='Naive'].copy()
def holm(p):
    o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for k,i in enumerate(o): run=max(run,(m-k)*p[i]); adj[i]=min(run,1)
    return adj
old['p_holm']=old.groupby(['Market','Model']).p.transform(lambda s:holm(s.values))
old.to_csv('dm_holm.csv',index=False)
t=old[old.Model.isin(['Chronos-2','TimesFM-2.5'])]
print(t.pivot_table(index=['Market','Model'],columns='h',values='p_holm').round(3))
print('sig5 raw',(old.p<.05).sum(),'holm',(old.p_holm<.05).sum(),'of',len(old))
