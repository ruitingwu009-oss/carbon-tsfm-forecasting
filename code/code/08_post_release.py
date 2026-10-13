import pandas as pd, numpy as np, pickle
exec(open('06b_dm_bartlett.py').read().split("H=[1,5")[0])
A=np.datetime64('2025-10-31')
H=[1,5,10,20,40,60]; rows=[]; dms=[]
for m,s in st.items():
    o=s['origins'];y=s['y'];P=s['P'];N=len(y);dt=s['dates'][o]
    sel=dt>=A; oo=o[sel]; ids=np.where(sel)[0]
    for mo in ['Naive','ARIMA','Chronos-2','TimesFM-2.5']:
        for h in H:
            ok=oo+h<N; yt=y[oo[ok]+h]; idx=ids[ok]; f=P[mo][idx,h-1]
            rows.append(dict(Market=m,Model=mo,h=h,MAPE=np.mean(np.abs(yt-f)/yt)*100,n=ok.sum()))
            if mo in ('Chronos-2','TimesFM-2.5'):
                z,p=dm_b((yt-f)**2-(yt-P['Naive'][idx,h-1])**2,h); dms.append(dict(Market=m,Model=mo,h=h,DM=z,p=p))
R=pd.DataFrame(rows); print(R.groupby(['Market','Model']).MAPE.mean().unstack().round(2)); print(R[R.h==1].groupby('Market').n.max())
Dd=pd.DataFrame(dms); print(Dd.pivot_table(index=['Market','Model'],columns='h',values='p').round(3)); print(Dd.pivot_table(index=['Market','Model'],columns='h',values='DM').round(2))
R.to_csv('postrelease31.csv',index=False); Dd.to_csv('postrelease31_dm.csv',index=False)
