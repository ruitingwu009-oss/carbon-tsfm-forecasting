import pandas as pd, numpy as np, pickle
exec(open('06b_dm_bartlett.py').read().split("H=[1,5")[0])
A,B=np.datetime64('2025-09-10'),np.datetime64('2026-08-04')
H=[1,5,10,20,40,60]; rows=[]; dms=[]
for m,s in st.items():
    o=s['origins'];y=s['y'];P=s['P'];N=len(y);dt=s['dates'][o]
    sel=(dt>=A)&(dt<=B); oo=o[sel]
    for mo in P:
        for h in H:
            ok=oo+h<N; yt=y[oo[ok]+h]; idx=np.where(sel)[0][ok]
            f=P[mo][idx,h-1]; rows.append(dict(Market=m,Model=mo,h=h,MAPE=np.mean(np.abs(yt-f)/yt)*100,n=ok.sum()))
            if mo in ('Chronos-2','TimesFM-2.5') and h in (1,10,20,40):
                z,p=dm_b((yt-f)**2-(yt-P['Naive'][idx,h-1])**2,h); dms.append(dict(Market=m,Model=mo,h=h,DM=z,p=p))
R=pd.DataFrame(rows); t=R.groupby(['Market','Model']).MAPE.mean().unstack().round(2)
print(t[['Naive','ARIMA','Chronos-2','TimesFM-2.5','VMD-Chronos-2','MLP','GBDT','VMD-AR']]); print(R.groupby('Market').n.max())
Dd=pd.DataFrame(dms); print(Dd.pivot_table(index=['Market','Model'],columns='h',values=['DM','p']).round(3))
R.to_csv('common_period.csv',index=False); Dd.to_csv('common_period_dm.csv',index=False)
