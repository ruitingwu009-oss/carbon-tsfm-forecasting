# Table A8: common calendar period with origins only and with origins and targets restricted to the window;
# post-publication sample with origins by horizon.
import numpy as np, pandas as pd, pickle
from scipy import stats
exec(open('16_metrics_dm_holm_both_estimators.py').read().split('met=[];dms=[]')[0])
A,B=np.datetime64('2025-09-10'),np.datetime64('2026-08-04')
rows=[];dms=[]
for design in ['origin','origin_target']:
    for m,s in st.items():
        o=s['origins'];y=s['y'];P=s['P'];N=len(y);dts=s['dates']
        for h in H:
            sel=(dts[o]>=A)&(dts[o]<=B)&(o+h<N)
            if design=='origin_target': sel&=np.r_[[dts[t+h]<=B if t+h<N else False for t in o]]
            idx=np.where(sel)[0]; oo=o[idx]; yt=y[oo+h]
            for mo in ['Naive','ARIMA','MLP','Chronos-2','TimesFM-2.5']:
                f=P[mo][idx,h-1]; rows.append(dict(design=design,Market=m,Model=mo,h=h,MAPE=np.mean(np.abs(yt-f)/yt)*100,n=len(idx),
                    o_first=str(dts[oo[0]])[:10],o_last=str(dts[oo[-1]])[:10],t_first=str(dts[oo[0]+h])[:10],t_last=str(dts[oo[-1]+h])[:10]))
                if mo in ('Chronos-2','TimesFM-2.5'):
                    z,p,_=dm((yt-f)**2-(yt-P['Naive'][idx,h-1])**2,h,'bart'); dms.append(dict(design=design,Market=m,Model=mo,h=h,DM=z,p=p))
R=pd.DataFrame(rows); Dd=pd.DataFrame(dms); R.to_csv('common_period_v5.csv',index=False); Dd.to_csv('common_period_dm_v5.csv',index=False)
for d in ['origin','origin_target']:
    print('==',d); print(R[R.design==d].groupby(['Market','Model']).MAPE.mean().unstack().round(2).to_string())
    print(R[(R.design==d)&(R.Model=='Naive')][['Market','h','n','o_first','o_last','t_first','t_last']].to_string(index=False))
    print(Dd[Dd.design==d].pivot_table(index=['Market','Model'],columns='h',values='p').round(3).to_string())

# ---- post-publication sample
A=np.datetime64('2025-10-31'); rows=[]
for m,s in st.items():
    o=s['origins'];y=s['y'];P=s['P'];N=len(y);dts=s['dates']
    for h in H:
        idx=np.where((dts[o]>=A)&(o+h<N))[0]; oo=o[idx]; yt=y[oo+h]
        for mo in ['Naive','ARIMA','Chronos-2','TimesFM-2.5']:
            rows.append(dict(Market=m,Model=mo,h=h,n=len(idx),MAPE=np.mean(np.abs(yt-P[mo][idx,h-1])/yt)*100,t_last=str(dts[oo[-1]+h])[:10]))
R=pd.DataFrame(rows); print(R.groupby(['Market','Model']).MAPE.mean().unstack().round(2)); print(R[R.Model=='Naive'].pivot_table(index='Market',columns='h',values='n'))
R.to_csv('postrelease_v5.csv',index=False)
