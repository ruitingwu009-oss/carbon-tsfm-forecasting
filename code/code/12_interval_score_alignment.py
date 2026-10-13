import pandas as pd, numpy as np, pickle
D='results3/'; Q='results4/'
rng=np.random.default_rng(0); MK=['EUA','CEA','Hubei','Guangdong']
# (month-clustered bootstrap of reversal ratios: see script 14)
# interval score (alpha=0.2), % of price, by horizon and averaged
rows=[]
for m in MK:
    q=pickle.load(open(Q+f'quantiles_{m}.pkl','rb')); o=q['origins'];y=q['y'];N=len(y)
    QQ={'Empirical random walk':q['naive_q'],'Chronos-2':q['chronos_q'],'TimesFM-2.5':q['tfm_qraw'][:,:,1:10]}
    for mo,A in QQ.items():
        for h in [1,5,10,20,40,60]:
            ok=o+h<N; yt=y[o[ok]+h]; lo=A[ok,h-1,0]; hi=A[ok,h-1,-1]
            IS=(hi-lo)+(2/0.2)*(lo-yt)*(yt<lo)+(2/0.2)*(yt-hi)*(yt>hi)
            rows.append(dict(Market=m,Model=mo,h=h,IS_pct=np.mean(IS/yt)*100))
I=pd.DataFrame(rows); I.to_csv('interval_score.csv',index=False)
print(I.groupby(['Market','Model']).IS_pct.mean().unstack().round(2))
# alignment audit: 3 random origins per market, h in {1,20,60}
aud=[]
for m in MK:
    s=pickle.load(open(D+f'forecasts_{m}.pkl','rb')); o=s['origins'];y=s['y'];dt=pd.to_datetime(s['dates']);N=len(y)
    for i in rng.choice(len(o),3,replace=False):
        t=o[i]
        for h in [1,20,60]:
            if t+h<N:
                aud.append(dict(Market=m,origin_index=t,input_start=str(dt[t-255].date()),input_end=str(dt[t].date()),h=h,target_date=str(dt[t+h].date()),
                    actual=y[t+h],naive=s['P']['Naive'][i,h-1],last_input=y[t],chronos2=s['P']['Chronos-2'][i,h-1],timesfm=s['P']['TimesFM-2.5'][i,h-1],arima=s['P']['ARIMA'][i,h-1]))
A=pd.DataFrame(aud); A.to_csv('alignment_audit.csv',index=False)
print('naive==last input:',np.allclose(A.naive,A.last_input)); print(A.head(6).to_string())
