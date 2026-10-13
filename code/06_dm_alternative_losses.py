import pandas as pd, numpy as np, pickle
from scipy import stats
D='results3/'
st={m:pickle.load(open(D+f'forecasts_{m}.pkl','rb')) for m in ['EUA','CEA','Hubei','Guangdong']}
def dm(d,h):
    N=len(d);db=d.mean()
    g=[np.sum((d[k:]-db)*(d[:N-k]-db))/N for k in range(h)]
    v=(g[0]+2*sum(g[1:]))/N
    if v<=0: v=(g[0]+2*sum((1-k/h)*g[k] for k in range(1,h)))/N
    s=db/np.sqrt(v)*np.sqrt((N+1-2*h+h*(h-1)/N)/N); return s,2*stats.t.sf(abs(s),N-1)
H=[1,5,10,20,40,60]; rows=[]
for m,s in st.items():
    o=s['origins'];y=s['y'];P=s['P'];N=len(y)
    for mo in P:
        if mo=='Naive':continue
        for h in H:
            ok=o+h<N;yt=y[o[ok]+h];e1=yt-P[mo][ok,h-1];e0=yt-P['Naive'][ok,h-1]
            for loss,f in [('squared',lambda e:e**2),('absolute',np.abs),('abs_pct',lambda e:np.abs(e)/yt*100)]:
                d=f(e1)-f(e0); z,p=dm(d,h); rows.append(dict(Market=m,Model=mo,h=h,loss=loss,DM=z,p=p))
R=pd.DataFrame(rows); R.to_csv('dm_losses.csv',index=False)
def c(r): return f"{r.DM:.2f}"+('***' if r.p<.01 else '**' if r.p<.05 else '*' if r.p<.1 else '')
R['cell']=R.apply(c,axis=1)
q=R[R.Model.isin(['Chronos-2','TimesFM-2.5'])]
print(q.pivot_table(index=['Market','Model','loss'],columns='h',values='cell',aggfunc='first').to_string())
# sign/significance agreement
w=R.pivot_table(index=['Market','Model','h'],columns='loss',values='p')
z=R.pivot_table(index=['Market','Model','h'],columns='loss',values='DM')
sig=lambda L:(w[L]<.1)*np.sign(z[L])
for L in ['absolute','abs_pct']: print(L,'agreement with squared (10% sig+sign):',(sig(L)==sig('squared')).mean().round(3))
