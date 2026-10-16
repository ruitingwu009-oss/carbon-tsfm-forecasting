# BDS test on raw and AR-filtered returns (Section 3.1, Table 1 note) and EUA returns around December rolls (Section 3.1).
import numpy as np, pandas as pd
from scipy import stats
def bds2(x,distance=1.5):
    x=np.asarray(x,float); n=len(x); eps=distance*x.std(ddof=1)
    I=(np.abs(x[:,None]-x[None,:])<eps).astype(float)
    def csum(J): m=len(J); return J[np.triu_indices(m,1)].mean()
    c=csum(I)
    k=((I.sum(1)**2).sum()-3*I.sum()+2*n)/(n*(n-1)*(n-2))
    m=2; var=4*(k**m+2*sum(k**(m-j)*c**(2*j) for j in range(1,m))+(m-1)**2*c**(2*m)-m**2*k*c**(2*m-2))
    J2=I[1:,1:]*I[:-1,:-1]; c2=csum(J2); c1=csum(I[1:,1:])
    s=np.sqrt(n-1)*(c2-c1**m)/np.sqrt(var); return s,2*stats.norm.sf(abs(s))
def ar_resid(r,maxlag=10):
    best=None
    for p in range(0,maxlag+1):
        Y=r[maxlag:]; Xm=np.column_stack([np.ones(len(Y))]+[r[maxlag-i:-i] for i in range(1,p+1)])
        b=np.linalg.lstsq(Xm,Y,rcond=None)[0]; e=Y-Xm@b; aic=len(Y)*np.log(e@e/len(Y))+2*(p+1)
        if best is None or aic<best[0]: best=(aic,p,e)
    return best[1],best[2]
def lb(e,L=10):
    n=len(e); e=e-e.mean(); a=[np.sum(e[k:]*e[:-k])/np.sum(e*e) for k in range(1,L+1)]
    Q=n*(n+2)*sum(ak**2/(n-k) for k,ak in enumerate(a,1)); return stats.chi2.sf(Q,L)
X=pd.read_excel("carbon_prices_clean.xlsx",sheet_name=None)
S={"EU ETS":X["EUA"].query("date >= '2018-01-01'"),"National ETS":X["CEA_national"],"Hubei":X["Hubei"],"Guangdong":X["Guangdong"]}
rows=[]
for k,df in S.items():
    r=np.diff(np.log(df.sort_values('date').close.values.astype(float)))
    s0,p0=bds2(r); p,e=ar_resid(r); s1,p1=bds2(e); s2,p2=bds2(np.abs(e)) if False else (np.nan,np.nan)
    rows.append(dict(Market=k,BDS_raw=s0,p_raw=p0,AR_order=p,BDS_ARresid=s1,p_ARresid=p1,LB10_resid_p=lb(e)))
R=pd.DataFrame(rows); print(R.round(4).to_string()); R.to_csv('bds_filtered.csv',index=False)

# ---- EUA continuous series: returns around December contract expiries
import pickle
s=pickle.load(open('results3/forecasts_EUA.pkl','rb')); y=s['y']; d=pd.to_datetime(s['dates']); r=np.diff(np.log(y))*100
df=pd.DataFrame({'date':d[1:],'ret_pct':r}); test_start=d[s['origins'][0]]
for yr in range(2018,2026):
    w_=df[(df.date>=f'{yr}-12-01')&(df.date<=f'{yr}-12-31')]
    print(yr,'largest |return| in December:',w_.loc[w_.ret_pct.abs().idxmax()].to_dict())
print('test period starts',test_start.date())
