exec(open('common.py').read())
import sys, time
rows=[]; store={}
for k,df in S.items():
    y=df.close.values.astype(float);N=len(y);split=int(N*.8)
    origins=np.arange(max(split,L),N-1)
    full,_=vmd(y,K=5)
    if full.shape[1]<N: full=np.column_stack([full[:,:1].repeat(N-full.shape[1],1),full])
    P={v:np.zeros((len(origins),HMAX)) for v in ['leaky_full','expanding_noresid','expanding_resid']}
    t0=time.time()
    for i,t in enumerate(origins):
        for m in full:
            w=m[t-L+1:t+1];c,phi=fit_ar(w,8);P['leaky_full'][i]+=ar_forecast(w,c,phi,HMAX)
        hist=y[:t+1]; mo,_=vmd(hist,K=5); hh=hist[-mo.shape[1]:]
        for j,m in enumerate(list(mo)+[hh-mo.sum(0)]):
            w=m[-L:];c,phi=fit_ar(w,8);f=ar_forecast(w,c,phi,HMAX)
            P['expanding_resid'][i]+=f
            if j<5: P['expanding_noresid'][i]+=f
        if i%200==0: print(k,i,len(origins),round(time.time()-t0),flush=True)
    store[k]=dict(origins=origins,y=y,P=P)
    for v,A in P.items():
        for h in H:
            ok=origins+h<N;yt=y[origins[ok]+h]
            rows.append(dict(Market=k,Variant=v,h=h,MAPE=np.mean(np.abs((yt-A[ok,h-1])/yt))*100))
    pd.DataFrame(rows).to_csv('leak2.csv',index=False)
pickle.dump(store,open('leak2.pkl','wb'))
print('DONE')
