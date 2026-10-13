import pandas as pd, numpy as np, pickle
exec(open('04_jump_regressions.py').read().split("if __name__")[0])
rng=np.random.default_rng(0)
rows=[]
for m in ['EUA','CEA','Hubei','Guangdong']:
    s=st[m]; o=s['origins'];y=s['y'];r=np.diff(np.log(y))
    test=r[o[0]-1:]   # returns within test period (r[t-1] is return into t)
    tr=r[:o[0]-1]
    for thr in [0.02,0.03,0.04,0.05]:
        idx=np.where(np.abs(test[:-1])>thr)[0]
        rr=-test[idx+1]/test[idx]
        bs=[np.mean(rng.choice(rr,len(rr))) for _ in range(5000)] if len(rr)>1 else [np.nan]
        rows.append(dict(Market=m,thr=thr,n_jumps=len(rr),share_days=len(rr)/len(test)*100,ratio=np.mean(rr) if len(rr) else np.nan,
                         lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5),median=np.median(rr) if len(rr) else np.nan,
                         share_reversed_half=np.mean(rr>=0.5) if len(rr) else np.nan))
    print(m,'sd test %.2f%%  sd train %.2f%%  |r| p90 %.2f%% p95 %.2f%%'%(test.std()*100,tr.std()*100,np.percentile(np.abs(test),90)*100,np.percentile(np.abs(test),95)*100))
J=pd.DataFrame(rows); J.to_csv('jump_ratio.csv',index=False); print(J.round(3).to_string())
# regression sensitivity
out=[]
for model in ['Chronos-2','TimesFM-2.5']:
    for h in [10,20]:
        for thr in [0.02,0.03,0.04,0.05]:
            for win in [3,5,10]:
                res,n=reg(['EUA','CEA','Hubei','Guangdong'],h,model,thr=thr,win=win)
                b,s,p=res['Jump']; out.append(dict(Model=model,h=h,def_=f'{int(thr*100)}%',win=win,coef=b,se=s,p=p,N=n))
        for k in [2.5,3.0]:
            res,n=reg(['EUA','CEA','Hubei','Guangdong'],h,model,win=5,volscale=k)
            b,s,p=res['Jump']; out.append(dict(Model=model,h=h,def_=f'{k}sd',win=5,coef=b,se=s,p=p,N=n))
O=pd.DataFrame(out); O.to_csv('jump_sens.csv',index=False)
print(O[O.Model=='Chronos-2'].pivot_table(index=['def_','win'],columns='h',values=['coef','p']).round(3).to_string())
print(O[O.Model=='TimesFM-2.5'].pivot_table(index=['def_','win'],columns='h',values=['coef','p']).round(3).to_string())
