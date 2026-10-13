# Next-day reversal ratio with month-clustered bootstrap intervals (Table 6 and Table A4)
import pandas as pd, numpy as np, pickle
rng = np.random.default_rng(0); D = 'results3/'
lab = {'EUA': 'EU ETS', 'Hubei': 'Hubei', 'Guangdong': 'Guangdong', 'CEA': 'National ETS'}
rows = []
for m in lab:
    s = pickle.load(open(D + f'forecasts_{m}.pkl', 'rb')); o = s['origins']; y = s['y']; dt = pd.to_datetime(s['dates']); r = np.diff(np.log(y))
    base = o[0] - 1; test = r[base:]
    for thr in [0.02, 0.03, 0.04, 0.05]:
        idx = np.where(np.abs(test[:-1]) > thr)[0]; rr = -test[idx + 1] / test[idx]
        mon = dt[base + idx + 1].to_period('M').astype(str).values; cl = pd.unique(mon); g = {c: rr[mon == c] for c in cl}
        bs = [np.concatenate([g[c] for c in rng.choice(cl, len(cl))]).mean() for _ in range(5000)]
        rows.append(dict(Market=lab[m], threshold=thr, n_jumps=len(rr), ratio=rr.mean(), lo95=np.percentile(bs, 2.5), hi95=np.percentile(bs, 97.5)))
pd.DataFrame(rows).to_csv('reversal_ratio_month_bootstrap.csv', index=False); print(pd.DataFrame(rows).round(2).to_string())
