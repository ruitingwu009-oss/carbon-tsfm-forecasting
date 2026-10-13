import pickle, numpy as np, pandas as pd, sys
R4 = sys.argv[1] if len(sys.argv) > 1 else 'results4'
H = [1, 5, 10, 20, 40, 60]
def nw_mean_se(x, lag):
    n = len(x); e = x - x.mean(); g0 = e @ e / n
    v = g0 + 2 * sum((1 - l / (lag + 1)) * (e[l:] @ e[:-l] / n) for l in range(1, lag + 1)) if lag > 0 else g0
    return np.sqrt(max(v, 1e-12) / n)
rows = []
for m in ['EUA', 'CEA', 'Hubei', 'Guangdong']:
    s = pickle.load(open(f'{R4}/quantiles_{m}.pkl', 'rb')); o = s['origins']; y = s['y']; N = len(y)
    Q = {'Empirical random walk': s['naive_q'], 'Chronos-2': s['chronos_q']}
    if 'tfm_qraw' in s: Q['TimesFM-2.5'] = s['tfm_qraw'][:, :, 1:10]   # columns 1..9 = quantiles 0.1..0.9 (column 0 = mean)
    for mo, q in Q.items():
        for h in H:
            ok = o + h < N; yt = y[o[ok] + h]; lo = q[ok, h - 1, 0]; hi = q[ok, h - 1, -1]
            hit = ((yt >= lo) & (yt <= hi)).astype(float); se = nw_mean_se(hit, h - 1)
            rows.append(dict(Market=m, Model=mo, h=h, n=int(ok.sum()), coverage=hit.mean() * 100,
                             lo95=(hit.mean() - 1.96 * se) * 100, hi95=(hit.mean() + 1.96 * se) * 100,
                             width=np.mean((hi - lo) / yt) * 100))
pd.DataFrame(rows).to_csv('coverage_by_horizon.csv', index=False); print(pd.DataFrame(rows).round(1).to_string())
