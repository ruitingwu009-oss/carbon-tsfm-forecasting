
# ===== carbon_tsfm_v3.py : full DM tables, sub-period robustness, Table 1 for all markets, saved forecasts =====
import numpy as np, pandas as pd, time, os, warnings, pickle, torch, shutil
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore'); np.random.seed(42)
OUT = 'results3'; os.makedirs(OUT, exist_ok=True)
H = [1, 5, 10, 20, 40, 60]; HMAX = 60; TRAIN_FRAC = 0.8; K_VMD = 5
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
X = pd.read_excel('carbon_prices_clean.xlsx', sheet_name=None)
series = {'EUA': X['EUA'].query("date >= '2018-01-01'"), 'CEA': X['CEA_national'], 'Hubei': X['Hubei'], 'Guangdong': X['Guangdong']}
series = {k: v.sort_values('date').reset_index(drop=True) for k, v in series.items()}
for k, v in series.items(): print(k, len(v), v.date.min().date(), v.date.max().date())
L = 256
# ---------------- VMD (numpy port of Dragomiretskiy & Zosso 2014) ----------------
def vmd(f, alpha=2000, tau=0.0, K=5, DC=0, init=1, tol=1e-7, Niter=500):
    f = np.asarray(f, float)
    if len(f) % 2: f = f[1:]
    T0 = len(f); fMirr = np.concatenate([np.flip(f[:T0 // 2]), f, np.flip(f[T0 // 2:])])
    T = len(fMirr); t = np.arange(1, T + 1) / T; freqs = t - 0.5 - 1 / T
    Alpha = alpha * np.ones(K)
    f_hat = np.fft.fftshift(np.fft.fft(fMirr)); f_hat_plus = f_hat.copy(); f_hat_plus[:T // 2] = 0
    u_hat_prev = np.zeros((T, K), complex); omega = 0.5 / K * np.arange(K)
    if DC: omega[0] = 0
    lam = np.zeros(T, complex); n = 0; uDiff = tol + 1
    while uDiff > tol and n < Niter:
        u_hat = u_hat_prev.copy()
        sum_uk = u_hat_prev.sum(1) - u_hat_prev[:, 0]
        for k in range(K):
            if k > 0: sum_uk = sum_uk + u_hat[:, k - 1] - u_hat_prev[:, k]
            u_hat[:, k] = (f_hat_plus - sum_uk - lam / 2) / (1 + Alpha[k] * (freqs - omega[k]) ** 2)
            if not (DC and k == 0):
                p = np.abs(u_hat[T // 2:, k]) ** 2
                omega[k] = freqs[T // 2:] @ p / p.sum()
        lam = lam + tau * (u_hat.sum(1) - f_hat_plus)
        uDiff = np.abs(np.sum((u_hat - u_hat_prev) * np.conj(u_hat - u_hat_prev))) / T
        u_hat_prev = u_hat; n += 1
    full = np.zeros((T, K), complex); full[T // 2:] = u_hat_prev[T // 2:]
    idx = np.arange(1, T // 2 + 1)[::-1]; full[idx] = np.conj(u_hat_prev[T // 2:])
    full[0] = np.conj(full[-1])
    u = np.real(np.fft.ifft(np.fft.ifftshift(full, axes=0), axis=0)).T
    u = u[:, T // 4: 3 * T // 4]
    order = np.argsort(omega)
    return u[order], omega[order]

# ---------------- AR helpers ----------------
def fit_ar(y, pmax=5):
    """OLS AR(p) with intercept, p chosen by AIC. Returns (c, phi)."""
    best = None
    for p in range(0, pmax + 1):
        if p == 0:
            c = y.mean(); res = y - c; k = 1; phi = np.array([])
        else:
            Y = y[p:]; Xm = np.column_stack([np.ones(len(Y))] + [y[p - i:-i] for i in range(1, p + 1)])
            b, *_ = np.linalg.lstsq(Xm, Y, rcond=None); res = Y - Xm @ b; c, phi = b[0], b[1:]; k = p + 1
        aic = len(res) * np.log(res.var() + 1e-12) + 2 * k
        if best is None or aic < best[0]: best = (aic, c, phi)
    return best[1], best[2]

def ar_forecast(y, c, phi, h):
    hist = list(y[-len(phi):]) if len(phi) else []; out = []
    for _ in range(h):
        v = c + sum(phi[i] * hist[-1 - i] for i in range(len(phi))); out.append(v); hist.append(v)
    return np.array(out)

# ---------------- models (each returns a path of HMAX forecasts) ----------------
def m_naive(win): return np.repeat(win[-1], HMAX)

def m_arima(win):  # ARIMA(p,1,0) on log prices, p by AIC
    lw = np.log(win); d = np.diff(lw); c, phi = fit_ar(d, 5)
    return np.exp(lw[-1] + np.cumsum(ar_forecast(d, c, phi, HMAX)))

def m_vmd_ar(win, K=5):  # leak-free: VMD on the current window only
    modes, _ = vmd(win, K=K)
    modes = np.vstack([modes, win - modes.sum(0)])  # residual component => exact reconstruction at the window end
    fc = np.zeros(HMAX)
    for m in modes:
        c, phi = fit_ar(m, 8); fc += ar_forecast(m, c, phi, HMAX)
    return fc

def feats(win):
    lw = np.log(win); r = np.diff(lw)
    return np.concatenate([r[-20:], [lw[-1] - lw[-5:].mean(), lw[-1] - lw[-20:].mean(), lw[-1] - lw[-60:].mean(), r[-20:].std()]])

# ---------------- evaluation ----------------
def mape(y, f): return np.mean(np.abs((y - f) / y)) * 100
def mae(y, f): return np.mean(np.abs(y - f))
def rmse(y, f): return np.sqrt(np.mean((y - f) ** 2))

def dm_test(e1, e2, h):
    """Diebold-Mariano with Harvey-Leybourne-Newbold correction, squared loss. Negative => model 1 better."""
    d = e1 ** 2 - e2 ** 2; N = len(d); dbar = d.mean()
    if N <= 2 * h: return np.nan, np.nan
    gam = [np.sum((d[k:] - dbar) * (d[:N - k] - dbar)) / N for k in range(h)]
    v = (gam[0] + 2 * sum(gam[1:])) / N
    if v <= 0: v = (gam[0] + 2 * sum((1 - k / h) * gam[k] for k in range(1, h))) / N   # Bartlett weights if not positive
    dm = dbar / np.sqrt(v) * np.sqrt((N + 1 - 2 * h + h * (h - 1) / N) / N)
    return dm, 2 * stats.t.sf(abs(dm), N - 1)

def adf(y, maxlag=10):
    """ADF with constant; lag by AIC. Returns t-stat (5% critical value = -2.86)."""
    dy = np.diff(y); best = None
    for p in range(maxlag + 1):
        Y = dy[p:]; cols = [np.ones(len(Y)), y[p:-1]] + [dy[p - i:-i] for i in range(1, p + 1)]
        Xm = np.column_stack(cols); b, *_ = np.linalg.lstsq(Xm, Y, rcond=None); res = Y - Xm @ b
        s2 = res @ res / (len(Y) - Xm.shape[1]); se = np.sqrt(s2 * np.linalg.inv(Xm.T @ Xm)[1, 1])
        aic = len(Y) * np.log(res @ res / len(Y)) + 2 * Xm.shape[1]
        if best is None or aic < best[0]: best = (aic, b[1] / se)
    return best[1]


# ---------- foundation models with quantiles ----------
from chronos import Chronos2Pipeline
chronos = Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map=DEVICE)
def chronos_q(windows, h=HMAX, batch=256):
    """returns (n, h, 9) quantiles for QL"""
    out = []
    for s in range(0, len(windows), batch):
        chunk = windows[s:s + batch]
        ctx = pd.concat([pd.DataFrame({'id': i, 'timestamp': pd.bdate_range(end='2030-01-01', periods=len(w)), 'target': np.asarray(w, float)}) for i, w in enumerate(chunk)])
        pdf = chronos.predict_df(ctx, prediction_length=h, quantile_levels=QL, id_column='id', timestamp_column='timestamp', target='target')
        qcols = []
        for q in QL:
            c = [c for c in pdf.columns if str(c) in (str(q), f'{q:.1f}')]
            qcols.append(c[0])
        pdf = pdf.sort_values(['id', 'timestamp'])
        out.append(np.stack([g[qcols].values for _, g in pdf.groupby('id', sort=True)]))
    return np.concatenate(out)
tfm = None
try:
    import timesfm
    tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained('google/timesfm-2.5-200m-pytorch')
    tfm.compile(timesfm.ForecastConfig(max_context=512, max_horizon=128, normalize_inputs=True,
        use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
except Exception as e:
    tfm = None; print('TimesFM skipped:', e)
def tfm_q(windows):
    pt, qt = tfm.forecast(horizon=HMAX, inputs=windows); qt = np.asarray(qt)
    print('TimesFM quantile array shape', qt.shape)
    return qt[:, :HMAX, 1:10]            # columns 1..9 = deciles 0.1..0.9
def naive_q(y, t, L):
    """empirical random-walk quantiles from historical h-day log returns in the window"""
    lw = np.log(y[t - L + 1:t + 1]); q = np.zeros((HMAX, 9))
    for h in range(1, HMAX + 1):
        rh = lw[h:] - lw[:-h]; q[h - 1] = y[t] * np.exp(np.quantile(rh, QL))
    return q
def prob_metrics(yt, Q):
    """Q: (n, 9). pinball loss averaged over levels (x2 ~ CRPS), 80% coverage & width"""
    pin = np.mean([np.mean(np.maximum(q * (yt - Q[:, i]), (q - 1) * (yt - Q[:, i]))) for i, q in enumerate(QL)]) * 2
    cov = np.mean((yt >= Q[:, 0]) & (yt <= Q[:, -1])) * 100
    wid = np.mean((Q[:, -1] - Q[:, 0]) / yt) * 100
    return pin, cov, wid


# ---------- PART 0: Table 1 statistics for all four markets ----------
from statsmodels.tsa.stattools import adfuller, bds
desc = []
for name, df in series.items():
    y = df.close.values.astype(float); N = len(y); split = int(N * TRAIN_FRAC)
    desc.append(dict(Market=name, Start=str(df.date.min().date()), End=str(df.date.max().date()), N=N,
        Mean=y.mean(), Std=y.std(), Min=y.min(), Max=y.max(), Skew=stats.skew(y), Kurt=stats.kurtosis(y),
        ADF_level_p=adfuller(y)[1], ADF_diff_p=adfuller(np.diff(y))[1],
        BDS_m2_p=float(np.atleast_1d(bds(np.diff(np.log(y)), max_dim=2)[1])[0]),
        Train=split, Test_start=str(df.date.iloc[max(split, L)].date())))
T1 = pd.DataFrame(desc).round(4); T1.to_csv(f'{OUT}/table1_descriptive_4markets.csv', index=False); print(T1.to_string())

# ---------- PART 1: all models, all markets (L = 256); forecasts are saved ----------
store, metrics, dm_rows, sub_rows, sub_dm = {}, [], [], [], []
for name, df in series.items():
    t0 = time.time()
    y = df.close.values.astype(float); dates = df.date.values; N = len(y); split = int(N * TRAIN_FRAC)
    origins = np.arange(max(split, L), N - 1); wins = [y[t - L + 1:t + 1] for t in origins]
    print(f'\n=== {name}: {len(origins)} origins', flush=True)
    tr = np.arange(L, split - HMAX)   # ML models: trained ONCE on the training period only
    Xtr = np.array([feats(y[t - L + 1:t + 1]) for t in tr]); Ytr = np.array([[np.log(y[t + h] / y[t]) for h in range(1, HMAX + 1)] for t in tr])
    sc = StandardScaler().fit(Xtr)
    gbdt = {h: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0).fit(Xtr, Ytr[:, h - 1]) for h in H}
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=500, early_stopping=True, random_state=0).fit(sc.transform(Xtr), Ytr[:, [h - 1 for h in H]] * 100)
    P = {m: np.full((len(origins), HMAX), np.nan) for m in ['Naive', 'ARIMA', 'GBDT', 'MLP', 'VMD-AR']}
    for i, w in enumerate(wins):   # ARIMA and VMD-AR: re-estimated at every origin on the last L observations
        P['Naive'][i] = m_naive(w); P['ARIMA'][i] = m_arima(w); P['VMD-AR'][i] = m_vmd_ar(w, K_VMD)
        x = feats(w)[None]; mo = mlp.predict(sc.transform(x))[0] / 100
        for j, h in enumerate(H):
            P['GBDT'][i, h - 1] = w[-1] * np.exp(gbdt[h].predict(x)[0]); P['MLP'][i, h - 1] = w[-1] * np.exp(mo[j])
    print('  classical done', f'{time.time()-t0:.0f}s', flush=True)
    P['Chronos-2'] = chronos_q(wins)[:, :, 4]
    modes = []
    for w in wins:
        u = vmd(w, K=K_VMD)[0]; modes.append(np.vstack([u, w - u.sum(0)]))
    flat = [m for ms in modes for m in ms]
    P['VMD-Chronos-2'] = chronos_q(flat)[:, :, 4].reshape(len(wins), K_VMD + 1, HMAX).sum(1)
    if tfm is not None: P['TimesFM-2.5'] = np.asarray(tfm.forecast(horizon=HMAX, inputs=wins)[0])[:, :HMAX]
    print('  foundation models done', f'{time.time()-t0:.0f}s', flush=True)
    store[name] = dict(origins=origins, dates=dates, y=y, P=P)
    pickle.dump(store[name], open(f'{OUT}/forecasts_{name}.pkl', 'wb'))
    # full-sample metrics and DM tests at ALL horizons
    for m, A in P.items():
        for h in H:
            ok = origins + h < N; yt = y[origins[ok] + h]; ft = A[ok, h - 1]
            metrics.append(dict(Market=name, Model=m, h=h, MAPE=mape(yt, ft), MAE=mae(yt, ft), RMSE=rmse(yt, ft), n=int(ok.sum())))
            for ref in ['Naive', 'ARIMA']:
                if m in ('Naive', ref): continue
                dm, p = dm_test(yt - A[ok, h - 1], yt - P[ref][ok, h - 1], h)
                dm_rows.append(dict(Market=name, Model=m, vs=ref, h=h, DM=dm, p=p))
    # sub-period robustness: split the test origins into two halves of equal length
    half = len(origins) // 2
    for s, idx in [('first', np.arange(half)), ('second', np.arange(half, len(origins)))]:
        o = origins[idx]; per = f'{str(dates[o[0]])[:10]} to {str(dates[o[-1]])[:10]}'
        for m, A in P.items():
            for h in H:
                ok = o + h < N; yt = y[o[ok] + h]; ft = A[idx][ok, h - 1]
                sub_rows.append(dict(Market=name, Subperiod=s, Period=per, Model=m, h=h, MAPE=mape(yt, ft), n=int(ok.sum())))
                if m != 'Naive' and h in (1, 10, 20, 40):
                    dm, p = dm_test(yt - ft, yt - P['Naive'][idx][ok, h - 1], h)
                    sub_dm.append(dict(Market=name, Subperiod=s, Period=per, Model=m, h=h, DM=dm, p=p))
    print(f'  {name} finished in {time.time()-t0:.0f}s', flush=True)

M = pd.DataFrame(metrics); M.to_csv(f'{OUT}/metrics_all_4markets.csv', index=False)
D = pd.DataFrame(dm_rows); D.to_csv(f'{OUT}/dm_all_horizons.csv', index=False)
S = pd.DataFrame(sub_rows); S.to_csv(f'{OUT}/subperiod_mape.csv', index=False)
SD = pd.DataFrame(sub_dm); SD.to_csv(f'{OUT}/subperiod_dm.csv', index=False)
t = M.pivot_table(index=['Market', 'Model'], columns='h', values='MAPE'); t['Avg'] = t.mean(1); print(t.round(2))
print(D[D.vs == 'Naive'].pivot_table(index=['Market', 'Model'], columns='h', values='p').round(3))
s = S.groupby(['Market', 'Subperiod', 'Period', 'Model']).MAPE.mean().unstack('Model').round(2); print(s)
shutil.make_archive('results3', 'zip', OUT); print('saved results3.zip')
from google.colab import files; files.download('results3.zip')
