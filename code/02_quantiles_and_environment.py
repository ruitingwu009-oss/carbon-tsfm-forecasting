# ===== carbon_tsfm_v4.py : revision run — per-origin quantile forecasts, versions, checkpoint revisions =====

# ===== carbon_tsfm_v3.py : full DM tables, sub-period robustness, Table 1 for all markets, saved forecasts =====
import numpy as np, pandas as pd, time, os, warnings, pickle, torch, shutil
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore'); np.random.seed(42)
OUT = 'results4'; os.makedirs(OUT, exist_ok=True)
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


# ---------- environment record (Reviewer 1, comment 8) ----------
import sys, platform, json, importlib.metadata as md
env = {'python': sys.version.split()[0], 'platform': platform.platform(),
       'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}
for pkg in ['chronos-forecasting', 'timesfm', 'torch', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'statsmodels', 'huggingface_hub', 'transformers']:
    try: env[pkg] = md.version(pkg)
    except Exception as e: env[pkg] = 'n/a'
try:
    from huggingface_hub import HfApi
    api = HfApi()
    for rid in ['amazon/chronos-2', 'google/timesfm-2.5-200m-pytorch']:
        info = api.model_info(rid)
        env[rid] = {'sha': info.sha, 'lastModified': str(info.last_modified), 'createdAt': str(getattr(info, 'created_at', ''))}
except Exception as e:
    env['hf_error'] = str(e)
json.dump(env, open(f'{OUT}/environment.json', 'w'), indent=2); print(json.dumps(env, indent=2))

# ---------- per-origin quantile forecasts (Reviewer 1, comments 5 and 6) ----------
for name, df in series.items():
    t0 = time.time()
    y = df.close.values.astype(float); N = len(y); split = int(N * TRAIN_FRAC)
    origins = np.arange(max(split, L), N - 1); wins = [y[t - L + 1:t + 1] for t in origins]
    out = dict(origins=origins, dates=df.date.values, y=y, QL=QL)
    out['chronos_q'] = chronos_q(wins).astype(np.float32)                    # (n, 60, 9): levels 0.1..0.9
    out['naive_q'] = np.stack([naive_q(y, t, L) for t in origins]).astype(np.float32)
    if tfm is not None:
        pt, qt = tfm.forecast(horizon=HMAX, inputs=wins)
        out['tfm_point'] = np.asarray(pt)[:, :HMAX].astype(np.float32)
        out['tfm_qraw'] = np.asarray(qt)[:, :HMAX, :].astype(np.float32)  # all quantile-head columns, as returned
        qr = out['tfm_qraw']
        for c in range(qr.shape[2]):
            print(f'  TimesFM column {c}: max |point - col| = {np.nanmax(np.abs(out["tfm_point"] - qr[:, :, c])):.6f}')
    pickle.dump(out, open(f'{OUT}/quantiles_{name}.pkl', 'wb'))
    print(name, 'done', f'{time.time()-t0:.0f}s', flush=True)

shutil.make_archive('results4', 'zip', OUT); print('saved results4.zip')
from google.colab import files; files.download('results4.zip')
