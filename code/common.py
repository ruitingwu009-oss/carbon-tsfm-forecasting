import numpy as np, pandas as pd, pickle
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


X=pd.read_excel("carbon_prices_clean.xlsx",sheet_name=None)
S={"EUA":X["EUA"].query("date >= '2018-01-01'"),"CEA":X["CEA_national"],"Hubei":X["Hubei"],"Guangdong":X["Guangdong"]}
S={k:v.sort_values('date').reset_index(drop=True) for k,v in S.items()}
H=[1,5,10,20,40,60];HMAX=60;L=256
