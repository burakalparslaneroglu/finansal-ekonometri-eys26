"""
generate_sample.py (upgraded / "edit")
======================================
Upgraded synthetic data-generating process (DGP) for the EYS'26 Day-3 module.

Design goals (vs. the previous constant-correlation generator):
  * Genuine ADCC dynamic conditional correlation  -> DCC/cDCC/ADCC are estimable
  * Asymmetry (c_true > 0)                         -> ADCC beats DCC (LR test)
  * A designed crisis window (COVID-like)          -> correlation & vol spike
  * Student-t innovations (nu ~ 7)                  -> excess kurtosis / fat tails
  * Heterogeneous univariate GARCH(1,1) per asset  -> realistic persistence spread
  * Sector block structure in unconditional corr   -> economically interpretable

Output schema matches the teaching app (returns + RV + BPV, business-day index),
so this file is a drop-in replacement for data/generate_sample.py.

The main time loop is intrinsically sequential (each Q_t depends on z_{t-1}) and
short (T=1500), so it is written in vectorised NumPy for RNG reproducibility;
the expensive estimation/backtest steps use the numba-jitted DCC loop elsewhere.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SEED         = 20260729
T            = 1500                      # business days (~6 years)
ASSETS       = ["BANKA", "SANAYI", "HOLD", "GAYRIM",
                "TEKNOLOJI", "ENERJI", "PERAK", "OTOMOT"]
N            = len(ASSETS)
N_INTRADAY   = 78                        # 5-min bars in a 6.5h session
START_DATE   = "2018-01-02"

# True ADCC correlation-dynamics parameters
A_TRUE, B_TRUE, C_TRUE = 0.040, 0.930, 0.030     # persistence a+b = 0.970
NU                     = 7.0                       # Student-t d.o.f.

# Per-asset GARCH(1,1): heterogeneous (alpha, beta) and target annual vol
GARCH_ALPHA = np.array([0.085, 0.060, 0.072, 0.055, 0.050, 0.078, 0.065, 0.058])
GARCH_BETA  = np.array([0.895, 0.925, 0.912, 0.930, 0.940, 0.905, 0.921, 0.933])
ANN_VOL     = np.array([0.29, 0.23, 0.26, 0.22, 0.27, 0.30, 0.24, 0.25])   # annualised
ANN_DRIFT   = np.array([0.095,0.090,0.092,0.088,0.098,0.094,0.090,0.093]) # ~homojen

# Crisis window (COVID-like): a common market crash + correlation-attractor shift
CRISIS_START = 555                        # ~ mid-March 2020 given START_DATE
CRISIS_PEAK  = 560
CRISIS_WIDTH = 5.5                         # Gaussian width (days) of the crash
CRISIS_LEN   = 35                          # elevated-attractor window length
CRASH_AMP    = 2.0                         # peak common shock (in z-units, negative)
VOL_MULT_PEAK = 2.7                        # peak exogenous volatility multiplier m_t

rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------
# Correlation targets
# ----------------------------------------------------------------------------
def _psd_correlation(C: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the set of valid correlation matrices."""
    C = 0.5 * (C + C.T)
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 1e-8, None)
    C = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


def build_normal_corr() -> np.ndarray:
    rho = 0.35
    R = rho * np.ones((N, N)) + (1.0 - rho) * np.eye(N)
    # Sector blocks (indices: 0 BANKA,1 SANAYI,2 HOLD,3 GAYRIM,4 TEKNO,5 ENERJI,6 PERAK,7 OTO)
    R[0, 1] = R[1, 0] = 0.72   # BANKA-SANAYI (financial-industrial)
    R[0, 2] = R[2, 0] = 0.68   # BANKA-HOLD
    R[2, 3] = R[3, 2] = 0.60   # HOLD-GAYRIM
    R[4, 5] = R[5, 4] = 0.55   # TEKNOLOJI-ENERJI
    R[6, 7] = R[7, 6] = 0.63   # PERAK-OTOMOT
    R[1, 7] = R[7, 1] = 0.58   # SANAYI-OTOMOT
    return _psd_correlation(R)


def build_crisis_corr() -> np.ndarray:
    """High, fairly homogeneous correlation attractor during systemic stress."""
    rho = 0.80
    R = rho * np.ones((N, N)) + (1.0 - rho) * np.eye(N)
    return _psd_correlation(R)


# ----------------------------------------------------------------------------
# Standardised multivariate-t draw with conditional correlation R
# ----------------------------------------------------------------------------
def draw_std_t(R_chol: np.ndarray, nu: float) -> np.ndarray:
    """Return z with Cov(z)=R and standardised Student-t marginals."""
    g = R_chol @ rng.standard_normal(N)          # MVN(0, R)
    w = rng.chisquare(nu)
    x = g / np.sqrt(w / nu)                        # cov = nu/(nu-2) * R
    return x * np.sqrt((nu - 2.0) / nu)            # cov = R


# ----------------------------------------------------------------------------
# Main DGP
# ----------------------------------------------------------------------------
def simulate():
    Rbar_n = build_normal_corr()
    Rbar_c = build_crisis_corr()

    # GARCH intercepts for target unconditional variance
    daily_var = (ANN_VOL / np.sqrt(252.0)) ** 2
    mu_daily  = ANN_DRIFT / 252.0
    omega = daily_var * (1.0 - GARCH_ALPHA - GARCH_BETA)

    tt = np.arange(T)
    # Exogenous, sustained crisis volatility multiplier m_t (asymmetric decay:
    # sharp rise, slower normalisation), producing weeks of elevated volatility.
    up  = np.exp(-((tt - CRISIS_PEAK) / CRISIS_WIDTH) ** 2)
    dn  = np.exp(-((tt - CRISIS_PEAK) / (CRISIS_WIDTH * 2.4)) ** 2)
    m   = 1.0 + (VOL_MULT_PEAK - 1.0) * np.where(tt <= CRISIS_PEAK, up, dn)
    m[tt < CRISIS_START] = 1.0
    m[tt > CRISIS_START + CRISIS_LEN] = 1.0
    # Small common negative shock (seeds joint co-movement / ADCC asymmetry)
    crash = -CRASH_AMP * up
    crash[tt < CRISIS_START] = 0.0
    crash[tt > CRISIS_START + CRISIS_LEN] = 0.0
    crash_load = np.linspace(0.92, 1.08, N)

    returns = np.zeros((T, N))
    sigmas  = np.zeros((T, N))
    R_path  = np.zeros((T, N, N))
    z_store = np.zeros((T, N))

    # Initialise
    Q = Rbar_n.copy()
    R_path[0] = Rbar_n
    z_prev = draw_std_t(np.linalg.cholesky(Rbar_n), NU)
    z_store[0] = z_prev
    sig2 = daily_var.copy()
    sigmas[0] = np.sqrt(sig2)
    returns[0] = mu_daily + sigmas[0] * z_prev

    for t in range(1, T):
        in_crisis = (CRISIS_START <= t <= CRISIS_START + CRISIS_LEN)
        Rbar_t = Rbar_c if in_crisis else Rbar_n

        n_prev = z_prev * (z_prev < 0.0)
        Q = ((1.0 - A_TRUE - B_TRUE) * Rbar_t
             + A_TRUE * np.outer(z_prev, z_prev)
             + B_TRUE * Q
             + C_TRUE * np.outer(n_prev, n_prev))
        R_t = _psd_correlation(Q)
        R_path[t] = R_t

        z = draw_std_t(np.linalg.cholesky(R_t), NU)
        if in_crisis:
            z = z + crash[t] * crash_load          # common market crash
        z_store[t] = z

        # Univariate GARCH(1,1) driven by CRISIS-FREE returns (divide out m),
        # then overlay the exogenous crisis multiplier on the conditional vol.
        r_prev_cf = (returns[t - 1] - mu_daily) / m[t - 1]
        sig2 = omega + GARCH_ALPHA * r_prev_cf ** 2 + GARCH_BETA * sig2
        sigmas[t] = np.sqrt(sig2) * m[t]
        returns[t] = mu_daily + sigmas[t] * z
        z_prev = z

    dates = pd.bdate_range(start=START_DATE, periods=T)
    df_ret = pd.DataFrame(returns, index=dates, columns=ASSETS)

    # ---- Synthetic realised measures via a genuine intraday aggregation ----
    # Each day: N_INTRADAY log-returns with total daily variance sigma_t^2,
    # plus i.i.d. microstructure noise; RV = sum r_intra^2, BPV = bipower.
    rv  = np.zeros((T, N))
    bpv = np.zeros((T, N))
    mn_scale = 0.20                                   # microstructure noise scale
    for i in range(N):
        intra_sig = sigmas[:, i, None] / np.sqrt(N_INTRADAY)       # (T,1)
        eff = rng.standard_normal((T, N_INTRADAY)) * intra_sig       # efficient
        noise = rng.standard_normal((T, N_INTRADAY)) * intra_sig * mn_scale
        obs = eff + noise
        rv[:, i]  = np.sum(obs ** 2, axis=1)
        abs_obs = np.abs(obs)
        bpv[:, i] = (np.pi / 2.0) * np.sum(abs_obs[:, 1:] * abs_obs[:, :-1], axis=1)

    df_rv  = pd.DataFrame(rv,  index=dates, columns=[f"{a}_RV"  for a in ASSETS])
    df_bpv = pd.DataFrame(bpv, index=dates, columns=[f"{a}_BPV" for a in ASSETS])

    return df_ret, df_rv, df_bpv, R_path, dates


if __name__ == "__main__":
    df_ret, df_rv, df_bpv, R_path, dates = simulate()

    print("=" * 68)
    print("DGP DİAGNOSTİKLERİ")
    print("=" * 68)
    print(f"Boyut: T={df_ret.shape[0]}, N={df_ret.shape[1]}")
    print(f"Tarih aralığı: {dates[0].date()} — {dates[-1].date()}")
    print(f"Kriz penceresi (indeks {CRISIS_START}): {dates[CRISIS_START].date()} "
          f"— {dates[CRISIS_START + CRISIS_LEN].date()}")

    print("\n--- Tanımlayıcı istatistikler (%, günlük) ---")
    hdr = f"{'Varlık':<10}{'Ort':>8}{'Std':>8}{'Min':>8}{'Maks':>8}{'Çarpık':>9}{'Basıklık':>10}"
    print(hdr)
    for i, a in enumerate(ASSETS):
        x = df_ret[a].values
        print(f"{a:<10}{x.mean()*100:>8.3f}{x.std()*100:>8.3f}{x.min()*100:>8.2f}"
              f"{x.max()*100:>8.2f}{stats.skew(x):>9.2f}{stats.kurtosis(x, fisher=False):>10.2f}")

    # Correlation diagnostics from the TRUE path
    iu = np.triu_indices(N, 1)
    mean_corr_series = np.array([R_path[t][iu].mean() for t in range(T)])
    pre = slice(CRISIS_START - 60, CRISIS_START - 10)
    peak_idx = CRISIS_START + np.argmax(mean_corr_series[CRISIS_START:CRISIS_START + CRISIS_LEN])
    print("\n--- Korelasyon dinamiği (GERÇEK yol) ---")
    print(f"Koşulsuz ort. korelasyon (normal dönem)  : {mean_corr_series[pre].mean():.3f}")
    print(f"Kriz zirvesi ort. korelasyon ({dates[peak_idx].date()}): {mean_corr_series[peak_idx]:.3f}")
    print(f"BANKA-SANAYI: normal {R_path[pre][:, 0, 1].mean():.3f} -> "
          f"kriz zirvesi {R_path[peak_idx, 0, 1]:.3f}")

    # Volatility spike check
    v_pre = df_ret.iloc[CRISIS_START-60:CRISIS_START-10].std().mean()*100
    v_crisis = df_ret.iloc[CRISIS_START:CRISIS_START+15].std().mean()*100
    print(f"\nGünlük std: normal {v_pre:.2f}% -> kriz {v_crisis:.2f}% "
          f"(x{v_crisis/v_pre:.1f})")

    # PSD check across the whole path
    min_eig = min(np.linalg.eigvalsh(R_path[t]).min() for t in range(T))
    print(f"Tüm R_t için min öz değer: {min_eig:.2e}  (>0 => geçerli)")

    # ---- Save (drop-in: dosyanın bulunduğu klasöre yazar) ----
    import json
    base = Path(__file__).parent
    out = base / "sample_returns.csv"
    pd.concat([df_ret, df_rv, df_bpv], axis=1).to_csv(out)

    # Kriz penceresi + üretim meta-verisi (app.py bunu st.session_state'e okur)
    meta = {
        "crisis_window": [str(dates[CRISIS_START].date()),
                          str(dates[CRISIS_START + CRISIS_LEN].date())],
        "seed": SEED, "T": int(T), "assets": ASSETS,
        "dgp": "ADCC + crisis + Student-t",
        "adcc_true": {"a": A_TRUE, "b": B_TRUE, "c": C_TRUE, "nu": NU},
    }
    (base / "sample_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nKaydedildi: {out}")
    print(f"Kriz penceresi meta: {meta['crisis_window']}")
    print(f"Sütunlar ({len(df_ret.columns)*3}): getiri + _RV + _BPV")
