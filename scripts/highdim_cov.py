"""
highdim_cov.py -- Part C of the Day-5 empirical layer.

Controlled panel DGP with a KNOWN K-factor covariance
        Sigma = B Sigma_f B' + Sigma_u        (Sigma_u diagonal, "sparse")
so that competing high-dimensional covariance estimators can be scored against
ground truth. We compare:
    * sample covariance
    * Ledoit-Wolf linear shrinkage (Ledoit-Wolf 2004)
    * POET (Fan-Liao-Mincheva 2013): K principal components + thresholded residual
on Frobenius error vs true Sigma and on out-of-sample minimum-variance-portfolio
variance. We also recover the factor count K by Bai-Ng (2002) IC, the
Marchenko-Pastur bulk-edge rule, and a simplified Onatski (2010) eigenvalue-gap
rule, and display the sample eigenvalue spectrum against the MP bulk edge.

Two regimes: N<T (c=0.2, sample cov well-defined) and N>T (c=1.5, sample cov
singular -> shrinkage/factor methods essential).
"""
from __future__ import annotations
import os
import sys

import numpy as np
from numba import njit
from sklearn.covariance import LedoitWolf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The factor-count estimators moved to the shared module so that the Day-3
# Factor-DCC tab and the Day-5 POET tab can use the same (corrected) code.
from factor_selection import bai_ng_ic, mp_threshold_count, onatski_ed  # noqa: E402

SEED = 20260731


# ----------------------------------------------------------------------
# DGP
# ----------------------------------------------------------------------
def simulate_panel(N, T, K, rng, factor_vol=None, idio_scale=1.0):
    """r_t = B f_t + u_t, with known Sigma = B Sigma_f B' + Sigma_u."""
    if factor_vol is None:
        # descending factor strengths (market + weaker factors), in daily-vol units
        factor_vol = np.linspace(0.020, 0.008, K)
    Sig_f = np.diag(factor_vol**2)
    B = rng.standard_normal((N, K)) * 0.6          # loadings
    # idiosyncratic variances: heterogeneous, ~ (0.5%-1.5% daily vol)
    idio_sd = (0.005 + 0.010 * rng.random(N)) * idio_scale
    Sig_u = idio_sd**2
    Sigma = B @ Sig_f @ B.T + np.diag(Sig_u)
    # draw returns
    f = rng.standard_normal((T, K)) @ np.linalg.cholesky(Sig_f).T
    u = rng.standard_normal((T, N)) * idio_sd
    R = f @ B.T + u
    return R, Sigma, B, Sig_f, Sig_u


# ----------------------------------------------------------------------
# Estimators
# ----------------------------------------------------------------------
def cov_sample(R):
    Rc = R - R.mean(0)
    return (Rc.T @ Rc) / R.shape[0]

def cov_lw_linear(R):
    return LedoitWolf(assume_centered=False).fit(R).covariance_

@njit(cache=True)
def _soft_threshold_resid(Res, C, omega):
    """Adaptive soft-thresholding of the off-diagonal residual covariance
    (correlation-scaled); diagonal preserved."""
    N = Res.shape[0]
    out = np.zeros((N, N))
    for i in range(N):
        out[i, i] = Res[i, i]
    for i in range(N):
        for j in range(N):
            if i != j:
                tau = C * omega * np.sqrt(abs(Res[i, i] * Res[j, j]))
                a = Res[i, j]
                if a > tau:
                    out[i, j] = a - tau
                elif a < -tau:
                    out[i, j] = a + tau
                # else 0
    return out

def cov_poet(R, K, C=0.5):
    """POET: keep K PCs of sample cov, soft-threshold residual. C tuned up if
    needed to keep positive definiteness."""
    S = cov_sample(R)
    N, T = S.shape[0], R.shape[0]
    w, V = np.linalg.eigh(S)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    low = (V[:, :K] * w[:K]) @ V[:, :K].T           # low-rank factor part
    Res = S - low
    omega = np.sqrt(np.log(N) / T) + 1.0/np.sqrt(N)
    for Cc in [C, 0.75, 1.0, 1.5, 2.0, 3.0]:
        Rt = _soft_threshold_resid(Res, Cc, omega)
        Sigma_hat = low + Rt
        ev = np.linalg.eigvalsh(Sigma_hat)
        if ev.min() > 1e-12:
            return Sigma_hat, Cc
    return low + np.diag(np.diag(Res)), np.nan       # fallback: diagonal residual


# ----------------------------------------------------------------------
# Factor-count recovery: bai_ng_ic / mp_threshold_count / onatski_ed are
# imported from factor_selection (see the import block at the top).  The
# copies that used to live here demeaned without standardising, could not
# return k=0, and ran Onatski on the covariance matrix while the MP rule ran
# on the correlation matrix — all three fixed in the shared module.
# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def frob_rel(Sig_hat, Sigma):
    return np.linalg.norm(Sig_hat - Sigma, "fro") / np.linalg.norm(Sigma, "fro")

def mvp_oos_var(Sig_hat, Sigma, T_train: int | None = None):
    """True variance of the MVP built from Sig_hat.

    If T_train is given and N >= T_train (sample covariance is singular by
    construction), return np.nan immediately rather than applying the 1e-12
    ridge, which is numerically unstable for a rank-deficient matrix.
    LW and POET estimators are always regularised and may still be called.
    """
    N = Sig_hat.shape[0]
    if T_train is not None and N >= T_train:
        return np.nan
    ones = np.ones(N)
    try:
        inv = np.linalg.solve(Sig_hat, ones)        # no epsilon: caller ensures PD
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(Sig_hat) @ ones
    w = inv / (ones @ inv)
    return float(w @ Sigma @ w)

def mvp_oracle_var(Sigma):
    N = Sigma.shape[0]; ones = np.ones(N)
    inv = np.linalg.solve(Sigma, ones); w = inv/(ones@inv)
    return float(w @ Sigma @ w)


# ----------------------------------------------------------------------
def run_regime(N, T, K, rng, label):
    R, Sigma, B, Sig_f, Sig_u = simulate_panel(N, T, K, rng)
    # split for OOS MVP: estimate on first half, score true var
    T_train = T // 2
    Rtr = R[:T_train]
    est = {}
    est["Sample"]    = cov_sample(Rtr)
    est["LW-linear"] = cov_lw_linear(Rtr)
    poet, Cused      = cov_poet(Rtr, K)
    est["POET(K)"]   = poet

    print(f"\n===== {label}: N={N}, T={T} (train {T_train}), K_true={K}, c=N/T_train={N/T_train:.2f} =====")
    orc = mvp_oracle_var(Sigma)
    print(f"{'Estimator':<12}{'Frob.rel':>10}{'MVP true var':>15}{'MVP/oracle':>12}")
    for name, Sh in est.items():
        fr = frob_rel(Sh, Sigma)
        # Sample covariance is rank-T_train when N >= T_train: MVP undefined
        t_arg = T_train if name == "Sample" else None
        mv = mvp_oos_var(Sh, Sigma, T_train=t_arg)
        if np.isnan(mv):
            print(f"{name:<12}{fr:>10.3f}{'---':>15}{'tanımsız':>12}   (N≥T_eğitim, tekil)")
        else:
            print(f"{name:<12}{fr:>10.3f}{mv:>15.3e}{mv/orc:>12.2f}")
    print(f"{'Oracle(Sigma)':<12}{0.0:>10.3f}{orc:>15.3e}{1.0:>12.2f}   (POET C={Cused})")

    # factor recovery (on full sample)
    kbn = bai_ng_ic(R, k_max=10)
    kmp, lam_plus, ev_corr = mp_threshold_count(R)
    kon = onatski_ed(R, k_max=10)
    print(f"Faktor sayisi geri kazanimi (K_true={K}):  "
          f"Bai-Ng={kbn}   MP-esigi={kmp} (lambda+={lam_plus:.2f})   Onatski={kon}")
    return ev_corr, lam_plus, N, T, K


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    # Regime 1: N < T
    run_regime(N=100, T=1000, K=3, rng=rng, label="Rejim 1 (N<T)")
    # Regime 2: N > T_train (sample cov singular)
    run_regime(N=150, T=200, K=3, rng=rng, label="Rejim 2 (N>T, ornek-kov tekil)")
    # Regime 3: very high dim
    run_regime(N=300, T=500, K=4, rng=rng, label="Rejim 3 (yuksek boyut)")
