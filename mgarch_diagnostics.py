"""
mgarch_diagnostics.py
=====================
Correlation-stage diagnostics for the DCC family, plus the univariate-stage
checks that must be cleared *before* any correlation diagnostic is interpreted.

Why this module exists
----------------------
The Engle-Sheppard (2001) artificial regression has a chi^2_{s+1} null
distribution when it is applied to CCC residuals standardised by a CONSTANT
correlation matrix.  Applied instead to DCC/ADCC-FILTERED residuals
eps_t = R_t^{-1/2} z_t that reference is invalid, for three reasons
(course notes 1.12.1):

  (i)   two-layer estimation error — R_t is itself estimated, and the
        artificial regression treats it as known;
  (ii)  pooling across pairs assumes one scalar sigma^2, but
        Var(eps_i eps_j) = (nu-2)/(nu-4) under a t(nu) innovation: at nu=7 the
        true scale is ~1.67, and for nu <= 4 the fourth moment does not exist,
        so the asymptotic chi^2 is not merely wrong but undefined;
  (iii) the lagged regressors are endogenous.

Therefore ``es_stat`` refuses to hand back a p-value unless the caller asks for
it explicitly, and the only p-values this module offers for filtered residuals
come from ``parametric_bootstrap_pvalue``, which re-estimates the FULL model
(N univariate GARCH fits *and* the correlation stage) on every bootstrap path.
Holding theta_hat fixed would make the bootstrap meaningless.

References
----------
Engle & Sheppard (2001), NBER WP 8554.
Engle & Ng (1993), Measuring and Testing the Impact of News on Volatility, JF.
Hosking (1980) JASA 75; Li & McLeod (1981) JRSS-B 43.
Nyblom (1989) JASA 84; Hansen (1992) JBES 10 (critical values).
Benjamini & Hochberg (1995) JRSS-B 57.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sstats

# ---------------------------------------------------------------------------
# Qualitative reading thresholds for pooled statistics on filtered residuals.
# Deliberately coarse: an order-of-magnitude reading is defensible where a
# formal p-value is not.
# ---------------------------------------------------------------------------
QUALITATIVE_LOW = 20.0
QUALITATIVE_HIGH = 100.0


def qualitative_reading(stat: float) -> str:
    """Order-of-magnitude verdict for a pooled statistic without a valid null."""
    if stat < QUALITATIVE_LOW:
        return "kalan dinamik izi yok"
    if stat > QUALITATIVE_HIGH:
        return "güçlü kalan dinamik"
    return "belirsiz, bootstrap gerekir"


# ---------------------------------------------------------------------------
# Engle-Sheppard artificial regression
# ---------------------------------------------------------------------------

def es_stat(eps, s, iu, return_pvalue: bool = False):
    """
    Engle-Sheppard (2001) constant-correlation artificial-regression statistic.

    Parameters
    ----------
    eps : (T, N) array
        Jointly standardised residuals.  For the legitimate version of the test
        these are CCC residuals, standardised by a CONSTANT R_bar^{-1/2}.
    s : int
        Number of lags in the artificial regression.
    iu : tuple of arrays
        ``np.triu_indices(N, k=1)``.
    return_pvalue : bool
        If True also return the chi^2_{s+1} tail probability.  Only pass True
        for CCC residuals; see the module docstring.

    Returns
    -------
    (stat, df) or (stat, df, p_value)
    """
    eps = np.asarray(eps, dtype=float)
    P = len(iu[0])
    Y = np.einsum("ti,tj->tij", eps, eps)[:, iu[0], iu[1]]   # (T, P)
    Tn = Y.shape[0]

    yv = np.concatenate([Y[s:, p] for p in range(P)])
    Z = np.vstack([
        np.column_stack([np.ones(Tn - s)] + [Y[s - j:Tn - j, p] for j in range(1, s + 1)])
        for p in range(P)
    ])
    ZtZ = Z.T @ Z
    d = np.linalg.solve(ZtZ, Z.T @ yv)
    sig2 = ((yv - Z @ d) ** 2).mean()
    stat = float(d @ ZtZ @ d / sig2)

    if return_pvalue:
        return stat, s + 1, float(sstats.chi2.sf(stat, s + 1))
    return stat, s + 1


def isqrt_apply(R, z):
    """eps_t = R_t^{-1/2} z_t using the symmetric square root at each t."""
    R = np.asarray(R, dtype=float)
    z = np.asarray(z, dtype=float)
    eps = np.empty_like(z)
    for t in range(len(z)):
        wv, V = np.linalg.eigh(R[t])
        eps[t] = (V * (1.0 / np.sqrt(wv))) @ (V.T @ z[t])
    return eps


def ccc_standardise(z):
    """eps = z R_bar^{-1/2} for the constant-correlation (CCC) null."""
    z = np.asarray(z, dtype=float)
    Rbar = np.corrcoef(z, rowvar=False)
    wv, V = np.linalg.eigh(Rbar)
    return z @ (V @ np.diag(1.0 / np.sqrt(wv)) @ V.T).T


# ---------------------------------------------------------------------------
# Pair-by-pair Ljung-Box with Benjamini-Hochberg FDR control
# ---------------------------------------------------------------------------

def pairwise_ljung_box(eps, cols, lags: int = 10, alpha: float = 0.05):
    """
    Ljung-Box test on each cross-product series Y_ij,t = eps_i,t eps_j,t,
    with Benjamini-Hochberg FDR control across the N(N-1)/2 pairs.

    Unlike the pooled ES regression this keeps the pairs separate, so it says
    WHICH pairs carry the residual dynamics.  The per-pair p-values are still
    only indicative (the residuals are filtered), but the FDR step at least
    controls the multiplicity that testing every pair introduces.

    Returns
    -------
    dict with keys: rows (list of dicts), n_reject, n_pairs, alpha, lags
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.multitest import multipletests

    eps = np.asarray(eps, dtype=float)
    N = eps.shape[1]
    iu = np.triu_indices(N, k=1)

    stats_, pvals, labels = [], [], []
    for i, j in zip(*iu):
        y = eps[:, i] * eps[:, j]
        lb = acorr_ljungbox(y, lags=[lags], return_df=True)
        stats_.append(float(lb["lb_stat"].iloc[0]))
        pvals.append(float(lb["lb_pvalue"].iloc[0]))
        labels.append(f"{cols[i]} × {cols[j]}")

    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")

    rows = [
        {"pair": lab, "lb_stat": s, "p_raw": p, "p_bh": float(pa), "reject": bool(r)}
        for lab, s, p, pa, r in zip(labels, stats_, pvals, p_adj, reject)
    ]
    rows.sort(key=lambda r: -r["lb_stat"])
    return {
        "rows": rows,
        "n_reject": int(reject.sum()),
        "n_pairs": len(rows),
        "alpha": alpha,
        "lags": lags,
    }


# ---------------------------------------------------------------------------
# Targeted sign-bias test for the correlation stage
# ---------------------------------------------------------------------------

def correlation_sign_bias(eps, z, iu=None):
    """
    Sign-bias test aimed at the asymmetry the pooled ES regression cannot see.

    Regress the filtered cross-products on the sign pattern of the PREVIOUS
    period's standardised shocks, pooled over pairs:

        Y_ij,t = w + b1 1{z_i,t-1<0} + b2 1{z_j,t-1<0}
                   + b3 1{z_i,t-1<0} 1{z_j,t-1<0} + e_ij,t

    H0: b1 = b2 = b3 = 0 (no leftover asymmetry).  A Wald statistic is
    returned; as with ES, its chi^2_3 reference is only nominal on filtered
    residuals, so use ``parametric_bootstrap_pvalue`` for inference.

    Returns
    -------
    dict with keys: stat, df, coef, nominal_p
    """
    eps = np.asarray(eps, dtype=float)
    z = np.asarray(z, dtype=float)
    T, N = eps.shape
    if iu is None:
        iu = np.triu_indices(N, k=1)

    neg = (z < 0.0).astype(float)
    y_parts, X_parts = [], []
    for i, j in zip(*iu):
        y_parts.append(eps[1:, i] * eps[1:, j])
        di, dj = neg[:-1, i], neg[:-1, j]
        X_parts.append(np.column_stack([np.ones(T - 1), di, dj, di * dj]))

    y = np.concatenate(y_parts)
    X = np.vstack(X_parts)

    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta
    sig2 = float(resid @ resid) / len(y)
    Vinv = XtX / sig2

    # Wald on the three slope coefficients (drop the intercept)
    idx = np.array([1, 2, 3])
    V_sub = np.linalg.inv(np.linalg.inv(Vinv)[np.ix_(idx, idx)])
    b_sub = beta[idx]
    stat = float(b_sub @ V_sub @ b_sub)

    return {
        "stat": stat,
        "df": 3,
        "coef": {"neg_i": float(beta[1]), "neg_j": float(beta[2]),
                 "neg_both": float(beta[3])},
        "nominal_p": float(sstats.chi2.sf(stat, 3)),
    }


# ---------------------------------------------------------------------------
# Hosking / Li-McLeod multivariate portmanteau
# ---------------------------------------------------------------------------

HLM_MAX_ASSETS = 6


def hosking_li_mcleod(eps, m: int = 5, max_assets: int = HLM_MAX_ASSETS):
    """
    Multivariate portmanteau on the vector of cross-products
    Y_t = (eps_i,t eps_j,t)_{i<j} in R^P, P = N(N-1)/2.

    Li-McLeod corrected statistic

        Q~ = T sum_{k=1..m} tr(C_k' C_0^{-1} C_k C_0^{-1}) + P^2 m(m+1)/(2T)

    is referred to chi^2 with  df = m P^2.

    The degrees of freedom explode with N: at N=8 and m=5, df = 5 * 28^2 = 3920,
    so the test has no power at realistic sample sizes and the estimate of C_0
    (a P x P matrix from T observations) is itself unreliable.  The function
    therefore refuses to compute above ``max_assets`` and says why.

    Returns
    -------
    dict with keys: available, reason, stat, df, p_value, P, m
    """
    eps = np.asarray(eps, dtype=float)
    T, N = eps.shape
    P = N * (N - 1) // 2

    if N > max_assets:
        return {
            "available": False,
            "reason": (f"N={N} > {max_assets}: sd = m·P² = {m}·{P}² = {m * P * P}; "
                       "bu serbestlik derecesinde test güç üretmez ve C_0 "
                       f"({P}×{P}) tahmini {T} gözlemle güvenilmezdir."),
            "stat": None, "df": m * P * P, "p_value": None, "P": P, "m": m,
        }

    iu = np.triu_indices(N, k=1)
    Y = np.einsum("ti,tj->tij", eps, eps)[:, iu[0], iu[1]]
    Y = Y - Y.mean(axis=0)

    C0 = Y.T @ Y / T
    C0_inv = np.linalg.inv(C0)

    total = 0.0
    for k in range(1, m + 1):
        Ck = Y[k:].T @ Y[:-k] / T
        total += float(np.trace(Ck.T @ C0_inv @ Ck @ C0_inv))

    df = m * P * P
    stat = T * total + P * P * m * (m + 1) / (2.0 * T)
    return {
        "available": True,
        "reason": "",
        "stat": float(stat),
        "df": int(df),
        "p_value": float(sstats.chi2.sf(stat, df)),
        "P": P, "m": m,
    }


# ---------------------------------------------------------------------------
# Univariate-stage diagnostics
# ---------------------------------------------------------------------------

# Hansen (1992) asymptotic critical values for the Nyblom (1989) joint L
# statistic, indexed by the number of parameters tested.
_NYBLOM_CRIT_05 = {1: 0.470, 2: 0.749, 3: 1.010, 4: 1.240, 5: 1.470,
                   6: 1.680, 7: 1.900, 8: 2.110, 9: 2.320, 10: 2.540}


def _garch11_scores(r, omega, alpha, beta):
    """
    Analytic per-observation scores of the Gaussian GARCH(1,1) QMLE.

        l_t     = -0.5 (log s_t + r_t^2 / s_t),   s_t = omega + alpha r_{t-1}^2 + beta s_{t-1}
        dl/dth  = 0.5 (r_t^2/s_t - 1) (1/s_t) ds_t/dth

    with ds_t/dth propagated by  ds_t/dth = x_t + beta ds_{t-1}/dth.
    """
    r = np.asarray(r, dtype=float)
    T = len(r)
    s = np.empty(T)
    s[0] = np.var(r)
    grad_s = np.zeros((T, 3))
    scores = np.zeros((T, 3))

    for t in range(1, T):
        s[t] = omega + alpha * r[t - 1] ** 2 + beta * s[t - 1]
        grad_s[t, 0] = 1.0 + beta * grad_s[t - 1, 0]
        grad_s[t, 1] = r[t - 1] ** 2 + beta * grad_s[t - 1, 1]
        grad_s[t, 2] = s[t - 1] + beta * grad_s[t - 1, 2]
        scores[t] = 0.5 * (r[t] ** 2 / s[t] - 1.0) / s[t] * grad_s[t]

    return scores[1:]


def nyblom_stability(r, omega, alpha, beta):
    """
    Nyblom (1989) joint parameter-constancy test for a fitted GARCH(1,1).

        L = (1/T) sum_t S_t' V^{-1} S_t,   S_t = sum_{i<=t} score_i,
        V = (1/T) sum_t score_t score_t'

    H0: constant parameters.  Rejection says the marginal model — not the
    correlation model — is the part that is misspecified, which is why this
    tab must be read before the correlation diagnostics.
    """
    scores = _garch11_scores(r, omega, alpha, beta)
    T, k = scores.shape
    V = scores.T @ scores / T
    S = np.cumsum(scores, axis=0)
    try:
        Vinv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        Vinv = np.linalg.pinv(V)
    L = float(np.einsum("tk,kl,tl->", S, Vinv, S) / (T * T))
    return {
        "stat": L,
        "k": k,
        "crit_05": _NYBLOM_CRIT_05.get(k),
        "reject_05": (L > _NYBLOM_CRIT_05[k]) if k in _NYBLOM_CRIT_05 else None,
    }


def engle_ng_sign_bias(z):
    """
    Engle & Ng (1993) joint sign-bias test on standardised residuals:

        z_t^2 = a0 + a1 1{z_{t-1}<0} + a2 1{z_{t-1}<0} z_{t-1}
                   + a3 1{z_{t-1}>=0} z_{t-1} + u_t

    H0: a1 = a2 = a3 = 0, LM = T R^2 ~ chi^2_3.  Unlike the correlation-stage
    diagnostics this one IS applied to residuals whose standardisation used a
    single estimated scalar path, so the chi^2_3 reference is the usual
    first-order-valid one.
    """
    z = np.asarray(z, dtype=float)
    y = z[1:] ** 2
    zl = z[:-1]
    neg = (zl < 0.0).astype(float)
    X = np.column_stack([np.ones(len(y)), neg, neg * zl, (1.0 - neg) * zl])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    lm = len(y) * r2
    return {"stat": float(lm), "df": 3, "p_value": float(sstats.chi2.sf(lm, 3)),
            "r2": float(r2)}


def univariate_diagnostics(z, sigmas, returns, univariate_models, cols, lags: int = 10):
    """
    Per-asset marginal-stage diagnostics: Ljung-Box on z and z^2, ARCH-LM,
    Engle-Ng sign bias, and Nyblom parameter constancy.

    Returns
    -------
    list of dicts, one per asset
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

    z = np.asarray(z, dtype=float)
    out = []
    for i, col in enumerate(cols):
        zi = z[:, i]
        lb = acorr_ljungbox(zi, lags=[lags], return_df=True)
        lb2 = acorr_ljungbox(zi ** 2, lags=[lags], return_df=True)
        arch_lm = het_arch(zi, nlags=lags)
        eng = engle_ng_sign_bias(zi)

        row = {
            "asset": col,
            "lb_z": float(lb["lb_stat"].iloc[0]),
            "lb_z_p": float(lb["lb_pvalue"].iloc[0]),
            "lb_z2": float(lb2["lb_stat"].iloc[0]),
            "lb_z2_p": float(lb2["lb_pvalue"].iloc[0]),
            "arch_lm": float(arch_lm[0]),
            "arch_lm_p": float(arch_lm[1]),
            "sign_bias": eng["stat"],
            "sign_bias_p": eng["p_value"],
        }

        try:
            params = univariate_models[i].params
            r_pct = np.asarray(returns[col].values, dtype=float) * 100.0
            nyb = nyblom_stability(r_pct - r_pct.mean(),
                                   float(params["omega"]),
                                   float(params["alpha[1]"]),
                                   float(params["beta[1]"]))
            row["nyblom"] = nyb["stat"]
            row["nyblom_crit"] = nyb["crit_05"]
            row["nyblom_reject"] = nyb["reject_05"]
        except Exception:                       # pragma: no cover - defensive
            row["nyblom"] = np.nan
            row["nyblom_crit"] = None
            row["nyblom_reject"] = None

        out.append(row)

    return out


# ---------------------------------------------------------------------------
# Parametric bootstrap
# ---------------------------------------------------------------------------

def simulate_dcc_path(garch_params, dcc_params, bar_Q, T, rng, model_type="DCC"):
    """
    Simulate a return path from a fitted DCC-family model.

    Parameters
    ----------
    garch_params : (N, 3) array of (omega, alpha, beta) in percent units
    dcc_params   : (a, b)
    bar_Q        : (N, N) correlation target
    T            : path length
    rng          : np.random.Generator
    model_type   : "DCC" or "DECO" (drives how R_t is formed)

    Returns
    -------
    (T, N) array of returns in DECIMAL units (matching the app's convention).
    """
    garch_params = np.asarray(garch_params, dtype=float)
    N = garch_params.shape[0]
    a, b = float(dcc_params[0]), float(dcc_params[1])
    omega, alpha, beta = garch_params[:, 0], garch_params[:, 1], garch_params[:, 2]

    Q = bar_Q.copy()
    s2 = omega / np.maximum(1.0 - alpha - beta, 1e-6)       # unconditional var
    r = np.zeros((T, N))
    z_prev = rng.standard_normal(N)
    is_deco = (model_type.upper() == "DECO")
    rho_lo = -1.0 / (N - 1) + 1e-6

    for t in range(T):
        if t > 0:
            s2 = omega + alpha * r[t - 1] ** 2 + beta * s2
            Q = (1.0 - a - b) * bar_Q + a * np.outer(z_prev, z_prev) + b * Q

        d = 1.0 / np.sqrt(np.diag(Q))
        if is_deco:
            rho = float(np.clip((d @ Q @ d - N) / (N * (N - 1)), rho_lo, 1.0 - 1e-6))
            R = (1.0 - rho) * np.eye(N) + rho * np.ones((N, N))
        else:
            R = Q * np.outer(d, d)
        np.fill_diagonal(R, 1.0)

        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(R)
            L = V * np.sqrt(np.maximum(w, 1e-12))

        z_prev = L @ rng.standard_normal(N)
        r[t] = np.sqrt(s2) * z_prev

    return r / 100.0            # garch params are in percent units


def parametric_bootstrap_pvalue(returns, model_type, statistic, observed, B=199,
                                seed=20260730, progress=None, n_jobs=None):
    """
    Bootstrap p-value for a correlation-stage statistic computed on FILTERED
    residuals.

    Every bootstrap path is simulated from the fitted model and then has the
    ENTIRE model re-estimated on it: N univariate GARCH fits plus the
    correlation stage.  Holding theta_hat fixed would ignore exactly the
    two-layer estimation error that invalidates the asymptotic chi^2 in the
    first place (course notes 1.12.1, step 3).

    Note on parallelism: the inner loop is dominated by `arch`'s optimiser, so
    numba's ``prange`` cannot be used here.  Replications are dispatched to a
    thread pool instead; the heavy parts (LAPACK, scipy.optimize) release the
    GIL.

    Parameters
    ----------
    returns : pd.DataFrame  (T, N) in decimals
    model_type : str        model to re-fit on each path
    statistic : callable    (eps, z, model) -> float
    observed : float        the statistic on the real data
    B : int                 number of replications (199 / 499 / 999)
    progress : callable or None
        Called as ``progress(done, B)`` after each replication.

    Returns
    -------
    dict with keys: p_value, observed, boot_stats, B, n_failed
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from dcc_garch import DCCGarch

    cols = list(returns.columns)
    T, N = returns.shape

    # --- fit the null model once -----------------------------------------
    base = DCCGarch(model_type=model_type)
    z0 = base.fit_univariate_garch(returns)
    base.fit_dcc(z0)

    gp = np.array([
        [float(m.params["omega"]), float(m.params["alpha[1]"]), float(m.params["beta[1]"])]
        for m in base.univariate_models
    ])
    d0 = 1.0 / np.sqrt(np.diag(base._bar_Q))
    bar_R = base._bar_Q * np.outer(d0, d0)
    np.fill_diagonal(bar_R, 1.0)
    ab = (float(base.dcc_params[0]), float(base.dcc_params[1]))

    seeds = np.random.SeedSequence(seed).spawn(B)

    def _one(ss):
        import pandas as pd
        rng = np.random.default_rng(ss)
        r = simulate_dcc_path(gp, ab, bar_R, T, rng, model_type=model_type)
        sim = pd.DataFrame(r, columns=cols)
        m = DCCGarch(model_type=model_type)
        zz = m.fit_univariate_garch(sim)        # re-estimate the marginals
        m.fit_dcc(zz)                           # ... and the correlation stage
        eps = isqrt_apply(np.asarray(m.R_seq), zz)
        return float(statistic(eps, zz, m))

    boot = []
    n_failed = 0
    workers = n_jobs if n_jobs else min(8, (len(seeds) or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_one, ss) for ss in seeds]
        for done, fut in enumerate(as_completed(futures), start=1):
            try:
                boot.append(fut.result())
            except Exception:                   # a path the optimiser failed on
                n_failed += 1
            if progress is not None:
                progress(done, B)

    boot_arr = np.asarray(boot, dtype=float)
    if boot_arr.size == 0:
        raise RuntimeError("Every bootstrap replication failed.")

    # (1 + #{boot >= observed}) / (1 + B_effective)  — never returns exactly 0
    p = (1.0 + float((boot_arr >= observed).sum())) / (1.0 + boot_arr.size)

    return {
        "p_value": float(p),
        "observed": float(observed),
        "boot_stats": boot_arr,
        "B": int(boot_arr.size),
        "n_failed": int(n_failed),
    }
