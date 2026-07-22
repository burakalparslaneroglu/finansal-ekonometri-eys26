##################################################################

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from scipy.optimize import bisect

def calculate_pelve_single(losses, alpha=0.05):
    """
    Calculates the Probability Equivalent Level of VaR and ES (PELVE) for a 1D array of losses.
    defined as the unique c in [1, 1/alpha] such that ES_{1 - c*alpha}(L) = VaR_{1 - alpha}(L).
    losses: 1D numpy array of losses (returns * -1)
    alpha: significance level (default 0.05)
    """
    losses = np.sort(losses)
    n = len(losses)

    # 1. Target Value: VaR_{1-alpha}(L)
    # Using interpolation to match standard quantile definitions
    var_target = np.quantile(losses, 1.0 - alpha)

    # 2. Define the objective function to find root for c
    # We want to solve: ES_{1 - c*alpha}(L) - VaR_{1 - alpha}(L) = 0
    # Note: ES_p(L) = Mean of losses exceeding quantile(p)
    def objective(c):
        p = 1.0 - c * alpha
        # Avoid boundary errors
        p = max(1e-8, min(p, 1.0 - 1e-8))
        q_val = np.quantile(losses, p)
        es_val = np.mean(losses[losses >= q_val])
        return es_val - var_target

    # Check boundary conditions
    obj_1 = objective(1.0) # ES_{1-alpha} vs VaR_{1-alpha} (usually positive, since ES > VaR)
    obj_max = objective(1.0 / alpha) # ES_{0} (mean of all losses) vs VaR_{1-alpha} (usually negative, since mean of all returns is smaller than 95% quantile)

    if obj_1 <= 0:
        return 1.0
    if obj_max >= 0:
        return 1.0 / alpha

    try:
        c_star = bisect(objective, 1.0, 1.0 / alpha, xtol=1e-5)
        return c_star
    except ValueError:
        # Fallback to grid search if bisection fails due to non-monotonic empirical anomalies
        c_grid = np.linspace(1.0, 1.0 / alpha, 500)
        obj_vals = np.array([objective(c) for c in c_grid])
        idx = np.argmin(np.abs(obj_vals))
        return c_grid[idx]

def calculate_pelve(returns_df, alpha=0.05, rolling_window=None):
    """
    Computes PELVE for a DataFrame of returns.
    If rolling_window is specified, returns a rolling PELVE time-series.
    """
    losses_df = -returns_df

    if rolling_window is None:
        pelve_results = {}
        for col in losses_df.columns:
            pelve_results[col] = calculate_pelve_single(losses_df[col].dropna().values, alpha)
        return pd.Series(pelve_results)
    else:
        # Rolling window estimation
        pelve_rolling = pd.DataFrame(index=returns_df.index[rolling_window:])
        for col in returns_df.columns:
            rolling_vals = []
            for t in range(rolling_window, len(returns_df)):
                window_losses = losses_df[col].iloc[t - rolling_window : t].values
                rolling_vals.append(calculate_pelve_single(window_losses, alpha))
            pelve_rolling[col] = rolling_vals
        return pelve_rolling

def calculate_var_es(returns, alpha=0.05, method="parametric_normal", df_t=5):
    """
    Computes Value at Risk (VaR) and Expected Shortfall (ES) for a 1D array of returns.
    Returns: (VaR, ES) as positive numbers representing loss.
    """
    losses = -np.asarray(returns)

    if method == "parametric_normal":
        mu = np.mean(losses)
        sigma = np.std(losses, ddof=1)
        var_val = mu + sigma * norm.ppf(1.0 - alpha)
        # ES for Normal distribution: mu + sigma * (phi(z_alpha) / alpha)
        z = norm.ppf(1.0 - alpha)
        es_val = mu + sigma * (norm.pdf(z) / alpha)

    elif method == "parametric_student_t":
        mu = np.mean(losses)
        sigma = np.std(losses, ddof=1)
        # Rescale sigma for t-distribution variance
        scale = sigma * np.sqrt((df_t - 2) / df_t)
        var_val = mu + scale * t.ppf(1.0 - alpha, df_t)
        # ES for Student-t
        x_q = t.ppf(1.0 - alpha, df_t)
        es_val = mu + scale * (t.pdf(x_q, df_t) / alpha) * ((df_t + x_q**2) / (df_t - 1))

    elif method == "historical":
        var_val = np.quantile(losses, 1.0 - alpha)
        es_val = np.mean(losses[losses >= var_val])

    else:
        raise ValueError(f"Unknown method: {method}")

    return var_val, es_val

def backtest_var(returns, var_forecasts, alpha=0.05):
    """
    Performs Kupiec POF and Christoffersen Independence tests for VaR backtesting.
    returns: 1D array of return series
    var_forecasts: 1D array of VaR forecasts (positive numbers)
    """
    losses = -np.asarray(returns)
    var_forecasts = np.asarray(var_forecasts)

    # Hits (violations): 1 if loss > VaR, 0 otherwise
    hits = (losses > var_forecasts).astype(int)
    N = len(hits)
    x = np.sum(hits)

    # 1. Kupiec POF Test (Unconditional Coverage)
    p_hat = x / N
    # Likelihood ratio statistic
    # LR = -2 * ln( ( (1 - alpha)^(N-x) * alpha^x ) / ( (1 - p_hat)^(N-x) * p_hat^x ) )
    # To handle 0 hits or 100% hits, we use clip
    p_hat_clip = np.clip(p_hat, 1e-8, 1.0 - 1e-8)
    lr_pof = -2.0 * ( (N - x) * np.log(1.0 - alpha) + x * np.log(alpha)
                      - (N - x) * np.log(1.0 - p_hat_clip) - x * np.log(p_hat_clip) )
    p_val_pof = 1.0 - norm.cdf(np.sqrt(max(0, lr_pof))) # Chi-sq with 1 df is same as Z^2
    # Standard chi-square with 1 df p-value:
    from scipy.stats import chi2
    p_val_pof = 1.0 - chi2.cdf(lr_pof, df=1)

    # 2. Christoffersen Independence Test
    # Counts transitions between state 0 (no violation) and 1 (violation)
    n00, n01, n10, n11 = 0, 0, 0, 0
    for i in range(1, len(hits)):
        if hits[i-1] == 0 and hits[i] == 0:
            n00 += 1
        elif hits[i-1] == 0 and hits[i] == 1:
            n01 += 1
        elif hits[i-1] == 1 and hits[i] == 0:
            n10 += 1
        elif hits[i-1] == 1 and hits[i] == 1:
            n11 += 1

    p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    p2 = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0

    # Likelihood under independence
    L_ind = ((1.0 - p2)**(n00 + n10)) * (p2**(n01 + n11))
    # Likelihood under dependence
    L_dep = ((1.0 - p01)**n00) * (p01**n01) * ((1.0 - p11)**n10) * (p11**n11)

    lr_ind = -2.0 * np.log(max(1e-10, L_ind / max(1e-10, L_dep)))
    p_val_ind = 1.0 - chi2.cdf(lr_ind, df=1)

    # Conditional Coverage (Kupiec + Christoffersen)
    lr_cc = lr_pof + lr_ind
    p_val_cc = 1.0 - chi2.cdf(lr_cc, df=2)

    return {
        "violations": x,
        "violation_rate": p_hat,
        "kupiec_stat": lr_pof,
        "kupiec_pvalue": p_val_pof,
        "independence_stat": lr_ind,
        "independence_pvalue": p_val_ind,
        "conditional_coverage_pvalue": p_val_cc
    }

def backtest_es_acerbi_szekely(returns, var_forecasts, es_forecasts, alpha=0.05):
    """
    Acerbi-Szekely (2014) ES backtests, CORRECTED.
      Z1: conditional-magnitude test, normalized by the VIOLATION COUNT N_T = sum I_t
          Z1 = (1/N_T) * sum_t ( L_t I_t / ES_t ) - 1
      Z2: frequency+magnitude test, normalized by N*alpha (expected # violations)
          Z2 = (1/(N*alpha)) * sum_t ( L_t I_t / ES_t ) - 1
    Under H0, E[Z1]=E[Z2]=0; strongly POSITIVE => risk underestimation (positive-loss/positive-ES convention: too-small ES inflates L_t/ES_t).
    (The previous implementation labeled Z2 as 'Z1'.)
    """
    losses = -np.asarray(returns)
    var_forecasts = np.asarray(var_forecasts)
    es_forecasts = np.clip(np.asarray(es_forecasts), 1e-6, None)

    hits = (losses > var_forecasts).astype(float)
    NT = hits.sum()
    num = float(np.sum(losses * hits / es_forecasts))
    z1_stat = num / NT - 1.0 if NT > 0 else np.nan          # <-- N_T (violation count)
    z2_stat = num / (len(losses) * alpha) - 1.0             # <-- N*alpha

    # Simulation-based p-values (standard-normal H0) for BOTH statistics
    n_sims, N = 1000, len(losses)
    sv = norm.ppf(1.0 - alpha); se = norm.pdf(sv) / alpha
    sim_z1 = np.empty(n_sims); sim_z2 = np.empty(n_sims)
    for b in range(n_sims):
        sl = np.random.normal(0.0, 1.0, N); sh = (sl > sv).astype(float)
        snt = sh.sum(); sn = float(np.sum(sl * sh / se))
        sim_z1[b] = sn / snt - 1.0 if snt > 0 else np.nan
        sim_z2[b] = sn / (N * alpha) - 1.0
    p1 = float(np.nanmean(sim_z1 >= z1_stat))  # right tail: positive Z = underestimation
    p2 = float(np.nanmean(sim_z2 >= z2_stat))  # right tail

    return {
        "z1_stat": z1_stat, "z1_pvalue": p1,     # TRUE Z1 (violation-count norm.)
        "z2_stat": z2_stat, "z2_pvalue": p2,
        "pvalue": p1,                            # backward-compat key
    }

def fissler_ziegel_loss(returns, var_forecasts, es_forecasts, alpha=0.05):
    """
    AL (Fissler-Ziegel, G1=0) joint VaR-ES score -- Taylor (2019, 2020).
    Inputs are in the LOSS convention (VaR>0, ES>0); they are converted internally
    to the RETURNS convention (Q=-VaR<0, ESr=-ES<0) required by the logarithmic AL
    parametrization: S = Q/ESr - I[y<=Q](Q-y)/(alpha*ESr) + ln(-ESr).
    (The previous positive-loss log-form was inconsistent: its minimizer had ES<0.)
    Returns the mean score (lower = better).
    """
    y = np.asarray(returns)
    Q = -np.asarray(var_forecasts)
    ESr = -np.clip(np.asarray(es_forecasts), 1e-6, None)
    ESr = np.minimum(ESr, Q - 1e-9)                 # enforce ES < Q (numeric safety)
    hit = (y <= Q).astype(float)
    S = Q / ESr - hit * (Q - y) / (alpha * ESr) + np.log(-ESr)   # const dropped
    return float(np.mean(S))


def al_score_series(returns, var_forecasts, es_forecasts, alpha=0.05):
    """Per-observation AL score (for skill scores / DM tests). Loss-convention inputs."""
    y = np.asarray(returns)
    Q = -np.asarray(var_forecasts)
    ESr = np.minimum(-np.clip(np.asarray(es_forecasts), 1e-6, None), Q - 1e-9)
    hit = (y <= Q).astype(float)
    return Q / ESr - hit * (Q - y) / (alpha * ESr) + np.log(-ESr)

def calculate_cornish_fisher_var(returns, alpha=0.05):
    """
    Cornish-Fisher expansion for mVaR accounting for skewness and excess kurtosis.
    z_CF = z_alpha + (1/6)(z_alpha^2 - 1)*S + (1/24)(z_alpha^3 - 3*z_alpha)*K
           - (1/36)(2*z_alpha^3 - 5*z_alpha)*S^2
    mVaR = mu + sigma * z_CF
    where S = skewness, K = excess kurtosis (kurt - 3)
    Returns: (mVaR, z_cf, skewness, excess_kurtosis) all as floats
    losses = -returns convention: returns mVaR as positive number (loss)
    """
    from scipy.stats import skew, kurtosis

    losses = -np.asarray(returns, dtype=float)
    mu = np.mean(losses)
    sigma = np.std(losses, ddof=1)

    S = skew(losses)
    # scipy kurtosis() returns excess kurtosis by default (Fisher definition)
    K = kurtosis(losses, fisher=True)

    z = norm.ppf(1.0 - alpha)  # e.g. 1.6449 for alpha=0.05

    z_cf = (z
            + (1.0 / 6.0) * (z**2 - 1.0) * S
            + (1.0 / 24.0) * (z**3 - 3.0 * z) * K
            - (1.0 / 36.0) * (2.0 * z**3 - 5.0 * z) * S**2)

    mvar = mu + sigma * z_cf
    return float(mvar), float(z_cf), float(S), float(K)


def calculate_evt_var_es(losses, alpha=0.05, threshold_quantile=0.90, threshold=None):
    """
    POT (Peaks-Over-Threshold) EVT estimator using Generalized Pareto Distribution.
    losses: 1D numpy array of losses (already positive, i.e. -returns)
    alpha: tail probability for VaR/ES
    threshold_quantile: quantile used to auto-select threshold u if threshold is None
    threshold: override u directly

    Steps:
    1. u = np.quantile(losses, threshold_quantile) if threshold is None
    2. Exceedances Y = losses[losses > u] - u
    3. Fit GPD to Y: shape xi, scale sigma via scipy.stats.genpareto.fit(Y, floc=0)
    4. VaR_alpha = u + (sigma/xi) * ((n/N_u * alpha)^(-xi) - 1)
       where n = len(losses), N_u = len(Y)
    5. ES_alpha = (VaR_alpha + sigma - xi*u) / (1 - xi)

    Returns dict with threshold, n_exceedances, exceedance_rate, xi, sigma,
    var, es, exceedances, and optionally warning flags.
    """
    from scipy.stats import genpareto

    losses = np.asarray(losses, dtype=float)
    n = len(losses)

    u = np.quantile(losses, threshold_quantile) if threshold is None else float(threshold)

    Y = losses[losses > u] - u
    N_u = len(Y)

    result = {
        'threshold': float(u),
        'n_exceedances': int(N_u),
        'exceedance_rate': N_u / n,
        'xi': np.nan,
        'sigma': np.nan,
        'var': np.nan,
        'es': np.nan,
        'exceedances': Y,
        'warning': None,
    }

    if N_u < 10:
        result['warning'] = f'Too few exceedances ({N_u}) above threshold {u:.4f}; estimates unreliable.'
        return result

    # Fit GPD with fixed location = 0
    xi, loc_, sigma = genpareto.fit(Y, floc=0)

    # POT VaR formula
    ratio = (n / N_u) * alpha  # = (alpha / exceedance_rate)
    if abs(xi) < 1e-10:
        # xi ≈ 0: log formula
        var_val = u + sigma * np.log(1.0 / ratio)
    else:
        var_val = u + (sigma / xi) * (ratio ** (-xi) - 1.0)

    # POT ES formula (valid only for xi < 1)
    if xi >= 1.0:
        es_val = np.nan
        result['warning'] = 'xi >= 1: ES is infinite under fitted GPD.'
    else:
        es_val = (var_val + sigma - xi * u) / (1.0 - xi)

    result['xi'] = float(xi)
    result['sigma'] = float(sigma)
    result['var'] = float(var_val)
    result['es'] = float(es_val) if not np.isnan(es_val) else np.nan
    return result


def calculate_covar(returns_i, returns_j, alpha=0.05, rolling_window=None):
    """
    CoVaR of asset i conditional on asset j being in distress.
    Uses linear quantile regression: quantile_alpha(r_i | r_j) = beta_0 + beta_1 * r_j

    CoVaR_{i|j} = beta_0_hat + beta_1_hat * VaR_alpha(r_j)
    DeltaCoVaR = CoVaR_{i|j,alpha} - CoVaR_{i|j,0.5}

    Implementation: statsmodels QuantReg
    returns_i, returns_j: 1D numpy arrays (returns, not losses)
    alpha: quantile level (e.g. 0.05 for 5% tail)
    rolling_window: if given, compute rolling CoVaR series

    Returns (static): dict with covar_alpha, covar_median, delta_covar,
                      beta0, beta1, var_j
    Returns (rolling): pd.DataFrame with columns ['covar', 'delta_covar']
    """
    from statsmodels.regression.quantile_regression import QuantReg

    returns_i = np.asarray(returns_i, dtype=float)
    returns_j = np.asarray(returns_j, dtype=float)

    def _fit_covar(ri, rj, q):
        X = np.column_stack([np.ones(len(rj)), rj])
        model = QuantReg(ri, X)
        res = model.fit(q=q, max_iter=1000)
        return float(res.params[0]), float(res.params[1])

    def _covar_static(ri, rj):
        var_j = float(np.quantile(rj, alpha))  # negative number (loss convention in return space)
        b0_a, b1_a = _fit_covar(ri, rj, alpha)
        b0_m, b1_m = _fit_covar(ri, rj, 0.5)
        covar_a = b0_a + b1_a * var_j
        covar_m = b0_m + b1_m * var_j
        delta_covar = covar_a - covar_m
        return {
            'covar_alpha': float(covar_a),
            'covar_median': float(covar_m),
            'delta_covar': float(delta_covar),
            'beta0': float(b0_a),
            'beta1': float(b1_a),
            'var_j': float(var_j),
        }

    if rolling_window is None:
        return _covar_static(returns_i, returns_j)

    # Rolling case
    records = []
    idx_range = range(rolling_window, len(returns_i) + 1)
    for end in idx_range:
        start = end - rolling_window
        ri_w = returns_i[start:end]
        rj_w = returns_j[start:end]
        res = _covar_static(ri_w, rj_w)
        records.append({'covar': res['covar_alpha'], 'delta_covar': res['delta_covar']})
    return pd.DataFrame(records)


def calculate_mes(returns_asset, returns_market, alpha=0.05):
    """
    Marginal Expected Shortfall: expected loss of asset i when market is in its worst alpha% days.
    MES = E[r_i | r_m <= VaR_{market,alpha}]

    returns_asset, returns_market: 1D numpy arrays (returns, not losses)
    alpha: tail probability (e.g. 0.05)

    Returns dict: mes, market_var, n_crisis_days, beta_tail
    """
    returns_asset = np.asarray(returns_asset, dtype=float)
    returns_market = np.asarray(returns_market, dtype=float)

    market_var = float(np.quantile(returns_market, alpha))  # alpha-quantile in return space
    crisis_mask = returns_market <= market_var

    n_crisis_days = int(np.sum(crisis_mask))
    if n_crisis_days == 0:
        return {
            'mes': np.nan,
            'market_var': market_var,
            'n_crisis_days': 0,
            'beta_tail': np.nan,
        }

    mes = float(np.mean(returns_asset[crisis_mask]))
    market_es = float(np.mean(returns_market[crisis_mask]))
    beta_tail = (mes / market_es) if abs(market_es) > 1e-12 else np.nan

    return {
        'mes': mes,
        'market_var': market_var,
        'n_crisis_days': n_crisis_days,
        'beta_tail': float(beta_tail) if not np.isnan(beta_tail) else np.nan,
    }


def berkowitz_pit_test(returns, var_forecasts, es_forecasts=None, alpha=0.05):
    """
    Berkowitz (2001) PIT (Probability Integral Transform) test.
    Under correct model: u_t = F_t(r_t) ~ U(0,1) → x_t = Phi^{-1}(u_t) ~ N(0,1)

    Here we approximate u_t empirically (fraction of rolling historical window below r_t).

    Test H0: x_t ~ iid N(0,1) using Ljung-Box on x_t and x_t^2.

    returns: 1D array
    var_forecasts: 1D array of VaR (positive = loss)

    Returns dict: pit_mean, pit_std, pit_acf1,
                  lb_pvalue_level, lb_pvalue_sq
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    # Empirical CDF: for each t, fraction of all observations <= r_t
    # (a simple full-sample ECDF approximation)
    ranks = np.array([np.mean(returns <= r) for r in returns])
    u_t = np.clip(ranks, 1e-4, 1.0 - 1e-4)
    x_t = norm.ppf(u_t)

    pit_mean = float(np.mean(x_t))
    pit_std = float(np.std(x_t, ddof=1))
    pit_acf1 = float(np.corrcoef(x_t[:-1], x_t[1:])[0, 1]) if n > 2 else np.nan

    # Ljung-Box on levels
    lb_level = acorr_ljungbox(x_t, lags=[10], return_df=True)
    lb_sq = acorr_ljungbox(x_t ** 2, lags=[10], return_df=True)

    lb_pvalue_level = float(lb_level['lb_pvalue'].iloc[-1])
    lb_pvalue_sq = float(lb_sq['lb_pvalue'].iloc[-1])

    return {
        'pit_mean': pit_mean,
        'pit_std': pit_std,
        'pit_acf1': pit_acf1,
        'lb_pvalue_level': lb_pvalue_level,
        'lb_pvalue_sq': lb_pvalue_sq,
    }


def basel_traffic_light(n_violations, n_obs, confidence=0.99):
    """
    Basel III/IV traffic light system for VaR backtesting.
    n_violations: number of VaR exceedances
    n_obs: total number of observations (typically 250 trading days)
    confidence: VaR confidence level (default 0.99)

    For 250-day window at 99% VaR:
      Green:  0-4 violations  → multiplier = 3
      Yellow: 5-9 violations  → multiplier = 3 + (n-4)*0.2
      Red:    10+ violations  → multiplier = 4

    Returns dict: zone, expected_violations, observed_violations,
                  multiplier, cumulative_pvalue
    """
    from scipy.stats import binom

    expected = n_obs * (1.0 - confidence)

    # Cumulative p-value: P(X >= n_violations) under H0
    # binom.sf(k, n, p) = P(X > k), so P(X >= k) = binom.sf(k-1, n, p)
    p_tail = 1.0 - confidence  # probability of a single violation
    cum_pvalue = float(binom.sf(n_violations - 1, n_obs, p_tail))

    if n_violations <= 4:
        zone = 'green'
        multiplier = 3.0
    elif n_violations <= 9:
        zone = 'yellow'
        multiplier = 3.0 + (n_violations - 4) * 0.2
    else:
        zone = 'red'
        multiplier = 4.0

    return {
        'zone': zone,
        'expected_violations': float(expected),
        'observed_violations': int(n_violations),
        'multiplier': float(multiplier),
        'cumulative_pvalue': cum_pvalue,
    }


# ============================================================================
# New backtesting tests: DQ, McNeil-Frey, Nolde-Ziegel
# ============================================================================

def dq_test(returns, var_forecasts, alpha=0.05, p=4):
    """
    Dynamic Quantile (DQ) test — Engle & Manganelli (2004).

    Hit_t = 1{L_t > VaR_t} - alpha  (centred, E=0 under H0).
    Regressor matrix X (n x q), n = T-p, q = p+2:
        columns: [1, Hit_{t-1}, ..., Hit_{t-p}, VaR_t]
    Test statistic: DQ = (Hit' X (X'X)^{-1} X' Hit) / (alpha*(1-alpha))
    DQ ~ chi2(q) under H0.  p < 0.05 → reject.

    Inputs: returns (1D, actual returns), var_forecasts (1D, positive loss convention).
    Returns: {"stat", "pvalue", "reject", "note"}.
    """
    from scipy.stats import chi2 as _chi2

    L = -np.asarray(returns, dtype=float)
    v = np.asarray(var_forecasts, dtype=float)
    T = len(L)

    Hit = (L > v).astype(float) - alpha   # centred indicator

    n = T - p
    q = p + 2                             # constant + p lags + VaR
    if n <= q + 1:
        return {"stat": np.nan, "pvalue": np.nan, "reject": False,
                "note": "Yetersiz gözlem"}

    # Build X (n × q): constant | lag-j Hit | VaR_t
    X = np.ones((n, q))
    for j in range(1, p + 1):
        X[:, j] = Hit[(p - j):(T - j)]   # Hit_{t-j}, t = p,...,T-1
    X[:, p + 1] = v[p:]                  # VaR_t,       t = p,...,T-1

    Hit_dep = Hit[p:]                     # Hit_t,        t = p,...,T-1

    try:
        XtX = X.T @ X
        note = ""
        if np.linalg.cond(XtX) > 1e12:
            XtX_inv = np.linalg.pinv(XtX)
            note = "Pseudo-ters (tekil X'X)"
        else:
            XtX_inv = np.linalg.inv(XtX)

        XtH    = X.T @ Hit_dep           # shape (q,)
        dq_stat = float(XtH @ XtX_inv @ XtH / (alpha * (1.0 - alpha)))
        pvalue  = float(1.0 - _chi2.cdf(dq_stat, df=q))
        return {"stat": dq_stat, "pvalue": pvalue, "reject": pvalue < 0.05, "note": note}
    except np.linalg.LinAlgError:
        return {"stat": np.nan, "pvalue": np.nan, "reject": False,
                "note": "Matris terslenemedi"}


def mcneil_frey_test(returns, var_forecasts, es_forecasts, alpha=0.05, n_boot=1000):
    """
    McNeil & Frey (2000) exceedance-residuals test for ES.

    Violation days: er_t = L_t - ES_t  (should be ≈ 0 under H0).
    H0: E[er] = 0.  H1: E[er] > 0  (ES underestimated, one-sided).
    p-value via centred bootstrap.

    Inputs: returns, var_forecasts, es_forecasts (all positive loss convention).
    Returns: {"stat" (t-stat), "pvalue", "reject", "note"}.
    """
    L = -np.asarray(returns, dtype=float)
    v = np.asarray(var_forecasts, dtype=float)
    e = np.clip(np.asarray(es_forecasts, dtype=float), 1e-8, None)

    hits   = L > v
    n_hits = int(hits.sum())

    if n_hits < 5:
        return {"stat": np.nan, "pvalue": np.nan, "reject": False,
                "note": f"Yetersiz ihlal (n={n_hits})"}

    er     = L[hits] - e[hits]          # exceedance residuals
    er_std = float(np.std(er, ddof=1))

    if er_std < 1e-12:
        return {"stat": np.nan, "pvalue": np.nan, "reject": False,
                "note": "Sıfır varyans"}

    t_obs = float(np.mean(er)) / (er_std / np.sqrt(n_hits))

    # Centred bootstrap (H0 world: zero mean)
    er0 = er - np.mean(er)
    rng = np.random.default_rng(42)
    t_boot = np.empty(n_boot)
    for b in range(n_boot):
        samp  = rng.choice(er0, size=n_hits, replace=True)
        s_std = float(np.std(samp, ddof=1))
        t_boot[b] = (float(np.mean(samp)) / (s_std / np.sqrt(n_hits))
                     if s_std > 0 else 0.0)

    pvalue = float(np.mean(t_boot >= t_obs))
    return {"stat": t_obs, "pvalue": pvalue, "reject": pvalue < 0.05,
            "note": f"n_ihlal={n_hits}"}


def nz_conditional_calibration(returns, var_forecasts, es_forecasts, alpha=0.05):
    """
    Nolde-Ziegel (2017) unconditional calibration test for (VaR, ES).

    Identification functions (loss convention, v=VaR>0, e=ES>0, L=-r):
        V1_t = 1{L_t > v_t} - alpha
        V2_t = (1/alpha)*1{L_t > v_t}*(L_t - v_t)  -  (e_t - v_t)

    SIGN NOTE: The task-prompt formula gave V2 = +(e-v) + (1/α)·hit·(L-v),
    which yields E[V2] = 2(ES-VaR) ≠ 0 under H0 and would always reject.
    The correct sign per NZ (2017) is -(e-v), implemented here.

    Wald statistic: T_stat = T · Vbar' · Omega^{-1} · Vbar ~ chi2(2) under H0.

    Returns: {"stat", "pvalue", "reject", "note"}.
    """
    from scipy.stats import chi2 as _chi2

    L = -np.asarray(returns, dtype=float)
    v = np.asarray(var_forecasts, dtype=float)
    e = np.asarray(es_forecasts, dtype=float)
    T = len(L)

    hit = (L > v).astype(float)
    V1  = hit - alpha
    V2  = (1.0 / alpha) * hit * (L - v) - (e - v)

    V    = np.column_stack([V1, V2])   # (T, 2)
    Vbar = V.mean(axis=0)              # (2,)
    Vc   = V - Vbar
    Omega = (Vc.T @ Vc) / T

    try:
        note = ""
        if np.linalg.cond(Omega) > 1e12:
            Omega_inv = np.linalg.pinv(Omega)
            note = "Pseudo-ters (tekil Ω)"
        else:
            Omega_inv = np.linalg.inv(Omega)

        T_stat = float(T * Vbar @ Omega_inv @ Vbar)
        pvalue = float(1.0 - _chi2.cdf(T_stat, df=2))
        return {"stat": T_stat, "pvalue": pvalue, "reject": pvalue < 0.05, "note": note}
    except np.linalg.LinAlgError:
        return {"stat": np.nan, "pvalue": np.nan, "reject": False,
                "note": "Matris terslenemedi"}


# Quick self-test
if __name__ == "__main__":
    from pathlib import Path
    _data_csv = Path(__file__).parent / "data" / "sample_returns.csv"
    df = pd.read_csv(_data_csv, index_col=0, parse_dates=True)
    _first_col = [c for c in df.columns if not c.endswith("_RV") and not c.endswith("_BPV")][0]
    returns = df[_first_col]

    # Compute static PELVE
    pelve_val = calculate_pelve_single(-returns.values)  # input is losses
    print(f"Static PELVE for {_first_col}:", pelve_val)

    # Compute GARCH-based VaR/ES and backtest
    from arch import arch_model
    model = arch_model(returns * 100, vol='Garch', p=1, q=1)
    res = model.fit(disp='off')

    cond_vol = res.conditional_volatility / 100
    # Parametric forecasts
    var_forecast = cond_vol * norm.ppf(0.95)
    es_forecast = cond_vol * (norm.pdf(norm.ppf(0.95)) / 0.05)

    # Backtest
    bt_var = backtest_var(returns.values, var_forecast.values, alpha=0.05)
    bt_es = backtest_es_acerbi_szekely(returns.values, var_forecast.values, es_forecast.values, alpha=0.05)
    fz_loss = fissler_ziegel_loss(returns.values, var_forecast.values, es_forecast.values, alpha=0.05)

    print("VaR Backtest:", bt_var)
    print("ES Backtest:", bt_es)
    print("Joint FZ Loss:", fz_loss)


##################################################################


# ============================================================================
# Taylor (2020) forecast combination for VaR and ES (via the AL/FZ joint score)
# ============================================================================
def min_score_combine(returns, var_mat, es_mat, alpha=0.05):
    """
    Taylor (2020) MINIMUM-SCORE combining (Eqs. 3-4). Combines VaR forecasts and
    the ES-VaR spacing with separate convex weights, estimated jointly by
    minimising the AL score. Inputs in LOSS convention.

    Parameters
    ----------
    returns : (T,) array
    var_mat, es_mat : (T, M) individual VaR / ES forecasts (loss convention, >0)

    Returns
    -------
    var_c, es_c : (T,) combined forecasts (loss convention); wQ, wS : weights
    """
    from scipy.optimize import minimize
    y = np.asarray(returns)
    Qm = -np.asarray(var_mat); Em = -np.asarray(es_mat)     # returns convention
    T, M = Qm.shape

    def _al(Qc, ESc):
        ESc = np.minimum(ESc, Qc - 1e-9)
        hit = (y <= Qc).astype(float)
        return (Qc / ESc - hit * (Qc - y) / (alpha * ESc) + np.log(-ESc)).mean()

    def obj(th):
        wQ, wS = th[:M], th[M:]
        Qc = Qm @ wQ
        return _al(Qc, Qc + (Em - Qm) @ wS)

    r = minimize(obj, np.r_[np.ones(M) / M, np.ones(M) / M], method="SLSQP",
                 bounds=[(0.0, 1.0)] * (2 * M),
                 constraints=[{"type": "eq", "fun": lambda th: th[:M].sum() - 1},
                              {"type": "eq", "fun": lambda th: th[M:].sum() - 1}],
                 options={"maxiter": 300, "ftol": 1e-10})
    wQ, wS = r.x[:M], r.x[M:]
    Qc = Qm @ wQ
    ESc = np.minimum(Qc + (Em - Qm) @ wS, Qc - 1e-9)
    return -Qc, -ESc, wQ, wS                                # back to loss convention


def relative_score_combine(returns, var_mat, es_mat, alpha=0.05):
    """
    Taylor (2020) RELATIVE-SCORE combining (Eqs. 5-7). Softmax weights over the
    cumulative in-sample AL score, single tuning parameter lambda (optimised).
    Inputs in LOSS convention. Returns var_c, es_c, weights w, lambda.
    """
    from scipy.optimize import minimize_scalar
    y = np.asarray(returns)
    Qm = -np.asarray(var_mat); Em = -np.asarray(es_mat)
    T, M = Qm.shape

    def _al_series(Qc, ESc):
        ESc = np.minimum(ESc, Qc - 1e-9)
        hit = (y <= Qc).astype(float)
        return Qc / ESc - hit * (Qc - y) / (alpha * ESc) + np.log(-ESc)

    cumS = np.array([_al_series(Qm[:, i], Em[:, i]).sum() for i in range(M)])
    c0 = cumS - cumS.min()

    def obj(lam):
        w = np.exp(-lam * c0); w /= w.sum()
        return _al_series(Qm @ w, Em @ w).mean()

    lam = max(minimize_scalar(obj, bounds=(0.0, 1e4), method="bounded").x, 0.0)
    w = np.exp(-lam * c0); w /= w.sum()
    return -(Qm @ w), -(Em @ w), w, lam
