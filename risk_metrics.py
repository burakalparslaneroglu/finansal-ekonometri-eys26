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
    Acerbi-Szekely (2014) ES backtest (Z1 statistic).
    Z1 is defined as:
    Z1 = sum_{t=1}^N ( (L_t * I_t) / (alpha * ES_t) ) / N - 1
    where I_t = 1 if L_t > VaR_t, and L_t is loss.
    Under the null hypothesis, E[Z1] = 0. A highly negative value indicates risk underestimation.
    returns: 1D array of returns
    var_forecasts: 1D array of VaR forecasts
    es_forecasts: 1D array of ES forecasts
    """
    losses = -np.asarray(returns)
    var_forecasts = np.asarray(var_forecasts)
    es_forecasts = np.asarray(es_forecasts)
    
    hits = (losses > var_forecasts).astype(int)
    
    # Prevent division by zero
    es_forecasts = np.clip(es_forecasts, 1e-6, None)
    
    # Z1 statistic
    z1_terms = (losses * hits) / (alpha * es_forecasts)
    z1_stat = np.mean(z1_terms) - 1.0
    
    # Since the distribution of Z1 is highly non-normal under the null, we use a simulation-based p-value
    # by simulating standardized losses from a normal distribution and calculating Z1
    n_sims = 1000
    N = len(losses)
    sim_stats = []
    
    for _ in range(n_sims):
        # Generate simulated losses under H0 (e.g., standard normal)
        sim_losses = np.random.normal(0, 1, N)
        # Scale to match forecast ES (average)
        sim_var = norm.ppf(1.0 - alpha)
        sim_es = norm.pdf(sim_var) / alpha
        
        sim_hits = (sim_losses > sim_var).astype(int)
        sim_z1 = np.mean((sim_losses * sim_hits) / (alpha * sim_es)) - 1.0
        sim_stats.append(sim_z1)
        
    sim_stats = np.array(sim_stats)
    # P-value is the proportion of simulated Z1 stats that are more negative than observed Z1
    p_value = np.mean(sim_stats <= z1_stat)
    
    return {
        "z1_stat": z1_stat,
        "pvalue": p_value
    }

def fissler_ziegel_loss(returns, var_forecasts, es_forecasts, alpha=0.05):
    """
    Computes the Fissler-Ziegel (FZ) joint VaR-ES loss function.
    Highly useful for forecast combinations and model comparisons since it is elicitable.
    Using g1(x) = x, g2(x) = -1/x, G2(x) = -log(-x)
    """
    losses = -np.asarray(returns)
    var_forecasts = np.asarray(var_forecasts)
    es_forecasts = np.asarray(es_forecasts)
    
    # Enforce positivity and logical bounds (ES >= VaR)
    # FZ loss expects positive VaR and ES representing losses
    var = np.clip(var_forecasts, 1e-6, None)
    es = np.clip(es_forecasts, var, None)
    
    hits = (losses > var).astype(int)
    
    # FZ formula
    # S = (I - alpha)*(-var) - I*(-losses) + (-1/es)*(es + 1/alpha*(losses - var)*I) - (-log(es))
    term1 = (hits - alpha) * (-var)
    term2 = hits * losses
    term3 = (-1.0 / es) * (es + (1.0 / alpha) * (losses - var) * hits)
    term4 = np.log(es)
    
    loss_series = term1 + term2 + term3 + term4
    return np.mean(loss_series)

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
