import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.covariance import ledoit_wolf
from scipy.optimize import minimize
import warnings

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

def estimate_har_rv(rv_series, lags=(1, 5, 22)):
    """
    Estimates the Heterogeneous Autoregressive model of Realized Volatility (HAR-RV) (Corsi, 2009).
    rv_series: pandas Series of realized volatility
    lags: tuple of (daily, weekly, monthly) lags (default: (1, 5, 22))
    Returns: statsmodels regression result object and the model design matrix DataFrame
    """
    n_obs = len(rv_series)
    max_lag = max(lags)
    
    # Compute averages of lags
    df_har = pd.DataFrame(index=rv_series.index)
    df_har["RV"] = rv_series
    
    # Daily lag (RV_{t-1})
    df_har["RV_d"] = rv_series.shift(1)
    
    # Weekly lag (average of past 5 days)
    df_har["RV_w"] = rv_series.shift(1).rolling(window=lags[1]).mean()
    
    # Monthly lag (average of past 22 days)
    df_har["RV_m"] = rv_series.shift(1).rolling(window=lags[2]).mean()
    
    # Drop NaNs due to rolling averages
    df_har_clean = df_har.dropna()
    
    # Regress
    X = df_har_clean[["RV_d", "RV_w", "RV_m"]]
    X = sm.add_constant(X)
    y = df_har_clean["RV"]
    
    model = sm.OLS(y, X)
    results = model.fit()
    
    return results, df_har_clean

def estimate_har_rv_j(rv_series, bpv_series, lags=(1, 5, 22)):
    """
    Estimates the HAR-RV-J model (HAR with Jumps separated using Bipower Variation).
    RV_t = beta_0 + beta_d * BPV_{t-1} + beta_w * BPV_{t-1:t-5} + beta_m * BPV_{t-1:t-22} + beta_j * J_{t-1} + eps_t
    """
    df_har = pd.DataFrame(index=rv_series.index)
    df_har["RV"] = rv_series
    df_har["BPV"] = bpv_series
    
    # Jump component: J_t = max(0, RV_t - BPV_t)
    df_har["J"] = (rv_series - bpv_series).clip(lower=0)
    
    # Daily BPV and J lags
    df_har["BPV_d"] = bpv_series.shift(1)
    df_har["J_d"] = df_har["J"].shift(1)
    
    # Weekly BPV lag
    df_har["BPV_w"] = bpv_series.shift(1).rolling(window=lags[1]).mean()
    
    # Monthly BPV lag
    df_har["BPV_m"] = bpv_series.shift(1).rolling(window=lags[2]).mean()
    
    df_har_clean = df_har.dropna()
    
    X = df_har_clean[["BPV_d", "BPV_w", "BPV_m", "J_d"]]
    X = sm.add_constant(X)
    y = df_har_clean["RV"]
    
    model = sm.OLS(y, X)
    results = model.fit()
    
    return results, df_har_clean

def poet_covariance(returns, k_factors=3, threshold=0.1):
    """
    Implements POET (Principal Orthogonal Complement Thresholding) covariance estimator (Fan et al., 2013).
    returns: DataFrame of asset returns (shape: n_obs x n_assets)
    k_factors: Number of principal factors to extract (default: 3)
    threshold: Threshold parameter for idiosyncratic matrix (default: 0.1)
    """
    X = np.asarray(returns)
    n_obs, n_assets = X.shape
    
    # 1. Demean returns
    X_mean = np.mean(X, axis=0)
    X_demeaned = X - X_mean
    
    # Sample covariance matrix
    Sigma_sample = np.cov(X_demeaned.T)
    
    # 2. Eigenvalue Decomposition of Sample Covariance
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma_sample)
    
    # Sort eigenvalues & eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[idx]
    
    # 3. Factor component
    # Keep the top k eigenvalues and eigenvectors
    V_k = eigenvectors[:, :k_factors]
    D_k = np.diag(eigenvalues[:k_factors])
    
    # Factor covariance matrix: F = V_k * D_k * V_k'
    Sigma_factor = V_k @ D_k @ V_k.T
    
    # 4. Idiosyncratic component
    Sigma_idiosyncratic = Sigma_sample - Sigma_factor
    
    # 5. Apply adaptive thresholding to idiosyncratic matrix
    # Soft thresholding helper: s_ij = sign(x) * max(0, |x| - tau)
    # Scale threshold by the variance of residuals to make it adaptive: tau_ij = threshold * sqrt(var_i * var_j)
    diag_vars = np.diag(Sigma_idiosyncratic)
    Sigma_poet_idio = np.zeros_like(Sigma_idiosyncratic)
    
    for i in range(n_assets):
        for j in range(n_assets):
            val = Sigma_idiosyncratic[i, j]
            if i == j:
                Sigma_poet_idio[i, j] = val # Keep diagonal unchanged
            else:
                tau = threshold * np.sqrt(max(1e-8, diag_vars[i] * diag_vars[j]))
                # Soft thresholding
                Sigma_poet_idio[i, j] = np.sign(val) * max(0, abs(val) - tau)
                
    # 6. Reconstruct POET covariance
    Sigma_poet = Sigma_factor + Sigma_poet_idio
    
    # Ensure positive definiteness (project eigenvalues to be positive if needed)
    evals, evecs = np.linalg.eigh(Sigma_poet)
    if np.any(evals < 0):
        evals_clipped = np.clip(evals, 1e-6, None)
        Sigma_poet = evecs @ np.diag(evals_clipped) @ evecs.T
        
    return Sigma_poet

def ledoit_wolf_covariance(returns):
    """
    Computes the Ledoit-Wolf shrinkage covariance estimator.
    returns: DataFrame of asset returns (shape: n_obs x n_assets)
    """
    X = np.asarray(returns)
    shrinked_cov, shrinkage = ledoit_wolf(X)
    return shrinked_cov, shrinkage

def estimate_har_rv_cj(rv_series, bpv_series, lags=(1, 5, 22)):
    """
    HAR-RV-CJ: separates continuous (BPV) and jump (J) components at multiple horizons.
    Andersen, Bollerslev, Diebold (2007, JASA).

    RV_{t+1} = beta_0
             + beta_cd * C_t + beta_cw * C_t^(w) + beta_cm * C_t^(m)
             + beta_jd * J_t
             + eps_{t+1}

    C_t = BPV_t (continuous path)
    J_t = max(0, RV_t - BPV_t) (jump component)
    C_t^(w) = mean of C over past 5 days
    C_t^(m) = mean of C over past 22 days

    Uses HAC standard errors (Newey-West, lags=5) via statsmodels.

    Returns: (results, df_har_cj) where df_har_cj has columns
    [RV, C_d, C_w, C_m, J_d, fitted, resid]
    """
    df = pd.DataFrame(index=rv_series.index)
    df["RV"] = rv_series.values
    df["C"] = np.maximum(0.0, bpv_series.values)          # continuous component
    df["J"] = np.maximum(0.0, rv_series.values - bpv_series.values)  # jump component

    # Lagged continuous components
    df["C_d"] = df["C"].shift(1)
    df["C_w"] = df["C"].shift(1).rolling(window=lags[1]).mean()
    df["C_m"] = df["C"].shift(1).rolling(window=lags[2]).mean()

    # Lagged daily jump
    df["J_d"] = df["J"].shift(1)

    df_clean = df.dropna()

    X = df_clean[["C_d", "C_w", "C_m", "J_d"]]
    X = sm.add_constant(X)
    y = df_clean["RV"]

    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})

    df_clean = df_clean.copy()
    df_clean["fitted"] = results.fittedvalues
    df_clean["resid"] = results.resid

    return results, df_clean[["RV", "C_d", "C_w", "C_m", "J_d", "fitted", "resid"]]


def estimate_har_rv_hac(rv_series, lags=(1, 5, 22), hac_lags=5):
    """
    HAR-RV with Newey-West HAC standard errors (Corsi 2009 + robust inference).

    Wraps estimate_har_rv but fits with HAC covariance.
    rv_series: pandas Series of realized volatility
    lags: (daily, weekly, monthly) horizons
    hac_lags: number of lags for Newey-West (default 5)

    Returns: (results_hac, df_har_clean) — same shape as estimate_har_rv
    """
    n_obs = len(rv_series)
    df_har = pd.DataFrame(index=rv_series.index)
    df_har["RV"] = rv_series
    df_har["RV_d"] = rv_series.shift(1)
    df_har["RV_w"] = rv_series.shift(1).rolling(window=lags[1]).mean()
    df_har["RV_m"] = rv_series.shift(1).rolling(window=lags[2]).mean()

    df_har_clean = df_har.dropna()

    X = df_har_clean[["RV_d", "RV_w", "RV_m"]]
    X = sm.add_constant(X)
    y = df_har_clean["RV"]

    model = sm.OLS(y, X)
    results_hac = model.fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})

    return results_hac, df_har_clean


# ---------------------------------------------------------------------------
# Numba-accelerated inner loops (with numpy fallback)
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @njit
    def _heavy_filter(rv_arr, omega, alpha, beta, sigma2_0):
        """Inner time loop for HEAVY model (numba version)."""
        T = len(rv_arr)
        sigma2 = np.empty(T)
        sigma2[0] = sigma2_0
        for t in range(1, T):
            sigma2[t] = omega + alpha * rv_arr[t - 1] + beta * sigma2[t - 1]
        return sigma2

    @njit
    def _garch_x_filter(eps_arr, rv_arr, omega, alpha, beta, gamma, sigma2_0):
        """Inner time loop for GARCH-X model (numba version)."""
        T = len(eps_arr)
        sigma2 = np.empty(T)
        sigma2[0] = sigma2_0
        for t in range(1, T):
            sigma2[t] = (omega
                         + alpha * eps_arr[t - 1] ** 2
                         + beta * sigma2[t - 1]
                         + gamma * rv_arr[t - 1])
        return sigma2
else:
    def _heavy_filter(rv_arr, omega, alpha, beta, sigma2_0):
        """Inner time loop for HEAVY model (numpy version)."""
        T = len(rv_arr)
        sigma2 = np.empty(T)
        sigma2[0] = sigma2_0
        for t in range(1, T):
            sigma2[t] = omega + alpha * rv_arr[t - 1] + beta * sigma2[t - 1]
        return sigma2

    def _garch_x_filter(eps_arr, rv_arr, omega, alpha, beta, gamma, sigma2_0):
        """Inner time loop for GARCH-X model (numpy version)."""
        T = len(eps_arr)
        sigma2 = np.empty(T)
        sigma2[0] = sigma2_0
        for t in range(1, T):
            sigma2[t] = (omega
                         + alpha * eps_arr[t - 1] ** 2
                         + beta * sigma2[t - 1]
                         + gamma * rv_arr[t - 1])
        return sigma2


def estimate_heavy(rv_series, returns_series=None):
    """
    HEAVY (High-frEquency-bAsed VolatilitY) model (Shephard-Sheppard, 2010):
      sigma2_t = omega + alpha * RV_{t-1} + beta * sigma2_{t-1}

    Estimation via QMLE (Gaussian log-likelihood).
    Initialises sigma2_0 = var(returns) if returns_series given, else mean(rv_series).
    Uses scipy.optimize.minimize with L-BFGS-B.

    rv_series: pandas Series of realized variance (daily)
    returns_series: optional pandas Series of daily returns for sigma2_0

    Returns dict with keys:
        'params'         : {'omega': w, 'alpha': a, 'beta': b}
        'sigma2_series'  : pd.Series of conditional variance (same index as rv_series)
        'loglik'         : float
        'persistence'    : alpha + beta
        'half_life_days' : log(0.5) / log(alpha + beta)
    """
    rv_arr = np.asarray(rv_series, dtype=float)
    T = len(rv_arr)

    if returns_series is not None:
        ret_arr = np.asarray(returns_series, dtype=float)
        sigma2_0 = float(np.var(ret_arr))
    else:
        sigma2_0 = float(np.mean(rv_arr))
    if sigma2_0 <= 0:
        sigma2_0 = 1e-6

    def neg_loglik(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 1:
            return 1e10
        sigma2 = _heavy_filter(rv_arr, omega, alpha, beta, sigma2_0)
        sigma2 = np.maximum(sigma2, 1e-12)
        ll = -0.5 * np.sum(np.log(sigma2) + rv_arr / sigma2)
        return -ll

    # Starting values
    omega0 = sigma2_0 * 0.05
    alpha0 = 0.10
    beta0  = 0.85
    x0 = np.array([omega0, alpha0, beta0])

    bounds = [(1e-9, None), (1e-6, 0.999), (1e-6, 0.999)]
    result = minimize(neg_loglik, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-12})

    omega, alpha, beta = result.x
    sigma2_path = _heavy_filter(rv_arr, omega, alpha, beta, sigma2_0)
    loglik = -result.fun

    persistence = alpha + beta
    half_life = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.inf

    return {
        'params': {'omega': omega, 'alpha': alpha, 'beta': beta},
        'sigma2_series': pd.Series(sigma2_path, index=rv_series.index, name='sigma2_heavy'),
        'loglik': loglik,
        'persistence': persistence,
        'half_life_days': half_life,
    }


def estimate_garch_x(returns_series, rv_series):
    """
    GARCH-X: sigma2_t = omega + alpha*eps2_{t-1} + beta*sigma2_{t-1} + gamma*RV_{t-1}

    QMLE Gaussian log-likelihood:
      L = -0.5 * sum_t (log(sigma2_t) + eps_t^2 / sigma2_t)

    Bounds: omega>0, alpha>0, beta>0, gamma>=0, alpha+beta+gamma < 1.
    Numerical Hessian used for standard errors.

    returns_series, rv_series: pandas Series aligned on the same index.

    Returns dict with keys:
        'params'         : {'omega':, 'alpha':, 'beta':, 'gamma':}
        'sigma2_series'  : pd.Series
        'loglik'         : float
        'gamma_tstat'    : float  (H0: gamma=0)
        'persistence'    : alpha+beta+gamma
    """
    # Align series
    aligned = pd.DataFrame({'ret': returns_series, 'rv': rv_series}).dropna()
    eps_arr = np.asarray(aligned['ret'], dtype=float)
    rv_arr  = np.asarray(aligned['rv'],  dtype=float)
    idx     = aligned.index
    T       = len(eps_arr)

    sigma2_0 = float(np.var(eps_arr))
    if sigma2_0 <= 0:
        sigma2_0 = 1e-6

    def neg_loglik(params):
        omega, alpha, beta, gamma = params
        if omega <= 0 or alpha <= 0 or beta <= 0 or gamma < 0:
            return 1e10
        if alpha + beta + gamma >= 1:
            return 1e10
        sigma2 = _garch_x_filter(eps_arr, rv_arr, omega, alpha, beta, gamma, sigma2_0)
        sigma2 = np.maximum(sigma2, 1e-12)
        ll = -0.5 * np.sum(np.log(sigma2) + eps_arr ** 2 / sigma2)
        return -ll

    x0 = np.array([1e-4, 0.05, 0.85, 0.05])
    bounds = [(1e-9, None), (1e-6, 0.999), (1e-6, 0.999), (0.0, 0.999)]
    result = minimize(neg_loglik, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 2000, 'ftol': 1e-12})

    omega, alpha, beta, gamma = result.x
    sigma2_path = _garch_x_filter(eps_arr, rv_arr, omega, alpha, beta, gamma, sigma2_0)
    loglik = -result.fun

    # Numerical Hessian for standard errors (approx_fprime approach)
    from scipy.optimize import approx_fprime
    eps_h = 1e-5 * np.abs(result.x) + 1e-8

    def grad_nll(p):
        return approx_fprime(p, neg_loglik, eps_h)

    # Finite-difference Hessian
    H = np.zeros((4, 4))
    g0 = grad_nll(result.x)
    for j in range(4):
        dx = np.zeros(4)
        dx[j] = eps_h[j]
        gj = grad_nll(result.x + dx)
        H[:, j] = (gj - g0) / eps_h[j]
    H = 0.5 * (H + H.T)  # symmetrize

    try:
        cov_params = np.linalg.inv(H)
        se_gamma = np.sqrt(max(cov_params[3, 3], 0.0))
    except np.linalg.LinAlgError:
        se_gamma = np.nan

    gamma_tstat = gamma / se_gamma if se_gamma and se_gamma > 0 else np.nan
    persistence = alpha + beta + gamma

    return {
        'params': {'omega': omega, 'alpha': alpha, 'beta': beta, 'gamma': gamma},
        'sigma2_series': pd.Series(sigma2_path, index=idx, name='sigma2_garchx'),
        'loglik': loglik,
        'gamma_tstat': gamma_tstat,
        'persistence': persistence,
    }


def estimate_realized_garch(returns_series, rv_series):
    """
    Realized GARCH two-equation system in log-log form (Hansen, Huang, Shek 2012).

    GARCH eq:    log(sigma2_t) = omega + beta*log(sigma2_{t-1}) + gamma*log(RV_{t-1})
    Measurement: log(RV_t)     = xi + phi*log(sigma2_t) + tau1*z_t + tau2*(z_t^2-1) + u_t

    z_t = r_t / sigma_t,   u_t ~ N(0, sigma_u^2)

    Joint QMLE over 8 parameters: [omega, beta, gamma, xi, phi, tau1, tau2, sigma_u].

    Returns dict with keys:
        'params'           : dict of all 8 parameters
        'log_sigma2_series': pd.Series of conditional log-variance
        'sigma2_series'    : pd.Series of conditional variance
        'loglik'           : float
        'persistence'      : beta + gamma
        'leverage_tau1'    : tau1
    """
    aligned = pd.DataFrame({'ret': returns_series, 'rv': rv_series}).dropna()
    r_arr  = np.asarray(aligned['ret'], dtype=float)
    rv_arr = np.asarray(aligned['rv'],  dtype=float)
    idx    = aligned.index
    T      = len(r_arr)

    # Use log-RV; guard against non-positive values
    rv_pos = np.maximum(rv_arr, 1e-12)
    log_rv = np.log(rv_pos)

    sigma2_0 = float(np.var(r_arr))
    if sigma2_0 <= 0:
        sigma2_0 = float(np.mean(rv_pos))
    log_sigma2_0 = np.log(max(sigma2_0, 1e-12))

    def neg_loglik(params):
        omega, beta, gamma, xi, phi, tau1, tau2, log_sigma_u = params
        sigma_u = np.exp(log_sigma_u)  # ensure positivity

        if not (0 < beta < 1 and 0 < gamma < 1 and beta + gamma < 1):
            return 1e10
        if phi <= 0:
            return 1e10

        # --- GARCH filter ---
        log_sigma2 = np.empty(T)
        log_sigma2[0] = log_sigma2_0
        for t in range(1, T):
            log_sigma2[t] = omega + beta * log_sigma2[t - 1] + gamma * log_rv[t - 1]

        sigma2 = np.exp(log_sigma2)
        sigma  = np.sqrt(np.maximum(sigma2, 1e-12))
        z      = r_arr / sigma

        # --- Return log-likelihood ---
        # Part 1: returns equation (standard normal)
        ll_ret = -0.5 * np.sum(log_sigma2 + z ** 2)

        # Part 2: measurement equation
        u = log_rv - xi - phi * log_sigma2 - tau1 * z - tau2 * (z ** 2 - 1)
        ll_meas = -0.5 * T * np.log(sigma_u ** 2) - 0.5 * np.sum(u ** 2) / (sigma_u ** 2)

        return -(ll_ret + ll_meas)

    # Starting values
    x0 = np.array([
        -0.1,    # omega
        0.7,     # beta
        0.2,     # gamma
        0.0,     # xi
        1.0,     # phi
        0.0,     # tau1
        0.0,     # tau2
        np.log(0.1),  # log(sigma_u)
    ])

    result = minimize(neg_loglik, x0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})

    omega, beta, gamma, xi, phi, tau1, tau2, log_sigma_u = result.x
    sigma_u = np.exp(log_sigma_u)

    # Reconstruct conditional variance path
    log_sigma2_path = np.empty(T)
    log_sigma2_path[0] = log_sigma2_0
    for t in range(1, T):
        log_sigma2_path[t] = omega + beta * log_sigma2_path[t - 1] + gamma * log_rv[t - 1]

    sigma2_path = np.exp(log_sigma2_path)
    loglik = -result.fun

    return {
        'params': {
            'omega': omega, 'beta': beta, 'gamma': gamma,
            'xi': xi, 'phi': phi, 'tau1': tau1, 'tau2': tau2,
            'sigma_u': sigma_u,
        },
        'log_sigma2_series': pd.Series(log_sigma2_path, index=idx, name='log_sigma2_rgarch'),
        'sigma2_series':     pd.Series(sigma2_path,     index=idx, name='sigma2_rgarch'),
        'loglik': loglik,
        'persistence': beta + gamma,
        'leverage_tau1': tau1,
    }


def compute_signature_plot(intraday_returns_df, freq_list=None):
    """
    Computes RV at multiple sampling frequencies to produce a volatility signature plot.

    intraday_returns_df: DataFrame where each row is one trading day and columns are
                         intraday 1-minute return intervals (e.g. 480 columns for 8-hour trading).
                         If None or empty, a theoretical (simulated) signature is returned.
    freq_list: list of sampling intervals in minutes relative to the 1-min base.
               Default: [1, 2, 5, 10, 15, 30, 60, 120]

    For each frequency f:
      - subsample every f-th return across the row
      - daily RV = sum of squared subsampled returns
      - average RV across all days

    Returns:
        pd.DataFrame with columns ['freq_min', 'mean_rv', 'annualized_vol_pct']
        and attribute-style access to 'optimal_freq' (argmin of second derivative of mean_rv).
    """
    if freq_list is None:
        freq_list = [1, 2, 5, 10, 15, 30, 60, 120]

    freq_list = sorted(freq_list)

    use_simulated = (
        intraday_returns_df is None
        or (hasattr(intraday_returns_df, '__len__') and len(intraday_returns_df) == 0)
    )

    if not use_simulated:
        data = np.asarray(intraday_returns_df, dtype=float)
        n_days, n_cols = data.shape
        records = []
        for f in freq_list:
            subsampled = data[:, ::f]          # take every f-th column
            daily_rv   = np.nansum(subsampled ** 2, axis=1)
            mean_rv    = float(np.nanmean(daily_rv))
            ann_vol    = float(np.sqrt(252.0 * mean_rv) * 100.0)
            records.append({'freq_min': f, 'mean_rv': mean_rv, 'annualized_vol_pct': ann_vol})
    else:
        # Theoretical signature plot under microstructure noise:
        # RV(f) = IV + 2 * eta2 / f   (Bandi-Russell decomposition)
        IV   = (0.01 ** 2)           # assume 1% daily vol
        eta2 = IV * 0.01             # noise-to-signal = 1%
        records = []
        for f in freq_list:
            mean_rv = IV + 2.0 * eta2 / f
            ann_vol = float(np.sqrt(252.0 * mean_rv) * 100.0)
            records.append({'freq_min': f, 'mean_rv': mean_rv, 'annualized_vol_pct': ann_vol})

    df_sig = pd.DataFrame(records)

    # Optimal frequency: inflection point = argmin of |second derivative| of mean_rv
    mean_rv_arr = df_sig['mean_rv'].values
    if len(mean_rv_arr) >= 3:
        d2 = np.abs(np.diff(mean_rv_arr, n=2))
        # index into freq_list: d2 has length len-2, corresponds to index 1..len-2
        opt_idx = int(np.argmin(d2)) + 1
        optimal_freq = int(df_sig['freq_min'].iloc[opt_idx])
    else:
        optimal_freq = int(df_sig['freq_min'].iloc[0])

    df_sig.attrs['optimal_freq'] = optimal_freq
    return df_sig


def marchenko_pastur_threshold(n_assets, n_obs, sigma2=1.0, eigenvalues=None):
    """
    Marchenko-Pastur upper bound for sample covariance eigenvalues under pure noise.

    kappa        = n_assets / n_obs
    lambda_plus  = sigma2 * (1 + sqrt(kappa))^2   (upper bulk edge)
    lambda_minus = sigma2 * (1 - sqrt(kappa))^2   (lower bulk edge)

    Parameters
    ----------
    n_assets    : int, number of assets (N)
    n_obs       : int, number of observations (T)
    sigma2      : float, noise variance (default 1.0)
    eigenvalues : array-like, optional. If provided, counts how many exceed lambda_plus.

    Returns dict:
        'kappa'               : N/T
        'lambda_plus'         : upper Marchenko-Pastur bound
        'lambda_minus'        : lower Marchenko-Pastur bound
        'n_signal_eigenvalues': count above lambda_plus (None if eigenvalues not given)
    """
    kappa = n_assets / n_obs
    sqrt_kappa = np.sqrt(kappa)

    lambda_plus  = sigma2 * (1.0 + sqrt_kappa) ** 2
    lambda_minus = sigma2 * (1.0 - sqrt_kappa) ** 2

    n_signal = None
    if eigenvalues is not None:
        eigs = np.asarray(eigenvalues, dtype=float)
        n_signal = int(np.sum(eigs > lambda_plus))

    return {
        'kappa': kappa,
        'lambda_plus': lambda_plus,
        'lambda_minus': lambda_minus,
        'n_signal_eigenvalues': n_signal,
    }


if __name__ == "__main__":
    from pathlib import Path
    _data_csv = Path(__file__).parent / "data" / "sample_returns.csv"
    df = pd.read_csv(_data_csv, index_col=0, parse_dates=True)
    
    _ret_cols = [c for c in df.columns if not c.endswith("_RV") and not c.endswith("_BPV")]
    _first = _ret_cols[0]

    # Test HAR-RV
    print("Testing HAR-RV...")
    har_res, _ = estimate_har_rv(df[f"{_first}_RV"])
    print(har_res.summary().tables[1])

    # Test HAR-RV-J
    print("\nTesting HAR-RV-J...")
    har_j_res, _ = estimate_har_rv_j(df[f"{_first}_RV"], df[f"{_first}_BPV"])
    print(har_j_res.summary().tables[1])

    # Test POET & Ledoit-Wolf
    print("\nTesting High-Dimensional Covariance estimators...")
    asset_cols = _ret_cols[:5]
    returns = df[asset_cols]
    
    poet_cov = poet_covariance(returns, k_factors=2, threshold=0.1)
    lw_cov, shrinkage = ledoit_wolf_covariance(returns)
    
    print("Sample Covariance Shape:", returns.cov().shape)
    print("POET Covariance Matrix (Asset 1-3):\n", poet_cov[:3, :3])
    print("Ledoit-Wolf Shrinkage intensity:", shrinkage)
