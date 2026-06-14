"""
dcc_garch.py
============
Multivariate GARCH models for financial econometrics.

Supported model types
---------------------
- "DCC"   : Dynamic Conditional Correlation (Engle, 2002)
- "cDCC"  : Corrected DCC (Aielli, 2013)
- "ADCC"  : Asymmetric DCC (Cappiello, Engle & Sheppard, 2006)
- "DECO"  : Dynamic Equicorrelation (Engle & Kelly, 2012)

All models share a two-step estimation routine:
  1.  fit_univariate_garch  – GARCH(1,1) per asset via the `arch` package
  2.  fit_dcc               – maximise the DCC composite likelihood

Optional numba acceleration is used for the inner time loop when the
`numba` package is available; the code falls back to pure NumPy otherwise.
"""

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from arch import arch_model

# ---------------------------------------------------------------------------
# Optional numba acceleration
# ---------------------------------------------------------------------------
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _dcc_loop_numpy(std_resid, a, b, c, bar_Q, N_bar, model_type_int):
    """
    Pure-NumPy inner loop that computes the DCC/cDCC/ADCC composite
    log-likelihood.

    Parameters
    ----------
    std_resid    : (T, N) array of standardised residuals
    a, b, c      : DCC scalar parameters (c=0 for DCC/cDCC)
    bar_Q        : (N, N) unconditional covariance of std_resid
    N_bar        : (N, N) unconditional mean of outer(n_t, n_t) (ADCC only)
    model_type_int : 0 = DCC, 1 = cDCC, 2 = ADCC

    Returns
    -------
    float  negative log-likelihood (positive = worse)
    """
    T, N = std_resid.shape
    Q = bar_Q.copy()
    loglike = 0.0

    for t in range(1, T):
        z = std_resid[t - 1]

        if model_type_int == 1:          # cDCC
            sqrt_diag = np.sqrt(np.diag(Q))
            P_mat = np.diag(sqrt_diag)
            z_star = P_mat @ z
            Q = (1.0 - a - b) * bar_Q + a * np.outer(z_star, z_star) + b * Q
        elif model_type_int == 2:        # ADCC
            n = z * (z < 0.0)
            Q = ((1.0 - a - b) * bar_Q
                 - c * N_bar
                 + a * np.outer(z, z)
                 + b * Q
                 + c * np.outer(n, n))
        else:                            # DCC
            Q = (1.0 - a - b) * bar_Q + a * np.outer(z, z) + b * Q

        # Normalise Q -> R
        inv_sqrt = 1.0 / np.sqrt(np.diag(Q))
        P_inv = np.diag(inv_sqrt)
        R = P_inv @ Q @ P_inv

        det_R = np.linalg.det(R)
        if det_R <= 0.0:
            return 1e10
        R_inv = np.linalg.inv(R)

        z_t = std_resid[t]
        loglike += 0.5 * (np.log(det_R) + z_t @ R_inv @ z_t - z_t @ z_t)

    return loglike


if HAS_NUMBA:
    from numba import njit as _njit

    @_njit
    def _dcc_loop_numba(std_resid, a, b, c, bar_Q, N_bar, model_type_int):
        """
        Numba-jitted inner loop (identical logic to _dcc_loop_numpy).
        model_type_int: 0=DCC, 1=cDCC, 2=ADCC
        """
        T, N = std_resid.shape
        Q = bar_Q.copy()
        loglike = 0.0

        for t in range(1, T):
            z = std_resid[t - 1]

            if model_type_int == 1:      # cDCC
                P_mat = np.diag(np.sqrt(np.diag(Q)))
                z_star = P_mat @ z
                Q = (1.0 - a - b) * bar_Q + a * np.outer(z_star, z_star) + b * Q
            elif model_type_int == 2:    # ADCC
                n = z * (z < 0.0)
                Q = ((1.0 - a - b) * bar_Q
                     - c * N_bar
                     + a * np.outer(z, z)
                     + b * Q
                     + c * np.outer(n, n))
            else:                        # DCC
                Q = (1.0 - a - b) * bar_Q + a * np.outer(z, z) + b * Q

            inv_sqrt = 1.0 / np.sqrt(np.diag(Q))
            P_inv = np.diag(inv_sqrt)
            R = P_inv @ Q @ P_inv

            det_R = np.linalg.det(R)
            if det_R <= 0.0:
                return 1e10
            R_inv = np.linalg.inv(R)

            z_t = std_resid[t]
            loglike += 0.5 * (np.log(det_R) + z_t @ R_inv @ z_t - z_t @ z_t)

        return loglike

    def _dcc_loop(std_resid, a, b, c, bar_Q, N_bar, model_type_int):
        """Dispatch to numba-jitted loop."""
        return _dcc_loop_numba(std_resid, a, b, c, bar_Q, N_bar, model_type_int)

else:
    def _dcc_loop(std_resid, a, b, c, bar_Q, N_bar, model_type_int):
        """Dispatch to pure-NumPy loop (numba not installed)."""
        return _dcc_loop_numpy(std_resid, a, b, c, bar_Q, N_bar, model_type_int)


# ---------------------------------------------------------------------------
# Integer codes for model types
# ---------------------------------------------------------------------------
_MODEL_INT = {"DCC": 0, "CDCC": 1, "ADCC": 2, "DECO": 0}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DCCGarch:
    """
    Multivariate GARCH via Dynamic Conditional Correlation variants.

    Supported model_type values
    ---------------------------
    "DCC"   – Engle (2002)
    "cDCC"  – Aielli (2013)
    "ADCC"  – Cappiello, Engle & Sheppard (2006)
    "DECO"  – Engle & Kelly (2012); equicorrelation averaging of DCC

    Workflow
    --------
    1.  model = DCCGarch(model_type="DCC")
    2.  std_resid = model.fit_univariate_garch(returns_df)
    3.  params   = model.fit_dcc(std_resid)
    4.  weights  = model.compute_mvp_weights()
    """

    def __init__(self, model_type="DCC"):
        """
        Initialise the DCCGarch estimator.

        Parameters
        ----------
        model_type : str
            One of "DCC", "cDCC", "ADCC", "DECO".
        """
        valid = {"DCC", "CDCC", "ADCC", "DECO"}
        self.model_type = model_type.upper()
        if self.model_type not in valid:
            raise ValueError(f"model_type must be one of {valid}; got '{model_type}'.")

        self.univariate_models = []
        self.std_resid = None
        self.sigmas = None
        self.dcc_params = None       # estimated [a, b] or [a, b, c]
        self.dcc_base_params = None  # for DECO: underlying DCC (a, b)
        self.Q_seq = None
        self.R_seq = None
        self.H_seq = None
        self._bar_Q = None
        self._N_bar = None
        self._equicorr_series = None  # for DECO

    # ------------------------------------------------------------------
    # Step 1 – Univariate GARCH
    # ------------------------------------------------------------------

    def fit_univariate_garch(self, returns):
        """
        Fit a GARCH(1,1) model to each column of *returns*.

        Parameters
        ----------
        returns : pd.DataFrame, shape (T, N)
            Asset returns (decimals, e.g. 0.01 for 1 %).

        Returns
        -------
        std_resid : np.ndarray, shape (T, N)
            Standardised residuals z_t = r_t / sigma_t.
        """
        n_obs, n_assets = returns.shape
        self.sigmas = np.zeros((n_obs, n_assets))
        self.std_resid = np.zeros((n_obs, n_assets))
        self.univariate_models = []

        for i in range(n_assets):
            col_name = returns.columns[i]
            model = arch_model(
                returns[col_name] * 100,
                vol='Garch', p=1, q=1, dist='normal', rescale=False
            )
            res = model.fit(disp='off')
            self.univariate_models.append(res)
            self.sigmas[:, i] = res.conditional_volatility / 100
            self.std_resid[:, i] = returns[col_name].values / self.sigmas[:, i]

        print("Univariate GARCH models fitted successfully.")
        return self.std_resid

    # ------------------------------------------------------------------
    # Log-likelihood (called by scipy.optimize)
    # ------------------------------------------------------------------

    def _dcc_loglike(self, params, std_resid):
        """
        Negative DCC composite log-likelihood for use with scipy.optimize.

        Parameters
        ----------
        params : array-like
            [a, b] for DCC/cDCC/DECO, or [a, b, c] for ADCC.
        std_resid : np.ndarray, shape (T, N)

        Returns
        -------
        float  Negative log-likelihood value.
        """
        if len(params) == 3:
            a, b, c = float(params[0]), float(params[1]), float(params[2])
        else:
            a, b = float(params[0]), float(params[1])
            c = 0.0

        # Parameter feasibility
        if a <= 0 or b <= 0 or c < 0 or (a + b + c) >= 1.0:
            return 1e10

        bar_Q = self._bar_Q
        N_bar = self._N_bar
        model_int = _MODEL_INT.get(self.model_type, 0)

        val = _dcc_loop(std_resid, a, b, c, bar_Q, N_bar, model_int)
        return val

    # ------------------------------------------------------------------
    # Step 2 – DCC parameter estimation
    # ------------------------------------------------------------------

    def fit_dcc(self, std_resid):
        """
        Estimate the DCC/cDCC/ADCC/DECO correlation parameters.

        For DECO the method first fits DCC(a, b), stores them as
        ``self.dcc_base_params``, then constructs the equicorrelation
        matrices from the DCC R_t sequence.

        Parameters
        ----------
        std_resid : np.ndarray, shape (T, N)
            Standardised residuals (from ``fit_univariate_garch``).

        Returns
        -------
        params : np.ndarray
            Estimated [a, b] or [a, b, c] vector.
        """
        self._bar_Q = np.cov(std_resid.T)

        # Unconditional N_bar for ADCC
        T, N = std_resid.shape
        neg_shocks = std_resid * (std_resid < 0)
        N_bar = np.zeros((N, N))
        for t in range(T):
            n = neg_shocks[t]
            N_bar += np.outer(n, n)
        N_bar /= T
        self._N_bar = N_bar

        is_adcc = (self.model_type == "ADCC")
        is_deco = (self.model_type == "DECO")

        if is_adcc:
            x0 = [0.03, 0.92, 0.02]
            bounds = ((1e-5, 0.15), (0.80, 0.999), (1e-5, 0.15))
            constraints = [{"type": "ineq", "fun": lambda p: 1.0 - p[0] - p[1] - p[2]}]
        else:
            # DCC, cDCC, DECO all use 2-param
            x0 = [0.03, 0.95]
            bounds = ((1e-5, 0.20), (0.80, 0.999))
            constraints = [{"type": "ineq", "fun": lambda p: 1.0 - p[0] - p[1]}]

        model_label = self.model_type if not is_deco else "DCC (for DECO)"
        print(f"Estimating {model_label} parameters...")

        res = minimize(
            self._dcc_loglike,
            x0,
            args=(std_resid,),
            bounds=bounds,
            constraints=constraints,
            method="L-BFGS-B",
        )

        if not res.success:
            # Retry with SLSQP which handles constraints more robustly
            res = minimize(
                self._dcc_loglike,
                x0,
                args=(std_resid,),
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
            )
        if not res.success:
            raise ValueError(f"DCC estimation failed: {res.message}")

        if is_deco:
            self.dcc_base_params = res.x
            # Temporarily switch to DCC to build R_t sequence
            saved_type = self.model_type
            self.model_type = "DCC"
            self.dcc_params = res.x
            self._compute_dynamic_matrices(std_resid)
            self.model_type = saved_type
            # Convert DCC R_t to DECO equicorrelation
            self._build_deco_matrices(std_resid)
            print(f"DECO estimation done. DCC base params (a,b): {self.dcc_base_params}")
        else:
            self.dcc_params = res.x
            label = "(alpha, beta)" if not is_adcc else "(alpha, beta, c_asym)"
            print(f"Estimation successful. Params {label}: {self.dcc_params}")
            self._compute_dynamic_matrices(std_resid)

        return self.dcc_params

    # ------------------------------------------------------------------
    # Dynamic matrix computation
    # ------------------------------------------------------------------

    def _compute_dynamic_matrices(self, std_resid):
        """
        Fill ``Q_seq``, ``R_seq``, and ``H_seq`` with time-varying matrices.

        Uses the fitted ``self.dcc_params`` and the model type to select
        the appropriate Q recursion (DCC / cDCC / ADCC).

        Parameters
        ----------
        std_resid : np.ndarray, shape (T, N)
        """
        T, N = std_resid.shape
        if len(self.dcc_params) == 3:
            a, b, c = self.dcc_params
        else:
            a, b = self.dcc_params
            c = 0.0

        bar_Q = self._bar_Q
        N_bar = self._N_bar

        self.Q_seq = np.zeros((T, N, N))
        self.R_seq = np.zeros((T, N, N))
        self.H_seq = np.zeros((T, N, N))

        # Initialise at t=0
        self.Q_seq[0] = bar_Q.copy()
        inv_sqrt0 = 1.0 / np.sqrt(np.diag(bar_Q))
        self.R_seq[0] = np.diag(inv_sqrt0) @ bar_Q @ np.diag(inv_sqrt0)
        D0 = np.diag(self.sigmas[0])
        self.H_seq[0] = D0 @ self.R_seq[0] @ D0

        for t in range(1, T):
            z = std_resid[t - 1]
            Q_prev = self.Q_seq[t - 1]

            if self.model_type == "CDCC":
                sqrt_diag = np.sqrt(np.diag(Q_prev))
                z_star = np.diag(sqrt_diag) @ z
                self.Q_seq[t] = (
                    (1.0 - a - b) * bar_Q
                    + a * np.outer(z_star, z_star)
                    + b * Q_prev
                )
            elif self.model_type == "ADCC":
                n = z * (z < 0.0)
                self.Q_seq[t] = (
                    (1.0 - a - b) * bar_Q
                    - c * N_bar
                    + a * np.outer(z, z)
                    + b * Q_prev
                    + c * np.outer(n, n)
                )
            else:  # DCC (also used as base for DECO)
                self.Q_seq[t] = (
                    (1.0 - a - b) * bar_Q
                    + a * np.outer(z, z)
                    + b * Q_prev
                )

            inv_sqrt = 1.0 / np.sqrt(np.diag(self.Q_seq[t]))
            self.R_seq[t] = np.diag(inv_sqrt) @ self.Q_seq[t] @ np.diag(inv_sqrt)

            D = np.diag(self.sigmas[t])
            self.H_seq[t] = D @ self.R_seq[t] @ D

        print("Dynamic covariance matrices computed.")

    # ------------------------------------------------------------------
    # DECO-specific builder
    # ------------------------------------------------------------------

    def _build_deco_matrices(self, std_resid):
        """
        Convert the DCC correlation matrices stored in ``R_seq`` into
        DECO equicorrelation matrices.

        The equicorrelation at time t is:
            rho_t = 2 / (N*(N-1)) * sum_{i<j} R_DCC[t, i, j]

        The DECO matrix is:
            R_DECO_t = (1 - rho_t)*I_N + rho_t * 1_N 1_N'

        Off-diagonal entries are clipped to [-1, 1].

        Parameters
        ----------
        std_resid : np.ndarray, shape (T, N)
        """
        T, N = std_resid.shape
        R_dcc = self.R_seq.copy()  # (T, N, N) DCC correlations

        pairs = N * (N - 1) / 2
        rho_t = np.array([
            np.sum(np.tril(R_dcc[t], -1)) / pairs
            for t in range(T)
        ])
        rho_t = np.clip(rho_t, -1.0 + 1e-8, 1.0 - 1e-8)

        ones = np.ones(N)
        R_deco = np.zeros((T, N, N))
        for t in range(T):
            rho = rho_t[t]
            mat = (1.0 - rho) * np.eye(N) + rho * np.outer(ones, ones)
            # Diagonal must remain exactly 1; off-diagonals clipped
            off = mat - np.diag(np.diag(mat))
            off = np.clip(off, -1.0, 1.0)
            R_deco[t] = off + np.eye(N)

        # Overwrite R_seq with DECO matrices; rebuild H_seq
        self.R_seq = R_deco
        self._equicorr_series = rho_t
        for t in range(T):
            D = np.diag(self.sigmas[t])
            self.H_seq[t] = D @ self.R_seq[t] @ D

        print("DECO equicorrelation matrices computed.")

    # ------------------------------------------------------------------
    # Portfolio methods
    # ------------------------------------------------------------------

    def compute_mvp_weights(self):
        """
        Compute time-varying Minimum Variance Portfolio (MVP) weights.

        The MVP weight vector at time t solves:
            min_w  w' H_t w   subject to 1' w = 1

        giving w_t = H_t^{-1} 1 / (1' H_t^{-1} 1).

        Returns
        -------
        weights : np.ndarray, shape (T, N)
        """
        if self.H_seq is None:
            raise ValueError("Model has not been fitted yet.")

        T, N, _ = self.H_seq.shape
        weights = np.zeros((T, N))
        ones = np.ones(N)

        for t in range(T):
            H_inv = np.linalg.inv(self.H_seq[t])
            denom = ones @ H_inv @ ones
            weights[t] = H_inv @ ones / denom

        return weights

    def compute_portfolio_vol(self, weights):
        """
        Compute time-varying portfolio volatility for a fixed weight vector.

        Parameters
        ----------
        weights : array-like, shape (N,)
            Fixed asset weights (need not sum to 1).

        Returns
        -------
        port_vol : np.ndarray, shape (T,)
            Portfolio volatility at each time step (same units as ``sigmas``).
        """
        if self.H_seq is None:
            raise ValueError("Model has not been fitted yet.")

        w = np.asarray(weights, dtype=float)
        T = self.H_seq.shape[0]
        port_vol = np.array([
            math.sqrt(float(w @ self.H_seq[t] @ w))
            for t in range(T)
        ])
        return port_vol

    # ------------------------------------------------------------------
    # Diagnostic / summary methods
    # ------------------------------------------------------------------

    def correlation_half_life(self):
        """
        Compute the half-life of correlation shocks in days.

        For DCC/cDCC/DECO (2-param) and ADCC (3-param):
            tau = log(0.5) / log(a + b)

        Returns
        -------
        float  Half-life in days.
        """
        if self.dcc_params is None and self.dcc_base_params is None:
            raise ValueError("Model has not been fitted yet.")

        params = self.dcc_base_params if self.model_type == "DECO" else self.dcc_params
        a, b = float(params[0]), float(params[1])
        return math.log(0.5) / math.log(a + b)

    def get_equicorrelation_series(self):
        """
        Return the scalar equicorrelation series (DECO models only).

        Returns
        -------
        rho_t : np.ndarray, shape (T,)

        Raises
        ------
        ValueError if the model is not of type DECO.
        """
        if self.model_type != "DECO":
            raise ValueError("get_equicorrelation_series() is only available for DECO models.")
        if self._equicorr_series is None:
            raise ValueError("Model has not been fitted yet.")
        return self._equicorr_series.copy()

    def get_summary_stats(self):
        """
        Return a dictionary of key model diagnostics.

        Returns
        -------
        dict with keys:
            alpha          – DCC alpha (a)
            beta           – DCC beta (b)
            c_asym         – asymmetry parameter c (0 if not ADCC)
            persistence    – a + b
            half_life_days – correlation half-life
            mean_corr      – time-averaged mean of off-diagonal R_t entries
            model_type     – self.model_type
            n_assets       – number of assets (N)
            n_obs          – number of observations (T)
        """
        if self.dcc_params is None and self.dcc_base_params is None:
            raise ValueError("Model has not been fitted yet.")

        params = self.dcc_base_params if self.model_type == "DECO" else self.dcc_params
        a = float(params[0])
        b = float(params[1])
        c = float(params[2]) if len(params) > 2 else 0.0

        T, N, _ = self.R_seq.shape
        # Mean of upper-triangular (off-diagonal) entries across time
        idx = np.triu_indices(N, k=1)
        mean_corr = float(np.mean([self.R_seq[t][idx] for t in range(T)]))

        return {
            "alpha": a,
            "beta": b,
            "c_asym": c,
            "persistence": a + b,
            "half_life_days": self.correlation_half_life(),
            "mean_corr": mean_corr,
            "model_type": self.model_type,
            "n_assets": N,
            "n_obs": T,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(42)
    T, N = 200, 3
    raw = rng.multivariate_normal(
        mean=np.zeros(N),
        cov=np.array([[1.0, 0.6, 0.3],
                      [0.6, 1.0, 0.4],
                      [0.3, 0.4, 1.0]]),
        size=T,
    )
    returns = pd.DataFrame(raw / 100, columns=["Asset_1", "Asset_2", "Asset_3"])

    for mtype in ["DCC", "cDCC", "ADCC", "DECO"]:
        print(f"\n{'='*50}")
        print(f"  Testing model_type = {mtype}")
        print(f"{'='*50}")
        model = DCCGarch(model_type=mtype)
        std_resid = model.fit_univariate_garch(returns)
        params = model.fit_dcc(std_resid)
        weights = model.compute_mvp_weights()
        print(f"  MVP weights (first row) : {weights[0].round(4)}")

        stats = model.get_summary_stats()
        print(f"  Summary stats           : {stats}")

        fixed_w = np.array([1 / N] * N)
        pvol = model.compute_portfolio_vol(fixed_w)
        print(f"  Equal-weight port vol (mean): {pvol.mean():.6f}")

        if mtype == "DECO":
            rho = model.get_equicorrelation_series()
            print(f"  Equicorrelation rho (mean): {rho.mean():.4f}")

    print("\nAll model types completed successfully.")
