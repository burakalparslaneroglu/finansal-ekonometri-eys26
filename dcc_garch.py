"""
dcc_garch.py
============
Multivariate GARCH models for financial econometrics.

Supported model types
---------------------
- "DCC"   : Dynamic Conditional Correlation (Engle, 2002)
- "cDCC"  : Corrected DCC (Aielli, 2013) — consistent E[z*_t z*'_t] targeting
- "ADCC"  : Asymmetric DCC (Cappiello, Engle & Sheppard, 2006)
- "DECO"  : Dynamic Equicorrelation (Engle & Kelly, 2012) — estimated on its
            OWN likelihood, not as a post-hoc average of a DCC fit.

All models share a two-step estimation routine:
  1.  fit_univariate_garch  – GARCH(1,1) per asset via the `arch` package
  2.  fit_dcc               – maximise the correlation-stage composite likelihood

Numerical notes
---------------
* The correlation-stage likelihood uses a Cholesky factorisation
  (``log|R| = 2 sum log diag(L)``, ``z'R^{-1}z = ||L^{-1}z||^2``) instead of
  ``det``/``inv``: for large N a determinant underflows to 0 and ``log(det)``
  returns ``-inf``.
* Q -> R normalisation is done by elementwise scaling
  (``R = Q * outer(d, d)``), never by building diag matrices and multiplying.
* DECO evaluates its likelihood in closed form from the scalar equicorrelation
  rho_t; no N x N inverse or determinant is ever formed (O(N) per observation
  after the O(N^2) Q update, versus O(N^3) for DCC).

Optional numba acceleration is used for the inner time loop when the
`numba` package is available; the code falls back to pure NumPy otherwise.
"""

import math
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

# NumPy 2.0 removed np.trapz in favour of np.trapezoid.  Releases of `arch`
# before 7.1 still call np.trapz at import time, so on a NumPy>=2 / arch<7.1
# combination the import below would fail.  The shim is version-conditional and
# is a no-op on any environment whose NumPy still exposes trapz (NumPy<2) or
# whose arch no longer needs it.  Pinned requirements (arch>=8) make it dead
# weight in the reference environment; it is kept only so the module still
# imports on a student machine with an older arch.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

from arch import arch_model

# ---------------------------------------------------------------------------
# Optional numba acceleration
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Integer codes for model types
# ---------------------------------------------------------------------------
# DECO gets its OWN code (3).  Mapping it to 0 (as an earlier version did) made
# the optimiser maximise the DCC likelihood and returned the DCC (a, b) as the
# DECO estimate — an algebraic identity, not a coincidence.
_MODEL_INT = {"DCC": 0, "CDCC": 1, "ADCC": 2, "DECO": 3}

# Feasibility guard for the equicorrelation: R = (1-rho)I + rho 11' is positive
# definite iff  -1/(N-1) < rho < 1.  Clipping at -1 (as an earlier version did)
# guards the wrong boundary and lets an invalid rho pass silently.
_RHO_EPS = 1e-8


def _rho_bounds(N):
    """Positive-definiteness interval for an N-dimensional equicorrelation."""
    lo = -1.0 / (N - 1) + _RHO_EPS if N > 1 else -1.0 + _RHO_EPS
    return lo, 1.0 - _RHO_EPS


# ---------------------------------------------------------------------------
# Correlation-stage kernels
# ---------------------------------------------------------------------------

def _deco_terms(rho, S_t, G_t, N):
    """
    Closed-form log|R_t| and z'R_t^{-1}z for an equicorrelation matrix.

        R_t          = (1-rho) I + rho 11'
        log|R_t|     = (N-1) log(1-rho) + log(1 + (N-1) rho)
        z'R_t^{-1}z  = (S_t - c_t G_t^2) / (1-rho),  c_t = rho / (1 + (N-1) rho)

    with S_t = sum_i z_it^2 and G_t = sum_i z_it.  No matrix is formed.
    """
    c_t = rho / (1.0 + (N - 1) * rho)
    logdet = (N - 1) * math.log(1.0 - rho) + math.log(1.0 + (N - 1) * rho)
    quad = (S_t - c_t * G_t * G_t) / (1.0 - rho)
    return logdet, quad


def _dcc_loop_numpy(std_resid, a, b, c, bar_Q, N_bar, model_type_int):
    """
    Pure-NumPy inner loop computing the correlation-stage negative
    log-likelihood.

    Parameters
    ----------
    std_resid    : (T, N) array of standardised residuals
    a, b, c      : correlation parameters (c=0 for DCC/cDCC/DECO)
    bar_Q        : (N, N) unconditional target for Q
    N_bar        : (N, N) unconditional mean of outer(n_t, n_t) (ADCC only)
    model_type_int : 0 = DCC, 1 = cDCC, 2 = ADCC, 3 = DECO

    Returns
    -------
    float  negative log-likelihood (positive = worse); 1e10 on infeasibility
    """
    T, N = std_resid.shape
    # The intercept (1-a-b)Q_bar (minus c*N_bar for ADCC) is constant in t;
    # hoisting it out of the loop removes one N x N product per observation.
    omega = (1.0 - a - b) * bar_Q
    if model_type_int == 2:
        omega = omega - c * N_bar
    Q = bar_Q.copy()
    loglike = 0.0
    rho_lo, rho_hi = _rho_bounds(N)

    for t in range(1, T):
        z = std_resid[t - 1]

        if model_type_int == 1:          # cDCC
            z_star = np.sqrt(np.diag(Q)) * z
            Q = omega + a * np.outer(z_star, z_star) + b * Q
        elif model_type_int == 2:        # ADCC
            n = z * (z < 0.0)
            Q = omega + a * np.outer(z, z) + b * Q + c * np.outer(n, n)
        else:                            # DCC (0) and DECO (3): same Q recursion
            Q = omega + a * np.outer(z, z) + b * Q

        q_diag = np.diag(Q)
        if np.any(q_diag <= 0.0):
            return 1e10
        inv_sqrt = 1.0 / np.sqrt(q_diag)

        z_t = std_resid[t]
        S_t = float(z_t @ z_t)

        if model_type_int == 3:          # DECO — closed form, no inv / no det
            # rho_t = mean of the off-diagonal entries of R_t = Q*outer(d,d).
            # sum_{i,j} d_i q_ij d_j = d'Qd and the diagonal contributes exactly
            # N, so the off-diagonal sum is d'Qd - N.
            rho = (float(inv_sqrt @ Q @ inv_sqrt) - N) / (N * (N - 1))
            if not (rho_lo < rho < rho_hi):
                return 1e10
            logdet, quad = _deco_terms(rho, S_t, float(z_t.sum()), N)
        else:
            # R = D Q D with D = diag(inv_sqrt), so chol(R) = D chol(Q) and
            #   log|R| = log|Q| - sum_i log q_ii
            #   z'R^{-1}z = u'Q^{-1}u  with  u = z * sqrt(q_ii)
            # i.e. R itself never has to be materialised.
            try:
                L = np.linalg.cholesky(Q)
            except np.linalg.LinAlgError:
                return 1e10
            logdet = (2.0 * float(np.sum(np.log(np.diag(L))))
                      - float(np.sum(np.log(q_diag))))
            u = z_t / inv_sqrt
            v = solve_triangular(L, u, lower=True, check_finite=False)
            quad = float(v @ v)

        loglike += 0.5 * (logdet + quad - S_t)

    return loglike


def _cdcc_target_numpy(std_resid, a, b, S_bar):
    """
    One pass of the Aielli (2013) cDCC targeting map:

        Q_t     = (1-a-b) S_bar + a z*_{t-1} z*'_{t-1} + b Q_{t-1}
        z*_t    = diag(Q_t)^{1/2} z_t
        S_new   = (1/T) sum_t z*_t z*'_t

    The result is rescaled to a correlation matrix before being returned.  The
    scale of Q_t is not identified — E[q_ii] = s_ii holds for ANY diagonal, so
    the raw map has a neutral direction along which the iteration drifts (and
    numerically explodes).  Fixing diag(S_bar) = 1 pins it down; only the
    off-diagonal target is estimated, which is what the correction is about.

    Returns the updated target S_new.
    """
    T, N = std_resid.shape
    Q = S_bar.copy()
    z_star = np.sqrt(np.diag(Q)) * std_resid[0]
    acc = np.outer(z_star, z_star)

    for t in range(1, T):
        Q = (1.0 - a - b) * S_bar + a * np.outer(z_star, z_star) + b * Q
        z_star = np.sqrt(np.diag(Q)) * std_resid[t]
        acc += np.outer(z_star, z_star)

    S_new = acc / T
    d = 1.0 / np.sqrt(np.diag(S_new))
    S_new = S_new * np.outer(d, d)
    np.fill_diagonal(S_new, 1.0)
    return S_new


if HAS_NUMBA:

    @_njit
    def _chol_logdet_quad_nb(R, z):
        """
        In-place Cholesky of R plus forward substitution for L v = z.

        Returns (log|R|, z'R^{-1}z, ok).  ``ok`` is False when R is not
        positive definite, in which case the caller bails out with 1e10.
        Written out longhand because numba cannot catch the LinAlgError that
        np.linalg.cholesky raises inside nopython code.
        """
        N = R.shape[0]
        L = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1):
                s = R[i, j]
                for k in range(j):
                    s -= L[i, k] * L[j, k]
                if i == j:
                    if s <= 0.0:
                        return 0.0, 0.0, False
                    L[i, i] = np.sqrt(s)
                else:
                    L[i, j] = s / L[j, j]

        logdet = 0.0
        for i in range(N):
            logdet += 2.0 * np.log(L[i, i])

        quad = 0.0
        v = np.zeros(N)
        for i in range(N):
            s = z[i]
            for k in range(i):
                s -= L[i, k] * v[k]
            vi = s / L[i, i]
            v[i] = vi
            quad += vi * vi

        return logdet, quad, True

    @_njit
    def _dcc_loop_numba(std_resid, a, b, c, bar_Q, N_bar, model_type_int):
        """Numba-jitted inner loop (identical logic to _dcc_loop_numpy)."""
        T, N = std_resid.shape
        # Constant intercept, hoisted out of the time loop.
        omega = (1.0 - a - b) * bar_Q
        if model_type_int == 2:
            omega = omega - c * N_bar
        Q = bar_Q.copy()
        loglike = 0.0
        if N > 1:
            rho_lo = -1.0 / (N - 1) + 1e-8
        else:
            rho_lo = -1.0 + 1e-8
        rho_hi = 1.0 - 1e-8

        d = np.zeros(N)
        u = np.zeros(N)
        z_star = np.zeros(N)

        for t in range(1, T):
            z = std_resid[t - 1]

            # Q update written out elementwise: one pass, no temporaries.
            if model_type_int == 1:      # cDCC
                for i in range(N):
                    z_star[i] = np.sqrt(Q[i, i]) * z[i]
                for i in range(N):
                    zi = z_star[i]
                    for j in range(N):
                        Q[i, j] = omega[i, j] + a * zi * z_star[j] + b * Q[i, j]
            elif model_type_int == 2:    # ADCC
                for i in range(N):
                    zi = z[i]
                    ni = zi if zi < 0.0 else 0.0
                    for j in range(N):
                        zj = z[j]
                        nj = zj if zj < 0.0 else 0.0
                        Q[i, j] = (omega[i, j] + a * zi * zj + b * Q[i, j]
                                   + c * ni * nj)
            else:                        # DCC (0) and DECO (3)
                for i in range(N):
                    zi = z[i]
                    for j in range(N):
                        Q[i, j] = omega[i, j] + a * zi * z[j] + b * Q[i, j]

            for i in range(N):
                if Q[i, i] <= 0.0:
                    return 1e10
                d[i] = 1.0 / np.sqrt(Q[i, i])

            z_t = std_resid[t]
            S_t = 0.0
            G_t = 0.0
            for i in range(N):
                S_t += z_t[i] * z_t[i]
                G_t += z_t[i]

            if model_type_int == 3:      # DECO — closed form
                # off-diagonal sum of R = D Q D, using symmetry
                s = 0.0
                for i in range(N):
                    di = d[i]
                    for j in range(i + 1, N):
                        s += di * Q[i, j] * d[j]
                rho = 2.0 * s / (N * (N - 1))
                if rho <= rho_lo or rho >= rho_hi:
                    return 1e10
                c_t = rho / (1.0 + (N - 1) * rho)
                logdet = (N - 1) * np.log(1.0 - rho) + np.log(1.0 + (N - 1) * rho)
                quad = (S_t - c_t * G_t * G_t) / (1.0 - rho)
            else:
                # Factorise Q, not R = D Q D: log|R| = log|Q| - sum log q_ii and
                # z'R^{-1}z = u'Q^{-1}u with u = z / d.  R is never built.
                for i in range(N):
                    u[i] = z_t[i] / d[i]
                logdet, quad, ok = _chol_logdet_quad_nb(Q, u)
                if not ok:
                    return 1e10
                for i in range(N):
                    logdet += 2.0 * np.log(d[i])

            loglike += 0.5 * (logdet + quad - S_t)

        return loglike

    @_njit
    def _cdcc_target_numba(std_resid, a, b, S_bar):
        """Numba-jitted Aielli targeting map (identical to _cdcc_target_numpy)."""
        T, N = std_resid.shape
        Q = S_bar.copy()
        z_star = np.sqrt(np.diag(Q)) * std_resid[0]
        acc = np.outer(z_star, z_star)
        for t in range(1, T):
            Q = (1.0 - a - b) * S_bar + a * np.outer(z_star, z_star) + b * Q
            z_star = np.sqrt(np.diag(Q)) * std_resid[t]
            acc += np.outer(z_star, z_star)

        S_new = acc / T
        d = np.zeros(N)
        for i in range(N):
            d[i] = 1.0 / np.sqrt(S_new[i, i])
        for i in range(N):
            for j in range(N):
                S_new[i, j] = S_new[i, j] * d[i] * d[j]
            S_new[i, i] = 1.0
        return S_new

    def _dcc_loop(std_resid, a, b, c, bar_Q, N_bar, model_type_int):
        """Dispatch to numba-jitted loop."""
        return _dcc_loop_numba(std_resid, a, b, c, bar_Q, N_bar, model_type_int)

    def _cdcc_target(std_resid, a, b, S_bar):
        """Dispatch to numba-jitted cDCC targeting map."""
        return _cdcc_target_numba(std_resid, a, b, S_bar)

else:  # pragma: no cover - exercised only without numba

    def _dcc_loop(std_resid, a, b, c, bar_Q, N_bar, model_type_int):
        """Dispatch to pure-NumPy loop (numba not installed)."""
        return _dcc_loop_numpy(std_resid, a, b, c, bar_Q, N_bar, model_type_int)

    def _cdcc_target(std_resid, a, b, S_bar):
        """Dispatch to pure-NumPy cDCC targeting map."""
        return _cdcc_target_numpy(std_resid, a, b, S_bar)


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
    "DECO"  – Engle & Kelly (2012)

    Workflow
    --------
    1.  model = DCCGarch(model_type="DCC")
    2.  std_resid = model.fit_univariate_garch(returns_df)
    3.  params   = model.fit_dcc(std_resid)
    4.  weights  = model.compute_mvp_weights()
    """

    # Outer iterations of the cDCC targeting fixed point (Aielli 2013).
    CDCC_MAX_OUTER = 3
    CDCC_TARGET_TOL = 1e-7

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
        self.dcc_params = None       # estimated [a, b] or [a, b, c] for THIS model
        self.dcc_base_params = None  # optional reference DCC (a, b) for DECO
        self.dcc_base_loglik = None  # its log-likelihood (comparison only)
        self.corr_loglik = None      # maximised correlation-stage log-likelihood
        self.Q_seq = None
        self.R_seq = None
        self.H_seq = None
        self._bar_Q = None
        self._N_bar = None
        self._equicorr_series = None  # for DECO

    # ------------------------------------------------------------------
    # Step 1 – Univariate GARCH
    # ------------------------------------------------------------------

    def fit_univariate_garch(self, returns, train_end=None):
        """
        Fit a GARCH(1,1) model to each column of *returns*.

        Parameters
        ----------
        returns : pd.DataFrame, shape (T, N)
            Asset returns (decimals, e.g. 0.01 for 1 %).
        train_end : int or None
            If given, estimate GARCH parameters on returns[:train_end] only,
            then fix them and filter the full series.  This prevents look-ahead
            bias in OOS evaluations (ZD-2 fix).  Default None trains on the
            full sample (legacy behaviour).

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
            r_pct = np.asarray(returns[col_name].values, dtype=float) * 100

            if train_end is not None:
                # ZD-2 fix: estimate on r[:train_end], fix params, filter all T
                model_is = arch_model(
                    r_pct[:train_end],
                    vol='Garch', p=1, q=1, dist='normal', rescale=False
                )
                res_is = model_is.fit(disp='off')
                model_full = arch_model(
                    r_pct, vol='Garch', p=1, q=1, dist='normal', rescale=False
                )
                res = model_full.fix(res_is.params)
            else:
                model = arch_model(
                    r_pct, vol='Garch', p=1, q=1, dist='normal', rescale=False
                )
                res = model.fit(disp='off')

            self.univariate_models.append(res)
            self.sigmas[:, i] = np.asarray(res.conditional_volatility) / 100
            self.std_resid[:, i] = np.asarray(returns[col_name].values, dtype=float) / self.sigmas[:, i]

        print("Univariate GARCH models fitted successfully.")
        return self.std_resid

    # ------------------------------------------------------------------
    # Log-likelihood (called by scipy.optimize)
    # ------------------------------------------------------------------

    def _dcc_loglike(self, params, std_resid):
        """
        Negative correlation-stage log-likelihood for use with scipy.optimize.

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

        model_int = _MODEL_INT[self.model_type]
        return _dcc_loop(std_resid, a, b, c, self._bar_Q, self._N_bar, model_int)

    # ------------------------------------------------------------------
    # Step 2 – correlation parameter estimation
    # ------------------------------------------------------------------

    def _optimise(self, z_est, x0, bounds, constraints):
        """Run SLSQP with a Nelder-Mead fallback; raise if both fail."""
        res = minimize(
            self._dcc_loglike, x0, args=(z_est,),
            bounds=bounds, constraints=constraints, method="SLSQP",
        )
        if not res.success:
            # Nelder-Mead: no gradient, more robust on flat regions
            res = minimize(
                self._dcc_loglike, x0, args=(z_est,), method="Nelder-Mead",
                options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 10000},
            )
        if not res.success:
            raise ValueError(f"{self.model_type} estimation failed: {res.message}")
        return res

    def fit_dcc(self, std_resid, train_end=None, deco_base_dcc=False):
        """
        Estimate the DCC / cDCC / ADCC / DECO correlation parameters.

        Every model type maximises ITS OWN likelihood.  In particular DECO is
        estimated from the equicorrelation likelihood (Engle & Kelly 2012,
        eq. 8) rather than by averaging a fitted DCC R_t.

        Parameters
        ----------
        std_resid : np.ndarray, shape (T, N)
            Standardised residuals (from ``fit_univariate_garch``).
        train_end : int or None
            If given, estimate parameters on std_resid[:train_end] only
            (ZD-2 fix — prevents look-ahead bias in OOS evaluations).
            bar_Q and N_bar are also computed from the in-sample slice.
        deco_base_dcc : bool
            DECO only.  When True an additional plain-DCC fit is run on the
            same residuals and stored in ``dcc_base_params`` /
            ``dcc_base_loglik`` for side-by-side reporting.  Off by default so
            that a DECO fit stays strictly cheaper than a DCC fit.

        Returns
        -------
        params : np.ndarray
            Estimated [a, b] or [a, b, c] vector for the requested model.
        """
        std_resid = np.ascontiguousarray(std_resid, dtype=np.float64)
        # Use in-sample slice for parameter estimation when train_end is set
        z_est = std_resid[:train_end] if train_end is not None else std_resid
        z_est = np.ascontiguousarray(z_est, dtype=np.float64)

        T_est, N = z_est.shape
        if N < 2:
            raise ValueError("At least two assets are required for a DCC-family model.")

        # Engle's targeting is the UNCENTRED second moment (1/T) sum z_t z_t',
        # not np.cov (which demeans and divides by T-1).
        self._bar_Q = z_est.T @ z_est / T_est

        # Unconditional N_bar for ADCC (in-sample only)
        neg_shocks = z_est * (z_est < 0)
        self._N_bar = neg_shocks.T @ neg_shocks / T_est

        is_adcc = (self.model_type == "ADCC")
        is_cdcc = (self.model_type == "CDCC")
        is_deco = (self.model_type == "DECO")

        if is_adcc:
            x0 = [0.03, 0.92, 0.02]
            bounds = ((1e-5, 0.15), (0.80, 0.999), (1e-5, 0.15))
            constraints = [{"type": "ineq", "fun": lambda p: 1.0 - p[0] - p[1] - p[2]}]
        else:
            # DCC, cDCC, DECO all use 2 parameters
            x0 = [0.03, 0.95]
            bounds = ((1e-5, 0.20), (0.80, 0.999))
            constraints = [{"type": "ineq", "fun": lambda p: 1.0 - p[0] - p[1]}]

        print(f"Estimating {self.model_type} parameters...")

        if is_cdcc:
            res = self._fit_cdcc_targeting(z_est, x0, bounds, constraints)
        else:
            res = self._optimise(z_est, x0, bounds, constraints)

        # _dcc_loglike returns the NEGATIVE correlation-stage log-likelihood,
        # so the maximised log-likelihood is -res.fun.  Used by the ADCC-vs-DCC
        # boundary-mixture LR test and by the model-comparison table.
        self.dcc_params = res.x
        self.corr_loglik = float(-res.fun)

        if is_deco and deco_base_dcc:
            # Reference DCC fit on the SAME residuals, for reporting only.
            saved = self.model_type
            self.model_type = "DCC"
            try:
                res_base = self._optimise(z_est, [0.03, 0.95], bounds, constraints)
                self.dcc_base_params = res_base.x
                self.dcc_base_loglik = float(-res_base.fun)
            finally:
                self.model_type = saved

        label = "(alpha, beta, c_asym)" if is_adcc else "(alpha, beta)"
        print(f"Estimation successful. Params {label}: {self.dcc_params}")

        self._compute_dynamic_matrices(std_resid)
        return self.dcc_params

    def _fit_cdcc_targeting(self, z_est, x0, bounds, constraints):
        """
        cDCC estimation with Aielli (2013) consistent targeting.

        The cDCC target is S_bar = E[z*_t z*'_t] with z*_t = diag(Q_t)^{1/2} z_t,
        NOT the DCC target E[z_t z_t'].  Because the map depends on (a, b) it is
        solved as an outer fixed point: optimise (a, b) given S_bar, refresh
        S_bar by iterating the targeting map at those (a, b), repeat.
        """
        res = self._optimise(z_est, x0, bounds, constraints)

        for _ in range(self.CDCC_MAX_OUTER):
            a, b = float(res.x[0]), float(res.x[1])
            S_bar = self._bar_Q
            delta = np.inf
            for _inner in range(20):
                S_new = _cdcc_target(z_est, a, b, S_bar)
                delta = float(np.max(np.abs(S_new - S_bar)))
                S_bar = S_new
                if delta < 1e-8:
                    break
            shift = float(np.max(np.abs(S_bar - self._bar_Q)))
            self._bar_Q = S_bar
            res = self._optimise(z_est, list(res.x), bounds, constraints)
            if shift < self.CDCC_TARGET_TOL:
                break

        return res

    # ------------------------------------------------------------------
    # Dynamic matrix computation
    # ------------------------------------------------------------------

    def _compute_dynamic_matrices(self, std_resid):
        """
        Fill ``Q_seq``, ``R_seq`` and ``H_seq`` with time-varying matrices.

        For DECO the equicorrelation rho_t is read straight off Q_t and R_t is
        built as (1-rho_t)I + rho_t 11' — the DCC R_t is never formed.
        """
        std_resid = np.ascontiguousarray(std_resid, dtype=np.float64)
        T, N = std_resid.shape
        if len(self.dcc_params) == 3:
            a, b, c = (float(v) for v in self.dcc_params)
        else:
            a, b = float(self.dcc_params[0]), float(self.dcc_params[1])
            c = 0.0

        bar_Q = self._bar_Q
        N_bar = self._N_bar
        is_deco = (self.model_type == "DECO")
        rho_lo, rho_hi = _rho_bounds(N)
        eye = np.eye(N)
        ones_mat = np.ones((N, N))

        self.Q_seq = np.zeros((T, N, N))
        self.R_seq = np.zeros((T, N, N))
        self.H_seq = np.zeros((T, N, N))
        equicorr = np.zeros(T) if is_deco else None

        def _q_to_r(Q, t):
            """Normalise Q_t to a correlation matrix (elementwise scaling)."""
            inv_sqrt = 1.0 / np.sqrt(np.diag(Q))
            if is_deco:
                rho = (float(inv_sqrt @ Q @ inv_sqrt) - N) / (N * (N - 1))
                rho = float(np.clip(rho, rho_lo, rho_hi))
                equicorr[t] = rho
                R = (1.0 - rho) * eye + rho * ones_mat
                np.fill_diagonal(R, 1.0)
                return R
            R = Q * np.outer(inv_sqrt, inv_sqrt)
            np.fill_diagonal(R, 1.0)
            return R

        # Initialise at t=0
        self.Q_seq[0] = bar_Q.copy()
        self.R_seq[0] = _q_to_r(bar_Q, 0)
        s0 = self.sigmas[0]
        self.H_seq[0] = self.R_seq[0] * np.outer(s0, s0)

        z_star = np.sqrt(np.diag(bar_Q)) * std_resid[0]   # cDCC only

        for t in range(1, T):
            z = std_resid[t - 1]
            Q_prev = self.Q_seq[t - 1]

            if self.model_type == "CDCC":
                self.Q_seq[t] = (
                    (1.0 - a - b) * bar_Q
                    + a * np.outer(z_star, z_star)
                    + b * Q_prev
                )
                z_star = np.sqrt(np.diag(self.Q_seq[t])) * std_resid[t]
            elif self.model_type == "ADCC":
                n = z * (z < 0.0)
                self.Q_seq[t] = (
                    (1.0 - a - b) * bar_Q
                    - c * N_bar
                    + a * np.outer(z, z)
                    + b * Q_prev
                    + c * np.outer(n, n)
                )
            else:  # DCC and DECO share the Q recursion
                self.Q_seq[t] = (
                    (1.0 - a - b) * bar_Q
                    + a * np.outer(z, z)
                    + b * Q_prev
                )

            self.R_seq[t] = _q_to_r(self.Q_seq[t], t)
            s = self.sigmas[t]
            self.H_seq[t] = self.R_seq[t] * np.outer(s, s)

        self._equicorr_series = equicorr
        print("Dynamic covariance matrices computed.")

    # ------------------------------------------------------------------
    # Portfolio methods
    # ------------------------------------------------------------------

    def compute_mvp_weights(self):
        """
        Compute time-varying Minimum Variance Portfolio (MVP) weights.

        The MVP weight vector at time t solves:
            min_w  w' H_t w   subject to 1' w = 1

        giving w_t = H_t^{-1} 1 / (1' H_t^{-1} 1).  Solved by Cholesky
        factorisation rather than an explicit inverse.

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
            try:
                cf = cho_factor(self.H_seq[t], lower=True, check_finite=False)
                x = cho_solve(cf, ones, check_finite=False)
            except np.linalg.LinAlgError:
                x = np.linalg.solve(self.H_seq[t], ones)
            weights[t] = x / x.sum()

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
        return np.sqrt(np.einsum("n,tnm,m->t", w, self.H_seq, w))

    # ------------------------------------------------------------------
    # Diagnostic / summary methods
    # ------------------------------------------------------------------

    def correlation_half_life(self):
        """
        Compute the half-life of correlation shocks in days:
            tau = log(0.5) / log(a + b)

        Uses this model's own (a, b) — including for DECO.
        """
        if self.dcc_params is None:
            raise ValueError("Model has not been fitted yet.")

        a, b = float(self.dcc_params[0]), float(self.dcc_params[1])
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
            alpha, beta, c_asym, persistence, half_life_days, mean_corr,
            model_type, n_assets, n_obs, corr_loglik, n_params,
            aic, bic, dcc_base_params (DECO reference fit, may be None)
        """
        if self.dcc_params is None:
            raise ValueError("Model has not been fitted yet.")

        params = self.dcc_params
        a = float(params[0])
        b = float(params[1])
        c = float(params[2]) if len(params) > 2 else 0.0

        T, N, _ = self.R_seq.shape
        # Mean of upper-triangular (off-diagonal) entries across time
        idx = np.triu_indices(N, k=1)
        mean_corr = float(np.mean(self.R_seq[:, idx[0], idx[1]]))

        k = len(params)
        ll = self.corr_loglik
        aic = float(-2.0 * ll + 2.0 * k) if ll is not None else None
        bic = float(-2.0 * ll + k * math.log(T)) if ll is not None else None

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
            "corr_loglik": ll,
            "n_params": k,
            "aic": aic,
            "bic": bic,
            "dcc_base_params": (None if self.dcc_base_params is None
                                else [float(v) for v in self.dcc_base_params]),
        }


# ---------------------------------------------------------------------------
# Reference (slow) DECO likelihood — used by the test suite
# ---------------------------------------------------------------------------

def deco_loglike_reference(std_resid, a, b, bar_Q=None):
    """
    Deliberately slow DECO negative log-likelihood: builds R_t explicitly and
    calls ``np.linalg.slogdet`` / ``np.linalg.solve``.

    Exists so the fast closed form in ``_dcc_loop`` can be validated against an
    implementation that shares none of its algebra.  Not used by the app.
    """
    z = np.ascontiguousarray(std_resid, dtype=np.float64)
    T, N = z.shape
    if bar_Q is None:
        bar_Q = z.T @ z / T

    Q = bar_Q.copy()
    nll = 0.0
    for t in range(1, T):
        zp = z[t - 1]
        Q = (1.0 - a - b) * bar_Q + a * np.outer(zp, zp) + b * Q
        d = 1.0 / np.sqrt(np.diag(Q))
        R_dcc = Q * np.outer(d, d)
        iu = np.triu_indices(N, k=1)
        rho = float(np.mean(R_dcc[iu]))
        R = (1.0 - rho) * np.eye(N) + rho * np.ones((N, N))
        np.fill_diagonal(R, 1.0)

        sign, logdet = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e10
        z_t = z[t]
        quad = float(z_t @ np.linalg.solve(R, z_t))
        nll += 0.5 * (logdet + quad - float(z_t @ z_t))

    return nll


# ---------------------------------------------------------------------------
# ADCC vs DCC likelihood-ratio test (boundary mixture)
# ---------------------------------------------------------------------------

def adcc_vs_dcc_lr_test(loglik_adcc, loglik_dcc, alpha=0.05):
    """
    Likelihood-ratio test of ADCC against DCC at the correlation stage.

    ADCC nests DCC under H0: c = 0 (no asymmetry). Because the asymmetry
    parameter satisfies c >= 0, the null value lies on the BOUNDARY of the
    parameter space, so the standard Wilks chi^2_1 result does NOT apply.
    Under the regularity conditions of Self & Liang (1987) / Andrews (2001)
    the asymptotic null distribution of

        LR = 2 * ( loglik_adcc - loglik_dcc )

    is the 50:50 mixture  0.5 * chi^2_0 + 0.5 * chi^2_1.

    Consequences versus the naive chi^2_1 test:
      * critical value at level ``alpha`` is ``chi2.isf(2*alpha, 1)``
        (e.g. 2.706 at alpha=0.05, NOT 3.841);
      * the p-value is HALF the naive one:  p = 0.5 * P(chi^2_1 > LR).

    In finite samples, with the ADCC positivity constraints, a parametric
    bootstrap from the fitted DCC (H0) model is the most reliable alternative.

    NOTE: both log-likelihoods must come from NESTED models fitted to the same
    residuals.  DECO does not nest DCC, so a DECO log-likelihood must never be
    passed here; the function raises if the values are obviously incompatible.

    Parameters
    ----------
    loglik_adcc, loglik_dcc : float
        Maximised correlation-stage log-likelihoods of the ADCC and DCC fits
        on the SAME data with the SAME marginals (see ``DCCGarch.corr_loglik``).
    alpha : float, optional
        Test level (default 0.05).

    Returns
    -------
    dict with keys:
        lr_stat, p_value, critical_value, alpha, reject, distribution, note
    """
    from scipy.stats import chi2

    if loglik_adcc is None or loglik_dcc is None:
        raise ValueError("Both log-likelihoods are required (got None).")

    lr = 2.0 * (float(loglik_adcc) - float(loglik_dcc))
    # LR >= 0 in theory (nested MLE); clamp tiny negatives from the optimiser.
    lr_clamped = max(lr, 0.0)

    # Mixture p-value: P(0.5*chi2_0 + 0.5*chi2_1 >= x) = 0.5 * P(chi2_1 >= x)
    # for x > 0, and = 1 at x = 0 (point mass of chi2_0).
    p_value = 0.5 * float(chi2.sf(lr_clamped, df=1)) if lr_clamped > 0.0 else 1.0

    # Critical value c: P(mixture > c) = alpha  =>  0.5 * P(chi2_1 > c) = alpha.
    critical_value = float(chi2.isf(2.0 * alpha, df=1)) if 0.0 < alpha < 0.5 else 0.0

    return {
        "lr_stat": float(lr),
        "p_value": float(p_value),
        "critical_value": critical_value,
        "alpha": float(alpha),
        "reject": bool(lr_clamped > critical_value),
        "distribution": "0.5*chi2_0 + 0.5*chi2_1 (boundary mixture)",
        "note": ("Boundary test (c>=0): Self-Liang 1987 / Andrews 2001. "
                 "Naive chi2_1 (crit 3.841) is too conservative; here crit=2.706 "
                 "and p is halved. Finite-sample: prefer a parametric bootstrap "
                 "from the DCC null. Only valid for NESTED models (ADCC vs DCC) "
                 "— never for DECO vs DCC."),
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
        params = model.fit_dcc(std_resid, deco_base_dcc=(mtype == "DECO"))
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
