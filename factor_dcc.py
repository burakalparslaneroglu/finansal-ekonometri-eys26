"""
factor_dcc.py
=============
Factor-DCC:  H_t = B Lambda_t B' + Omega_t   (course notes 1.10.3)

Stages
------
0.  Choose K (see ``factor_selection``).
1.  Loadings B — either the first K eigenvectors of the correlation matrix of
    the standardised returns ("pca") or OLS on user-supplied observed factor
    returns ("observed").
2.  Factors      f_t = (B'B)^{-1} B' r_t.
3.  Factor covariance Lambda_t — univariate GARCH(1,1) on each of the K
    factors plus a K-dimensional DCC (the existing ``DCCGarch`` class).
4.  Idiosyncratic Omega — diag(var(e_i)) by default, optionally a GARCH(1,1)
    per residual giving a time-varying Omega_t.

Why Omega is mandatory
----------------------
B Lambda_t B' is rank K < N and therefore singular.  H_t^{-1}, and hence the
minimum-variance portfolio, exists only because of Omega (notes 1.10.3, after
Remark 1.8).  Omega's diagonal is floored at 1e-10 for that reason.

Numerical efficiency
--------------------
H_t is NEVER formed and inverted.  Woodbury / the matrix determinant lemma give

    M_t         = Lambda_t^{-1} + B' Omega_t^{-1} B          (K x K)
    H_t^{-1}    = Omega_t^{-1} - Omega_t^{-1} B M_t^{-1} B' Omega_t^{-1}
    log det H_t = log det Omega_t + log det Lambda_t + log det M_t

at cost O(N K^2 + K^3) instead of O(N^3).  This is also what makes the model
usable at N=200: storing H_t explicitly for T=2500, N=200 would need ~800 MB.

Identification
--------------
For any invertible K x K matrix Q,  B -> B Q  and  Lambda_t -> Q^{-1} Lambda_t Q^{-1'}
leave H_t unchanged.  Under the PCA option the INDIVIDUAL loadings are
therefore not interpretable — only the factor space and the variance shares
are.  The UI states this explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OMEGA_FLOOR = 1e-10


class FactorDCC:
    """
    Factor-DCC estimator.

    Workflow
    --------
    >>> model = FactorDCC(K=3)
    >>> model.fit(returns_df)
    >>> w = model.mvp_weights()
    """

    def __init__(self, K: int = 3, loading_mode: str = "pca",
                 factor_model: str = "DCC", idio_garch: bool = False):
        if K < 1:
            raise ValueError("K must be at least 1.")
        if loading_mode not in ("pca", "observed"):
            raise ValueError("loading_mode must be 'pca' or 'observed'.")
        self.K = int(K)
        self.loading_mode = loading_mode
        self.factor_model = factor_model
        self.idio_garch = bool(idio_garch)

        self.B = None                # (N, K) loadings
        self.factors = None          # (T, K)
        self.Lambda_seq = None       # (T, K, K)
        self.Omega_diag = None       # (T, N) idiosyncratic variances
        self.resid = None            # (T, N)
        self.cols = None
        self.index = None
        self.var_share = None        # explained variance share per factor
        self.factor_dcc = None       # the fitted DCCGarch on the factors

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, returns: pd.DataFrame, factor_returns: pd.DataFrame | None = None):
        """
        Estimate the factor structure and the factor covariance path.

        Parameters
        ----------
        returns : (T, N) DataFrame of asset returns (decimals)
        factor_returns : (T, K) DataFrame, required when loading_mode='observed'
        """
        from factor_selection import standardise
        from dcc_garch import DCCGarch

        R = np.asarray(returns.values, dtype=float)
        T, N = R.shape
        self.cols = list(returns.columns)
        self.index = returns.index

        if self.K >= N:
            raise ValueError(f"K={self.K} must be smaller than N={N}.")

        # --- stage 1: loadings ------------------------------------------
        if self.loading_mode == "pca":
            X = standardise(R)
            C = X.T @ X / T
            w, V = np.linalg.eigh(C)
            order = np.argsort(w)[::-1]
            w, V = w[order], V[:, order]
            self.B = np.ascontiguousarray(V[:, :self.K])
            self.var_share = (w[:self.K] / w.sum()).astype(float)
            F = None
        else:
            if factor_returns is None:
                raise ValueError("loading_mode='observed' requires factor_returns.")
            F = np.asarray(factor_returns.values, dtype=float)
            if F.shape[0] != T:
                raise ValueError("factor_returns must have the same length as returns.")
            self.K = F.shape[1]
            Fc = np.column_stack([np.ones(T), F])
            coef, *_ = np.linalg.lstsq(Fc, R, rcond=None)
            self.B = np.ascontiguousarray(coef[1:].T)      # (N, K)
            tot = R.var(axis=0, ddof=0).sum()
            self.var_share = np.array([
                float((self.B[:, k] ** 2).sum() * F[:, k].var(ddof=0) / tot)
                for k in range(self.K)
            ])

        # --- stage 2: factors -------------------------------------------
        if F is not None:
            self.factors = F
        else:
            BtB = self.B.T @ self.B
            self.factors = R @ self.B @ np.linalg.inv(BtB).T   # (T, K)

        # --- stage 4a: idiosyncratic residuals --------------------------
        self.resid = R - self.factors @ self.B.T

        # --- stage 3: Lambda_t ------------------------------------------
        fcols = [f"F{k+1}" for k in range(self.K)]
        fdf = pd.DataFrame(self.factors, index=returns.index, columns=fcols)

        if self.K == 1:
            # A single factor has no correlation stage: Lambda_t is its GARCH
            # variance path.
            from arch import arch_model
            am = arch_model(fdf.iloc[:, 0].values * 100, vol="Garch", p=1, q=1,
                            dist="normal", rescale=False)
            res = am.fit(disp="off")
            sig = np.asarray(res.conditional_volatility) / 100.0
            self.Lambda_seq = (sig ** 2).reshape(T, 1, 1)
            self.factor_dcc = None
        else:
            fdcc = DCCGarch(model_type=self.factor_model)
            fz = fdcc.fit_univariate_garch(fdf)
            fdcc.fit_dcc(fz)
            self.Lambda_seq = np.asarray(fdcc.H_seq)
            self.factor_dcc = fdcc

        # --- stage 4b: Omega --------------------------------------------
        if self.idio_garch:
            from arch import arch_model
            omg = np.zeros((T, N))
            for i in range(N):
                am = arch_model(self.resid[:, i] * 100, vol="Garch", p=1, q=1,
                                dist="normal", rescale=False)
                res = am.fit(disp="off")
                omg[:, i] = (np.asarray(res.conditional_volatility) / 100.0) ** 2
            self.Omega_diag = np.maximum(omg, OMEGA_FLOOR)
        else:
            v = np.maximum(self.resid.var(axis=0, ddof=0), OMEGA_FLOOR)
            self.Omega_diag = np.tile(v, (T, 1))

        return self

    # ------------------------------------------------------------------
    # Woodbury machinery
    # ------------------------------------------------------------------

    def _woodbury_parts(self, t: int):
        """Return (Omega_inv_diag, B, Lambda_t, M_t) for time t."""
        om_inv = 1.0 / self.Omega_diag[t]
        Lam = self.Lambda_seq[t]
        BtOiB = self.B.T @ (self.B * om_inv[:, None])
        M = np.linalg.inv(Lam) + BtOiB
        return om_inv, self.B, Lam, M

    def H_inv_at(self, t: int):
        """H_t^{-1} via Woodbury — O(N K^2 + K^3), no N x N inverse."""
        om_inv, B, _Lam, M = self._woodbury_parts(t)
        OiB = B * om_inv[:, None]                    # (N, K)
        return np.diag(om_inv) - OiB @ np.linalg.solve(M, OiB.T)

    def H_at(self, t: int):
        """H_t = B Lambda_t B' + Omega_t, built explicitly (plots only)."""
        return self.B @ self.Lambda_seq[t] @ self.B.T + np.diag(self.Omega_diag[t])

    def logdet_H(self, t: int) -> float:
        """log det H_t via the matrix determinant lemma."""
        om_inv, _B, Lam, M = self._woodbury_parts(t)
        sign_l, ld_lam = np.linalg.slogdet(Lam)
        sign_m, ld_m = np.linalg.slogdet(M)
        if sign_l <= 0 or sign_m <= 0:
            raise np.linalg.LinAlgError("Lambda_t or M_t is not positive definite.")
        ld_omega = float(np.sum(np.log(self.Omega_diag[t])))
        return ld_omega + float(ld_lam) + float(ld_m)

    # ------------------------------------------------------------------
    # Portfolio and summary
    # ------------------------------------------------------------------

    def mvp_weights(self):
        """
        w_t = H_t^{-1} 1 / (1' H_t^{-1} 1), computed with Woodbury so that
        neither H_t nor its inverse is ever materialised.
        """
        T = self.Lambda_seq.shape[0]
        N = self.B.shape[0]
        ones = np.ones(N)
        W = np.zeros((T, N))

        for t in range(T):
            om_inv, B, _Lam, M = self._woodbury_parts(t)
            OiB = B * om_inv[:, None]
            x = om_inv * ones - OiB @ np.linalg.solve(M, OiB.T @ ones)
            W[t] = x / x.sum()
        return W

    def conditional_vol(self):
        """Per-asset conditional volatility path, sqrt(diag(H_t))."""
        # diag(B Lambda_t B') without forming the product:
        #   (B Lambda_t B')_ii = sum_kl B_ik Lambda_kl B_il
        common = np.einsum("nk,tkl,nl->tn", self.B, self.Lambda_seq, self.B)
        return np.sqrt(common + self.Omega_diag)

    def factor_correlation(self):
        """Time-varying factor correlation R_t of the K factors (K>=2)."""
        if self.factor_dcc is None:
            return None
        return np.asarray(self.factor_dcc.R_seq)

    def summary(self) -> dict:
        """Key numbers for the UI."""
        T, N = self.Omega_diag.shape
        idio_share = float(np.mean(self.Omega_diag.mean(axis=0)
                                   / self.conditional_vol().mean(axis=0) ** 2))
        out = {
            "K": int(self.K),
            "N": int(N),
            "T": int(T),
            "loading_mode": self.loading_mode,
            "factor_model": self.factor_model if self.K > 1 else "GARCH(1,1)",
            "idio_garch": self.idio_garch,
            "var_share": [float(v) for v in self.var_share],
            "var_share_total": float(np.sum(self.var_share)),
            "mean_idio_share": idio_share,
        }
        if self.factor_dcc is not None:
            s = self.factor_dcc.get_summary_stats()
            out["factor_alpha"] = s["alpha"]
            out["factor_beta"] = s["beta"]
            out["factor_persistence"] = s["persistence"]
            out["factor_half_life"] = s["half_life_days"]
        return out
