"""
go_garch.py
===========
Generalised Orthogonal GARCH — van der Weide (2002), course notes 1.10.1.

Model
-----
    r_t = Z y_t,        E[y_t y_t' | F_{t-1}] = V_t = diag(h_{1t}, ..., h_{Nt})
    H_t = Z V_t Z'

Z is a constant, invertible N x N mixing matrix.  It is factorised as

    Z = P Lambda^{1/2} U

where (P, Lambda) is the eigendecomposition of the unconditional covariance of
r_t and U is orthogonal.  Two ways of pinning U down are offered:

* ``method="pca"``  – U = I.  This is O-GARCH (Alexander 2001): the components
  are the principal components, orthogonal only UNCONDITIONALLY.
* ``method="ica"``  – U from FastICA on the whitened data (the practical
  estimator of van der Weide 2002; see also Broda & Paolella 2009).  Asking for
  statistically INDEPENDENT components is stronger than asking for uncorrelated
  ones, which is what the model's conditional-diagonality assumption needs.

Each component then gets its own univariate GARCH(1,1), so V_t is diagonal by
construction.

Contrast with Factor-DCC (notes, Remark 1.8)
--------------------------------------------
GO-GARCH keeps all N components: Z is square and invertible, H_t is full rank
on its own and needs no idiosyncratic term.  Factor-DCC keeps K < N factors, so
B Lambda_t B' is singular and an Omega_t is indispensable.  GO-GARCH buys
parsimony by restricting the ROTATION (one constant Z, N univariate GARCHs, no
correlation stage at all); Factor-DCC buys it by restricting the RANK.

Identification
--------------
Component order, sign and scale are not identified: (Z, V_t) and
(Z D S, S D^{-1} V_t D^{-1} S) give the same H_t for any permutation/sign
matrix S and positive diagonal D.  Individual columns of Z must therefore not
be given an economic reading; only H_t and the variance shares are invariant.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


class GOGarch:
    """
    GO-GARCH estimator.

    Workflow
    --------
    >>> model = GOGarch(method="ica")
    >>> model.fit(returns_df)
    >>> w = model.mvp_weights()
    """

    def __init__(self, method: str = "ica", random_state: int = 20260730):
        if method not in ("ica", "pca"):
            raise ValueError("method must be 'ica' or 'pca'.")
        self.method = method
        self.random_state = random_state

        self.Z = None            # (N, N) mixing matrix
        self.Z_inv = None
        self.U = None            # (N, N) rotation
        self.components = None   # (T, N) y_t
        self.h_seq = None        # (T, N) component conditional variances
        self.cols = None
        self.index = None
        self.loglik = None
        self.var_share = None
        self.univariate_models = []
        self._converged = None

    # ------------------------------------------------------------------

    def fit(self, returns: pd.DataFrame):
        """
        Estimate Z, extract the components and fit a GARCH(1,1) to each.

        Parameters
        ----------
        returns : (T, N) DataFrame of asset returns (decimals)
        """
        from arch import arch_model

        R = np.asarray(returns.values, dtype=float)
        T, N = R.shape
        if N < 2:
            raise ValueError("GO-GARCH requires at least two assets.")
        self.cols = list(returns.columns)
        self.index = returns.index

        mu = R.mean(axis=0)
        Rc = R - mu

        # --- unconditional whitening: S = P Lambda^{1/2} -----------------
        Sigma = Rc.T @ Rc / T
        w, P = np.linalg.eigh(Sigma)
        order = np.argsort(w)[::-1]
        w, P = w[order], P[:, order]
        if w.min() <= 0:
            raise np.linalg.LinAlgError(
                "Unconditional covariance is singular — drop collinear assets."
            )
        lam_sqrt = np.sqrt(w)
        whitened = Rc @ P / lam_sqrt          # (T, N), identity covariance

        # --- rotation U ---------------------------------------------------
        if self.method == "pca":
            self.U = np.eye(N)
            self._converged = True
        else:
            from sklearn.decomposition import FastICA
            ica = FastICA(n_components=N, whiten="unit-variance",
                          random_state=self.random_state, max_iter=1000, tol=1e-6)
            ica.fit(whitened)
            # FastICA returns an orthogonal mixing matrix on already-white data;
            # re-orthogonalise defensively (polar decomposition).
            A = np.asarray(ica.mixing_, dtype=float)
            Ua, _s, Vt = np.linalg.svd(A)
            self.U = Ua @ Vt
            self._converged = bool(getattr(ica, "n_iter_", 0) < 1000)

        # Z = P Lambda^{1/2} U ; y_t = Z^{-1} r_t = U' whitened_t
        self.Z = (P * lam_sqrt) @ self.U
        self.Z_inv = np.linalg.inv(self.Z)
        self.components = whitened @ self.U

        # --- univariate GARCH per component -------------------------------
        h = np.zeros((T, N))
        self.univariate_models = []
        for i in range(N):
            am = arch_model(self.components[:, i] * 100, vol="Garch", p=1, q=1,
                            dist="normal", rescale=False)
            res = am.fit(disp="off")
            self.univariate_models.append(res)
            h[:, i] = (np.asarray(res.conditional_volatility) / 100.0) ** 2
        self.h_seq = h

        # --- log-likelihood (Gaussian, up to the -T*N/2*log(2pi) constant) --
        sign, logdet_Z = np.linalg.slogdet(self.Z)
        y = self.components
        self.loglik = float(
            -T * logdet_Z
            - 0.5 * np.sum(np.log(h) + y ** 2 / h)
        )

        # variance share of each component in total unconditional variance
        contrib = (self.Z ** 2) * h.mean(axis=0)      # (N, N): asset x component
        self.var_share = contrib.sum(axis=0) / contrib.sum()

        return self

    # ------------------------------------------------------------------
    # H_t access — Z is constant, so nothing needs to be stored per t
    # ------------------------------------------------------------------

    def H_at(self, t: int):
        """H_t = Z V_t Z' (built explicitly; for plots)."""
        return (self.Z * self.h_seq[t]) @ self.Z.T

    def H_inv_at(self, t: int):
        """H_t^{-1} = Z^{-1'} V_t^{-1} Z^{-1} — one constant inverse, no per-t solve."""
        Zi = self.Z_inv
        return Zi.T @ (Zi / self.h_seq[t][:, None])

    def logdet_H(self, t: int) -> float:
        """log det H_t = 2 log|det Z| + sum_i log h_it."""
        _sign, logdet_Z = np.linalg.slogdet(self.Z)
        return 2.0 * float(logdet_Z) + float(np.sum(np.log(self.h_seq[t])))

    def conditional_vol(self):
        """Per-asset conditional volatility path, sqrt(diag(H_t))."""
        # diag(Z V_t Z')_ii = sum_k Z_ik^2 h_kt
        return np.sqrt(self.h_seq @ (self.Z ** 2).T)

    def conditional_correlation(self):
        """Time-varying correlation matrices implied by H_t."""
        T = self.h_seq.shape[0]
        N = self.Z.shape[0]
        R = np.zeros((T, N, N))
        for t in range(T):
            H = self.H_at(t)
            d = 1.0 / np.sqrt(np.diag(H))
            R[t] = H * np.outer(d, d)
            np.fill_diagonal(R[t], 1.0)
        return R

    def mvp_weights(self):
        """
        w_t = H_t^{-1} 1 / (1' H_t^{-1} 1), evaluated as
        Z^{-1'} V_t^{-1} Z^{-1} 1 — one triangular-free constant inverse and a
        diagonal scaling per t.
        """
        T, N = self.h_seq.shape
        ones = np.ones(N)
        u = self.Z_inv @ ones                    # constant across t
        W = np.zeros((T, N))
        for t in range(T):
            x = self.Z_inv.T @ (u / self.h_seq[t])
            W[t] = x / x.sum()
        return W

    def summary(self) -> dict:
        """Key numbers for the UI."""
        T, N = self.h_seq.shape
        rows = []
        for i, res in enumerate(self.univariate_models):
            p = res.params
            a = float(p["alpha[1]"])
            b = float(p["beta[1]"])
            rows.append({
                "component": f"y{i+1}",
                "alpha": a,
                "beta": b,
                "persistence": a + b,
                "half_life": (math.log(0.5) / math.log(a + b)) if 0 < a + b < 1 else np.nan,
                "var_share": float(self.var_share[i]),
            })
        return {
            "method": self.method,
            "N": int(N),
            "T": int(T),
            "n_params": int(3 * N),      # N GARCH(1,1)s; Z is profiled out
            "loglik": self.loglik,
            "ica_converged": self._converged,
            "components": rows,
        }
