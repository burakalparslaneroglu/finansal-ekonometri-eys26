"""
factor_selection.py
===================
Shared factor-count selection for the Day-3 Factor-DCC tab and the Day-5 POET
tab.  Previously these estimators lived only inside
``scripts/highdim_cov.py`` (a standalone script the app never imported), which
meant the POET tab offered nothing but a manual ``k_factors`` slider.

Three inconsistencies present in the script version are fixed here:

1. ``bai_ng_ic`` only demeaned the columns.  Bai & Ng (2002) assume a
   STANDARDISED panel, so the columns are now divided by their standard
   deviation as well — otherwise a single high-variance asset dominates the
   principal components and the criterion over-counts factors.
2. ``bai_ng_ic`` started the candidate set at k=1, so "no factors" could never
   be selected.  k=0 is now a candidate.
3. ``onatski_ed`` worked on the sample COVARIANCE matrix while
   ``mp_threshold_count`` worked on the CORRELATION matrix, so the three
   criteria were not on a comparable scale.  All three now use the correlation
   matrix of the standardised panel.

References
----------
Bai, J. & Ng, S. (2002). Determining the Number of Factors in Approximate
    Factor Models. Econometrica 70(1), 191-221.
Onatski, A. (2010). Determining the Number of Factors from Empirical
    Distribution of Eigenvalues. REStat 92(4), 1004-1016.
Marchenko, V. & Pastur, L. (1967); bulk-edge rule as used in RMT screening.
"""

from __future__ import annotations

import numpy as np

METHODS = ("Bai-Ng ICp1", "Onatski ED", "Marchenko-Pastur", "Manuel")


def standardise(R):
    """Return the column-standardised panel X = (R - mean) / std."""
    R = np.asarray(R, dtype=float)
    sd = R.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (R - R.mean(axis=0)) / sd


def correlation_matrix(R):
    """Sample correlation matrix of the panel (X'X / T on standardised data)."""
    X = standardise(R)
    T = X.shape[0]
    return X.T @ X / T


def bai_ng_ic(R, k_max: int = 10, return_curve: bool = False):
    """
    Bai-Ng (2002) IC_{p1} on the STANDARDISED panel, with k=0 in the candidate
    set.

        IC_p1(k) = log V(k) + k * (N+T)/(N*T) * log( N*T/(N+T) )

    where V(k) is the mean squared residual after removing k principal
    components.

    Returns
    -------
    int, or (int, np.ndarray) when ``return_curve`` is True — the IC value for
    each k in 0..k_max.
    """
    X = standardise(R)
    T, N = X.shape
    k_max = int(min(k_max, min(N, T) - 1))
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    g = (N + T) / (N * T) * np.log(N * T / (N + T))
    ics = []
    for k in range(0, k_max + 1):
        if k == 0:
            resid = X
        else:
            resid = X - (U[:, :k] * s[:k]) @ Vt[:k, :]
        V_k = float(np.mean(resid ** 2))
        V_k = max(V_k, 1e-300)
        ics.append(np.log(V_k) + k * g)

    ics = np.asarray(ics)
    k_hat = int(np.argmin(ics))          # index == k because k starts at 0
    return (k_hat, ics) if return_curve else k_hat


def mp_threshold_count(R):
    """
    Number of eigenvalues of the sample CORRELATION matrix above the
    Marchenko-Pastur bulk edge  lambda_+ = (1 + sqrt(N/T))^2.

    Returns
    -------
    (count, lambda_plus, eigenvalues_ascending)
    """
    R = np.asarray(R, dtype=float)
    T, N = R.shape
    C = correlation_matrix(R)
    ev = np.linalg.eigvalsh(C)
    c = N / T
    lam_plus = (1.0 + np.sqrt(c)) ** 2
    return int((ev > lam_plus).sum()), float(lam_plus), ev


ED_MIN_WINDOW = 3          # fewest eigenvalues the delta regression can use


def onatski_ed(R, k_max: int = 10, max_iter: int = 100, return_ok: bool = False):
    """
    Onatski (2010) Eigenvalue-Difference estimator, iterative form, applied to
    the CORRELATION matrix so it is comparable with the other two criteria.

    delta is recalibrated by regressing eigenvalues lam_j..lam_{j+4} on
    ((j-1)..(j+3))^{2/3};  k_hat(delta) = max{i <= k_max : lam_i - lam_{i+1} >= delta}.

    The window is clipped to the eigenvalues that actually exist.  With a short
    panel (small N) fewer than ED_MIN_WINDOW are left and delta cannot be
    calibrated at all; ``return_ok`` then reports False so callers can display
    "not available" instead of a meaningless zero.
    """
    C = correlation_matrix(R)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    n = len(ev)
    k_max = int(min(k_max, n - 2))
    if k_max < 1:
        return (0, False) if return_ok else 0

    j = k_max + 1
    k_prev = -1
    k_hat = 0
    ok = False
    for _ in range(max_iter):
        j = max(1, min(j, n - 1))
        win = min(5, n - j + 1)
        if win < ED_MIN_WINDOW:
            break
        y = ev[j - 1:j - 1 + win]
        xreg = (np.arange(j, j + win) - 1.0) ** (2.0 / 3.0)
        xc = xreg - xreg.mean()
        denom = float(xc @ xc)
        if denom <= 0:
            break
        beta = float(xc @ (y - y.mean())) / denom
        delta = 2.0 * abs(beta)
        diffs = ev[:k_max] - ev[1:k_max + 1]
        cand = np.where(diffs >= delta)[0]
        k_hat = int(cand.max() + 1) if len(cand) else 0
        ok = True
        if k_hat == k_prev:
            break
        k_prev = k_hat
        j = k_hat + 1

    return (k_hat, ok) if return_ok else k_hat


def select_k(R, method: str = "Bai-Ng ICp1", k_max: int = 10, manual: int = 1):
    """
    Run every criterion and return both the chosen K and the full comparison.

    Returns
    -------
    dict with keys:
        k          – the K implied by ``method``
        method     – echo of the request
        bai_ng, onatski, mp        – each criterion's K
        lambda_plus, eigenvalues   – MP bulk edge and the correlation spectrum
        ic_curve                   – Bai-Ng IC by k (index == k)
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}; got '{method}'.")

    k_bn, ic_curve = bai_ng_ic(R, k_max=k_max, return_curve=True)
    k_mp, lam_plus, ev = mp_threshold_count(R)
    k_on, on_ok = onatski_ed(R, k_max=k_max, return_ok=True)

    chosen = {
        "Bai-Ng ICp1": k_bn,
        "Onatski ED": k_on,
        "Marchenko-Pastur": k_mp,
        "Manuel": int(manual),
    }[method]

    return {
        "k": int(chosen),
        "method": method,
        "bai_ng": int(k_bn),
        "onatski": int(k_on),
        "onatski_ok": bool(on_ok),
        "mp": int(k_mp),
        "lambda_plus": float(lam_plus),
        "eigenvalues": np.sort(ev)[::-1],
        "ic_curve": ic_curve,
    }
