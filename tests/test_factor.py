"""
Task C acceptance criteria — factor selection and Factor-DCC.
"""

import time

import numpy as np
import pandas as pd
import pytest

import factor_selection as fs
from factor_dcc import FactorDCC


def simulate_factor_panel(N, T, K, rng, factor_vol=None, idio_scale=1.0):
    """r_t = B f_t + u_t with a known K-factor covariance."""
    if factor_vol is None:
        factor_vol = np.linspace(0.020, 0.008, K)
    Sig_f = np.diag(factor_vol ** 2)
    B = rng.standard_normal((N, K)) * 0.6
    idio_sd = (0.005 + 0.010 * rng.random(N)) * idio_scale
    Sigma = B @ Sig_f @ B.T + np.diag(idio_sd ** 2)
    f = rng.standard_normal((T, K)) @ np.linalg.cholesky(Sig_f).T
    u = rng.standard_normal((T, N)) * idio_sd
    return f @ B.T + u, Sigma, B


def as_df(R):
    cols = [f"A{i+1}" for i in range(R.shape[1])]
    return pd.DataFrame(R, index=pd.bdate_range("2015-01-01", periods=R.shape[0]),
                        columns=cols)


# --- C.5.1 : all three criteria recover K ----------------------------------

@pytest.mark.parametrize("K", [1, 3, 5])
def test_all_criteria_recover_known_K(K):
    rng = np.random.default_rng(100 + K)
    R, _Sigma, _B = simulate_factor_panel(N=80, T=800, K=K, rng=rng)
    sel = fs.select_k(R, method="Bai-Ng ICp1", k_max=10)
    assert sel["bai_ng"] == K
    assert sel["onatski"] == K
    assert sel["mp"] == K


def test_bai_ng_can_select_zero_factors():
    """Fix #2: k=0 must be reachable when there is no factor structure."""
    rng = np.random.default_rng(55)
    R = rng.standard_normal((1500, 40)) * 0.01
    assert fs.bai_ng_ic(R, k_max=10) == 0


def test_bai_ng_standardises_the_panel():
    """
    Fix #1: without standardisation a single high-variance asset dominates the
    principal components.  Rescaling one column must not change the answer.
    """
    rng = np.random.default_rng(66)
    R, _S, _B = simulate_factor_panel(N=60, T=600, K=3, rng=rng)
    k_plain = fs.bai_ng_ic(R, k_max=10)

    R_scaled = R.copy()
    R_scaled[:, 0] *= 500.0
    assert fs.bai_ng_ic(R_scaled, k_max=10) == k_plain == 3


@pytest.mark.parametrize("N", [3, 4, 5, 6, 8])
def test_onatski_handles_small_panels(N):
    """
    Regression: the fixed 5-eigenvalue window used to raise a matmul shape
    error whenever the panel had fewer eigenvalues than the window.
    """
    rng = np.random.default_rng(N)
    R = rng.standard_normal((400, N)) * 0.01
    k, ok = fs.onatski_ed(R, k_max=min(10, N - 1), return_ok=True)
    assert isinstance(k, int) and k >= 0
    assert isinstance(ok, bool)
    sel = fs.select_k(R, method="Bai-Ng ICp1", k_max=min(10, N - 1))
    assert "onatski_ok" in sel


def test_all_criteria_use_the_correlation_matrix():
    """
    Fix #3: Onatski and MP must be scale-invariant, i.e. read off the same
    correlation matrix, so the three counts stay comparable.
    """
    rng = np.random.default_rng(77)
    R, _S, _B = simulate_factor_panel(N=60, T=600, K=3, rng=rng)
    R_scaled = R * np.linspace(1.0, 100.0, R.shape[1])

    assert fs.onatski_ed(R, k_max=10) == fs.onatski_ed(R_scaled, k_max=10)
    assert fs.mp_threshold_count(R)[0] == fs.mp_threshold_count(R_scaled)[0]


# --- C.5.2 / C.5.3 : Woodbury and the determinant lemma --------------------

@pytest.fixture(scope="module")
def fitted_small():
    rng = np.random.default_rng(2026)
    R, _S, _B = simulate_factor_panel(N=10, T=600, K=3, rng=rng)
    return FactorDCC(K=3).fit(as_df(R))


def test_woodbury_matches_explicit_inverse(fitted_small):
    for t in (0, 137, 599):
        fast = fitted_small.H_inv_at(t)
        slow = np.linalg.inv(fitted_small.H_at(t))
        assert np.allclose(fast, slow, rtol=1e-9, atol=1e-12), t


def test_determinant_lemma_matches_slogdet(fitted_small):
    for t in (0, 137, 599):
        sign, ld = np.linalg.slogdet(fitted_small.H_at(t))
        assert sign > 0
        assert np.isclose(fitted_small.logdet_H(t), ld, rtol=1e-9)


# --- C.5.4 : H_t positive definite, MVP weights sum to one ----------------

def test_H_positive_definite_and_weights_sum_to_one(fitted_small):
    for t in (0, 250, 599):
        assert np.linalg.eigvalsh(fitted_small.H_at(t)).min() > 0
    w = fitted_small.mvp_weights()
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.isfinite(w).all()


def test_omega_keeps_H_invertible_when_rank_deficient():
    """
    B Lambda_t B' alone is rank K < N and singular; only Omega makes H_t
    invertible.  Verify the rank claim explicitly.
    """
    rng = np.random.default_rng(4)
    R, _S, _B = simulate_factor_panel(N=8, T=400, K=2, rng=rng)
    m = FactorDCC(K=2).fit(as_df(R))

    low_rank = m.B @ m.Lambda_seq[10] @ m.B.T
    assert np.linalg.matrix_rank(low_rank, tol=1e-12) <= 2
    assert np.linalg.eigvalsh(low_rank).min() < 1e-10
    assert np.linalg.eigvalsh(m.H_at(10)).min() > 0


def test_conditional_vol_matches_diagonal_of_H(fitted_small):
    cv = fitted_small.conditional_vol()
    for t in (0, 300):
        assert np.allclose(cv[t] ** 2, np.diag(fitted_small.H_at(t)), rtol=1e-10)


def test_observed_factor_mode():
    rng = np.random.default_rng(8)
    R, _S, B = simulate_factor_panel(N=12, T=500, K=2, rng=rng)
    f = rng.standard_normal((500, 2)) * 0.01
    R_obs = f @ B[:, :2].T + rng.standard_normal((500, 12)) * 0.004

    fdf = pd.DataFrame(f, columns=["MKT", "SMB"])
    m = FactorDCC(K=2, loading_mode="observed").fit(as_df(R_obs), factor_returns=fdf)
    assert m.B.shape == (12, 2)
    assert np.allclose(m.mvp_weights().sum(axis=1), 1.0)


# --- C.5.5 : the large-N fit must be fast ---------------------------------

@pytest.mark.slow
def test_large_panel_fit_under_30s():
    rng = np.random.default_rng(31415)
    R, _S, _B = simulate_factor_panel(N=200, T=2500, K=5, rng=rng)
    t0 = time.perf_counter()
    m = FactorDCC(K=5).fit(as_df(R))
    w = m.mvp_weights()
    elapsed = time.perf_counter() - t0
    assert np.allclose(w.sum(axis=1), 1.0)
    assert elapsed < 30.0, f"Factor-DCC took {elapsed:.1f}s at N=200, T=2500"
