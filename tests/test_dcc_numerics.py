"""
Task E1-E5 — numerical correctness and efficiency of the correlation stage.
"""

import numpy as np
import pytest

from conftest import as_returns, simulate_dcc
from dcc_garch import (
    DCCGarch,
    _cdcc_target,
    _dcc_loop,
    _dcc_loop_numpy,
    _MODEL_INT,
    HAS_NUMBA,
)


def _naive_loglike(z, a, b, c, bar_Q, N_bar, model):
    """
    Deliberately literal reimplementation using det/inv and diag-matrix
    products — the code shape E3/E4 replaced.  Serves as the oracle.
    """
    T, N = z.shape
    Q = bar_Q.copy()
    nll = 0.0
    for t in range(1, T):
        zp = z[t - 1]
        if model == "CDCC":
            P = np.diag(np.sqrt(np.diag(Q)))
            zs = P @ zp
            Q = (1 - a - b) * bar_Q + a * np.outer(zs, zs) + b * Q
        elif model == "ADCC":
            n = zp * (zp < 0.0)
            Q = ((1 - a - b) * bar_Q - c * N_bar + a * np.outer(zp, zp)
                 + b * Q + c * np.outer(n, n))
        else:
            Q = (1 - a - b) * bar_Q + a * np.outer(zp, zp) + b * Q
        Pi = np.diag(1.0 / np.sqrt(np.diag(Q)))
        R = Pi @ Q @ Pi
        zt = z[t]
        nll += 0.5 * (np.log(np.linalg.det(R))
                      + zt @ np.linalg.inv(R) @ zt - zt @ zt)
    return nll


@pytest.mark.parametrize("model", ["DCC", "CDCC", "ADCC"])
def test_cholesky_path_matches_det_inv_path(model):
    """E3/E4 must be a pure refactor: same value as det/inv, rtol=1e-10."""
    rng = np.random.default_rng(4)
    N = 6
    z = np.ascontiguousarray(simulate_dcc(500, N, 0.05, 0.90, rng))
    bar_Q = z.T @ z / z.shape[0]
    neg = z * (z < 0)
    N_bar = neg.T @ neg / z.shape[0]
    c = 0.03 if model == "ADCC" else 0.0

    fast = _dcc_loop(z, 0.04, 0.90, c, bar_Q, N_bar, _MODEL_INT[model])
    slow = _naive_loglike(z, 0.04, 0.90, c, bar_Q, N_bar, model)
    assert np.isclose(fast, slow, rtol=1e-10, atol=0.0)


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
@pytest.mark.parametrize("code", [0, 1, 2, 3])
def test_numba_and_numpy_kernels_agree(code):
    """The two dispatch branches must not drift apart."""
    rng = np.random.default_rng(13)
    N = 5
    z = np.ascontiguousarray(simulate_dcc(400, N, 0.05, 0.90, rng))
    bar_Q = z.T @ z / z.shape[0]
    neg = z * (z < 0)
    N_bar = neg.T @ neg / z.shape[0]
    c = 0.02 if code == 2 else 0.0

    nb = _dcc_loop(z, 0.04, 0.92, c, bar_Q, N_bar, code)
    np_ = _dcc_loop_numpy(z, 0.04, 0.92, c, bar_Q, N_bar, code)
    assert np.isclose(nb, np_, rtol=1e-10, atol=0.0)


def test_e3_determinant_underflows_where_cholesky_does_not():
    """
    E3 motivation: for a strongly correlated R at large N, det(R) underflows to
    exactly 0, so log(det(R)) is -inf.  The Cholesky log-determinant stays
    finite and exact.
    """
    N, rho = 400, 0.9
    R = (1.0 - rho) * np.eye(N) + rho * np.ones((N, N))
    np.fill_diagonal(R, 1.0)

    assert np.linalg.det(R) == 0.0, "test setup: determinant should underflow"

    L = np.linalg.cholesky(R)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    exact = (N - 1) * np.log(1.0 - rho) + np.log(1.0 + (N - 1) * rho)
    assert np.isfinite(logdet)
    assert np.isclose(logdet, exact, rtol=1e-10)


def test_e3_kernel_stays_finite_at_large_N():
    """The DCC kernel must return a usable value where det/inv would not."""
    N, T = 80, 300
    rng = np.random.default_rng(21)
    R_bar = 0.9 * np.ones((N, N))
    np.fill_diagonal(R_bar, 1.0)
    L = np.linalg.cholesky(R_bar)
    z = np.ascontiguousarray(rng.standard_normal((T, N)) @ L.T)
    bar_Q = z.T @ z / T
    N_bar = np.zeros((N, N))

    val = _dcc_loop(z, 0.03, 0.95, 0.0, bar_Q, N_bar, _MODEL_INT["DCC"])
    assert np.isfinite(val) and val < 1e9


def test_e2_target_is_uncentred_second_moment():
    """E2: bar_Q = z'z/T, not np.cov (which demeans and divides by T-1)."""
    rng = np.random.default_rng(8)
    z = simulate_dcc(300, 4, 0.05, 0.90, rng) + 0.3   # deliberately off-centre
    m = DCCGarch("DCC")
    m.sigmas = np.ones_like(z)
    m.fit_dcc(z)

    expected = z.T @ z / z.shape[0]
    assert np.allclose(m._bar_Q, expected)
    assert not np.allclose(m._bar_Q, np.cov(z.T))


def test_e1_cdcc_target_is_a_fixed_point():
    """
    E1: after fitting cDCC, bar_Q must be a fixed point of the Aielli map
    S -> (1/T) sum diag(Q_t)^{1/2} z_t z_t' diag(Q_t)^{1/2}, and must differ
    from the plain DCC target.
    """
    rng = np.random.default_rng(2026)
    z = np.ascontiguousarray(simulate_dcc(1500, 4, 0.06, 0.90, rng))

    m = DCCGarch("cDCC")
    m.sigmas = np.ones_like(z)
    m.fit_dcc(z)

    a, b = float(m.dcc_params[0]), float(m.dcc_params[1])
    S_next = _cdcc_target(z, a, b, m._bar_Q)
    drift = float(np.max(np.abs(S_next - m._bar_Q)))
    assert drift < 1e-6, f"cDCC target not a fixed point (drift {drift:.2e})"

    dcc_target = z.T @ z / z.shape[0]
    assert float(np.max(np.abs(m._bar_Q - dcc_target))) > 1e-4, (
        "cDCC target collapsed onto the DCC target"
    )


def test_e5_mvp_weights_match_explicit_inverse():
    """E5: cho_solve path must reproduce the H^{-1}1 / 1'H^{-1}1 formula."""
    rng = np.random.default_rng(17)
    z = simulate_dcc(300, 4, 0.05, 0.90, rng)
    m = DCCGarch("DCC")
    zz = m.fit_univariate_garch(as_returns(z))
    m.fit_dcc(zz)

    w = m.compute_mvp_weights()
    ones = np.ones(4)
    for t in (0, 50, 299):
        Hi = np.linalg.inv(m.H_seq[t])
        expected = Hi @ ones / (ones @ Hi @ ones)
        assert np.allclose(w[t], expected, rtol=1e-9, atol=1e-12)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_r_seq_is_valid_correlation_matrix():
    """Diagonal exactly 1, symmetric, positive definite, for every model."""
    rng = np.random.default_rng(31)
    z = simulate_dcc(300, 4, 0.05, 0.90, rng)
    returns = as_returns(z)
    for mt in ("DCC", "cDCC", "ADCC", "DECO"):
        m = DCCGarch(mt)
        zz = m.fit_univariate_garch(returns)
        m.fit_dcc(zz)
        R = m.R_seq
        assert np.allclose(np.diagonal(R, axis1=1, axis2=2), 1.0), mt
        assert np.allclose(R, np.transpose(R, (0, 2, 1))), mt
        assert np.linalg.eigvalsh(R).min() > 0, mt


def test_summary_reports_information_criteria():
    rng = np.random.default_rng(23)
    z = simulate_dcc(300, 3, 0.05, 0.90, rng)
    m = DCCGarch("ADCC")
    zz = m.fit_univariate_garch(as_returns(z))
    m.fit_dcc(zz)
    s = m.get_summary_stats()
    assert s["n_params"] == 3
    assert s["aic"] == pytest.approx(-2 * s["corr_loglik"] + 2 * 3)
    assert s["bic"] == pytest.approx(-2 * s["corr_loglik"] + 3 * np.log(s["n_obs"]))
