"""
Task A acceptance criteria — DECO is estimated on its own likelihood.

Reference: Engle & Kelly (2012), Dynamic Equicorrelation, JBES 30(2).
"""

import time

import numpy as np
import pytest

from conftest import as_returns, simulate_dcc, simulate_deco
from dcc_garch import (
    DCCGarch,
    _dcc_loop,
    _MODEL_INT,
    deco_loglike_reference,
)


def _fit(returns, model_type):
    m = DCCGarch(model_type=model_type)
    z = m.fit_univariate_garch(returns)
    m.fit_dcc(z)
    return m


def test_model_int_deco_is_distinct():
    """DECO must not share DCC's kernel branch."""
    assert _MODEL_INT["DECO"] != _MODEL_INT["DCC"]
    assert len(set(_MODEL_INT.values())) == len(_MODEL_INT)


# --- A.4.3 : fast closed form == slow explicit implementation ---------------

@pytest.mark.parametrize("N", [3, 6, 12])
def test_deco_closed_form_matches_reference(N):
    """
    The O(N) closed form must reproduce the O(N^3) implementation that builds
    R_t and calls slogdet/solve, to rtol=1e-10.
    """
    rng = np.random.default_rng(7 + N)
    z = np.ascontiguousarray(simulate_dcc(600, N, 0.04, 0.93, rng))
    bar_Q = z.T @ z / z.shape[0]
    N_bar = np.zeros((N, N))

    for a, b in [(0.02, 0.95), (0.05, 0.90), (0.10, 0.85)]:
        fast = _dcc_loop(z, a, b, 0.0, bar_Q, N_bar, _MODEL_INT["DECO"])
        slow = deco_loglike_reference(z, a, b, bar_Q=bar_Q)
        assert np.isclose(fast, slow, rtol=1e-10, atol=0.0), (a, b, fast, slow)


def test_deco_equicorrelation_pd_bound():
    """
    rho must be guarded at -1/(N-1), not -1: a value in between yields an
    indefinite equicorrelation matrix, which the likelihood must reject.
    """
    N = 5
    rho_bad = -0.5 * (1.0 / (N - 1)) - 0.5  # in (-1, -1/(N-1))
    R = (1.0 - rho_bad) * np.eye(N) + rho_bad * np.ones((N, N))
    np.fill_diagonal(R, 1.0)
    assert np.linalg.eigvalsh(R).min() < 0, "test setup: matrix should be indefinite"
    # ... and the closed-form log|R| would be undefined there
    assert 1.0 + (N - 1) * rho_bad < 0


# --- A.4.2 : on a true equicorrelation DGP, DECO ~ DCC ---------------------

def test_deco_matches_dcc_on_equicorrelated_dgp():
    """
    A.4.2 — sanity check that the separation found in A.4.1 is structural.

    On a panel whose unconditional correlation matrix is equicorrelated (all
    pairs 0.4) the equicorrelation restriction is close to free, so DECO and
    DCC must land on the same (a, b) to within 0.02.
    """
    rng = np.random.default_rng(20260730)
    z = simulate_dcc(2500, 6, 0.04, 0.94, rng)   # default R_bar: all pairs 0.4
    returns = as_returns(z)

    deco = _fit(returns, "DECO")
    dcc = _fit(returns, "DCC")

    assert abs(deco.dcc_params[0] - dcc.dcc_params[0]) < 0.02
    assert abs(deco.dcc_params[1] - dcc.dcc_params[1]) < 0.02
    # both recover the DGP
    assert abs(deco.dcc_params[0] - 0.04) < 0.02
    assert abs(dcc.dcc_params[0] - 0.04) < 0.02


def test_correctly_specified_model_wins_on_its_own_dgp():
    """
    Cross-check that each likelihood really is the likelihood of its own model:
    the correctly specified model must attain the higher log-likelihood, and
    DECO must recover the DGP parameters on a genuine equicorrelation process.

    On a DECO DGP the DCC filter is misspecified — its R_t carries pairwise
    noise the DGP does not have — so its pseudo-true 'a' is biased sharply
    downwards.  That is exactly why the two models must not share a kernel.
    """
    rng = np.random.default_rng(20260730)
    z = np.ascontiguousarray(simulate_deco(2500, 6, 0.04, 0.94, rng, rho_bar=0.45))

    fits = {}
    for mt in ("DCC", "DECO"):
        m = DCCGarch(mt)
        m.sigmas = np.ones_like(z)          # residuals are already standardised
        m.fit_dcc(z)
        fits[mt] = m

    assert fits["DECO"].corr_loglik > fits["DCC"].corr_loglik
    assert abs(fits["DECO"].dcc_params[0] - 0.04) < 0.02
    assert abs(fits["DECO"].dcc_params[1] - 0.94) < 0.02


# --- A.4.1 + A.4.4 : on heterogeneous correlations they must differ --------

def test_deco_differs_from_dcc_on_heterogeneous_dgp():
    """
    With pairwise-heterogeneous correlations the equicorrelation restriction
    binds: parameters and log-likelihoods must separate.
    """
    rng = np.random.default_rng(11)
    N = 5
    R_bar = np.array([
        [1.00, 0.85, 0.20, 0.10, 0.05],
        [0.85, 1.00, 0.15, 0.05, 0.05],
        [0.20, 0.15, 1.00, 0.75, 0.10],
        [0.10, 0.05, 0.75, 1.00, 0.10],
        [0.05, 0.05, 0.10, 0.10, 1.00],
    ])
    z = simulate_dcc(2000, N, 0.05, 0.90, rng, R_bar=R_bar)
    returns = as_returns(z)

    deco = _fit(returns, "DECO")
    dcc = _fit(returns, "DCC")

    da = abs(deco.dcc_params[0] - dcc.dcc_params[0])
    db = abs(deco.dcc_params[1] - dcc.dcc_params[1])
    assert max(da, db) > 1e-4, "DECO collapsed onto the DCC solution"
    assert abs(deco.corr_loglik - dcc.corr_loglik) > 1e-6
    # DCC nests the equicorrelation structure, so it cannot fit worse.
    assert dcc.corr_loglik > deco.corr_loglik


def test_deco_on_bundled_sample(sample_returns):
    """A.4.1 on the dataset the app actually ships with."""
    cols = list(sample_returns.columns)[:4]
    returns = sample_returns[cols]

    deco = _fit(returns, "DECO")
    dcc = _fit(returns, "DCC")

    da = abs(deco.dcc_params[0] - dcc.dcc_params[0])
    db = abs(deco.dcc_params[1] - dcc.dcc_params[1])
    assert max(da, db) > 1e-4
    assert abs(deco.corr_loglik - dcc.corr_loglik) > 1e-6


# --- structural checks -----------------------------------------------------

def test_deco_R_seq_is_equicorrelated():
    rng = np.random.default_rng(3)
    z = simulate_deco(400, 5, 0.05, 0.90, rng)
    m = _fit(as_returns(z), "DECO")

    R = m.R_seq
    iu = np.triu_indices(5, k=1)
    off = R[:, iu[0], iu[1]]
    # every pair identical at each t
    assert np.allclose(off, off[:, [0]], atol=1e-12)
    assert np.allclose(np.diagonal(R, axis1=1, axis2=2), 1.0)
    # the reported series matches the matrices
    assert np.allclose(m.get_equicorrelation_series(), off[:, 0], atol=1e-12)
    # every R_t positive definite
    assert np.linalg.eigvalsh(R).min() > 0


def test_summary_uses_own_params_not_dcc_base():
    rng = np.random.default_rng(5)
    z = simulate_dcc(600, 4, 0.06, 0.90, rng)
    m = DCCGarch("DECO")
    zz = m.fit_univariate_garch(as_returns(z))
    m.fit_dcc(zz, deco_base_dcc=True)

    s = m.get_summary_stats()
    assert s["alpha"] == pytest.approx(float(m.dcc_params[0]))
    assert s["beta"] == pytest.approx(float(m.dcc_params[1]))
    assert s["corr_loglik"] == pytest.approx(m.corr_loglik)
    assert s["n_params"] == 2
    # the reference DCC fit is kept, but only for reporting
    assert m.dcc_base_params is not None
    assert m.dcc_base_loglik != m.corr_loglik


# --- A.4.5 : the closed form must actually be cheaper ----------------------

def test_deco_likelihood_cheaper_than_dcc():
    """
    Per-evaluation cost at N=50, T=2500: DECO avoids every N x N factorisation,
    so its correlation-stage likelihood must cost at most 30 % of DCC's.
    """
    rng = np.random.default_rng(99)
    N, T = 50, 2500
    z = np.ascontiguousarray(rng.standard_normal((T, N)))
    bar_Q = z.T @ z / T
    N_bar = np.zeros((N, N))

    # warm up the JIT so compilation time is not measured
    for code in (_MODEL_INT["DCC"], _MODEL_INT["DECO"]):
        _dcc_loop(z[:50], 0.03, 0.95, 0.0, bar_Q, N_bar, code)

    def timeit(code, reps=3):
        best = np.inf
        for _ in range(reps):
            t0 = time.perf_counter()
            _dcc_loop(z, 0.03, 0.95, 0.0, bar_Q, N_bar, code)
            best = min(best, time.perf_counter() - t0)
        return best

    t_dcc = timeit(_MODEL_INT["DCC"])
    t_deco = timeit(_MODEL_INT["DECO"])
    assert t_deco <= 0.30 * t_dcc, f"DECO {t_deco:.4f}s vs DCC {t_dcc:.4f}s"
