"""Shared pytest fixtures for the EYS'26 test suite."""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

# Make the repo root (python/) importable regardless of where pytest is invoked.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def sample_returns():
    """The dataset bundled with the app (data/sample_returns.csv)."""
    path = os.path.join(ROOT, "data", "sample_returns.csv")
    if not os.path.exists(path):
        pytest.skip("sample_returns.csv not available")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    cols = [c for c in df.columns if not c.endswith("_RV") and not c.endswith("_BPV")]
    return df[cols]


def simulate_dcc(T, N, a, b, rng, R_bar=None):
    """
    Draw standardised residuals from a scalar DCC data-generating process.

    Returns z of shape (T, N) with unit unconditional variances.
    """
    if R_bar is None:
        R_bar = 0.4 * np.ones((N, N))
        np.fill_diagonal(R_bar, 1.0)
    Q = R_bar.copy()
    z = np.zeros((T, N))
    z_prev = rng.standard_normal(N)
    z[0] = z_prev
    for t in range(1, T):
        Q = (1.0 - a - b) * R_bar + a * np.outer(z_prev, z_prev) + b * Q
        d = 1.0 / np.sqrt(np.diag(Q))
        R = Q * np.outer(d, d)
        np.fill_diagonal(R, 1.0)
        L = np.linalg.cholesky(R)
        z_prev = L @ rng.standard_normal(N)
        z[t] = z_prev
    return z


def simulate_deco(T, N, a, b, rng, rho_bar=0.4):
    """
    Draw standardised residuals from a genuine EQUICORRELATION DGP: the
    correlation matrix is (1-rho_t) I + rho_t 11' at every t, with rho_t driven
    by the scalar DECO recursion.
    """
    R_bar = rho_bar * np.ones((N, N))
    np.fill_diagonal(R_bar, 1.0)
    Q = R_bar.copy()
    z = np.zeros((T, N))
    z_prev = rng.standard_normal(N)
    z[0] = z_prev
    lo = -1.0 / (N - 1) + 1e-6
    for t in range(1, T):
        Q = (1.0 - a - b) * R_bar + a * np.outer(z_prev, z_prev) + b * Q
        d = 1.0 / np.sqrt(np.diag(Q))
        rho = float(np.clip((d @ Q @ d - N) / (N * (N - 1)), lo, 1.0 - 1e-6))
        R = (1.0 - rho) * np.eye(N) + rho * np.ones((N, N))
        np.fill_diagonal(R, 1.0)
        L = np.linalg.cholesky(R)
        z_prev = L @ rng.standard_normal(N)
        z[t] = z_prev
    return z


def as_returns(z, scale=0.01):
    """Wrap standardised residuals as a returns DataFrame the app can fit."""
    cols = [f"A{i+1}" for i in range(z.shape[1])]
    idx = pd.bdate_range("2015-01-01", periods=z.shape[0])
    return pd.DataFrame(z * scale, index=idx, columns=cols)
