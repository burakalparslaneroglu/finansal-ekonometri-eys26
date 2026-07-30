"""
Task E6 — GO-GARCH (van der Weide 2002).
"""

import numpy as np
import pandas as pd
import pytest

from go_garch import GOGarch


def _panel(T=800, N=4, seed=11):
    """Returns with a genuine common-factor structure and GARCH components."""
    rng = np.random.default_rng(seed)
    y = np.zeros((T, N))
    s2 = np.ones(N) * 1.0
    for t in range(T):
        s2 = 0.05 + 0.08 * (y[t - 1] ** 2 if t else np.ones(N)) + 0.88 * s2
        y[t] = np.sqrt(s2) * rng.standard_normal(N)
    Z = rng.standard_normal((N, N)) * 0.5 + np.eye(N)
    R = y @ Z.T * 0.01
    idx = pd.bdate_range("2015-01-01", periods=T)
    return pd.DataFrame(R, index=idx, columns=[f"A{i+1}" for i in range(N)])


@pytest.fixture(scope="module")
def fitted_ica():
    return GOGarch(method="ica").fit(_panel())


@pytest.fixture(scope="module")
def fitted_pca():
    return GOGarch(method="pca").fit(_panel())


def test_rotation_is_orthogonal(fitted_ica, fitted_pca):
    N = fitted_ica.Z.shape[0]
    assert np.allclose(fitted_ica.U @ fitted_ica.U.T, np.eye(N), atol=1e-8)
    assert np.allclose(fitted_pca.U, np.eye(N))


def test_components_recover_the_data(fitted_ica):
    """r_t = Z y_t must hold up to the mean that was removed."""
    R = _panel().values
    recon = fitted_ica.components @ fitted_ica.Z.T + R.mean(axis=0)
    assert np.allclose(recon, R, atol=1e-10)


def test_components_are_unconditionally_uncorrelated(fitted_ica):
    C = np.corrcoef(fitted_ica.components, rowvar=False)
    off = C - np.diag(np.diag(C))
    assert np.abs(off).max() < 1e-6
    assert np.allclose(np.diag(C), 1.0, atol=1e-6)


def test_H_is_positive_definite_and_matches_correlation(fitted_ica):
    for t in (0, 400, 799):
        H = fitted_ica.H_at(t)
        assert np.linalg.eigvalsh(H).min() > 0
        assert np.allclose(H, H.T)
    R = fitted_ica.conditional_correlation()
    assert np.allclose(np.diagonal(R, axis1=1, axis2=2), 1.0)
    assert np.abs(R).max() <= 1.0 + 1e-10


def test_H_inv_matches_explicit_inverse(fitted_ica):
    for t in (0, 250, 799):
        assert np.allclose(fitted_ica.H_inv_at(t),
                           np.linalg.inv(fitted_ica.H_at(t)),
                           rtol=1e-8, atol=1e-10)


def test_logdet_matches_slogdet(fitted_ica):
    for t in (0, 250, 799):
        sign, ld = np.linalg.slogdet(fitted_ica.H_at(t))
        assert sign > 0
        assert np.isclose(fitted_ica.logdet_H(t), ld, rtol=1e-10)


def test_conditional_vol_matches_diag_H(fitted_ica):
    cv = fitted_ica.conditional_vol()
    for t in (0, 500):
        assert np.allclose(cv[t] ** 2, np.diag(fitted_ica.H_at(t)), rtol=1e-10)


def test_mvp_weights_sum_to_one_and_match_formula(fitted_ica):
    W = fitted_ica.mvp_weights()
    assert np.allclose(W.sum(axis=1), 1.0)
    ones = np.ones(fitted_ica.Z.shape[0])
    for t in (0, 300, 799):
        Hi = np.linalg.inv(fitted_ica.H_at(t))
        assert np.allclose(W[t], Hi @ ones / (ones @ Hi @ ones), rtol=1e-8)


def test_summary_reports_component_persistence(fitted_ica):
    s = fitted_ica.summary()
    assert s["n_params"] == 3 * s["N"]
    assert len(s["components"]) == s["N"]
    assert np.isclose(sum(c["var_share"] for c in s["components"]), 1.0)
    for c in s["components"]:
        assert 0.0 < c["persistence"] < 1.0


def test_go_garch_is_advertised_only_where_implemented():
    """
    E6: the hero subtitle used to promise GO-GARCH with no implementation
    behind it.  Now the claim must be backed by the module and the tab.
    """
    import pathlib
    import conftest
    root = pathlib.Path(conftest.ROOT)
    app = (root / "app.py").read_text(encoding="utf-8")
    tab = (root / "tabs" / "tab_day3.py").read_text(encoding="utf-8")

    assert "GO-GARCH" in app
    assert "_run_go_garch" in tab
    assert "from go_garch import GOGarch" in tab


def test_rejects_single_asset():
    df = _panel(N=4).iloc[:, :1]
    with pytest.raises(ValueError):
        GOGarch().fit(df)
