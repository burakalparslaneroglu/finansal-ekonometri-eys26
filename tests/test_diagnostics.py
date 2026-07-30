"""
Task B — correlation-stage diagnostics.

The central point: no asymptotic p-value is handed back for statistics computed
on DCC/ADCC-filtered residuals (course notes 1.12.1).
"""

import inspect

import numpy as np
import pytest

from conftest import as_returns, simulate_dcc
import mgarch_diagnostics as md


@pytest.fixture(scope="module")
def filtered():
    """DCC fit on a DCC DGP, with its filtered residuals."""
    from dcc_garch import DCCGarch

    rng = np.random.default_rng(404)
    z = simulate_dcc(800, 4, 0.05, 0.90, rng)
    returns = as_returns(z)
    m = DCCGarch("DCC")
    zz = np.asarray(m.fit_univariate_garch(returns))
    m.fit_dcc(zz)
    eps = md.isqrt_apply(np.asarray(m.R_seq), zz)
    return {"model": m, "z": zz, "eps": eps, "returns": returns,
            "cols": list(returns.columns)}


# --- B.1 / B.2 -------------------------------------------------------------

def test_es_stat_withholds_pvalue_by_default(filtered):
    """The p-value must be opt-in, not the default return shape."""
    iu = np.triu_indices(4, k=1)
    out = md.es_stat(filtered["eps"], 5, iu)
    assert len(out) == 2, "a p-value leaked into the default return"

    out_p = md.es_stat(filtered["eps"], 5, iu, return_pvalue=True)
    assert len(out_p) == 3
    assert 0.0 <= out_p[2] <= 1.0


def test_es_stat_signature_defaults_to_no_pvalue():
    sig = inspect.signature(md.es_stat)
    assert sig.parameters["return_pvalue"].default is False


def test_diagnostics_tab_requests_no_pvalue():
    """
    Guard against a regression in the tab: the filtered-residual path must not
    call es_stat with return_pvalue=True.
    """
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "tabs" / "tab_day3.py"
    text = src.read_text(encoding="utf-8")

    # the only legitimate p-value call is the CCC one in _run_ccc_test
    n_pvalue_calls = text.count("return_pvalue=True")
    assert n_pvalue_calls == 1, (
        f"expected exactly one return_pvalue=True (the CCC test), found {n_pvalue_calls}"
    )
    assert "_run_diagnostics" in text
    diag_body = text.split("def _run_diagnostics")[1].split("@st.cache_data")[0]
    code_only = "\n".join(ln for ln in diag_body.splitlines()
                          if not ln.strip().startswith("#"))
    assert "return_pvalue" not in code_only


def test_qualitative_reading_thresholds():
    assert md.qualitative_reading(5.0) == "kalan dinamik izi yok"
    assert md.qualitative_reading(50.0) == "belirsiz, bootstrap gerekir"
    assert md.qualitative_reading(500.0) == "güçlü kalan dinamik"


# --- B.3 additions ---------------------------------------------------------

def test_pairwise_ljung_box_fdr(filtered):
    res = md.pairwise_ljung_box(filtered["eps"], filtered["cols"], lags=10)
    assert res["n_pairs"] == 6                       # N=4 -> 4*3/2
    assert len(res["rows"]) == 6
    # BH-adjusted p-values are never smaller than the raw ones
    for r in res["rows"]:
        assert r["p_bh"] >= r["p_raw"] - 1e-12
    # rows are sorted by strength
    stats = [r["lb_stat"] for r in res["rows"]]
    assert stats == sorted(stats, reverse=True)


def test_sign_bias_detects_asymmetry():
    """
    On an ADCC DGP (correlations rise after joint negative shocks) the targeted
    sign-bias statistic must be larger than on a symmetric DCC DGP.
    """
    from dcc_garch import DCCGarch

    def _stat(z):
        m = DCCGarch("DCC")
        m.sigmas = np.ones_like(z)
        m.fit_dcc(z)
        eps = md.isqrt_apply(np.asarray(m.R_seq), z)
        return md.correlation_sign_bias(eps, z)["stat"]

    rng = np.random.default_rng(77)
    N, T = 4, 1500
    R_bar = 0.35 * np.ones((N, N))
    np.fill_diagonal(R_bar, 1.0)

    # symmetric DCC path
    z_sym = simulate_dcc(T, N, 0.05, 0.90, rng, R_bar=R_bar)

    # asymmetric path: extra correlation loading after joint down moves
    rng2 = np.random.default_rng(78)
    Q = R_bar.copy()
    z_asy = np.zeros((T, N))
    zp = rng2.standard_normal(N)
    a, b, c = 0.02, 0.90, 0.10
    N_bar = 0.25 * R_bar
    for t in range(T):
        n = zp * (zp < 0)
        Q = ((1 - a - b) * R_bar - c * N_bar + a * np.outer(zp, zp)
             + b * Q + c * np.outer(n, n))
        d = 1.0 / np.sqrt(np.diag(Q))
        R = Q * np.outer(d, d)
        np.fill_diagonal(R, 1.0)
        w, V = np.linalg.eigh(R)
        L = V * np.sqrt(np.maximum(w, 1e-10))
        zp = L @ rng2.standard_normal(N)
        z_asy[t] = zp

    assert _stat(z_asy) > _stat(z_sym)


def test_hosking_li_mcleod_disabled_above_six_assets():
    rng = np.random.default_rng(9)
    eps = rng.standard_normal((500, 8))
    res = md.hosking_li_mcleod(eps, m=5)
    assert res["available"] is False
    assert res["df"] == 5 * 28 ** 2 == 3920          # notes: N=8, m=5 -> 3920
    assert "3920" in res["reason"]

    ok = md.hosking_li_mcleod(rng.standard_normal((500, 4)), m=5)
    assert ok["available"] is True
    assert ok["df"] == 5 * 6 ** 2
    assert 0.0 <= ok["p_value"] <= 1.0


# --- univariate stage ------------------------------------------------------

def test_engle_ng_detects_leverage():
    """A GJR-type path must show more sign bias than a symmetric GARCH path."""
    rng = np.random.default_rng(5)
    T = 3000

    def _path(gamma):
        s2 = np.zeros(T)
        r = np.zeros(T)
        s2[0] = 1.0
        for t in range(1, T):
            neg = 1.0 if r[t - 1] < 0 else 0.0
            s2[t] = 0.05 + (0.05 + gamma * neg) * r[t - 1] ** 2 + 0.90 * s2[t - 1]
            r[t] = np.sqrt(s2[t]) * rng.standard_normal()
        return r / np.sqrt(s2)      # standardised by the TRUE symmetric filter

    sym = md.engle_ng_sign_bias(_path(0.0))
    asy = md.engle_ng_sign_bias(_path(0.12))
    assert asy["stat"] > sym["stat"]
    assert 0.0 <= asy["p_value"] <= 1.0


def test_nyblom_flags_a_break():
    """Nyblom L must be larger when the GARCH intercept jumps mid-sample."""
    rng = np.random.default_rng(3)
    T = 2000
    omega, alpha, beta = 0.05, 0.08, 0.88

    def _sim(break_at=None, factor=6.0):
        s2 = np.zeros(T)
        r = np.zeros(T)
        s2[0] = omega / (1 - alpha - beta)
        for t in range(1, T):
            om = omega * (factor if (break_at is not None and t > break_at) else 1.0)
            s2[t] = om + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
            r[t] = np.sqrt(s2[t]) * rng.standard_normal()
        return r

    stable = md.nyblom_stability(_sim(), omega, alpha, beta)
    broken = md.nyblom_stability(_sim(break_at=T // 2), omega, alpha, beta)
    assert broken["stat"] > stable["stat"]
    assert stable["k"] == 3 and stable["crit_05"] == pytest.approx(1.010)


def test_univariate_diagnostics_shape(filtered):
    m = filtered["model"]
    rows = md.univariate_diagnostics(
        filtered["z"], m.sigmas, filtered["returns"], m.univariate_models,
        filtered["cols"])
    assert len(rows) == 4
    for r in rows:
        assert 0.0 <= r["lb_z_p"] <= 1.0
        assert 0.0 <= r["arch_lm_p"] <= 1.0
        assert np.isfinite(r["nyblom"])


# --- bootstrap -------------------------------------------------------------

def test_simulated_path_has_the_intended_dynamics():
    """The bootstrap DGP must reproduce volatility clustering and correlation."""
    rng = np.random.default_rng(31)
    N = 3
    gp = np.tile(np.array([0.05, 0.08, 0.90]), (N, 1))
    bar_R = 0.5 * np.ones((N, N))
    np.fill_diagonal(bar_R, 1.0)

    r = md.simulate_dcc_path(gp, (0.05, 0.92), bar_R, 3000, rng)
    assert r.shape == (3000, N)
    assert np.isfinite(r).all()
    # volatility clustering: squared returns are autocorrelated
    r2 = r[:, 0] ** 2
    ac1 = np.corrcoef(r2[1:], r2[:-1])[0, 1]
    assert ac1 > 0.05
    # correlation is in the right neighbourhood
    assert 0.3 < np.corrcoef(r, rowvar=False)[0, 1] < 0.7


@pytest.mark.slow
def test_parametric_bootstrap_refits_every_path():
    """
    A small end-to-end bootstrap: the p-value must be a valid probability and
    the null draws must vary (which they only can if theta is re-estimated).
    """
    rng = np.random.default_rng(12)
    z = simulate_dcc(400, 3, 0.05, 0.90, rng)
    returns = as_returns(z)
    iu = np.triu_indices(3, k=1)

    def _statfn(eps, zz, m):
        return md.es_stat(eps, 5, iu)[0]

    res = md.parametric_bootstrap_pvalue(returns, "DCC", _statfn, observed=10.0,
                                         B=8, n_jobs=4)
    assert 0.0 < res["p_value"] <= 1.0
    assert res["B"] >= 1
    assert np.std(res["boot_stats"]) > 0
