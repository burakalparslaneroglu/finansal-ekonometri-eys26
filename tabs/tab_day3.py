##################################################################

"""
3. Gün -- Çok Değişkenli Oynaklık: DCC, cDCC, ADCC, DECO & MVP Portföy
EYS'26 -- Pamukkale Üniversitesi
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLOT_TEMPLATE = "plotly_dark"
# Renk-körü-güvenli qualitative palet (Okabe & Ito 2008); koyu temada görünür.
COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999",
]
# Isı-haritaları için algısal-düzgün, renk-körü-güvenli sıralı harita.
HEATMAP_CMAP = "Viridis"


# ─── SENTINEL ────────────────────────────────────────────────────────────────

class _NoCompute(Exception):
    """Raised inside a tab's try block to skip remaining display code."""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

from ._utils import asset_cols as _asset_cols   # shared with Day 5
from ._utils import df_key as _df_hash          # sha1 content hash, not hash()


def _metric_card(label: str, value: str, tip: str = "") -> str:
    """Return an HTML metric card div matching the app theme."""
    tip_html = f'<div class="label">{tip}</div>' if tip else ""
    return f"""
<div class="metric-card">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  {tip_html}
</div>"""


# ---------------------------------------------------------------------------
# Cached estimation functions
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _run_dcc_model(asset_tuple: tuple, model_type: str, df_hash: str) -> dict:
    """
    Fit a DCC/cDCC/ADCC model for the given assets.

    Parameters
    ----------
    asset_tuple : tuple of str
        Column names to include in the model.
    model_type : str
        One of "DCC", "cDCC", "ADCC".
    df_hash : str
        Short string hash of the DataFrame (cache key discriminator).

    Returns
    -------
    dict with keys:
        params, stats, corr_series, sigmas, index, cols, R_seq, H_seq
    """
    from dcc_garch import DCCGarch

    df = st.session_state.returns_df
    cols = list(asset_tuple)
    returns = df[cols]

    dcc = DCCGarch(model_type=model_type)
    std_resid = dcc.fit_univariate_garch(returns)
    params = dcc.fit_dcc(std_resid)
    stats = dcc.get_summary_stats()
    weights = dcc.compute_mvp_weights()

    corr_series = {}
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            label = f"{cols[i]} vs {cols[j]}"
            corr_series[label] = dcc.R_seq[:, i, j]

    return {
        "params": params,
        "stats": stats,
        "weights": weights,
        "corr_series": corr_series,
        "sigmas": dcc.sigmas,
        "index": returns.index,
        "cols": cols,
        "R_seq": dcc.R_seq,
        "H_seq": dcc.H_seq,
    }


@st.cache_data(show_spinner=False)
def _run_deco_model(asset_tuple: tuple, df_hash: str) -> dict:
    """
    Fit a DECO model for the given assets.

    Parameters
    ----------
    asset_tuple : tuple of str
        Column names to include in the model.
    df_hash : str
        Short string hash of the DataFrame (cache key discriminator).

    Returns
    -------
    dict with keys: stats, rho_series, index, sample_mean_corr,
                  dcc_pair_min, dcc_pair_max, n_pairs
    """
    from dcc_garch import DCCGarch

    df = st.session_state.returns_df
    cols = list(asset_tuple)
    returns = df[cols]

    deco = DCCGarch(model_type="DECO")
    std_resid = deco.fit_univariate_garch(returns)
    # deco_base_dcc: run a reference DCC fit purely so the two parameter sets
    # can be shown side by side.  It is OFF in the default estimation path, so
    # a plain DECO fit stays cheaper than a DCC fit.
    deco.fit_dcc(std_resid, deco_base_dcc=True)
    stats = deco.get_summary_stats()
    rho_series = deco.get_equicorrelation_series()

    corr_mat = returns.corr().values
    n = len(cols)
    idx_upper = np.triu_indices(n, k=1)
    sample_mean_corr = float(np.mean(corr_mat[idx_upper]))

    # DCC modeli kendi std_resid'ini kullanır (DECO'nunkiyle karışmamalı)
    dcc = DCCGarch(model_type="DCC")
    dcc_std_resid = dcc.fit_univariate_garch(returns)
    dcc.fit_dcc(dcc_std_resid)
    T_dcc = dcc.R_seq.shape[0]
    # Tüm ikili korelasyon serileri: şekil (T, M), M = N*(N-1)/2
    dcc_pair_corr = np.asarray(dcc.R_seq)[:, idx_upper[0], idx_upper[1]]

    return {
        "stats": stats,
        "dcc_stats": dcc.get_summary_stats(),
        "rho_series": rho_series,
        "index": returns.index,
        "cols": cols,
        "sample_mean_corr": sample_mean_corr,
        "dcc_pair_min": dcc_pair_corr.min(axis=1),   # (T,) en düşük ikili kor.
        "dcc_pair_max": dcc_pair_corr.max(axis=1),   # (T,) en yüksek ikili kor.
        "n_pairs": dcc_pair_corr.shape[1],            # N*(N-1)/2
        "R_deco": np.asarray(deco.R_seq),
        "R_dcc": np.asarray(dcc.R_seq),
    }


@st.cache_data(show_spinner=False)
def _run_comparison(asset_tuple: tuple, df_hash: str) -> dict:
    """
    Fit DCC, cDCC, and DECO on the same assets for comparison.

    Parameters
    ----------
    asset_tuple : tuple of str
        Column names (typically 3 for speed).
    df_hash : str
        Short string hash of the DataFrame (cache key discriminator).

    Returns
    -------
    dict mapping model_type -> summary stats dict
    """
    from dcc_garch import DCCGarch

    df = st.session_state.returns_df
    cols = list(asset_tuple)
    returns = df[cols]
    results = {}
    N = len(cols)
    idx_u = np.triu_indices(N, k=1)

    for mtype in ("DCC", "cDCC", "DECO"):
        m = DCCGarch(model_type=mtype)
        std_resid = m.fit_univariate_garch(returns)
        m.fit_dcc(std_resid)
        stats = m.get_summary_stats()
        # Çiftler-arası std: her çiftin zaman-ortalaması korelasyonunun standart sapması.
        # DCC: her çift kendi dinamiğine sahip → std > 0.
        # DECO: tüm çiftler aynı rho_t'ye kısıtlanmış → std = 0 (tanımlı).
        pair_means = np.array([m.R_seq[:, i, j].mean() for i, j in zip(*idx_u)])
        stats["corr_cross_std"] = float(pair_means.std())
        results[mtype] = {
            "stats": stats,
            "R_seq": m.R_seq.copy(),   # (T, N, N)
            "index": returns.index,
        }

    return results


@st.cache_data(show_spinner=False)
def _run_mvp(asset_tuple: tuple, model_type: str, df_hash: str) -> dict:
    """
    Fit a DCC-family model and compute MVP weight paths.

    Returns unconstrained (analytic) and long-only (Jagannathan-Ma, SLSQP) dynamic
    weights, plus a static Ledoit-Wolf shrinkage benchmark, and inputs needed for
    cost-aware net-of-fee performance metrics.

    Returns
    -------
    dict with keys: weights_unc, weights_lo, w_lw, lw_delta, H_seq, sigmas,
    index, cols, stats, returns
    """
    import numpy as np
    from scipy.optimize import minimize
    from sklearn.covariance import LedoitWolf
    from dcc_garch import DCCGarch

    df = st.session_state.returns_df
    cols = list(asset_tuple)
    returns = df[cols]
    N = len(cols)
    one = np.ones(N)

    dcc = DCCGarch(model_type=model_type)
    std_resid = dcc.fit_univariate_garch(returns)
    dcc.fit_dcc(std_resid)
    H = np.asarray(dcc.H_seq)
    T = H.shape[0]
    stats = dcc.get_summary_stats()

    w_unc = np.asarray(dcc.compute_mvp_weights())

    def _mvp_long(Ht, w0):
        r = minimize(
            lambda w: w @ Ht @ w, w0, jac=lambda w: 2.0 * Ht @ w, method="SLSQP",
            bounds=[(0.0, 1.0)] * N,
            constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
            options={"maxiter": 100, "ftol": 1e-12},
        )
        w = np.clip(r.x, 0.0, None)
        s = w.sum()
        return w / s if s > 0 else one / N

    w_lo = np.zeros((T, N))
    w0 = one / N
    for t in range(T):                       # warm-start'lı long-only QP yolu
        w0 = _mvp_long(H[t], w0)
        w_lo[t] = w0

    lw = LedoitWolf().fit(returns.values)     # statik büzülme benchmark (long-only)
    w_lw_vec = _mvp_long(lw.covariance_, one / N)

    return {
        "weights_unc": w_unc,
        "weights_lo": w_lo,
        "w_lw": w_lw_vec,
        "lw_delta": float(lw.shrinkage_),
        "H_seq": H,
        "sigmas": np.asarray(dcc.sigmas),
        "index": returns.index,
        "cols": cols,
        "stats": stats,
        "returns": returns.values,
    }


# ---------------------------------------------------------------------------
# Diagnostics & tests (batch-4 additions)
# ---------------------------------------------------------------------------

# The Engle-Sheppard machinery now lives in mgarch_diagnostics, where the
# p-value is opt-in.  See §1.12.1: the chi^2_{s+1} reference is valid for CCC
# residuals only, never for DCC/ADCC-filtered ones.
from mgarch_diagnostics import (            # noqa: E402
    ccc_standardise as _ccc_standardise,
    es_stat as _es_stat,
    isqrt_apply as _isqrt_apply,
    qualitative_reading as _qual,
)


@st.cache_data(show_spinner=False)
def _run_ccc_test(asset_tuple: tuple, df_hash: str) -> dict:
    """CCC (Bollerslev 1990) baz + Engle-Sheppard sabit-korelasyon testi (s=1,5,10)."""
    import numpy as np
    from dcc_garch import DCCGarch

    df = st.session_state.returns_df
    cols = list(asset_tuple)
    returns = df[cols]
    N = len(cols)
    iu = np.triu_indices(N, k=1)

    dcc = DCCGarch(model_type="DCC")
    z = np.asarray(dcc.fit_univariate_garch(returns))          # CCC = tek değişkenli GARCH artıkları
    Rbar = np.corrcoef(z, rowvar=False)
    eps = _ccc_standardise(z)
    # CCC residuals: the constant-correlation null is exactly the one the
    # chi^2_{s+1} reference is derived for, so the p-value IS legitimate here.
    rows = [(s,) + _es_stat(eps, s, iu, return_pvalue=True) for s in (1, 5, 10)]
    return {
        "rows": rows,
        "mean_corr": float(Rbar[iu].mean()),
        "N": N, "T": int(len(z)), "P": int(len(iu[0])),
        "n_params": 3 * N + N * (N - 1) // 2,
    }


@st.cache_data(show_spinner=False)
def _run_diagnostics(asset_tuple: tuple, df_hash: str) -> dict:
    """
    DCC yeterlilik tanısı: ES istatistiği CCC/DCC/ADCC-filtrelenmiş artıklara
    (s=5), **p-değeri olmadan** (§1.12.1).  Ek olarak çift-bazlı Ljung-Box +
    BH-FDR, işaret-yanlılığı ve Hosking-Li-McLeod portmanteau.
    """
    import numpy as np
    from dcc_garch import DCCGarch
    from mgarch_diagnostics import (
        correlation_sign_bias, hosking_li_mcleod, pairwise_ljung_box,
    )

    df = st.session_state.returns_df
    cols = list(asset_tuple)
    returns = df[cols]
    N = len(cols)
    iu = np.triu_indices(N, k=1)

    def _diag(mt):
        m = DCCGarch(model_type=mt)
        zz = np.asarray(m.fit_univariate_garch(returns))
        m.fit_dcc(zz)
        eps = _isqrt_apply(np.asarray(m.R_seq), zz)
        # return_pvalue omitted on purpose: these residuals are filtered.
        return _es_stat(eps, 5, iu), zz, eps

    out = {"es": {}}
    out["es"]["DCC"], zdcc, eps_dcc = _diag("DCC")
    out["es"]["ADCC"], _, _ = _diag("ADCC")
    epsC = _ccc_standardise(zdcc)                              # CCC: DCC ile aynı z
    out["es"]["CCC"] = _es_stat(epsC, 5, iu)

    out["pairwise_lb"] = pairwise_ljung_box(eps_dcc, cols, lags=10, alpha=0.05)
    out["sign_bias"] = correlation_sign_bias(eps_dcc, zdcc, iu)
    out["hlm"] = hosking_li_mcleod(eps_dcc, m=5)
    out["cols"] = cols
    return out


@st.cache_data(show_spinner=False)
def _run_univariate_diagnostics(asset_tuple: tuple, df_hash: str) -> list:
    """Tek değişkenli aşama: LB(z), LB(z²), ARCH-LM, Engle-Ng, Nyblom."""
    import numpy as np
    from dcc_garch import DCCGarch
    from mgarch_diagnostics import univariate_diagnostics

    df = st.session_state.returns_df
    cols = list(asset_tuple)
    returns = df[cols]

    m = DCCGarch(model_type="DCC")
    z = np.asarray(m.fit_univariate_garch(returns))
    return univariate_diagnostics(z, m.sigmas, returns, m.univariate_models, cols)


@st.cache_data(show_spinner=False)
def _run_cl(asset_tuple: tuple, df_hash: str) -> dict:
    """İkili bileşik olabilirlik (Pakel vd. 2021) ile (a,b) vs tam DCC."""
    import numpy as np
    from scipy.optimize import minimize
    from numba import njit
    from dcc_garch import DCCGarch

    df = st.session_state.returns_df
    cols = list(asset_tuple)
    returns = df[cols]
    N = len(cols)
    iu = np.triu_indices(N, k=1)

    dcc = DCCGarch(model_type="DCC")
    z = np.asarray(dcc.fit_univariate_garch(returns))
    dcc.fit_dcc(z)
    aF, bF = np.asarray(dcc.dcc_params)[:2]
    Rbar = np.corrcoef(z, rowvar=False)
    pairs = np.array(list(zip(*iu)), dtype=np.int64)
    rho = np.array([Rbar[i, j] for i, j in pairs])

    @njit(cache=True, fastmath=True)
    def _pair_cl_nll(a, b, z, pairs, rho):
        T, N = z.shape
        P = pairs.shape[0]
        nll = 0.0
        for k in range(P):
            i = pairs[k, 0]; j = pairs[k, 1]; r = rho[k]
            q11 = 1.0; q22 = 1.0; q12 = r
            for t in range(1, T):
                zi = z[t - 1, i]; zj = z[t - 1, j]
                q11 = (1 - a - b) + a * zi * zi + b * q11
                q22 = (1 - a - b) + a * zj * zj + b * q22
                q12 = (1 - a - b) * r + a * zi * zj + b * q12
                rt = q12 / np.sqrt(q11 * q22)
                det = 1.0 - rt * rt
                u = z[t, i]; v = z[t, j]
                nll += 0.5 * (np.log(det) + (u * u - 2 * rt * u * v + v * v) / det - (u * u + v * v))
        return nll

    def obj(p):
        a, b = p
        return _pair_cl_nll(a, b, z, pairs, rho) if (a > 0 and b > 0 and a + b < 0.9999) else 1e12

    r = minimize(obj, np.array([0.04, 0.92]), method="L-BFGS-B", bounds=[(1e-6, 0.5), (1e-6, 0.999)])
    aCL, bCL = r.x
    return {"full": (float(aF), float(bF)), "cl": (float(aCL), float(bCL)), "P": int(len(iu[0]))}


@st.cache_data(show_spinner=False)
def _run_factor_selection(asset_tuple: tuple, k_max: int, df_hash: str) -> dict:
    """Bai-Ng / Onatski / Marchenko-Pastur factor counts on the same panel."""
    from factor_selection import select_k

    df = st.session_state.returns_df
    R = df[list(asset_tuple)].values
    return select_k(R, method="Bai-Ng ICp1", k_max=k_max)


@st.cache_data(show_spinner=False)
def _run_factor_dcc(asset_tuple: tuple, K: int, loading_mode: str,
                    idio_garch: bool, df_hash: str) -> dict:
    """Fit the Factor-DCC model and return everything the tab plots."""
    import numpy as np
    from factor_dcc import FactorDCC

    df = st.session_state.returns_df
    cols = list(asset_tuple)
    returns = df[cols]

    factors = None
    if loading_mode == "observed":
        factors = st.session_state.get("factors_df")
        if factors is None or factors.empty:
            raise ValueError(
                "Gözlenen faktör seçilebilmesi için veri yüklerken 'Faktör "
                "sütunları' belirtilmelidir (Görev D)."
            )
        factors = factors.reindex(returns.index).dropna()
        returns = returns.loc[factors.index]

    model = FactorDCC(K=K, loading_mode=loading_mode, idio_garch=idio_garch)
    model.fit(returns, factor_returns=factors)

    T = model.Lambda_seq.shape[0]
    t_mid = T // 2
    return {
        "summary": model.summary(),
        "B": model.B,
        "factors": model.factors,
        "cond_vol": model.conditional_vol(),
        "weights": model.mvp_weights(),
        "Lambda_seq": model.Lambda_seq,
        "factor_R": model.factor_correlation(),
        "index": returns.index,
        "cols": list(returns.columns),
        "H_sample": model.H_at(t_mid),
        "H_sample_t": t_mid,
        "logdet_sample": model.logdet_H(t_mid),
        "omega_mean": model.Omega_diag.mean(axis=0),
    }


@st.cache_data(show_spinner=False)
def _run_go_garch(asset_tuple: tuple, method: str, df_hash: str) -> dict:
    """Fit GO-GARCH and return the component paths plus implied correlations."""
    import numpy as np
    from go_garch import GOGarch

    df = st.session_state.returns_df
    cols = list(asset_tuple)
    returns = df[cols]

    model = GOGarch(method=method)
    model.fit(returns)

    R = model.conditional_correlation()
    iu = np.triu_indices(len(cols), k=1)
    return {
        "summary": model.summary(),
        "Z": model.Z,
        "components": model.components,
        "h_seq": model.h_seq,
        "cond_vol": model.conditional_vol(),
        "mean_corr": R[:, iu[0], iu[1]].mean(axis=1),
        "R_sample": R[len(R) // 2],
        "R_sample_t": len(R) // 2,
        "weights": model.mvp_weights(),
        "index": returns.index,
        "cols": cols,
    }


def _stress_window(index, returns_values):
    """Şekillerde gölgelenecek stres penceresi (start, end) tarihleri.

    Önce st.session_state['crisis_window'] (DGP meta-verisi) denenir; yoksa
    kesitsel ortalama |getiri| yumuşatılmış tepesi etrafında otomatik tespit.
    """
    import numpy as np
    cw = st.session_state.get("crisis_window", None)
    if cw is not None:
        return cw
    m = np.abs(np.asarray(returns_values)).mean(axis=1)
    k = 5
    ms = np.convolve(m, np.ones(k) / k, mode="same")
    pk = int(np.argmax(ms))
    half = 18
    a = max(0, pk - half)
    b = min(len(index) - 1, pk + half)
    return (index[a], index[b])


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render():
    """Entry point called by app.py for Day 3."""

    try:
        import numba  # noqa: F401
    except ImportError:
        st.info("Numba bulunamadı -- saf NumPy kullanılıyor (daha yavaş)")

    df = st.session_state.returns_df
    asset_cols = _asset_cols(df)
    dh = _df_hash()
    stress = _stress_window(df.index, df[asset_cols].values)

    (tab_theory, tab_model, tab_deco, tab_factor, tab_go, tab_compare,
     tab_portfolio, tab_diag) = st.tabs([
        "\U0001f4d6 Teori",
        "⚙️ Model Tahmini",
        "\U0001f504 DECO Modeli",
        "\U0001f9ee Faktör-DCC",
        "\U0001f501 GO-GARCH",
        "\U0001f4ca Model Karşılaştırması",
        "\U0001f4bc MVP Portföy",
        "\U0001f9ea Tanılar & Testler",
    ])

    # =========================================================================
    # TAB 1 -- THEORY
    # =========================================================================
    with tab_theory:
        st.markdown("### DCC-GARCH Ailesi: Teorik Özet")

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("DCC (Engle 2002)", expanded=False):
                st.markdown(r"""
**Temel cerceve:**
$$H_t = D_t R_t D_t$$

$D_t = \text{diag}(\sigma_{1t},\ldots,\sigma_{Nt})$

**Q güncelleme:**
$$Q_t = (1-a-b)\bar{Q} + a\,z_{t-1}z_{t-1}^\top + b\,Q_{t-1}$$

**R normalleşme:**
$$R_t = Q_t^{*-1} Q_t Q_t^{*-1}$$
$$Q_t^* = \text{diag}(\sqrt{q_{11,t}},\ldots,\sqrt{q_{NN,t}})$$

**Kısıtlar:** $a>0,\; b>0,\; a+b<1$

**İki aşamalı QMLE:**
1. Her varlık için GARCH(1,1)
2. $a$ ve $b$ DCC log-olabilirligi ile
""")

        with col2:
            with st.expander("cDCC (Aielli 2013)", expanded=False):
                st.markdown(r"""
**Düzeltilmiş Q güncelleme:**
$$Q_t = (1-a-b)\bar{Q} + a\,\tilde{z}_{t-1}\tilde{z}_{t-1}^\top + b\,Q_{t-1}$$

$$\tilde{z}_{t-1} = Q_{t-1}^{*1/2}\,z_{t-1}$$

**Neden gerekli?**

Standart DCC'de $\bar{Q}$ tahmini asimptotik sapmaya
yol açar. cDCC, $Q_{t-1}^{*1/2}$ ile standardize ederek
$\bar{Q}$ tahminini tutarlı kilar.

**Sonuc:** Büyük $N$'de ve yüksek kalıcılikta DCC ile
cDCC belirgin biçimde ayrışır.
""")

        col3, col4 = st.columns(2)

        with col3:
            with st.expander("ADCC (Cappiello et al. 2006)", expanded=False):
                st.markdown(r"""
**Asimetrik Q güncelleme:**
$$Q_t = (1-a-b)\bar{Q} - c\bar{N} + a\,z_{t-1}z_{t-1}^\top + b\,Q_{t-1} + c\,n_{t-1}n_{t-1}^\top$$

**Negatif şok vektörü:**
$$n_t = z_t \odot \mathbf{1}[z_t < 0]$$

**Koşulsuz ortalama:**
$$\bar{N} = \frac{1}{T}\sum_{t=1}^{T} n_t n_t^\top$$

**Kısıtlar:** $a>0,\; b>0,\; c\geq 0,\; a+b+c < 1$

Piyasalar dustugunde korelasyonlar artar;
$c > 0$ bunu modelleştirir.
""")

        with col4:
            with st.expander("DECO (Engle-Kelly 2012)", expanded=False):
                st.markdown(r"""
**Ekikorelasyon (equicorrelation):**
$$\bar\rho_t = \frac{2}{N(N-1)}\sum_{i<j}\frac{q_{ij,t}}{\sqrt{q_{ii,t}q_{jj,t}}}$$

**DECO korelasyon matrisi:**
$$R_t^{\text{DECO}} = (1-\bar\rho_t)I_N + \bar\rho_t\,\mathbf{1}_N\mathbf{1}_N^\top$$

**Kapalı biçim — matris tersi/determinantı yok:**
$$\log|R_t| = (N-1)\log(1-\bar\rho_t) + \log\bigl(1+(N-1)\bar\rho_t\bigr)$$
$$z_t'R_t^{-1}z_t = \frac{S_t - c_t G_t^2}{1-\bar\rho_t},\quad
c_t=\frac{\bar\rho_t}{1+(N-1)\bar\rho_t}$$

$S_t=\sum_i z_{it}^2$, $G_t=\sum_i z_{it}$.

**Algoritma:** $Q_t$ özyinelemesi DCC ile **aynıdır**; değişen tek şey
olabilirliğin değerlendirildiği matristir. DECO kendi olabilirliğini
maksimize eder — DCC'nin $R_t$'sinin ortalaması *değildir*.

**Kısıt:** $\bar\rho_t > -\tfrac{1}{N-1}$ (pozitif tanımlılık).
""")

        st.divider()

        col5, col6 = st.columns(2)

        with col5:
            with st.expander("GO-GARCH (van der Weide 2002)", expanded=False):
                st.markdown(r"""
**Model:**
$$r_t = Z\,y_t,\qquad
\mathbb{E}[y_ty_t'\mid\mathcal F_{t-1}] = V_t = \mathrm{diag}(h_{1t},\dots,h_{Nt})$$
$$H_t = Z V_t Z'$$

**Ayrıştırma:** $Z = P\Lambda^{1/2}U$ — $(P,\Lambda)$ koşulsuz kovaryansın
özayrışması, $U$ ortogonal döndürme.

| $U$ nasıl bulunur | Ne varsayar |
|---|---|
| $U=I$ (PCA / O-GARCH) | bileşenler yalnızca **koşulsuz** ilişkisiz |
| FastICA | bileşenler **bağımsız** — koşullu köşegenliğe daha yakın |

**Korelasyon aşaması yoktur:** $N$ tek değişkenli GARCH ve sabit bir $Z$.

**Kimliklendirme:** bileşen sırası, işareti ve ölçeği belirsizdir;
$Z$'nin sütunları ekonomik olarak yorumlanamaz.
""")

        with col6:
            with st.expander("Faktör-DCC (§1.10.3, Not 1.8)", expanded=False):
                st.markdown(r"""
**Model:**
$$H_t = B\Lambda_t B' + \Omega_t$$

$B$: $N\times K$ yükler, $\Lambda_t$: $K$-boyutlu DCC, $\Omega_t$: köşegen
idiyosenkratik kovaryans.

**$\Omega$ neden zorunlu?** $B\Lambda_tB'$ rank-$K$ tekildir; $H_t^{-1}$ ve
dolayısıyla MVP yalnızca $\Omega$ sayesinde tanımlıdır.

**Woodbury (zorunlu):**
$$M_t = \Lambda_t^{-1} + B'\Omega_t^{-1}B \quad (K\times K)$$
$$H_t^{-1} = \Omega_t^{-1} - \Omega_t^{-1}BM_t^{-1}B'\Omega_t^{-1}$$
$$\log\det H_t = \log\det\Omega_t + \log\det\Lambda_t + \log\det M_t$$

Maliyet $O(NK^2+K^3)$, $O(N^3)$ değil.

**GO-GARCH ile karşıtlık:** GO-GARCH **döndürmeyi** kısıtlar ($Z$ kare,
tersinir, $\Omega$ yok); Faktör-DCC **rankı** kısıtlar ($K<N$, $\Omega$
zorunlu).
""")

        st.divider()

        with st.expander("Korelasyon Yarı-Ömrü (Half-Life)", expanded=False):
            st.markdown(r"""
$$\tau_{1/2} = \frac{\ln(0.5)}{\ln(a+b)}$$

| $a+b$ | Yarı-Ömür |
|-------|-----------|
| 0.990 | 69 gün |
| 0.975 | 27 gün |
| 0.950 | 14 gün |
| 0.900 | 7 gün  |
| 0.800 | 3 gün  |

$a+b \to 1$ iken korelasyon şokları kalıcı hale gelir.
Finansal serilerde tipik değer $a+b \in [0.97, 0.99]$.
""")

        with st.expander("İki Aşamalı Tahmin (Two-Step QMLE)", expanded=False):
            st.markdown(r"""
| Adım | İşlem | Yöntem | Hedef |
|------|--------|--------|-------|
| **1** | Her varlık için GARCH(1,1) | QMLE | $\hat{\sigma}_{it}$, $z_{it}$ |
| **2** | DCC parametrelerini tahmin et | Bileşik QMLE | $\hat{a}$, $\hat{b}$ (ve $\hat{c}$) |

**DCC log-olabilirlik (2. adım):**
$$\ell(a,b) = -\frac{1}{2}\sum_{t=1}^{T}\bigl[\log|R_t| + z_t^\top R_t^{-1} z_t - z_t^\top z_t\bigr]$$

**Neden iki adım?** Tam ortak tahmin $O(N^2)$ parametre gerektirir.
GARCH + DCC bunu $2N + 2$'ye indirir.
""")

        with st.expander("Model Karşılaştırma Tablosu", expanded=False):
            st.markdown(r"""
| Model | Param. | Gözlem başına maliyet | Avantajlar | Dezavantajlar | Büyük N |
|-------|--------|---|------------|---------------|---------|
| **DCC** | $a,b$ (2) | $O(N^3)$ | Basit, yorumlanabilir | $\bar{Q}$ tutarsız | Orta ($N<50$) |
| **cDCC** | $a,b$ (2) | $O(N^3)$ | Tutarlı $\bar{Q}$ hedefi | Daha yavaş | Orta ($N<50$) |
| **ADCC** | $a,b,c$ (3) | $O(N^3)$ | Asimetriyi yakalar | Kısıt karmaşık | Orta ($N<30$) |
| **DECO** | $a,b$ (2) | $O(N)$ | Kapalı biçim, hızlı | Eşkorelasyon kısıtı | Büyük ($N>50$) |
| **GO-GARCH** | $3N$ | $O(N^2)$ | Korelasyon aşaması yok | $Z$ sabit; yorumlanamaz | Orta |
| **Faktör-DCC** | $\approx 3K{+}2$ | $O(NK^2{+}K^3)$ | Rank kısıtı + Woodbury | $K$ seçimine duyarlı | Çok büyük |

**Dikkat — sık yapılan hata:** DECO'nun DCC'ye üstünlüğü **parametre
sayısında değildir**: skaler DCC de DECO da $2$ parametre kullanır. Kazanç,
gözlem başına $O(N)$ ve $O(N^3)$ hesap maliyeti arasındaki farktadır
(§1.10.2) — DECO olabilirliği hiçbir $N\times N$ ters/determinant
gerektirmez.
""")

        # ── Notebook indirme ─────────────────────────────────────────────────
        st.divider()
        try:
            from pathlib import Path as _Path
            _nb_path = _Path(__file__).parent.parent / "notebooks" / "gun3_dcc.ipynb"
            st.download_button(
                label="📥 Jupyter Not Defteri İndir (gun3_dcc.ipynb)",
                data=_nb_path.read_bytes(),
                file_name="gun3_dcc.ipynb",
                mime="application/json",
                use_container_width=True,
            )
        except FileNotFoundError:
            st.caption("📓 `notebooks/gun3_dcc.ipynb` bulunamadı.")
        except Exception as _e:
            st.caption(f"Notebook hatası: {_e}")

    # =========================================================================
    # TAB 2 -- MODEL ESTIMATION
    # =========================================================================
    with tab_model:
        try:
            st.markdown("### DCC / cDCC / ADCC Model Tahmini")

            col_a, col_b = st.columns(2)
            with col_a:
                selected_assets = st.multiselect(
                    "Varlıklar",
                    asset_cols,
                    default=[],
                    key="t2_assets",
                )
            with col_b:
                model_type = st.selectbox(
                    "Model Türü",
                    ["DCC", "cDCC", "ADCC"],
                    key="t2_model",
                )

            if len(selected_assets) < 2:
                st.warning("En az 2 varlık seçiniz.")
                raise _NoCompute

            pairs = [
                f"{selected_assets[i]} vs {selected_assets[j]}"
                for i in range(len(selected_assets))
                for j in range(i + 1, len(selected_assets))
            ]
            selected_pair = st.selectbox("Korelasyon Çifti Grafigi", pairs, key="t2_pair")

            # ── Build cache key ──
            _key = f"d3_model_{'_'.join(sorted(selected_assets))}_{model_type}_{dh}"

            # ── Hesapla button ──
            if st.button("🔄 Hesapla", key="d3_model_run", type="primary"):
                with st.spinner(f"{model_type} modeli tahmin ediliyor..."):
                    try:
                        result = _run_dcc_model(tuple(selected_assets), model_type, dh)
                        st.session_state[_key] = result
                    except Exception as _exc:
                        st.error(f"Model tahmini başarısız oldu: `{_exc}`")
                        raise _NoCompute

            # ── Gate: require result ──
            if _key not in st.session_state:
                st.info("⬆️ Parametreleri seçip **🔄 Hesapla**'ya tıklayın.")
                raise _NoCompute

            result = st.session_state[_key]

            stats = result["stats"]
            params = result["params"]
            corr_series = result["corr_series"]
            sigmas = result["sigmas"]
            idx = result["index"]
            cols = result["cols"]
            R_seq = np.asarray(result["R_seq"])
            iu_h = np.triu_indices(len(cols), k=1)

            is_adcc = (model_type == "ADCC")
            alpha_v = stats["alpha"]
            beta_v = stats["beta"]
            c_v = stats.get("c_asym", 0.0)
            persist = stats["persistence"]
            hl = stats["half_life_days"]
            mean_corr = stats["mean_corr"]

            if is_adcc:
                metric_defs = [
                    ("alpha (DCC alpha)", f"{alpha_v:.4f}", "Kısa dönem korelasyon tepkisi"),
                    ("beta (DCC beta)", f"{beta_v:.4f}", "Korelasyon kalıcılığı"),
                    ("c (Asimetri)", f"{c_v:.4f}", "Negatif şok etkisi"),
                    ("alpha + beta", f"{persist:.4f}", "1'e yakınsa yüksek kalıcılik"),
                    ("Yarı-Ömür (gün)", f"{hl:.1f}", "Korelasyon şokları"),
                    ("Ort. Korelasyon", f"{mean_corr:.4f}", "Zaman ortalaması"),
                ]
            else:
                metric_defs = [
                    ("alpha (DCC alpha)", f"{alpha_v:.4f}", "Kısa dönem korelasyon tepkisi"),
                    ("beta (DCC beta)", f"{beta_v:.4f}", "Korelasyon kalıcılığı"),
                    ("alpha + beta", f"{persist:.4f}", "1'e yakınsa yüksek kalıcılik"),
                    ("Yarı-Ömür (gün)", f"{hl:.1f}", "Korelasyon şokları"),
                    ("Ort. Korelasyon", f"{mean_corr:.4f}", "Zaman ortalaması"),
                ]

            metric_cols = st.columns(len(metric_defs))
            for mc, (lbl, val, tip) in zip(metric_cols, metric_defs):
                with mc:
                    st.markdown(_metric_card(lbl, val, tip), unsafe_allow_html=True)

            st.markdown("")

            pair_data = corr_series.get(selected_pair, np.array([]))
            mean_rho = R_seq[:, iu_h[0], iu_h[1]].mean(axis=1)
            fig_corr = go.Figure()
            fig_corr.add_vrect(
                x0=stress[0], x1=stress[1], fillcolor="gray", opacity=0.15, line_width=0,
                annotation_text="stres", annotation_position="top left",
            )
            fig_corr.add_trace(go.Scatter(
                x=idx, y=pair_data, mode="lines", name=selected_pair,
                line=dict(color=COLORS[1], width=1.6),
            ))
            fig_corr.add_trace(go.Scatter(
                x=idx, y=mean_rho, mode="lines", name="ortalama (tüm çiftler)",
                line=dict(color=COLORS[4], width=1.4, dash="dot"),
            ))
            fig_corr.update_layout(
                template=PLOT_TEMPLATE,
                title=f"Dinamik Koşullu Korelasyon -- {selected_pair} ({model_type})",
                xaxis_title="Tarih",
                yaxis_title="Korelasyon rho",
                height=380,
                margin=dict(l=20, r=20, t=50, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Koşullu korelasyon matrisi (anlık) -- ortak renk ölçeği (Viridis)
            st.markdown("#### Koşullu Korelasyon Matrisi $R_t$ (anlık)")
            zmin_hm = float(np.nanmin(R_seq[:, iu_h[0], iu_h[1]]))
            snap = st.select_slider(
                "Tarih (anlık matris)", options=list(range(len(idx))),
                value=int(np.argmax(mean_rho)),
                format_func=lambda i: str(pd.Timestamp(idx[i]).date()), key="t2_snap",
            )
            Rsnap = R_seq[snap]
            fig_hm = go.Figure(go.Heatmap(
                z=Rsnap, x=cols, y=cols, colorscale=HEATMAP_CMAP, zmin=zmin_hm, zmax=1.0,
                text=np.round(Rsnap, 2), texttemplate="%{text}", textfont=dict(size=10),
                colorbar=dict(title="rho"),
            ))
            fig_hm.update_layout(
                template=PLOT_TEMPLATE,
                title=f"R_t @ {pd.Timestamp(idx[snap]).date()}  "
                      f"(ort. çift korelasyon = {mean_rho[snap]:.3f})",
                height=430, yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

            n_sel = len(cols)
            fig_vol = make_subplots(rows=1, cols=n_sel, subplot_titles=cols, shared_yaxes=True)
            for i, col in enumerate(cols):
                fig_vol.add_trace(
                    go.Scatter(
                        x=idx,
                        y=sigmas[:, i] * 100,
                        mode="lines",
                        name=col,
                        line=dict(color=COLORS[i % len(COLORS)], width=1.0),
                    ),
                    row=1, col=i + 1,
                )
            fig_vol.update_layout(
                template=PLOT_TEMPLATE,
                title="Koşullu Standart Sapma (%) -- Marjinal GARCH(1,1)",
                height=280,
                showlegend=False,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_vol, use_container_width=True)

            with st.expander("Tahmin Detayları", expanded=False):
                st.json({
                    "model_type": model_type,
                    "alpha": round(alpha_v, 6),
                    "beta": round(beta_v, 6),
                    "c_asym": round(c_v, 6),
                    "persistence": round(persist, 6),
                    "half_life_days": round(hl, 2),
                    "mean_corr": round(mean_corr, 6),
                    "n_assets": stats["n_assets"],
                    "n_obs": stats["n_obs"],
                })
        except _NoCompute:
            pass

    # =========================================================================
    # TAB 3 -- DECO MODEL
    # =========================================================================
    with tab_deco:
        try:
            st.markdown("### DECO -- Dinamik Ekikorelasyon Modeli")

            st.info(
                "**DECO'nun kazancı parametre sayısında değildir.** Skaler DCC "
                "de DECO da iki parametre (a, b) kullanır. Fark hesap "
                "maliyetindedir: DECO olabilirliği ρ̄_t üzerinden kapalı "
                "biçimde değerlendirilir ve hiçbir N×N ters/determinant "
                "gerektirmez → gözlem başına O(N), DCC'de O(N³) (§1.10.2). "
                "DECO burada **kendi olabilirliğini** maksimize eder; DCC'nin "
                "R_t'sinin sonradan alınmış ortalaması değildir."
            )

            col_da, col_db = st.columns(2)
            with col_da:
                deco_assets = st.multiselect(
                    "Varlıklar (DECO)",
                    asset_cols,
                    default=[],
                    key="t3_assets",
                )
            with col_db:
                show_dcc_compare = st.checkbox(
                    "DCC çiftsel korelasyon bandıyla karşılaştır",
                    value=True,
                    key="t3_compare",
                )

            if len(deco_assets) < 2:
                st.warning("En az 2 varlık seçiniz.")
                raise _NoCompute

            # ── Build cache key ──
            _key = f"d3_deco_{'_'.join(sorted(deco_assets))}_{dh}"

            # ── Hesapla button ──
            if st.button("🔄 Hesapla", key="d3_deco_run", type="primary"):
                with st.spinner("DECO modeli tahmin ediliyor..."):
                    try:
                        deco_res = _run_deco_model(tuple(deco_assets), dh)
                        st.session_state[_key] = deco_res
                    except Exception as _exc:
                        st.error(f"Model tahmini başarısız oldu: `{_exc}`")
                        raise _NoCompute

            # ── Gate: require result ──
            if _key not in st.session_state:
                st.info("⬆️ Parametreleri seçip **🔄 Hesapla**'ya tıklayın.")
                raise _NoCompute

            deco_res = st.session_state[_key]

            rho = deco_res["rho_series"]
            deco_idx = deco_res["index"]
            sample_mean = deco_res["sample_mean_corr"]
            dcc_pair_min = deco_res["dcc_pair_min"]
            dcc_pair_max = deco_res["dcc_pair_max"]
            n_pairs      = deco_res["n_pairs"]
            deco_stats = deco_res["stats"]

            roll_win = 60
            rho_roll = pd.Series(rho, index=deco_idx).rolling(roll_win, min_periods=1).mean().values

            fig_rho = go.Figure()
            fig_rho.add_trace(go.Scatter(
                x=deco_idx, y=rho,
                mode="lines",
                name="DECO rho_t",
                line=dict(color=COLORS[0], width=1.2),
                fill="tozeroy",
                fillcolor="rgba(167,139,250,0.08)",
            ))
            fig_rho.add_trace(go.Scatter(
                x=deco_idx, y=rho_roll,
                mode="lines",
                name=f"{roll_win}g kayan ort.",
                line=dict(color=COLORS[1], width=1.8, dash="dash"),
            ))
            fig_rho.add_hline(
                y=sample_mean,
                line_dash="dot",
                line_color=COLORS[3],
                annotation_text=f"Ornek ort. kor. ({sample_mean:.3f})",
                annotation_position="bottom right",
            )
            fig_rho.update_layout(
                template=PLOT_TEMPLATE,
                title="DECO Ekikorelasyon Serisi rho_t",
                xaxis_title="Tarih",
                yaxis_title="rho_t",
                height=380,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_rho, use_container_width=True)

            if show_dcc_compare:
                _idx_list = list(deco_idx)
                fig_cmp = go.Figure()
                # Gölgeli bant: DCC ikili korelasyonlarının kesitsel [min, max] aralığı
                fig_cmp.add_trace(go.Scatter(
                    x=_idx_list + _idx_list[::-1],
                    y=list(dcc_pair_max) + list(dcc_pair_min[::-1]),
                    fill="toself",
                    fillcolor="rgba(86,180,233,0.18)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"DCC ikili kor. aralığı ({n_pairs} çift, min–maks)",
                    hoverinfo="skip",
                ))
                # DECO rho_t = DCC ikili korelasyonlarının kesitsel ortalaması
                fig_cmp.add_trace(go.Scatter(
                    x=deco_idx, y=rho,
                    mode="lines",
                    name="DECO ρ_t  (= DCC çiftleri kesitsel ort.)",
                    line=dict(color=COLORS[0], width=2.2),
                ))
                fig_cmp.update_layout(
                    template=PLOT_TEMPLATE,
                    title=f"DECO ρ_t vs DCC Çiftsel Korelasyon Bandı ({n_pairs} çift)",
                    xaxis_title="Tarih",
                    yaxis_title="Korelasyon",
                    height=320,
                    margin=dict(l=20, r=20, t=50, b=30),
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

            # ── DECO R_t vs DCC R_t: eşkorelasyon kısıtının maliyeti ─────
            st.markdown("#### Eşkorelasyon Kısıtının Maliyeti: $R_t$ Karşılaştırması")
            R_deco = np.asarray(deco_res["R_deco"])
            R_dcc = np.asarray(deco_res["R_dcc"])
            dcols = deco_res["cols"]
            iu_d = np.triu_indices(len(dcols), k=1)
            mean_dcc = R_dcc[:, iu_d[0], iu_d[1]].mean(axis=1)

            snap_d = st.select_slider(
                "Tarih (anlık matris)", options=list(range(len(deco_idx))),
                value=int(np.argmax(mean_dcc)),
                format_func=lambda i: str(pd.Timestamp(deco_idx[i]).date()),
                key="t3_snap",
            )
            zmin_d = float(min(R_dcc[snap_d][iu_d].min(), R_deco[snap_d][iu_d].min()))

            hm1, hm2 = st.columns(2)
            for _col, _R, _ttl in (
                (hm1, R_dcc[snap_d], "DCC R_t (çiftler serbest)"),
                (hm2, R_deco[snap_d], "DECO R_t (tek ρ̄_t)"),
            ):
                with _col:
                    _fig = go.Figure(go.Heatmap(
                        z=_R, x=dcols, y=dcols, colorscale=HEATMAP_CMAP,
                        zmin=zmin_d, zmax=1.0, text=np.round(_R, 2),
                        texttemplate="%{text}", textfont=dict(size=9),
                        colorbar=dict(title="ρ"),
                    ))
                    _fig.update_layout(
                        template=PLOT_TEMPLATE, title=_ttl, height=360,
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=10, r=10, t=50, b=20),
                    )
                    st.plotly_chart(_fig, use_container_width=True)
            st.caption(
                f"{pd.Timestamp(deco_idx[snap_d]).date()} — DCC çiftleri "
                f"[{R_dcc[snap_d][iu_d].min():.3f}, {R_dcc[snap_d][iu_d].max():.3f}] "
                f"aralığına yayılırken DECO hepsini {rho[snap_d]:.3f}'e kısıtlar. "
                "Aradaki yayılım, eşkorelasyon varsayımının bilgi maliyetidir."
            )

            st.markdown("#### DECO Parametre Özeti")
            deco_a = deco_stats["alpha"]
            deco_b = deco_stats["beta"]
            deco_p = deco_stats["persistence"]
            deco_hl = deco_stats["half_life_days"]
            deco_mc = deco_stats["mean_corr"]
            dcc_stats_ref = deco_res["dcc_stats"]

            deco_table = pd.DataFrame([
                {
                    "Model": "DECO",
                    "alpha": f"{deco_a:.4f}",
                    "beta": f"{deco_b:.4f}",
                    "alpha+beta": f"{deco_p:.4f}",
                    "Yarı-Ömür (gün)": f"{deco_hl:.1f}",
                    "logL": f"{deco_stats['corr_loglik']:.2f}",
                    "#par": deco_stats["n_params"],
                    "Ort. Korelasyon": f"{deco_mc:.4f}",
                },
                {
                    "Model": "DCC (referans)",
                    "alpha": f"{dcc_stats_ref['alpha']:.4f}",
                    "beta": f"{dcc_stats_ref['beta']:.4f}",
                    "alpha+beta": f"{dcc_stats_ref['persistence']:.4f}",
                    "Yarı-Ömür (gün)": f"{dcc_stats_ref['half_life_days']:.1f}",
                    "logL": f"{dcc_stats_ref['corr_loglik']:.2f}",
                    "#par": dcc_stats_ref["n_params"],
                    "Ort. Korelasyon": f"{dcc_stats_ref['mean_corr']:.4f}",
                },
            ])
            st.dataframe(deco_table, use_container_width=True, hide_index=True)
            st.caption(
                f"Örnek koşulsuz ortalama korelasyon: {sample_mean:.4f}. "
                "İki satır **ayrı olabilirliklerden** gelir; aynı çıkmaları "
                "beklenmez. DCC, DECO'yu içine alan daha geniş bir model "
                "olduğundan logL'i her zaman ≥ DECO'nunkidir — bu bir üstünlük "
                "kanıtı değil, iç içe geçmenin (nesting) sonucudur. İkisi "
                "**iç içe geçmiş değildir** ki LR testi uygulanabilsin: "
                "karşılaştırma için AIC/BIC ya da tahmin performansı kullanın."
            )

            st.markdown("""
**Temel Bulgular:**
- N büyüdükçe DECO'nun hesaplama avantajı (gözlem başına O(N) vs O(N³))
  belirginleşir.
- ρ̄_t'nin yüksek olduğu dönemler piyasa krizlerine işaret eder.
- DECO tek bir skaler korelasyon varsayımı yaptığından portföy çeşitlendirme
  stratejilerinde muhafazakâr bir alt sınır sunar.
""")
        except _NoCompute:
            pass

    # =========================================================================
    # TAB 4 -- FACTOR-DCC
    # =========================================================================
    with tab_factor:
        try:
            st.markdown("### Faktör-DCC — $H_t = B\\Lambda_t B' + \\Omega_t$")
            st.info(
                "Kimliklendirme: herhangi bir tersinir $Q$ için $B\\to BQ$, "
                "$\\Lambda_t\\to Q^{-1}\\Lambda_tQ^{-1\\prime}$ dönüşümü $H_t$'yi "
                "değiştirmez. **PCA seçeneğinde tekil yükler yorumlanamaz**; "
                "yalnızca faktör uzayı ve açıklanan varyans payları raporlanabilir."
            )

            fa_c1, fa_c2 = st.columns(2)
            with fa_c1:
                f_assets = st.multiselect("Varlıklar (en az 3)", asset_cols,
                                          default=[], key="tf_assets")
            with fa_c2:
                f_mode = st.selectbox(
                    "Yükler (B)",
                    ["PCA (istatistiksel)", "Gözlenen faktörlere OLS"],
                    key="tf_mode",
                )
            loading_mode = "pca" if f_mode.startswith("PCA") else "observed"

            if len(f_assets) < 3:
                st.warning("En az 3 varlık seçiniz.")
                raise _NoCompute

            k_max = int(min(10, len(f_assets) - 1))
            sel = _run_factor_selection(tuple(f_assets), k_max, dh)

            st.markdown("#### Aşama 0 — Faktör sayısı $K$")
            kc1, kc2 = st.columns([2, 1])
            with kc1:
                crit_df = pd.DataFrame([
                    {"Ölçüt": "Bai-Ng ICp1", "K": str(sel["bai_ng"]),
                     "Not": "standartlaştırılmış panel, k=0 aday kümesinde"},
                    {"Ölçüt": "Onatski ED",
                     "K": str(sel["onatski"]) if sel["onatski_ok"] else "—",
                     "Not": ("özdeğer farkı, yinelemeli δ" if sel["onatski_ok"]
                             else "N çok küçük: δ kalibrasyonu için yeterli "
                                  "özdeğer yok")},
                    {"Ölçüt": "Marchenko-Pastur", "K": str(sel["mp"]),
                     "Not": f"λ₊ = {sel['lambda_plus']:.3f}"},
                ])
                st.dataframe(crit_df, use_container_width=True, hide_index=True)
                st.caption("Üç ölçüt de aynı **korelasyon** matrisi üzerinde "
                           "çalışır; ölçek bakımından karşılaştırılabilirler.")
            with kc2:
                k_method = st.selectbox(
                    "K seçimi", ["Bai-Ng ICp1", "Onatski ED",
                                 "Marchenko-Pastur", "Manuel"], key="tf_kmethod")
                k_manual = st.number_input("Manuel K", 1, k_max, 1, key="tf_kman")
                idio_g = st.checkbox("İdiyosenkratik GARCH(1,1) (Ω_t)",
                                     value=False, key="tf_idio")

            K_map = {"Bai-Ng ICp1": sel["bai_ng"], "Onatski ED": sel["onatski"],
                     "Marchenko-Pastur": sel["mp"], "Manuel": int(k_manual)}
            K = max(1, min(int(K_map[k_method]), k_max))
            if K_map[k_method] < 1 and k_method != "Manuel":
                st.warning(f"{k_method} ölçütü K=0 verdi (faktör yapısı yok). "
                           "Grafikler için K=1 kullanılıyor.")

            ev = sel["eigenvalues"]
            fig_scree = go.Figure()
            fig_scree.add_trace(go.Bar(
                x=list(range(1, len(ev) + 1)), y=ev, name="özdeğer",
                marker_color=COLORS[1],
            ))
            fig_scree.add_hline(
                y=sel["lambda_plus"], line_dash="dash", line_color=COLORS[5],
                annotation_text=f"MP kenarı λ₊ = {sel['lambda_plus']:.2f}",
                annotation_position="top right",
            )
            fig_scree.update_layout(
                template=PLOT_TEMPLATE, title="Korelasyon Matrisi Özdeğer Spektrumu",
                xaxis_title="Sıra", yaxis_title="Özdeğer", height=300,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_scree, use_container_width=True)

            _key = f"d3_fdcc_{'_'.join(sorted(f_assets))}_{K}_{loading_mode}_{idio_g}_{dh}"
            if st.button("🔄 Hesapla", key="d3_fdcc_run", type="primary"):
                with st.spinner(f"Faktör-DCC tahmin ediliyor (K={K})..."):
                    try:
                        st.session_state[_key] = _run_factor_dcc(
                            tuple(f_assets), K, loading_mode, idio_g, dh)
                    except Exception as _exc:
                        st.error(f"Model tahmini başarısız oldu: `{_exc}`")
                        raise _NoCompute

            if _key not in st.session_state:
                st.info("⬆️ Parametreleri seçip **🔄 Hesapla**'ya tıklayın.")
                raise _NoCompute

            fres = st.session_state[_key]
            fsum = fres["summary"]
            fidx = fres["index"]

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(_metric_card("K", f"{fsum['K']}", fsum["loading_mode"]),
                            unsafe_allow_html=True)
            with m2:
                st.markdown(_metric_card("Açıklanan varyans",
                                         f"{fsum['var_share_total']:.1%}",
                                         "ilk K bileşen"), unsafe_allow_html=True)
            with m3:
                st.markdown(_metric_card("Ort. idiyosenkratik pay",
                                         f"{fsum['mean_idio_share']:.1%}",
                                         "Ω / H köşegeni"), unsafe_allow_html=True)
            with m4:
                _hl = fsum.get("factor_half_life")
                st.markdown(_metric_card("Faktör yarı-ömrü",
                                         "—" if _hl is None else f"{_hl:.1f} g",
                                         fsum["factor_model"]),
                            unsafe_allow_html=True)
            st.markdown("")

            fig_f = go.Figure()
            for k in range(fsum["K"]):
                fig_f.add_trace(go.Scatter(
                    x=fidx, y=fres["factors"][:, k], mode="lines",
                    name=f"F{k+1} ({fsum['var_share'][k]:.1%})",
                    line=dict(color=COLORS[k % len(COLORS)], width=1.0),
                ))
            fig_f.add_vrect(x0=stress[0], x1=stress[1], fillcolor="gray",
                            opacity=0.15, line_width=0)
            fig_f.update_layout(
                template=PLOT_TEMPLATE, title="Çıkarılan Faktör Getirileri f_t",
                xaxis_title="Tarih", yaxis_title="Getiri", height=320,
                margin=dict(l=20, r=20, t=50, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            st.plotly_chart(fig_f, use_container_width=True)

            fR = fres["factor_R"]
            if fR is not None and fsum["K"] >= 2:
                iu_f = np.triu_indices(fsum["K"], k=1)
                fig_fr = go.Figure()
                for k, (i, j) in enumerate(zip(*iu_f)):
                    fig_fr.add_trace(go.Scatter(
                        x=fidx, y=fR[:, i, j], mode="lines",
                        name=f"F{i+1}–F{j+1}",
                        line=dict(color=COLORS[k % len(COLORS)], width=1.2),
                    ))
                fig_fr.update_layout(
                    template=PLOT_TEMPLATE,
                    title="Faktörler Arası Dinamik Korelasyon Λ_t (DCC)",
                    xaxis_title="Tarih", yaxis_title="Korelasyon", height=300,
                    margin=dict(l=20, r=20, t=50, b=30),
                )
                st.plotly_chart(fig_fr, use_container_width=True)

            st.markdown("#### Woodbury doğrulaması (bu tahminde)")
            t_s = fres["H_sample_t"]
            H_s = np.asarray(fres["H_sample"])
            sign_s, ld_direct = np.linalg.slogdet(H_s)
            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("log det H_t (lemma)", f"{fres['logdet_sample']:.6f}")
            wc2.metric("log det H_t (slogdet)", f"{ld_direct:.6f}")
            wc3.metric("Mutlak fark",
                       f"{abs(fres['logdet_sample'] - ld_direct):.2e}")
            st.caption(
                f"t = {pd.Timestamp(fidx[t_s]).date()}. Uygulamada $H_t$ hiç "
                "kurulmaz ve tersi alınmaz; burada yalnızca doğrulama amacıyla "
                "tek bir $t$ için açıkça hesaplanmıştır. Maliyet "
                f"$O(NK^2+K^3)$, $O(N^3)$ değil."
            )

            with st.expander("Yükler B (yalnızca faktör uzayı yorumlanabilir)",
                             expanded=False):
                st.dataframe(
                    pd.DataFrame(fres["B"], index=fres["cols"],
                                 columns=[f"F{k+1}" for k in range(fsum["K"])]
                                 ).round(4),
                    use_container_width=True,
                )
        except _NoCompute:
            pass

    # =========================================================================
    # TAB 5 -- GO-GARCH
    # =========================================================================
    with tab_go:
        try:
            st.markdown("### GO-GARCH — $r_t = Z y_t$, $H_t = Z V_t Z'$")
            st.info(
                "Bileşen sırası, işareti ve ölçeği kimliklendirilemez: "
                "$(Z,V_t)$ ile $(ZDS,\\,S D^{-1}V_tD^{-1}S)$ aynı $H_t$'yi "
                "verir. $Z$'nin sütunlarına ekonomik anlam yüklemeyin; "
                "$H_t$ ve varyans payları değişmezdir."
            )

            gc1, gc2 = st.columns(2)
            with gc1:
                g_assets = st.multiselect("Varlıklar (2-8)", asset_cols,
                                          default=[], key="tg_assets")
            with gc2:
                g_method = st.selectbox(
                    "Döndürme $U$",
                    ["ICA (FastICA — bağımsız bileşenler)",
                     "PCA (U = I — O-GARCH)"],
                    key="tg_method",
                )
            method = "ica" if g_method.startswith("ICA") else "pca"

            if not (2 <= len(g_assets) <= 8):
                st.warning("2 ile 8 arasında varlık seçiniz.")
                raise _NoCompute

            _key = f"d3_go_{'_'.join(sorted(g_assets))}_{method}_{dh}"
            if st.button("🔄 Hesapla", key="d3_go_run", type="primary"):
                with st.spinner(f"GO-GARCH tahmin ediliyor ({method})..."):
                    try:
                        st.session_state[_key] = _run_go_garch(
                            tuple(g_assets), method, dh)
                    except Exception as _exc:
                        st.error(f"Model tahmini başarısız oldu: `{_exc}`")
                        raise _NoCompute

            if _key not in st.session_state:
                st.info("⬆️ Parametreleri seçip **🔄 Hesapla**'ya tıklayın.")
                raise _NoCompute

            gres = st.session_state[_key]
            gsum = gres["summary"]
            gidx = gres["index"]

            g1, g2, g3 = st.columns(3)
            with g1:
                st.markdown(_metric_card("Yöntem", gsum["method"].upper(),
                                         "U döndürmesi"), unsafe_allow_html=True)
            with g2:
                st.markdown(_metric_card("Parametre", f"{gsum['n_params']}",
                                         "3N (Z profillenmiş)"),
                            unsafe_allow_html=True)
            with g3:
                st.markdown(_metric_card("logL", f"{gsum['loglik']:.1f}",
                                         "Gauss, sabit hariç"),
                            unsafe_allow_html=True)
            st.markdown("")

            comp_df = pd.DataFrame(gsum["components"])
            comp_df = comp_df.assign(
                alpha=comp_df["alpha"].round(4), beta=comp_df["beta"].round(4),
                persistence=comp_df["persistence"].round(4),
                half_life=comp_df["half_life"].round(1),
                var_share=(comp_df["var_share"] * 100).round(1),
            ).rename(columns={
                "component": "Bileşen", "alpha": "α", "beta": "β",
                "persistence": "α+β", "half_life": "Yarı-Ömür (gün)",
                "var_share": "Varyans payı (%)",
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # The components are whitened: their UNCONDITIONAL variance is 1 by
            # construction, so sqrt(h_it) is a unit-free multiple of that, not a
            # percentage return volatility.  Plotting it as "%" would show a
            # meaningless 100-300 axis.
            fig_gc = go.Figure()
            for i in range(min(gsum["N"], len(COLORS))):
                fig_gc.add_trace(go.Scatter(
                    x=gidx, y=np.sqrt(gres["h_seq"][:, i]), mode="lines",
                    name=f"y{i+1}", line=dict(color=COLORS[i % len(COLORS)],
                                              width=1.0),
                ))
            fig_gc.add_hline(y=1.0, line_dash="dot", line_color="gray",
                             annotation_text="koşulsuz düzey = 1",
                             annotation_position="bottom right")
            fig_gc.add_vrect(x0=stress[0], x1=stress[1], fillcolor="gray",
                             opacity=0.15, line_width=0)
            fig_gc.update_layout(
                template=PLOT_TEMPLATE,
                title="Bileşen Koşullu Standart Sapmaları √h_it (birimsiz)",
                xaxis_title="Tarih", yaxis_title="√h_it", height=320,
                margin=dict(l=20, r=20, t=50, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            st.plotly_chart(fig_gc, use_container_width=True)
            st.caption(
                "Bileşenler beyazlatılmıştır: koşulsuz varyansları tanım gereği "
                "1'dir. Dolayısıyla √h_it bir yüzde getiri oynaklığı değil, "
                "koşulsuz düzeye göre bir çarpandır. Varlık bazında yüzde "
                "oynaklık için H_t'nin köşegenine bakınız."
            )

            fig_gv = go.Figure()
            for i, cname in enumerate(gres["cols"]):
                fig_gv.add_trace(go.Scatter(
                    x=gidx, y=gres["cond_vol"][:, i] * 100, mode="lines",
                    name=cname, line=dict(color=COLORS[i % len(COLORS)], width=1.0),
                ))
            fig_gv.update_layout(
                template=PLOT_TEMPLATE,
                title="Varlık Koşullu Standart Sapması √diag(H_t) (%)",
                xaxis_title="Tarih", yaxis_title="Volatilite (%)", height=300,
                margin=dict(l=20, r=20, t=50, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            st.plotly_chart(fig_gv, use_container_width=True)

            fig_gr = go.Figure()
            fig_gr.add_vrect(x0=stress[0], x1=stress[1], fillcolor="gray",
                             opacity=0.15, line_width=0)
            fig_gr.add_trace(go.Scatter(
                x=gidx, y=gres["mean_corr"], mode="lines",
                name="GO-GARCH ort. korelasyon",
                line=dict(color=COLORS[0], width=1.6),
            ))
            fig_gr.update_layout(
                template=PLOT_TEMPLATE,
                title="H_t'nin İma Ettiği Ortalama Koşullu Korelasyon",
                xaxis_title="Tarih", yaxis_title="Korelasyon", height=300,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_gr, use_container_width=True)
            st.caption(
                "GO-GARCH'ta ayrı bir korelasyon aşaması yoktur: korelasyon "
                "dinamiği tamamen bileşen varyanslarının $h_{it}$ değişiminden "
                "ve sabit $Z$'den doğar. DCC ile karşıtlık budur."
            )

            if gsum["method"] == "ica" and gsum["ica_converged"] is False:
                st.warning("FastICA maksimum yineleme sayısına ulaştı; "
                           "sonuçlar başlangıç değerine duyarlı olabilir.")
        except _NoCompute:
            pass

    # =========================================================================
    # TAB 6 -- MODEL COMPARISON
    # =========================================================================
    with tab_compare:
        try:
            st.markdown("### Model Karşılaştırması: DCC, cDCC, DECO")
            st.caption("Hız için sabit 3 varlık önerilir; aşağıdan değiştirebilirsiniz.")

            cmp_assets = st.multiselect(
                "Karşılaştırma İçin Varlıklar (önerilen: 3)",
                asset_cols,
                default=[],
                key="t4_assets",
            )

            if len(cmp_assets) < 2:
                st.warning("En az 2 varlık seçiniz.")
                raise _NoCompute

            # ── Build cache key ──
            _key = f"d3_cmp_{'_'.join(sorted(cmp_assets))}_{dh}"

            # ── Hesapla button ──
            if st.button("🔄 Hesapla", key="d3_cmp_run", type="primary"):
                with st.spinner("DCC, cDCC ve DECO tahmin ediliyor..."):
                    try:
                        cmp_results = _run_comparison(tuple(cmp_assets), dh)
                        st.session_state[_key] = cmp_results
                    except Exception as _exc:
                        st.error(f"Model tahmini başarısız oldu: `{_exc}`")
                        raise _NoCompute

            # ── Gate: require result ──
            if _key not in st.session_state:
                st.info("⬆️ Parametreleri seçip **🔄 Hesapla**'ya tıklayın.")
                raise _NoCompute

            cmp_results = st.session_state[_key]

            rows = []
            for mtype, entry in cmp_results.items():
                s = entry["stats"]
                rows.append({
                    "Model": mtype,
                    "alpha": round(s["alpha"], 4),
                    "beta": round(s["beta"], 4),
                    "alpha+beta": round(s["persistence"], 4),
                    "Yarı-Ömür (gün)": round(s["half_life_days"], 1),
                    "logL": round(s["corr_loglik"], 2),
                    "#par": s["n_params"],
                    "AIC": round(s["aic"], 1),
                    "BIC": round(s["bic"], 1),
                    "Ort. Korelasyon": round(s["mean_corr"], 4),
                    "Çiftler Arası Std": round(s.get("corr_cross_std", 0.0), 4),
                })

            df_cmp = pd.DataFrame(rows)
            st.markdown("#### Parametre Karşılaştırması")
            st.caption(
                "**logL yorumu.** DCC ve cDCC iç içe geçmemiş ama aynı "
                "korelasyon uzayında; DECO ise DCC'nin **kısıtlı** halidir "
                "(tüm çiftler tek ρ̄_t). Bu nedenle DCC'nin logL'i DECO'nunkinden "
                "büyük çıkar — bu bir üstünlük kanıtı değil, kısıtın maliyetidir. "
                "Modeller aynı sayıda parametre kullandığından AIC/BIC sıralaması "
                "logL sıralamasıyla aynıdır; asıl karar ölçütü örneklem-dışı "
                "tahmin performansı ve N ölçeklenmesidir."
            )

            def _highlight_persist(val):
                try:
                    v = float(val)
                    if v > 0.98:
                        return "color: #f472b6; font-weight: 700"
                    if v > 0.95:
                        return "color: #fbbf24"
                    return "color: #34d399"
                except Exception:
                    return ""

            st.dataframe(
                df_cmp.style.map(_highlight_persist, subset=["alpha+beta"]),
                use_container_width=True,
                hide_index=True,
            )

            fig_bar = go.Figure(go.Bar(
                x=[r["Model"] for r in rows],
                y=[r["Ort. Korelasyon"] for r in rows],
                marker_color=COLORS[:len(rows)],
                text=[f"{r['Ort. Korelasyon']:.4f}" for r in rows],
                textposition="outside",
            ))
            fig_bar.update_layout(
                template=PLOT_TEMPLATE,
                title="Modellere Gore Ortalama Çift Korelasyon",
                xaxis_title="Model",
                yaxis_title="Ortalama Korelasyon",
                height=280,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_hl = go.Figure(go.Bar(
                x=[r["Model"] for r in rows],
                y=[r["Yarı-Ömür (gün)"] for r in rows],
                marker_color=[COLORS[(i + 3) % len(COLORS)] for i in range(len(rows))],
                text=[f"{r['Yarı-Ömür (gün)']:.1f} gün" for r in rows],
                textposition="outside",
            ))
            fig_hl.update_layout(
                template=PLOT_TEMPLATE,
                title="Korelasyon Soku Yarı-Ömür Karşılaştırması",
                xaxis_title="Model",
                yaxis_title="Yarı-Ömür (gün)",
                height=280,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_hl, use_container_width=True)

            # DCC vs ADCC: asimetri etkisi (ortalama koşullu korelasyon ve farkı)
            st.markdown("#### DCC vs ADCC: Asimetri Etkisi")
            try:
                res_d = _run_dcc_model(tuple(cmp_assets), "DCC", dh)
                res_a = _run_dcc_model(tuple(cmp_assets), "ADCC", dh)
                iu_c = np.triu_indices(len(cmp_assets), k=1)
                Rd = np.asarray(res_d["R_seq"]); Ra = np.asarray(res_a["R_seq"])
                mD = Rd[:, iu_c[0], iu_c[1]].mean(axis=1)
                mA = Ra[:, iu_c[0], iu_c[1]].mean(axis=1)
                cidx = res_d["index"]
                fig_da = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                    vertical_spacing=0.09,
                    subplot_titles=("Ortalama koşullu korelasyon", "Fark (ADCC - DCC)"),
                )
                for _fr in (1, 2):
                    fig_da.add_vrect(x0=stress[0], x1=stress[1], fillcolor="gray",
                                     opacity=0.15, line_width=0, row=_fr, col=1)
                fig_da.add_trace(go.Scatter(x=cidx, y=mD, name="DCC",
                                            line=dict(color=COLORS[7], width=1.4)), row=1, col=1)
                fig_da.add_trace(go.Scatter(x=cidx, y=mA, name="ADCC",
                                            line=dict(color=COLORS[2], width=1.4)), row=1, col=1)
                fig_da.add_trace(go.Scatter(x=cidx, y=(mA - mD), name="ADCC - DCC", fill="tozeroy",
                                            line=dict(color=COLORS[2], width=1.0),
                                            showlegend=False), row=2, col=1)
                fig_da.add_hline(y=0.0, line_color="gray", line_width=0.7, row=2, col=1)
                fig_da.update_layout(
                    template=PLOT_TEMPLATE, height=460, margin=dict(l=20, r=20, t=50, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
                )
                st.plotly_chart(fig_da, use_container_width=True)
                st.caption("ADCC, ortak düşüşlerdeki asimetri (c>0) nedeniyle kriz ve sonrasında "
                           "sistematik olarak daha yüksek korelasyon üretir.")

                # ── ADCC vs DCC: sınır-karışım LR testi ──────────────────────
                from dcc_garch import adcc_vs_dcc_lr_test
                ll_d = res_d["stats"].get("corr_loglik")
                ll_a = res_a["stats"].get("corr_loglik")
                if ll_d is not None and ll_a is not None:
                    lrt = adcc_vs_dcc_lr_test(ll_a, ll_d, alpha=0.05)
                    st.markdown("**Sınır-Karışım LR Testi (ADCC vs DCC)**")
                    lc1, lc2, lc3 = st.columns(3)
                    lc1.metric("LR = 2(ℓ_ADCC − ℓ_DCC)", f"{lrt['lr_stat']:.2f}")
                    lc2.metric("Kritik değer (%5)", f"{lrt['critical_value']:.2f}")
                    lc3.metric("p-değeri", f"{lrt['p_value']:.2e}")
                    _vd = ("H₀: c=0 **reddedildi** → ADCC anlamlı biçimde üstün"
                           if lrt["reject"] else
                           "H₀: c=0 reddedilemedi → ADCC anlamlı üstünlük göstermiyor")
                    st.caption(
                        "H₀: c=0. Asimetri parametresi c≥0 parametre uzayının sınırında "
                        "olduğundan null dağılım ½χ²₀+½χ²₁'dir (düz χ²₁ değil; kritik değer "
                        "2.71, 3.84 değil; p yarıya iner — Self-Liang 1987 / Andrews 2001). "
                        + _vd + ". Sonlu örneklemde DCC-null parametrik bootstrap tercih edilir."
                    )
            except Exception as exc:
                st.info(f"DCC/ADCC karşılaştırması hesaplanamadı: `{exc}`")

            # ── DCC vs DECO: ekikorelasyon etkisi ────────────────────────────
            st.markdown("#### DCC vs DECO: Ekikorelasyon Etkisi")
            try:
                R_dcc  = np.asarray(cmp_results["DCC"]["R_seq"])
                R_deco = np.asarray(cmp_results["DECO"]["R_seq"])
                cidx_d = cmp_results["DCC"]["index"]
                iu = np.triu_indices(len(cmp_assets), k=1)
                # DECO rho_t: tüm çiftler aynı → ilk çifti al
                rho_deco = R_deco[:, iu[0][0], iu[1][0]]
                fig_dd = go.Figure()
                # DCC ikili korelasyonları (ince renkli çizgiler)
                for k, (i, j) in enumerate(zip(*iu)):
                    lbl = f"DCC {cmp_assets[i]}-{cmp_assets[j]}"
                    fig_dd.add_trace(go.Scatter(
                        x=cidx_d, y=R_dcc[:, i, j],
                        mode="lines", name=lbl,
                        line=dict(width=1.0, color=COLORS[k % len(COLORS)]),
                        opacity=0.65,
                    ))
                # DECO rho_t (kalın beyaz)
                fig_dd.add_trace(go.Scatter(
                    x=cidx_d, y=rho_deco,
                    mode="lines", name="DECO ρ_t (tüm çiftler)",
                    line=dict(color="#ffffff", width=2.5),
                ))
                fig_dd.update_layout(
                    template=PLOT_TEMPLATE,
                    title="DCC İkili Korelasyonlar vs DECO ρ_t",
                    xaxis_title="Tarih", yaxis_title="Korelasyon",
                    height=380, margin=dict(l=20, r=20, t=50, b=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1),
                )
                st.plotly_chart(fig_dd, use_container_width=True)
                st.caption(
                    "DECO tüm ikili korelasyonları tek bir ρ_t'ye (kesitsel ortalama) "
                    "kısıtlar. Çiftler Arası Std sütunu bu kısıtın maliyetini özetler: "
                    "DCC'de > 0, DECO'da tanım gereği 0."
                )
            except Exception as exc:
                st.info(f"DCC/DECO karşılaştırması hesaplanamadı: `{exc}`")

            st.markdown("#### Hangi Modeli Ne Zaman Kullanmalı?")
            st.markdown(r"""
| Model | Kullanım Durumu |
|-------|----------------|
| **DCC** | Hızlı prototipleme, az varlık ($N<20$), yorumlanabilirlik |
| **cDCC** | Tutarlı parametre tahmini; yüksek kalıcılik |
| **ADCC** | Asimetrik korelasyon dinamikleri (kriz dönemlerinde) |
| **DECO** | Büyük portföy ($N>50$), gerçek zamanli risk, $O(1)$ ölçekleme |

**Genel kural:** $\alpha + \beta$ modeller arasında büyük fark yoksa DCC yeterlidir.
Büyük $N$ için DECO'yu tercih edin.
""")
        except _NoCompute:
            pass

    # =========================================================================
    # TAB 5 -- MVP PORTFOLIO
    # =========================================================================
    with tab_portfolio:
        try:
            st.markdown("### Minimum Varyans Portföy (MVP) -- DCC Tabanlı")
            st.markdown(r"""
$$w_t = \frac{H_t^{-1}\mathbf{1}}{\mathbf{1}^\top H_t^{-1}\mathbf{1}}$$
""")

            col_pa, col_pb = st.columns(2)
            with col_pa:
                mvp_assets = st.multiselect(
                    "Varlıklar (2-8)",
                    asset_cols,
                    default=[],
                    key="t5_assets",
                )
            with col_pb:
                mvp_model = st.selectbox(
                    "Model Türü",
                    ["DCC", "cDCC", "ADCC", "DECO"],
                    key="t5_model",
                )

            if not (2 <= len(mvp_assets) <= 8):
                st.warning("2 ile 8 arasinda varlık seçiniz.")
                raise _NoCompute

            # ── Build cache key ──
            _key = f"d3_mvp_{'_'.join(sorted(mvp_assets))}_{mvp_model}_{dh}"

            # ── Hesapla button ──
            if st.button("🔄 Hesapla", key="d3_mvp_run", type="primary"):
                with st.spinner(f"MVP ağırlıkları hesaplanıyor ({mvp_model})..."):
                    try:
                        mvp_res = _run_mvp(tuple(mvp_assets), mvp_model, dh)
                        st.session_state[_key] = mvp_res
                    except Exception as _exc:
                        st.error(f"Model tahmini başarısız oldu: `{_exc}`")
                        raise _NoCompute

            # ── Gate: require result ──
            if _key not in st.session_state:
                st.info("⬆️ Parametreleri seçip **🔄 Hesapla**'ya tıklayın.")
                raise _NoCompute

            mvp_res = st.session_state[_key]

            weights_unc = mvp_res["weights_unc"]
            weights_lo = mvp_res["weights_lo"]
            w_lw = mvp_res["w_lw"]
            lw_delta = mvp_res["lw_delta"]
            H_seq = mvp_res["H_seq"]
            mvp_idx = mvp_res["index"]
            mvp_cols = mvp_res["cols"]
            mvp_ret = mvp_res["returns"]
            T_mvp, N_mvp = weights_unc.shape
            ann = np.sqrt(252)
            ew = np.full(N_mvp, 1.0 / N_mvp)

            col_v, col_c = st.columns(2)
            with col_v:
                variant = st.radio(
                    "Ağırlık Kısıtı", ["Kısıtsız", "Long-only (Jagannathan-Ma)"],
                    horizontal=True, key="t5_variant",
                )
            with col_c:
                cost_bps = st.slider("İşlem maliyeti (bps / birim devir)", 0, 50, 10, 1, key="t5_cost")
            cost = cost_bps / 1e4
            is_long = variant.startswith("Long")
            W_sel = weights_lo if is_long else weights_unc

            fig_w = go.Figure()
            fig_w.add_vrect(x0=stress[0], x1=stress[1], fillcolor="gray", opacity=0.15,
                            line_width=0, annotation_text="stres", annotation_position="top left")
            for i, col in enumerate(mvp_cols):
                c = COLORS[i % len(COLORS)]
                fig_w.add_trace(go.Scatter(
                    x=mvp_idx, y=W_sel[:, i], mode="lines", name=col, stackgroup="one",
                    line=dict(color=c, width=0.5), fillcolor=c,
                ))
            fig_w.update_layout(
                template=PLOT_TEMPLATE, title=f"Dinamik MVP Ağırlıkları -- {mvp_model} ({variant})",
                xaxis_title="Tarih", yaxis_title="Ağırlık", height=380,
                margin=dict(l=20, r=20, t=50, b=30), yaxis=dict(tickformat=".0%"),
            )
            st.plotly_chart(fig_w, use_container_width=True)

            def _perf(W):
                if W.ndim == 1:
                    W = np.tile(W, (T_mvp, 1))
                pv = np.sqrt(np.einsum("tn,tnm,tm->t", W, H_seq, W))
                dW = np.abs(np.diff(W, axis=0)).sum(1)
                gross = np.einsum("tn,tn->t", W[:-1], mvp_ret[1:])
                net = gross - cost * dW
                _sh = lambda x: float(np.mean(x) / np.std(x)) * ann if np.std(x) > 0 else float("nan")
                return dict(vol=float(np.mean(pv)) * ann * 100, sharpe=_sh(gross), net=_sh(net),
                            turn=float(np.mean(dW)) * 100, maxw=float(np.max(np.mean(W, axis=0))))

            p_unc = _perf(weights_unc); p_lo = _perf(weights_lo)
            p_lw = _perf(w_lw); p_ew = _perf(ew)
            p_sel = p_lo if is_long else p_unc

            pv_sel = np.sqrt(np.einsum("tn,tnm,tm->t", W_sel, H_seq, W_sel))
            pv_ew = np.sqrt(np.einsum("n,tnm,m->t", ew, H_seq, ew))
            fig_pv = go.Figure()
            fig_pv.add_vrect(x0=stress[0], x1=stress[1], fillcolor="gray", opacity=0.15, line_width=0)
            fig_pv.add_trace(go.Scatter(x=mvp_idx, y=pv_sel * 100, name="MVP",
                                        line=dict(color=COLORS[4], width=1.5)))
            fig_pv.add_trace(go.Scatter(x=mvp_idx, y=pv_ew * 100, name="Eşit Ağırlık",
                                        line=dict(color=COLORS[1], width=1.4, dash="dash")))
            fig_pv.update_layout(
                template=PLOT_TEMPLATE,
                title=f"Portföy Oynaklığı (%/gün) -- MVP vs 1/N ({mvp_model})",
                xaxis_title="Tarih", yaxis_title="Volatilite (%)", height=280,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_pv, use_container_width=True)

            metric_defs_mvp = [
                ("Yıllık Vol. (%)", f"{p_sel['vol']:.2f}"),
                ("Brüt Sharpe", f"{p_sel['sharpe']:.3f}"),
                (f"Net Sharpe ({cost_bps}bps)", f"{p_sel['net']:.3f}"),
                ("Günlük Devir (%)", f"{p_sel['turn']:.1f}"),
            ]
            m_cols = st.columns(len(metric_defs_mvp))
            for mc, (lbl, val) in zip(m_cols, metric_defs_mvp):
                with mc:
                    st.markdown(_metric_card(lbl, val), unsafe_allow_html=True)
            st.markdown("")

            st.markdown(f"#### Strateji Karşılaştırması (maliyet: {cost_bps} bps)")

            def _row(name, p):
                return {
                    "Strateji": name, "Yıllık Vol. (%)": f"{p['vol']:.1f}",
                    "Brüt Sharpe": f"{p['sharpe']:.3f}", "Net Sharpe": f"{p['net']:.3f}",
                    "Devir (%)": f"{p['turn']:.1f}", "Maks. Ort. Ağırlık": f"{p['maxw']:.3f}",
                }
            cmp_df = pd.DataFrame([
                _row("Eşit Ağırlık (1/N)", p_ew),
                _row(f"{mvp_model}-MVP (kısıtsız)", p_unc),
                _row(f"{mvp_model}-MVP (long-only)", p_lo),
                _row(f"LW-Statik (long-only, δ={lw_delta:.2f})", p_lw),
            ])
            st.dataframe(cmp_df, use_container_width=True, hide_index=True)
            st.caption(
                "Dinamik kovaryans oynaklığı düşürür; ancak kısıtsız günlük dengeleme deviri ve "
                "maliyeti net Sharpe'ı erozyona uğratır. Long-only kısıt (Jagannathan-Ma) deviri "
                "kırıp net rekabetçiliği geri getirir; Ledoit-Wolf büzülmesi benzer etkidedir."
            )

            with st.expander("Ağırlık Özet İstatistikleri (seçili varyant)", expanded=False):
                df_w_desc = pd.DataFrame(W_sel, index=mvp_idx, columns=mvp_cols)
                st.dataframe(
                    df_w_desc.describe().round(4).style.background_gradient(cmap="viridis"),
                    use_container_width=True,
                )
        except _NoCompute:
            pass

    # =========================================================================
    # TAB 6 -- DIAGNOSTICS & TESTS
    # =========================================================================
    with tab_diag:
        try:
            st.markdown("### Tanılar & Testler")
            st.markdown(
                "Sabit-korelasyon testi, DCC yeterlilik tanısı ve yüksek-boyut için "
                "bileşik-olabilirlik tahmini. Tümü seçili varlık kümesinde hesaplanır."
            )
            diag_assets = st.multiselect(
                "Varlıklar (2-8)", asset_cols,
                default=[], key="t6_assets",
            )
            if not (2 <= len(diag_assets) <= 8):
                st.warning("2 ile 8 arasında varlık seçiniz.")
            else:
                da = tuple(diag_assets)

                # ── Build cache key ──
                _key = f"d3_diag_{'_'.join(sorted(diag_assets))}_{dh}"

                # ── Hesapla button ──
                if st.button("🔄 Hesapla", key="d3_diag_run", type="primary"):
                    with st.spinner("Tanılar hesaplanıyor..."):
                        try:
                            uni = _run_univariate_diagnostics(da, dh)
                            ccc = _run_ccc_test(da, dh)
                            dg = _run_diagnostics(da, dh)
                            cl = _run_cl(da, dh)
                            st.session_state[_key] = {"uni": uni, "ccc": ccc,
                                                      "dg": dg, "cl": cl}
                        except Exception as _exc:
                            st.error(f"Tanı hesabı başarısız oldu: `{_exc}`")
                            raise _NoCompute

                # ── Gate: require result ──
                if _key not in st.session_state:
                    st.info("⬆️ Parametreleri seçip **🔄 Hesapla**'ya tıklayın.")
                else:
                    _diag_data = st.session_state[_key]
                    uni = _diag_data["uni"]
                    ccc = _diag_data["ccc"]
                    dg = _diag_data["dg"]
                    cl = _diag_data["cl"]

                    sub_uni, sub_corr, sub_boot, sub_hd = st.tabs([
                        "1️⃣ Tek Değişkenli Aşama",
                        "2️⃣ Korelasyon Aşaması",
                        "3️⃣ Parametrik Bootstrap",
                        "4️⃣ Yüksek Boyut",
                    ])

                    # ── 1) UNIVARIATE STAGE ─────────────────────────────
                    with sub_uni:
                        st.markdown("#### Marjinal GARCH(1,1) Yeterliliği")
                        st.markdown(
                            "İki aşamalı QMLE'de korelasyon tanıları, marjinallerin "
                            "doğru belirtilmiş olduğunu **varsayar**. Marjinal "
                            "aşama reddediliyorsa $z_t$ zaten bozuktur ve "
                            "korelasyon tanıları yorumlanamaz — bu yüzden bu "
                            "sekme önce gelir."
                        )
                        uni_df = pd.DataFrame([{
                            "Varlık": r["asset"],
                            "LB(z,10)": f"{r['lb_z']:.1f}",
                            "p": f"{r['lb_z_p']:.3f}",
                            "LB(z²,10)": f"{r['lb_z2']:.1f}",
                            "p ": f"{r['lb_z2_p']:.3f}",
                            "ARCH-LM(10)": f"{r['arch_lm']:.1f}",
                            "p  ": f"{r['arch_lm_p']:.3f}",
                            "Engle-Ng (χ²₃)": f"{r['sign_bias']:.1f}",
                            "p   ": f"{r['sign_bias_p']:.3f}",
                            "Nyblom L": ("—" if not np.isfinite(r.get("nyblom", np.nan))
                                         else f"{r['nyblom']:.2f}"),
                            "Nyblom %5": ("—" if r.get("nyblom_crit") is None
                                          else f"{r['nyblom_crit']:.2f}"),
                        } for r in uni])
                        st.dataframe(uni_df, use_container_width=True, hide_index=True)
                        st.caption(
                            "Bu tabloda p-değerleri **meşrudur**: testler tek "
                            "değişkenli, tek bir tahmin katmanına dayalı "
                            "artıklara uygulanır. LB(z²) ve ARCH-LM reddi → "
                            "oynaklık denklemi yetersiz; Engle-Ng reddi → "
                            "asimetri (GJR/EGARCH gerekir); Nyblom L > kritik "
                            "değer → parametreler örneklem boyunca sabit değil "
                            "(yapısal kırılma)."
                        )

                    # ── 2) CORRELATION STAGE ────────────────────────────
                    with sub_corr:
                        st.markdown("#### A. Sabit Koşullu Korelasyon (CCC) — Engle-Sheppard")
                        st.markdown(
                            r"$H_0:\ R_t=\bar R\ \forall t$. ES yapay-regresyonu "
                            r"$\hat\delta' X'X\hat\delta/\hat\sigma^2 \sim \chi^2_{s+1}$."
                        )
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(_metric_card("CCC parametre", f"{ccc['n_params']}",
                                                     "3N + N(N-1)/2"), unsafe_allow_html=True)
                        with c2:
                            st.markdown(_metric_card("Ort. koşulsuz kor.", f"{ccc['mean_corr']:.3f}"),
                                        unsafe_allow_html=True)
                        with c3:
                            st.markdown(_metric_card("Gözlem (T x N)", f"{ccc['T']} x {ccc['N']}"),
                                        unsafe_allow_html=True)
                        es_df = pd.DataFrame([
                            {"Gecikme s": s, "chi2 (df=s+1)": f"{stat:.1f}", "p-değeri": f"{p:.2e}",
                             "Karar": "CCC reddedilir" if p < 0.01 else "reddedilemez"}
                            for s, stat, _df, p in ccc["rows"]
                        ])
                        st.dataframe(es_df, use_container_width=True, hide_index=True)
                        st.success(
                            "**Bu p-değerleri geçerlidir.** Test, CCC artıklarına "
                            "(sabit $\\bar R^{-1/2}$ ile standardize) uygulanır — "
                            "χ²_{s+1} referansının türetildiği durumun ta kendisi. "
                            "CCC reddi, DCC genellemesinin ampirik gerekçesidir "
                            "(Engle-Sheppard 2001; Tse 2000)."
                        )

                        st.divider()
                        st.markdown("#### B. DCC Yeterlilik Tanısı (filtreleme sonrası)")
                        st.markdown(
                            r"Model-standardize artıklar $\varepsilon_t=R_t^{-1/2}z_t$ "
                            r"IID + $\mathrm{Cov}=I$ olmalı; aynı ES istatistiği "
                            r"filtrelenmiş artıklara uygulanır ($s=5$)."
                        )
                        dg_es = dg["es"]
                        dg_df = pd.DataFrame([
                            {"Filtre": nm,
                             "İstatistik (s=5)": f"{dg_es[nm][0]:.1f}",
                             "Nitel okuma": _qual(dg_es[nm][0])}
                            for nm in ("CCC", "DCC", "ADCC")
                        ])
                        st.dataframe(dg_df, use_container_width=True, hide_index=True)
                        st.warning(
                            "**p-değeri ve biçimsel karar sütunu bilinçli olarak "
                            "yoktur (§1.12.1).** Filtrelenmiş artıklara uygulanan "
                            "bu istatistiğin χ²₆ referansı geçersizdir:\n\n"
                            "1. **İki katmanlı tahmin hatası.** $R_t$ kendisi "
                            "tahmin edilmiştir; yapay regresyon onu biliniyormuş "
                            "gibi ele alır.\n"
                            "2. **Havuzlama tek-skaler-$\\hat\\sigma^2$ varsayar.** "
                            "$t(\\nu)$ yenilik altında "
                            "$\\mathrm{Var}(\\varepsilon_i\\varepsilon_j)=(\\nu-2)/(\\nu-4)$; "
                            "$\\nu=7$'de gerçek ölçek ≈ 1.67, $\\nu\\le 4$'te "
                            "dördüncü moment yoktur ve asimptotik χ² **tanımsızdır**.\n"
                            "3. **Gecikmeli regresörler içseldir.**\n\n"
                            "Nitel okuma eşikleri: < 20 → kalan dinamik izi yok; "
                            "> 100 → güçlü kalan dinamik; arası → belirsiz, "
                            "bootstrap gerekir (3. sekme)."
                        )

                        st.divider()
                        st.markdown("#### C. Çift-Bazlı Ljung-Box + Benjamini-Hochberg FDR")
                        plb = dg["pairwise_lb"]
                        st.caption(
                            f"DCC-filtrelenmiş $\\varepsilon_i\\varepsilon_j$ "
                            f"serilerinde LB({plb['lags']}); {plb['n_pairs']} çift "
                            f"üzerinde BH-FDR (α={plb['alpha']}). Havuzlanmış ES "
                            "testinin aksine bu, **hangi** çiftin kalıntı "
                            "taşıdığını söyler."
                        )
                        pc1, pc2 = st.columns([1, 2])
                        with pc1:
                            st.markdown(_metric_card(
                                "Reddedilen çift",
                                f"{plb['n_reject']} / {plb['n_pairs']}",
                                "BH-FDR sonrası"), unsafe_allow_html=True)
                        with pc2:
                            st.dataframe(pd.DataFrame([{
                                "Çift": r["pair"],
                                "LB": f"{r['lb_stat']:.1f}",
                                "p (ham)": f"{r['p_raw']:.3f}",
                                "p (BH)": f"{r['p_bh']:.3f}",
                                "Red": "✔" if r["reject"] else "",
                            } for r in plb["rows"][:5]]),
                                use_container_width=True, hide_index=True)
                        st.caption("En güçlü 5 çift gösterilmektedir.")

                        st.divider()
                        st.markdown("#### D. İşaret-Yanlılığı Testi (hedefli)")
                        sb = dg["sign_bias"]
                        st.markdown(
                            r"$Y_{ij,t}= \omega + b_1\mathbf{1}\{z_{i,t-1}<0\} "
                            r"+ b_2\mathbf{1}\{z_{j,t-1}<0\} "
                            r"+ b_3\mathbf{1}\{z_{i,t-1}<0\}\mathbf{1}\{z_{j,t-1}<0\}$"
                        )
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        sc1.metric("Wald (df=3)", f"{sb['stat']:.1f}")
                        sc2.metric("b₁ (i negatif)", f"{sb['coef']['neg_i']:.4f}")
                        sc3.metric("b₂ (j negatif)", f"{sb['coef']['neg_j']:.4f}")
                        sc4.metric("b₃ (ikisi de)", f"{sb['coef']['neg_both']:.4f}")
                        st.caption(
                            "Havuzlanmış ES testi **doğrusal** kalıntıyı yakalar, "
                            "eksik asimetriyi yakalayamaz; bu test tam olarak onu "
                            "hedefler. p-değeri burada da verilmez — aynı üç "
                            "gerekçe geçerlidir; bootstrap sekmesini kullanın."
                        )

                        st.divider()
                        st.markdown("#### E. Hosking-Li-McLeod Portmanteau")
                        hlm = dg["hlm"]
                        if hlm["available"]:
                            hc1, hc2, hc3 = st.columns(3)
                            hc1.metric("Q̃", f"{hlm['stat']:.1f}")
                            hc2.metric("sd = m·P²", f"{hlm['df']}")
                            hc3.metric("p (nominal)", f"{hlm['p_value']:.3f}")
                            st.caption(
                                f"P = N(N−1)/2 = {hlm['P']}, m = {hlm['m']}. "
                                "Nominal p, aynı iki katmanlı tahmin hatası "
                                "nedeniyle yalnızca göstergedir."
                            )
                        else:
                            st.info(f"⏭️ Devre dışı — {hlm['reason']}")

                    # ── 3) PARAMETRIC BOOTSTRAP ─────────────────────────
                    with sub_boot:
                        st.markdown("#### Parametrik Bootstrap p-Değeri")
                        st.markdown(
                            "Filtrelenmiş artıklardaki istatistiklerin geçerli "
                            "referans dağılımını üretmenin tek yolu: sığdırılmış "
                            "modelden yol simüle et ve **her yolda modeli baştan "
                            "kestir**."
                        )
                        st.warning(
                            "⚠️ **Yavaş.** Her replikasyonda N tek değişkenli "
                            "GARCH **ve** korelasyon aşaması yeniden kestirilir; "
                            "$\\hat\\theta$ sabitlenirse bootstrap anlamsızdır "
                            "(§1.12.1, 3. adım). B=199, N=5, T=1500 için kabaca "
                            "**dakikalar** mertebesinde sürer. Ders sırasında "
                            "önceden çalıştırmanız önerilir."
                        )
                        bc1, bc2, bc3 = st.columns(3)
                        with bc1:
                            B = st.selectbox("Replikasyon B", [199, 499, 999],
                                             key="t6_B")
                        with bc2:
                            b_model = st.selectbox("Null model", ["DCC", "ADCC"],
                                                   key="t6_bmodel")
                        with bc3:
                            b_stat = st.selectbox(
                                "İstatistik", ["Engle-Sheppard (s=5)",
                                               "İşaret-yanlılığı"], key="t6_bstat")

                        _bkey = f"d3_boot_{'_'.join(sorted(diag_assets))}_{B}_{b_model}_{b_stat}_{dh}"
                        if st.button("🎲 Bootstrap'i Çalıştır", key="d3_boot_run"):
                            from mgarch_diagnostics import (
                                correlation_sign_bias, es_stat,
                                parametric_bootstrap_pvalue,
                            )
                            _iu = np.triu_indices(len(diag_assets), k=1)
                            if b_stat.startswith("Engle"):
                                def _statfn(eps, z, m):
                                    return es_stat(eps, 5, _iu)[0]
                                observed = dg["es"][b_model][0]
                            else:
                                def _statfn(eps, z, m):
                                    return correlation_sign_bias(eps, z, _iu)["stat"]
                                observed = dg["sign_bias"]["stat"]

                            bar = st.progress(0.0, text="Bootstrap çalışıyor…")
                            try:
                                res_b = parametric_bootstrap_pvalue(
                                    df[list(diag_assets)], b_model, _statfn,
                                    observed, B=B,
                                    progress=lambda d, tot: bar.progress(
                                        d / tot, text=f"Bootstrap {d}/{tot}"),
                                )
                                st.session_state[_bkey] = res_b
                            except Exception as _exc:
                                st.error(f"Bootstrap başarısız: `{_exc}`")
                            finally:
                                bar.empty()

                        if _bkey in st.session_state:
                            rb = st.session_state[_bkey]
                            rc1, rc2, rc3 = st.columns(3)
                            rc1.metric("Gözlenen istatistik", f"{rb['observed']:.1f}")
                            rc2.metric("Bootstrap p", f"{rb['p_value']:.4f}")
                            rc3.metric("Başarılı replikasyon",
                                       f"{rb['B']}" + (f" (−{rb['n_failed']})"
                                                       if rb["n_failed"] else ""))
                            fig_b = go.Figure()
                            fig_b.add_trace(go.Histogram(
                                x=rb["boot_stats"], nbinsx=40,
                                marker_color=COLORS[1], name="bootstrap null"))
                            fig_b.add_vline(
                                x=rb["observed"], line_color=COLORS[5],
                                line_width=2,
                                annotation_text="gözlenen",
                                annotation_position="top right")
                            fig_b.update_layout(
                                template=PLOT_TEMPLATE,
                                title="Bootstrap Null Dağılımı",
                                xaxis_title="İstatistik", yaxis_title="Frekans",
                                height=320, margin=dict(l=20, r=20, t=50, b=30),
                            )
                            st.plotly_chart(fig_b, use_container_width=True)
                            st.caption(
                                "p = (1 + #{boot ≥ gözlenen}) / (1 + B). Ekleme, "
                                "p'nin tam olarak 0 çıkmasını engeller "
                                "(Davison-Hinkley)."
                            )

                    # ── 4) HIGH DIMENSION ───────────────────────────────
                    with sub_hd:
                        st.markdown("#### İkili Bileşik Olabilirlik (Pakel vd. 2021)")
                        st.markdown(
                            r"$(a,b)$ skaler ve çiftlerde ortak $\Rightarrow$ ikili "
                            r"alt-olabilirlikler $c\ell(a,b)=\sum_{i<j}\ell_{ij}$ "
                            r"tutarlı; $O(N^3)$ matris tersi yok."
                        )
                        cl_df = pd.DataFrame([
                            {"Yöntem": "Tam DCC (ML)", "a": f"{cl['full'][0]:.4f}",
                             "b": f"{cl['full'][1]:.4f}",
                             "a+b": f"{cl['full'][0] + cl['full'][1]:.4f}"},
                            {"Yöntem": f"İkili CL ({cl['P']} çift)", "a": f"{cl['cl'][0]:.4f}",
                             "b": f"{cl['cl'][1]:.4f}",
                             "a+b": f"{cl['cl'][0] + cl['cl'][1]:.4f}"},
                        ])
                        st.dataframe(cl_df, use_container_width=True, hide_index=True)
                        st.caption("İkili CL, DCC dinamiğini tutarlı geri kazanır "
                                   "(küçük verimlilik kaybıyla). Hesaplama avantajı "
                                   "asimptotiktir: yüzlerce varlıkta uygulanabilir tek yol.")
        except _NoCompute:
            pass


##################################################################
