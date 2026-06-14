"""
5. Gün - Gerçekleşen Oynaklık & Büyük Boyut Kovaryans
HAR-RV, TSRV, Imza Grafigi, Sıçramalar, HEAVY, GARCH-X, Realized GARCH
POET, Ledoit-Wolf, Marchenko-Pastur, MVP Karşılaştırması - EYS-26
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from scipy.stats import norm as _sp_norm

PLOT_TEMPLATE = "plotly_dark"
COLORS = [
    "#a78bfa", "#34d399", "#f472b6", "#60a5fa",
    "#fbbf24", "#fb923c", "#38bdf8", "#4ade80",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _asset_cols(df):
    return [c for c in df.columns if not c.endswith("_RV") and not c.endswith("_BPV")]


def _get_rv_bpv(df, asset):
    """Return (rv_series, bpv_series), synthesising from returns if columns missing."""
    if f"{asset}_RV" in df.columns:
        rv = df[f"{asset}_RV"].dropna()
    else:
        rv = df[asset].pow(2).rolling(5).mean().dropna()
    if f"{asset}_BPV" in df.columns:
        bpv = df[f"{asset}_BPV"].dropna()
    else:
        r = df[asset]
        bpv = (np.pi / 2 * r.abs() * r.shift(1).abs()).rolling(5).mean().dropna()
    common = rv.index.intersection(bpv.index)
    return rv.loc[common], bpv.loc[common]


def _cov_to_corr(cov):
    d = np.sqrt(np.diag(cov))
    d_inv = np.where(d > 1e-12, 1.0 / d, 0.0)
    return np.outer(d_inv, d_inv) * cov


def _metric_card(label, value, color="#a78bfa"):
    return (
        '<div style="background:#1e1e2e;border:1px solid #333;border-radius:12px;'
        'padding:1rem 1.2rem;text-align:center;color:white;margin:4px 0;">'
        f'<div style="font-size:0.75rem;opacity:0.65;">{label}</div>'
        f'<div style="font-size:1.5rem;font-weight:700;color:{color};">{value}</div>'
        '</div>'
    )


def _step_card(n, title, body):
    return (
        '<div style="background:#1e1e2e;border:1px solid #444;border-radius:10px;'
        'padding:0.9rem 1.1rem;margin:6px 0;color:white;">'
        '<span style="background:linear-gradient(90deg,#667eea,#764ba2);color:white;'
        'padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700;'
        f'margin-right:8px;">Adım {n}</span>'
        f'<strong>{title}</strong>'
        f'<div style="font-size:0.85rem;opacity:0.8;margin-top:4px;">{body}</div>'
        '</div>'
    )


def _insight_card(text, color="#a78bfa"):
    return (
        f'<div style="background:#1e1e2e;border-left:4px solid {color};'
        'border-radius:8px;padding:0.8rem 1rem;margin:6px 0;'
        f'color:white;font-size:0.85rem;">{text}</div>'
    )


# ---------------------------------------------------------------------------
# Cached computations
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_har(model_type, asset, df_key):
    from realized_volatility import (
        estimate_har_rv, estimate_har_rv_hac,
        estimate_har_rv_j, estimate_har_rv_cj,
    )
    df = st.session_state.returns_df
    rv, bpv = _get_rv_bpv(df, asset)
    if model_type == "HAR-RV":
        res, dfm = estimate_har_rv(rv)
    elif model_type == "HAR-RV (HAC)":
        res, dfm = estimate_har_rv_hac(rv)
    elif model_type == "HAR-RV-J":
        res, dfm = estimate_har_rv_j(rv, bpv)
    else:
        res, dfm = estimate_har_rv_cj(rv, bpv)
    return res, dfm, rv, bpv


@st.cache_data(show_spinner=False)
def _cached_heavy(asset, df_key):
    from realized_volatility import estimate_heavy
    df = st.session_state.returns_df
    rv, _ = _get_rv_bpv(df, asset)
    ret = df[asset].reindex(rv.index).dropna()
    rv = rv.reindex(ret.index).dropna()
    return estimate_heavy(rv, ret)


@st.cache_data(show_spinner=False)
def _cached_garch_x(asset, df_key):
    from realized_volatility import estimate_garch_x
    df = st.session_state.returns_df
    rv, _ = _get_rv_bpv(df, asset)
    ret = df[asset].reindex(rv.index).dropna()
    rv = rv.reindex(ret.index).dropna()
    return estimate_garch_x(ret, rv)


@st.cache_data(show_spinner=False)
def _cached_realized_garch(asset, df_key):
    from realized_volatility import estimate_realized_garch
    df = st.session_state.returns_df
    rv, _ = _get_rv_bpv(df, asset)
    ret = df[asset].reindex(rv.index).dropna()
    rv = rv.reindex(ret.index).dropna()
    return estimate_realized_garch(ret, rv)


@st.cache_data(show_spinner=False)
def _cached_garch_baseline(asset, df_key):
    from arch import arch_model
    df = st.session_state.returns_df
    ret = df[asset].dropna() * 100
    m = arch_model(ret, vol="Garch", p=1, q=1, dist="normal")
    res = m.fit(disp="off")
    sigma2 = (res.conditional_volatility.values / 100) ** 2
    alpha = float(res.params.get("alpha[1]", 0))
    beta = float(res.params.get("beta[1]", 0))
    return {
        "sigma2_series": pd.Series(sigma2, index=ret.index, name="sigma2_garch11"),
        "persistence": alpha + beta,
        "loglik": float(res.loglikelihood),
        "params": {"alpha": alpha, "beta": beta},
        "gamma_tstat": float("nan"),
    }


@st.cache_data(show_spinner=False)
def _cached_covariance(n_assets, k_factors, threshold, df_key):
    from realized_volatility import poet_covariance, ledoit_wolf_covariance
    df = st.session_state.returns_df
    cols = _asset_cols(df)[:n_assets]
    returns_sub = df[cols].dropna()
    sample_cov = returns_sub.cov().values
    poet_cov_mat = poet_covariance(returns_sub, k_factors=k_factors, threshold=threshold)
    lw_cov_mat, shrinkage = ledoit_wolf_covariance(returns_sub)
    corr_mat = returns_sub.corr().values
    eigs_corr = np.sort(np.linalg.eigvalsh(corr_mat))[::-1]
    return sample_cov, poet_cov_mat, lw_cov_mat, shrinkage, list(cols), returns_sub, eigs_corr


@st.cache_data(show_spinner=False)
def _cached_signature(eta2_scale, df_key):
    freq_list = [1, 2, 3, 5, 10, 15, 20, 30, 60, 120]
    IV = 0.0001        # 1% daily vol^2
    eta2 = IV * eta2_scale
    rv_5 = IV + 2.0 * eta2 / 5
    records = []
    for f in freq_list:
        mean_rv = IV + 2.0 * eta2 / f
        ann_vol = float(np.sqrt(252.0 * mean_rv) * 100.0)
        iv_pct = float(np.sqrt(252.0 * IV) * 100.0)
        noise_pct = float(np.sqrt(252.0 * 2.0 * eta2 / f) * 100.0)
        bias_pct = (mean_rv - rv_5) / rv_5 * 100
        records.append({
            "freq_min": f,
            "mean_rv": mean_rv,
            "annualized_vol_pct": ann_vol,
            "iv_component_pct": iv_pct,
            "noise_component_pct": noise_pct,
            "relative_bias_pct": bias_pct,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------

def render():
    (
        tab_theory, tab_sig, tab_har,
        tab_hf, tab_cov, tab_integrated,
    ) = st.tabs([
        "Teori",
        "Volatilite Imza Grafigi",
        "HAR-RV Modeli",
        "Yüksek Frekanslı Modeller",
        "Büyük Boyut Kovaryans",
        "Entegre Analiz: HAR + POET MVP",
    ])

    # =========================================================================
    # TAB 1: TEORi
    # =========================================================================
    with tab_theory:
        st.markdown("### Gerçekleşen Oynaklık & Büyük Boyut Kovaryans: Teori")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Gerçekleşen Varyans")
            st.markdown(r"""
**Tanim (Andersen & Bollerslev, 1998):**
$$RV_t = \sum_{j=1}^{M} r_{t,j}^2 \;\xrightarrow{p}\; \int_0^1 \sigma_t^2(s)\,ds$$
$r_{t,j}$: $M$ eşit aralıklı gun-ici getiri. $M\to\infty$ ile RV, entegre varyansa yakinsar.

---
**İki Ölçekli RV -- TSRV (Zhang, Mykland & Ait-Sahalia, 2005):**
$$\widehat{RV}^{TS} = \widehat{RV}^{(yavash)} - \frac{n_{yavash}}{n_{hızlı}}\,\widehat{RV}^{(hızlı)}$$
Yavash frekans bias'ini yok eder; hızlı frekans gürültüsunu cikarir. $O(n^{-1/6})$ yakinshma hizi.

---
**Realized Kernel (Barndorff-Nielsen et al., 2008, Econometrica):**
$$RK = \sum_{h=-H}^{H} k\!\left(\tfrac{h}{H}\right)\hat{\gamma}_h, \quad k(\cdot):\text{Parzen cekirdegi}$$
$\hat{\gamma}_h$: oto-kovaryans; cekirdek ağırlıklar gürültüyu sondurur.

---
**Oynaklık Imza Grafigi:**
Ornekleme frekansi yukseldikce mikroyapi gürültüsu artar ve RV pozitif yanli olur.
**5 dakika kurali:** Bandi & Russell (2006).
Teorik: $RV(f) = IV + 2\eta^2/f$ (gürültü-sinyal ayristirmasi).

---
**Bagimsiz Kuvvet Varyasyonu (BPV):**
$$BPV_t = \frac{\pi}{2}\sum_{j=2}^{M}|r_{t,j}|\,|r_{t,j-1}|$$
Sıçramalardan bagimsiz: $BPV_t \xrightarrow{p} \int_0^1\sigma_t^2(s)\,ds$.

**Sıçrama bilesheni:**
$$J_t = \max(0,\; RV_t - BPV_t)$$

**BNS Sıçrama Testi (Barndorff-Nielsen & Shephard, 2006):**
$$z_t = \frac{RV_t - BPV_t}{\sqrt{(\pi^2/4+\pi-5)\,\max(1,\,\widehat{TQ}_t/BPV_t^2)/M}}$$
$z_t > z_{0.999}$ => sıçrama anlamli.
""")

        with col2:
            st.markdown("#### HAR-RV Ailesi")
            st.markdown(r"""
**HAR-RV (Corsi, 2009):**
$$RV_t^{(d)} = \beta_0 + \beta_d RV_{t-1}^{(d)} + \beta_w RV_{t-1}^{(w)} + \beta_m RV_{t-1}^{(m)} + \varepsilon_t$$
$RV^{(w)}$: 5 günlük, $RV^{(m)}$: 22 günlük ortalama. Uzun bellegi basit OLS ile yakalar. R2 ~ 0.40-0.65.

---
**HAR-RV-J (Andersen, Bollerslev & Diebold, 2007):**
$$RV_t = \beta_0 + \beta_d BPV_{t-1} + \beta_w BPV_{t-1}^{(w)} + \beta_m BPV_{t-1}^{(m)} + \beta_j J_{t-1} + \varepsilon_t$$

---
**HAR-RV-CJ (Andersen et al., 2007, JASA):**
$$RV_t = \beta_0 + \beta_{cd}C_t + \beta_{cw}C_t^{(w)} + \beta_{cm}C_t^{(m)} + \beta_{jd}J_t + \varepsilon_t$$
$C_t = BPV_t$ sürekli yol bilesheni.

---
**HEAVY (Shephard & Sheppard, 2010):**
$$\sigma_t^2 = \omega + \alpha\cdot RV_{t-1} + \beta\cdot\sigma_{t-1}^2$$
Gun-ici RV'yi doğrudan girdi olarak kullanir.

---
**GARCH-X:**
$$\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2 + \gamma\cdot RV_{t-1}$$
$\gamma > 0$: RV ek bilgi tasiyorsa anlamli.

---
**Realized GARCH (Hansen, Huang & Shek, 2012):**
$$\ln\sigma_t^2 = \omega + \beta\ln\sigma_{t-1}^2 + \gamma\ln RV_{t-1}$$
$$\ln RV_t = \xi + \phi\ln\sigma_t^2 + \tau_1 z_t + \tau_2(z_t^2-1) + u_t$$
$\tau_1 < 0$: kaldirac etkisi.
""")
            st.markdown("**Model Karşılaştırması:**")
            st.dataframe(pd.DataFrame([
                {"Model": "HAR-RV",         "Veri": "Günlük RV",    "Sıçrama": "Hayir", "Tipik R2": "0.40-0.60"},
                {"Model": "HAR-RV-J",       "Veri": "RV + BPV",     "Sıçrama": "Evet",  "Tipik R2": "0.45-0.65"},
                {"Model": "HAR-RV-CJ",      "Veri": "RV + BPV",     "Sıçrama": "Evet",  "Tipik R2": "0.45-0.65"},
                {"Model": "HEAVY",          "Veri": "RV + Getiri",   "Sıçrama": "Hayir", "Tipik R2": "-"},
                {"Model": "GARCH-X",        "Veri": "RV + Getiri",   "Sıçrama": "Hayir", "Tipik R2": "-"},
                {"Model": "Realized GARCH", "Veri": "RV + Getiri",   "Sıçrama": "Evet",  "Tipik R2": "-"},
            ]), hide_index=True, use_container_width=True)

        with col3:
            st.markdown("#### Büyük Boyut Kovaryans")
            st.markdown(r"""
**Marchenko-Pastur Dagilimi:**
$$\lambda_\pm = \sigma^2\!\left(1 \pm \sqrt{\kappa}\right)^2, \quad \kappa = N/T$$
$[\lambda_-, \lambda_+]$: saf gürültü oz değerlerinin destek araligi.
$\lambda > \lambda_+$ => sinyal oz değeri (gerçek faktor).
$\kappa$ buyudukce (N/T yukselince) gürültü daha shishar.

---
**POET - Principal Orthogonal Complement Thresholding**
*(Fan, Liao & Mincheva, 2013, Ann. Statist.):*
$$\hat{\Sigma} = \hat{B}\hat{\Lambda}_k\hat{B}' + \hat{\Sigma}_u^{\mathcal{T}}$$
1. PCA: ilk $k$ oz vektor ($\hat{B}$) + oz değer ($\hat{\Lambda}_k$)
2. Idiyosenkratik: $\hat{\Sigma}_u = \hat{\Sigma} - \hat{B}\hat{\Lambda}_k\hat{B}'$
3. Yumusak esikleme: kucuk off-diag elemanlar sifirlanir
4. $N > T$ durumunda dahi tutarlı ve pozitif tanimli.

---
**Ledoit-Wolf Buzulme (2004, J. Multivar. Anal.):**
$$\hat{\Sigma}_{LW} = (1-\delta)\hat{S} + \delta\hat{\mu}\,\mathbf{I}_N$$
$\delta$: analitik optimal buzulme yogunlugu.

**Dogrusal Olmayan LW (Oracle, 2020):**
$$\hat{d}_i^{NL} = d_i / |\tilde{\mu}(d_i)|^2$$
Her oz değeri ayri buzur (Stieltjes donusumu).
""")
            st.markdown("**N/T oranina göre yöntem secimi:**")
            st.dataframe(pd.DataFrame([
                {"N/T": "< 0.10",    "Yöntem": "Ornek Kovaryans", "Not": "Yeterli gozlem"},
                {"N/T": "0.10-0.50", "Yöntem": "Ledoit-Wolf",     "Not": "Iyi genel amacli"},
                {"N/T": "0.50-1.00", "Yöntem": "POET (k>=2)",     "Not": "Faktor yapisi var"},
                {"N/T": "> 1.00",    "Yöntem": "POET zorunlu",    "Not": "Ornek tekil"},
            ]), hide_index=True, use_container_width=True)

    # =========================================================================
    # TAB 2: Volatilite Imza Grafigi
    # =========================================================================
    with tab_sig:
        st.markdown("### Volatilite Imza Grafigi (Signature Plot)")
        df = st.session_state.returns_df
        ac = _asset_cols(df)
        df_key = id(df)

        col_a, col_b, col_c = st.columns([2, 3, 2])
        with col_a:
            sig_asset = st.selectbox("Varlık", ac, key="sig_asset")
        with col_b:
            eta2_scale = st.slider(
                "Gürültü/Sinyal Orani (eta2/IV)", 0.01, 0.50, 0.15, 0.01,
                key="sig_eta2",
                help="Mikroyapi gürültüsunun entegre varyansa orani. Yüksek => daha güçlü U-şekli."
            )
        with col_c:
            show_iv = st.toggle("Gerçek IV referans cizgisi", value=True, key="sig_show_iv")

        sig_df = _cached_signature(eta2_scale, df_key)

        # Main signature plot
        fig_sig = go.Figure()
        iv_pct = float(sig_df["iv_component_pct"].iloc[0])
        if show_iv:
            fig_sig.add_hline(
                y=iv_pct, line_dash="dot", line_color="#34d399", line_width=1.5,
                annotation_text=f"Gerçek IV ({iv_pct:.2f}%)",
                annotation_position="bottom right",
                annotation_font_color="#34d399",
            )
        fig_sig.add_trace(go.Scatter(
            x=sig_df["freq_min"], y=sig_df["annualized_vol_pct"],
            mode="lines+markers", name="Gözlemlenen RV (yıllıkl. %)",
            line=dict(color=COLORS[0], width=2.5), marker=dict(size=7),
        ))
        fig_sig.add_trace(go.Scatter(
            x=sig_df["freq_min"], y=sig_df["noise_component_pct"],
            mode="lines", name="Gürültü bilesheni",
            line=dict(color=COLORS[2], width=1.4, dash="dash"),
            fill="tozeroy", fillcolor="rgba(244,114,182,0.10)",
        ))
        fig_sig.add_vline(
            x=5, line_dash="dash", line_color="#fbbf24", line_width=1.8,
            annotation_text="5 dk optimal",
            annotation_position="top right",
            annotation_font_color="#fbbf24",
        )
        fig_sig.update_layout(
            template=PLOT_TEMPLATE, height=380,
            title=f"Volatilite Imza Grafigi (eta2/IV = {eta2_scale:.2f})",
            xaxis_title="Ornekleme Araligi (dakika)",
            yaxis_title="Yıllıkl. Oynaklık (%)",
            margin=dict(l=20, r=20, t=55, b=35),
            legend=dict(orientation="h", y=-0.20),
        )
        st.plotly_chart(fig_sig, use_container_width=True)

        st.markdown("**Frekans Karşılaştırma Tablosu**")
        tbl = sig_df[["freq_min", "mean_rv", "annualized_vol_pct", "relative_bias_pct"]].copy()
        tbl.columns = ["Frekans (dk)", "Ortalama RV", "Yıllıkl. Vol (%)", "Göreli Yanlılık (%)"]
        st.dataframe(tbl.style.format({
            "Ortalama RV": "{:.6f}",
            "Yıllıkl. Vol (%)": "{:.3f}",
            "Göreli Yanlılık (%)": "{:+.2f}",
        }), use_container_width=True, hide_index=True)

        st.divider()

        rv_s, bpv_s = _get_rv_bpv(df, sig_asset)
        ann_rv = np.sqrt(rv_s * 252) * 100

        fig_rv_ts = go.Figure()
        fig_rv_ts.add_trace(go.Scatter(
            x=rv_s.index, y=ann_rv.values,
            mode="lines", name="sqrt(RV*252)*100 (%)",
            line=dict(color=COLORS[0], width=1.5),
        ))
        fig_rv_ts.update_layout(
            template=PLOT_TEMPLATE, height=280,
            title=f"Gerçekleşen Oynaklık Zaman Serisi -- {sig_asset}",
            xaxis_title="Tarih", yaxis_title="Yıllıkl. Oynaklık (%)",
            margin=dict(l=20, r=20, t=50, b=30),
        )
        st.plotly_chart(fig_rv_ts, use_container_width=True)

        # Jump analysis
        st.markdown("#### Sıçrama Analizi")
        bpv_aligned = bpv_s.reindex(rv_s.index).fillna(rv_s)
        J_t = (rv_s - bpv_aligned).clip(lower=0)
        J_frac = (J_t / rv_s.replace(0, np.nan)).fillna(0)
        J_frac_roll = J_frac.rolling(22, min_periods=1).mean()

        col_j1, col_j2 = st.columns(2)
        with col_j1:
            fig_jt = go.Figure()
            fig_jt.add_trace(go.Scatter(
                x=rv_s.index, y=rv_s.values,
                mode="lines", name="RV",
                line=dict(color="#60a5fa", width=1), opacity=0.55,
            ))
            fig_jt.add_trace(go.Bar(
                x=J_t.index, y=J_t.values,
                name="J_t (Sıçrama)",
                marker_color="#f87171", opacity=0.80,
            ))
            fig_jt.update_layout(
                template=PLOT_TEMPLATE, height=280,
                title="Sıçrama Bilesheni J_t = max(0, RV-BPV)",
                xaxis_title="Tarih", yaxis_title="Varyans",
                margin=dict(l=20, r=20, t=50, b=30),
                barmode="overlay",
            )
            st.plotly_chart(fig_jt, use_container_width=True)

        with col_j2:
            fig_jfrac = go.Figure()
            fig_jfrac.add_trace(go.Scatter(
                x=J_frac_roll.index, y=J_frac_roll.values * 100,
                mode="lines", name="J/RV 22g ort. (%)",
                line=dict(color=COLORS[1], width=1.8),
            ))
            fig_jfrac.update_layout(
                template=PLOT_TEMPLATE, height=280,
                title="Sıçrama Payi (J_t / RV_t) - 22G Hareketli Ort.",
                xaxis_title="Tarih", yaxis_title="J/RV (%)",
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_jfrac, use_container_width=True)

        total_jumps = int((J_t > 0).sum())
        mean_j_rv = float(J_frac[J_t > 0].mean() * 100) if total_jumps > 0 else 0.0
        max_j_date = str(J_t.idxmax())[:10] if len(J_t) > 0 else "-"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(_metric_card("Toplam Sıçrama Gunu", str(total_jumps), "#f472b6"), unsafe_allow_html=True)
        with c2:
            st.markdown(_metric_card("Ortalama J/RV (%)", f"{mean_j_rv:.2f}", "#a78bfa"), unsafe_allow_html=True)
        with c3:
            st.markdown(_metric_card("En Büyük Sıçrama Tarihi", max_j_date, "#fbbf24"), unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: HAR-RV Modeli
    # =========================================================================
    with tab_har:
        st.markdown("### HAR-RV Model Tahmini")
        df = st.session_state.returns_df
        ac = _asset_cols(df)
        df_key = id(df)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            har_asset = st.selectbox("Varlık", ac, key="har3_asset")
        with col_b:
            har_model = st.selectbox(
                "Model",
                ["HAR-RV", "HAR-RV (HAC)", "HAR-RV-J", "HAR-RV-CJ"],
                key="har3_model",
            )
        with col_c:
            forecast_horizon = st.slider("Öngörü Ufku (gün)", 1, 22, 5, key="har3_horizon")
            st.caption(f"Secili ufuk: {forecast_horizon} gün")

        with st.spinner("Model tahmin ediliyor..."):
            res, dfm, rv_s, bpv_s = _cached_har(har_model, har_asset, df_key)

        if "fitted" in dfm.columns:
            fitted = dfm["fitted"]
            resid = dfm["resid"]
        else:
            fitted = res.fittedvalues
            resid = res.resid

        r2 = res.rsquared
        adj_r2 = res.rsquared_adj
        n_obs = int(res.nobs)

        st.markdown("#### Tahmin Sonuclari")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(_metric_card("R2", f"{r2:.4f}"), unsafe_allow_html=True)
        with m2:
            st.markdown(_metric_card("Duz. R2", f"{adj_r2:.4f}", "#34d399"), unsafe_allow_html=True)
        with m3:
            st.markdown(_metric_card("Gozlem", str(n_obs), "#60a5fa"), unsafe_allow_html=True)
        st.markdown("")

        params_df = pd.DataFrame({
            "Katsayı": res.params,
            "Std. Hata": res.bse,
            "t-ist.": res.tvalues,
            "p-değeri": res.pvalues,
        }).round(6)

        def _color_pval(v):
            try:
                fv = float(v)
                if fv < 0.001:
                    return "color:#f87171;font-weight:700"
                if fv < 0.05:
                    return "color:#f97316"
                return "color:#34d399"
            except Exception:
                return ""

        st.markdown("**Katsayı Tablosu** (kırmızı => p<0.001, turuncu => p<0.05, yesil => anlaml degil)")
        st.dataframe(
            params_df.style.map(_color_pval, subset=["p-değeri"]),
            use_container_width=True,
        )

        if har_model == "HAR-RV-CJ":
            st.info(
                "HAR-RV-CJ: beta_cd/cw/cm sürekli yol (BPV); beta_jd günlük sıçrama. "
                "beta_jd anlamsizligi sıçramalarin kalici olmadigini gosterir."
            )

        st.divider()

        if "RV" in dfm.columns:
            actual_y = dfm["RV"]
        else:
            actual_y = rv_s.reindex(dfm.index)

        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(
            x=actual_y.index, y=actual_y.values,
            mode="lines", name="Gerçek RV",
            line=dict(color="#60a5fa", width=1.0), opacity=0.75,
        ))
        fig_fit.add_trace(go.Scatter(
            x=fitted.index, y=fitted.values,
            mode="lines", name=f"{har_model} Fitted",
            line=dict(color=COLORS[0], width=2.0),
        ))
        fig_fit.update_layout(
            template=PLOT_TEMPLATE, height=380,
            title=f"Gerçek RV vs. {har_model} Fitted -- {har_asset}",
            xaxis_title="Tarih", yaxis_title="Gerçekleşen Varyans",
            margin=dict(l=20, r=20, t=55, b=30),
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fig_fit, use_container_width=True)

        st.markdown("**Kalikti Otokorelasyon Fonksiyonu (ACF)**")
        resid_vals = np.asarray(resid, dtype=float)
        resid_vals = resid_vals[np.isfinite(resid_vals)]
        acf_vals = sm.tsa.acf(resid_vals, nlags=20, fft=True)
        n_res = len(resid_vals)
        conf_band = 1.96 / np.sqrt(max(n_res, 1))
        lags_range = list(range(1, 21))
        acf_plot = acf_vals[1:21]
        acf_colors = ["#f87171" if abs(v) > conf_band else "#60a5fa" for v in acf_plot]

        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(
            x=lags_range, y=acf_plot,
            marker_color=acf_colors, name="ACF",
        ))
        fig_acf.add_hline(y=conf_band,  line_dash="dot", line_color="#fbbf24", line_width=1)
        fig_acf.add_hline(y=-conf_band, line_dash="dot", line_color="#fbbf24", line_width=1)
        fig_acf.update_layout(
            template=PLOT_TEMPLATE, height=280,
            title=f"Kalikti ACF -- {har_model} (+/-{conf_band:.3f} = 95% bant)",
            xaxis_title="Gecikme", yaxis_title="ACF",
            margin=dict(l=20, r=20, t=50, b=30),
        )
        st.plotly_chart(fig_acf, use_container_width=True)
        st.caption(
            "Kırmızı barlar 95% guven bandinin disinda -- kalikitilarda otokorelasyon var. "
            "HAC standart hatalar kullanmak onerilir (HAR-RV HAC secenegi)."
        )

    # =========================================================================
    # TAB 4: Yüksek Frekanslı Modeller
    # =========================================================================
    with tab_hf:
        st.markdown("### Yüksek Frekanslı Oynaklık Modelleri")
        df = st.session_state.returns_df
        ac = _asset_cols(df)
        df_key = id(df)

        col_a, col_b = st.columns([2, 3])
        with col_a:
            hf_asset = st.selectbox("Varlık", ac, key="hf_asset")
        with col_b:
            selected_models = st.multiselect(
                "Modeller",
                ["GARCH(1,1) baseline", "HEAVY", "GARCH-X", "Realized GARCH"],
                default=["GARCH(1,1) baseline", "HEAVY", "GARCH-X"],
                key="hf_models",
            )

        if not selected_models:
            st.info("Lutfen en az bir model secin.")
            st.stop()

        model_results = {}
        progress_bar = st.progress(0, text="Modeller hesaplaniyor...")
        total_m = len(selected_models)

        for i, mname in enumerate(selected_models):
            progress_bar.progress(i / total_m, text=f"{mname} tahmini...")
            try:
                if mname == "GARCH(1,1) baseline":
                    model_results[mname] = _cached_garch_baseline(hf_asset, df_key)
                elif mname == "HEAVY":
                    model_results[mname] = _cached_heavy(hf_asset, df_key)
                elif mname == "GARCH-X":
                    model_results[mname] = _cached_garch_x(hf_asset, df_key)
                elif mname == "Realized GARCH":
                    model_results[mname] = _cached_realized_garch(hf_asset, df_key)
            except Exception as e:
                st.warning(f"{mname} hatasi: {e}")

        progress_bar.progress(1.0, text="Tamamlandi.")

        if not model_results:
            st.error("Hicbir model tahmin edilemedi.")
            st.stop()

        MODEL_COLORS = {
            "GARCH(1,1) baseline": COLORS[3],
            "HEAVY":               COLORS[0],
            "GARCH-X":             COLORS[1],
            "Realized GARCH":      COLORS[2],
        }

        # sigma2 time series overlay
        fig_hf = go.Figure()
        for mname, r in model_results.items():
            s2 = r["sigma2_series"]
            ann_vol = np.sqrt(np.maximum(s2, 0) * 252) * 100
            fig_hf.add_trace(go.Scatter(
                x=s2.index, y=ann_vol.values,
                mode="lines", name=mname,
                line=dict(color=MODEL_COLORS.get(mname, "#fff"), width=1.6),
                opacity=0.85,
            ))
        rv_s, _ = _get_rv_bpv(df, hf_asset)
        ann_rv_ref = np.sqrt(rv_s * 252) * 100
        fig_hf.add_trace(go.Scatter(
            x=rv_s.index, y=ann_rv_ref.values,
            mode="lines", name="sqrt(RV*252)*100 (referans)",
            line=dict(color="#94a3b8", width=0.8, dash="dot"), opacity=0.5,
        ))
        fig_hf.update_layout(
            template=PLOT_TEMPLATE, height=380,
            title=f"Koşullu Oynaklık Karşılaştırması -- {hf_asset} (yıllıkl. %)",
            xaxis_title="Tarih", yaxis_title="Yıllıkl. Oynaklık (%)",
            margin=dict(l=20, r=20, t=55, b=35),
            legend=dict(orientation="h", y=-0.22),
        )
        st.plotly_chart(fig_hf, use_container_width=True)

        # Comparison table
        st.markdown("**Model Karşılaştırma Tablosu**")
        rows_cmp = []
        for mname, r in model_results.items():
            pers = r.get("persistence", float("nan"))
            hl = np.log(0.5) / np.log(pers) if (0 < pers < 1) else float("inf")
            loglik = r.get("loglik", float("nan"))
            p = r.get("params", {})
            if mname == "GARCH-X":
                note = f"gamma={p.get('gamma',0):.5f} (t={r.get('gamma_tstat', float('nan')):.2f})"
            elif mname == "Realized GARCH":
                note = f"gamma={p.get('gamma', float('nan')):.4f}, tau1={r.get('leverage_tau1', float('nan')):.3f}"
            elif mname == "HEAVY":
                note = f"alpha={p.get('alpha',0):.4f}, beta={p.get('beta',0):.4f}"
            else:
                note = f"alpha={p.get('alpha',0):.4f}, beta={p.get('beta',0):.4f}"
            rows_cmp.append({
                "Model": mname,
                "Kalicilik (a+b)": f"{pers:.4f}" if not (isinstance(pers, float) and pers != pers) else "-",
                "Yarı-Ömür (gün)": f"{hl:.1f}" if hl != float("inf") and hl == hl else "inf",
                "Log-lik": f"{loglik:,.1f}" if loglik == loglik else "-",
                "Parametre Notu": note,
            })
        st.dataframe(pd.DataFrame(rows_cmp), use_container_width=True, hide_index=True)

        # Persistence bar chart
        pers_vals = {m: r.get("persistence", 0) for m, r in model_results.items()}
        fig_pers = go.Figure(go.Bar(
            y=list(pers_vals.keys()),
            x=list(pers_vals.values()),
            orientation="h",
            marker_color=[MODEL_COLORS.get(m, COLORS[0]) for m in pers_vals],
        ))
        fig_pers.add_vline(x=1.0, line_dash="dash", line_color="#f87171",
                           annotation_text="a+b=1 (sonsuz bellek)", annotation_font_color="#f87171")
        fig_pers.update_layout(
            template=PLOT_TEMPLATE, height=280,
            title="Model Kaliciligi (alpha + beta)",
            xaxis_title="Kalicilik", xaxis=dict(range=[0, 1.05]),
            margin=dict(l=20, r=20, t=50, b=30),
        )
        st.plotly_chart(fig_pers, use_container_width=True)

        # Insight cards
        st.markdown("**Temel Icgoruler**")
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            if "HEAVY" in model_results:
                p_h = model_results["HEAVY"].get("persistence", 0)
                p_g = model_results.get("GARCH(1,1) baseline", {}).get("persistence", 0)
                cmp = f"GARCH: {p_g:.3f} vs HEAVY: {p_h:.3f}" if p_g else f"Kalicilik: {p_h:.3f}"
                st.markdown(_insight_card(
                    f"<strong>HEAVY Modeli</strong><br>{cmp}<br>"
                    "RV'yi doğrudan kullanan HEAVY, GARCH'a göre yeni oynaklk bilgisine daha hızlı tepki verir.",
                    COLORS[0]
                ), unsafe_allow_html=True)
        with ic2:
            if "GARCH-X" in model_results:
                gv = model_results["GARCH-X"]["params"].get("gamma", 0)
                tv = model_results["GARCH-X"].get("gamma_tstat", float("nan"))
                tv_str = f"{tv:.2f}" if tv == tv else "NaN"
                sig = "ANLAMLI" if (tv == tv and abs(tv) > 1.96) else "anlamlsiz"
                st.markdown(_insight_card(
                    f"<strong>GARCH-X</strong><br>gamma={gv:.5f} (t={tv_str}) => {sig}<br>"
                    "gamma>0 ve anlamliysa RV, getiri karesinin otesinde ek bilgi tasir.",
                    COLORS[1]
                ), unsafe_allow_html=True)
        with ic3:
            if "Realized GARCH" in model_results:
                tau1 = model_results["Realized GARCH"].get("leverage_tau1", float("nan"))
                lev = "Kaldirac etkisi var (tau1<0)" if (tau1 == tau1 and tau1 < 0) else "Kaldirac etkisi zayif"
                tau_str = f"{tau1:.3f}" if tau1 == tau1 else "NaN"
                st.markdown(_insight_card(
                    f"<strong>Realized GARCH</strong><br>tau1={tau_str} => {lev}<br>"
                    "İki denklemli sistem: ölçüm hatasini ve asimetriyi birlikte modeller.",
                    COLORS[2]
                ), unsafe_allow_html=True)

    # =========================================================================
    # TAB 5: Büyük Boyut Kovaryans
    # =========================================================================
    with tab_cov:
        st.markdown("### Büyük Boyut Kovaryans Tahmini")
        df = st.session_state.returns_df
        ac = _asset_cols(df)
        max_n = min(len(ac), 8)
        df_key = id(df)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            n_assets_cov = st.slider("Varlık Sayisi", 2, max_n, min(5, max_n), key="cov5_n")
        with col_b:
            k_max = max(1, n_assets_cov - 1)
            k_factors = st.slider("Faktor Sayisi (POET)", 1, k_max, min(3, k_max), key="cov5_k")
        with col_c:
            threshold = st.slider("Esik (POET)", 0.01, 0.50, 0.10, 0.01, key="cov5_thr")

        with st.spinner("Kovaryans matrisleri hesaplaniyor..."):
            (
                sample_cov, poet_cov_mat, lw_cov_mat,
                shrinkage, cols, returns_sub, eigs_corr,
            ) = _cached_covariance(n_assets_cov, k_factors, threshold, df_key)

        from realized_volatility import marchenko_pastur_threshold as _mp_thresh
        T_obs = len(returns_sub)
        kappa = n_assets_cov / T_obs
        mp_info = _mp_thresh(n_assets_cov, T_obs, sigma2=1.0, eigenvalues=eigs_corr)
        n_signal = mp_info["n_signal_eigenvalues"] or 0
        n_noise = n_assets_cov - n_signal

        st.markdown(
            f"LW buzulme yogunlugu delta = **{shrinkage:.4f}** | "
            f"kappa = N/T = {n_assets_cov}/{T_obs} = **{kappa:.4f}** | "
            f"MP lambda+ = **{mp_info['lambda_plus']:.4f}**"
        )

        # Marchenko-Pastur eigenvalue plot
        st.markdown("#### Marchenko-Pastur Sınıri & Oz Değer Spektrumu")
        eig_colors = [
            "#a78bfa" if v > mp_info["lambda_plus"] else "#6b7280"
            for v in eigs_corr
        ]
        fig_mp = go.Figure()
        fig_mp.add_trace(go.Scatter(
            x=list(range(1, n_assets_cov + 1)), y=list(eigs_corr),
            mode="markers",
            marker=dict(color=eig_colors, size=12, symbol="circle"),
            name="Oz Değerler",
        ))
        fig_mp.add_hline(
            y=mp_info["lambda_plus"], line_dash="dash", line_color="#fbbf24", line_width=2,
            annotation_text=f"lambda+ = {mp_info['lambda_plus']:.3f} (MP ust sınır)",
            annotation_position="top right", annotation_font_color="#fbbf24",
        )
        fig_mp.add_hline(
            y=mp_info["lambda_minus"], line_dash="dot", line_color="#94a3b8", line_width=1.2,
            annotation_text=f"lambda- = {mp_info['lambda_minus']:.3f}",
            annotation_position="bottom right", annotation_font_color="#94a3b8",
        )
        fig_mp.update_layout(
            template=PLOT_TEMPLATE, height=380,
            title=f"Korelasyon Matrisi Oz Değerleri -- {n_signal} sinyal (mor), {n_noise} gürültü (gri)",
            xaxis_title="Oz Değer Sirasi", yaxis_title="Oz Değer Büyüklugu",
            margin=dict(l=20, r=20, t=55, b=35),
        )
        st.plotly_chart(fig_mp, use_container_width=True)

        # Heatmaps
        st.markdown("#### Kovaryans Matrisi Isi Haritalari (Korelasyon Formunda)")
        corr_sample = _cov_to_corr(sample_cov)
        corr_poet = _cov_to_corr(poet_cov_mat)
        corr_lw = _cov_to_corr(lw_cov_mat)

        fig_heat = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Ornek Kovaryans", "POET", "Ledoit-Wolf"],
            horizontal_spacing=0.04,
        )
        for col_i, mat in enumerate([corr_sample, corr_poet, corr_lw], start=1):
            fig_heat.add_trace(
                go.Heatmap(
                    z=np.round(mat, 4), x=cols, y=cols,
                    colorscale="RdBu_r", zmin=-1, zmax=1,
                    showscale=(col_i == 3),
                    colorbar=dict(x=1.02, title="Cor") if col_i == 3 else None,
                ),
                row=1, col=col_i,
            )
        fig_heat.update_layout(
            template=PLOT_TEMPLATE, height=380,
            title="Korelasyon Matrisi Karşılaştırması",
            margin=dict(l=20, r=20, t=60, b=30),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Condition number & Frobenius distance
        st.markdown("**Matris Kalite Gostergeleri**")
        rows_cond = []
        for mname, cov_m in [("Ornek", sample_cov), ("POET", poet_cov_mat), ("Ledoit-Wolf", lw_cov_mat)]:
            cond = np.linalg.cond(cov_m)
            frob = np.linalg.norm(cov_m - sample_cov, "fro") if mname != "Ornek" else 0.0
            rows_cond.append({
                "Tahminci": mname,
                "Kosul Sayisi": f"{cond:.2f}",
                "Frobenius (ornekten uzaklik)": f"{frob:.6f}",
            })
        st.dataframe(pd.DataFrame(rows_cond), use_container_width=True, hide_index=True)

        # LW eigenvalue shrinkage visualization
        st.markdown("**Ledoit-Wolf Buzulme: Oz Değer Karşılaştırması**")
        eigs_sample_cov = np.sort(np.linalg.eigvalsh(sample_cov))[::-1]
        eigs_lw_cov = np.sort(np.linalg.eigvalsh(lw_cov_mat))[::-1]
        idx_r = list(range(1, n_assets_cov + 1))
        fig_shrink = go.Figure()
        fig_shrink.add_trace(go.Bar(
            x=idx_r, y=list(eigs_sample_cov),
            name="Ornek Oz Değerleri", marker_color=COLORS[3], opacity=0.7,
        ))
        fig_shrink.add_trace(go.Bar(
            x=idx_r, y=list(eigs_lw_cov),
            name="LW Buzulmus Oz Değerleri", marker_color=COLORS[0], opacity=0.85,
        ))
        fig_shrink.update_layout(
            template=PLOT_TEMPLATE, height=280,
            title=f"Oz Değer Buzulmesi -- delta = {shrinkage:.4f}",
            xaxis_title="Oz Değer Sirasi", yaxis_title="Büyükluk",
            barmode="group", margin=dict(l=20, r=20, t=50, b=30),
        )
        st.plotly_chart(fig_shrink, use_container_width=True)

    # =========================================================================
    # TAB 6: Entegre Analiz: HAR + POET MVP
    # =========================================================================
    with tab_integrated:
        st.markdown("### Entegre Analiz: HAR-RV + Büyük Boyut Kovaryans + MVP Portföy")
        df = st.session_state.returns_df
        ac = _asset_cols(df)
        max_n6 = min(len(ac), 8)
        df_key = id(df)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            n_int = st.slider("Varlık Sayisi", 3, max_n6, min(5, max_n6), key="int_n")
        with col_b:
            har_choice = st.selectbox("HAR Modeli", ["HAR-RV", "HAR-RV-J"], key="int_har")
        with col_c:
            cov_choice = st.selectbox("Kovaryans", ["POET", "Ledoit-Wolf", "Ornek"], key="int_cov")

        int_assets = ac[:n_int]

        # Step cards
        st.markdown(_step_card(1, "HAR-RV ile Volatilite Tahmini",
            f"Her varlık için {har_choice} modeli tahmin edilir; R2, beta_d, beta_w, beta_m gosterilir."),
            unsafe_allow_html=True)
        st.markdown(_step_card(2, "Kovaryans Matrisi Tahmini",
            f"Secili yöntem: {cov_choice}. Günlük getiriler kullanilir."),
            unsafe_allow_html=True)
        st.markdown(_step_card(3, "MVP Ağırlıkları",
            "w = Sigma^-1 * 1 / (1' * Sigma^-1 * 1) -- minimum varyans portföy."),
            unsafe_allow_html=True)
        st.markdown(_step_card(4, "Portföy VaR/ES",
            "Portföy getirilerinden normal VaR ve ES hesaplanır."),
            unsafe_allow_html=True)
        st.markdown(_step_card(5, "Geriye Donuk Değerlendirme",
            "Eşit ağırlık, Ornek-MVP, POET-MVP, LW-MVP portföyleri karşılaştırılır."),
            unsafe_allow_html=True)

        st.divider()

        # Step 1: HAR summaries
        st.markdown("#### Adım 1 -- HAR-RV Özet")
        har_rows = []
        with st.spinner("HAR modelleri tahmin ediliyor..."):
            for ast in int_assets:
                try:
                    res_h, _, _, _ = _cached_har(har_choice, ast, df_key)
                    p = res_h.params
                    har_rows.append({
                        "Varlık": ast,
                        "R2": f"{res_h.rsquared:.4f}",
                        "beta_0 (sabit)": f"{float(p.iloc[0]):.6f}",
                        "beta_d (günlük)": f"{float(p.iloc[1]):.4f}",
                        "beta_w (haftalik)": f"{float(p.iloc[2]):.4f}",
                        "beta_m (aylik)": f"{float(p.iloc[3]):.4f}" if len(p) > 3 else "-",
                    })
                except Exception as e:
                    har_rows.append({"Varlık": ast, "R2": f"Hata: {e}"})
        st.dataframe(pd.DataFrame(har_rows), use_container_width=True, hide_index=True)

        # Step 2-3: Covariance + MVP weights
        st.markdown("#### Adım 2-3 -- Kovaryans & MVP Ağırlıkları")
        k_int = min(3, n_int - 1)
        with st.spinner("Kovaryans & MVP ağırlıklar..."):
            (
                sample_cov_i, poet_cov_i, lw_cov_i,
                shrink_i, cols_i, returns_sub_i, _,
            ) = _cached_covariance(n_int, k_int, 0.10, df_key)

        cov_map = {
            "POET": poet_cov_i,
            "Ledoit-Wolf": lw_cov_i,
            "Ornek": sample_cov_i,
        }
        selected_cov_mat = cov_map[cov_choice]
        ones = np.ones(n_int)
        reg = 1e-8 * np.eye(n_int)
        H_inv = np.linalg.inv(selected_cov_mat + reg)
        w_mvp = H_inv @ ones / (ones @ H_inv @ ones)

        fig_w = go.Figure(go.Bar(
            x=int_assets, y=list(w_mvp),
            marker_color=[COLORS[i % len(COLORS)] for i in range(n_int)],
        ))
        fig_w.add_hline(
            y=1.0 / n_int, line_dash="dot", line_color="#94a3b8",
            annotation_text="Eşit ağırlık", annotation_font_color="#94a3b8"
        )
        fig_w.update_layout(
            template=PLOT_TEMPLATE, height=280,
            title=f"MVP Ağırlıkları -- {cov_choice}",
            xaxis_title="Varlık", yaxis_title="Ağırlık",
            margin=dict(l=20, r=20, t=50, b=30),
        )
        st.plotly_chart(fig_w, use_container_width=True)

        # Step 4-5: Portfolio comparison
        st.markdown("#### Adım 4-5 -- Portföy Performans Karşılaştırması")

        def _port_metrics(w_arr, ret_mat):
            pr = ret_mat @ w_arr
            ann_vol = float(np.std(pr) * np.sqrt(252) * 100)
            sharpe = float(np.mean(pr) / np.std(pr) * np.sqrt(252)) if np.std(pr) > 0 else float("nan")
            cum = np.cumprod(1 + pr)
            roll_max = np.maximum.accumulate(cum)
            dd = (cum - roll_max) / np.where(roll_max > 0, roll_max, 1)
            max_dd = float(dd.min()) * 100
            mu_p = float(np.mean(pr))
            sig_p = float(np.std(pr))
            var95 = -(mu_p - 1.6449 * sig_p) * 100
            es95 = -(mu_p - sig_p * _sp_norm.pdf(_sp_norm.ppf(0.05)) / 0.05) * 100
            return {
                "Ann. Vol (%)": f"{ann_vol:.2f}",
                "Sharpe": f"{sharpe:.3f}" if sharpe == sharpe else "-",
                "Max DD (%)": f"{max_dd:.2f}",
                "VaR 95% (%)": f"{var95:.3f}",
                "ES 95% (%)": f"{es95:.3f}",
            }

        rets_np = returns_sub_i.values
        port_rows = []
        port_series_dict = {}
        w_eq = np.full(n_int, 1.0 / n_int)

        for pname, pcov in [
            ("Eşit Ağırlık", None),
            ("Ornek-MVP",    sample_cov_i),
            ("POET-MVP",     poet_cov_i),
            ("LW-MVP",       lw_cov_i),
        ]:
            if pcov is None:
                w_p = w_eq.copy()
            else:
                try:
                    Hi = np.linalg.inv(pcov + reg)
                    w_p = Hi @ ones / (ones @ Hi @ ones)
                except np.linalg.LinAlgError:
                    w_p = w_eq.copy()
            m_metrics = _port_metrics(w_p, rets_np)
            port_rows.append({"Portföy": pname, **m_metrics})
            port_series_dict[pname] = rets_np @ w_p

        st.dataframe(pd.DataFrame(port_rows), use_container_width=True, hide_index=True)

        # Cumulative return chart
        PORT_COLORS = {
            "Eşit Ağırlık": "#94a3b8",
            "Ornek-MVP":    COLORS[3],
            "POET-MVP":     COLORS[0],
            "LW-MVP":       COLORS[1],
        }
        fig_cum = go.Figure()
        idx_ret = returns_sub_i.index
        for pname, pret in port_series_dict.items():
            cum_ret = np.cumprod(1 + pret)
            fig_cum.add_trace(go.Scatter(
                x=idx_ret, y=list(cum_ret),
                mode="lines", name=pname,
                line=dict(color=PORT_COLORS.get(pname, "#fff"), width=1.8),
            ))
        fig_cum.update_layout(
            template=PLOT_TEMPLATE, height=380,
            title="Kumulatif Getiri Karşılaştırması (Statik MVP Ağırlıkları)",
            xaxis_title="Tarih", yaxis_title="Kumulatif Getiri (1 = baslangic)",
            margin=dict(l=20, r=20, t=55, b=35),
            legend=dict(orientation="h", y=-0.22),
        )
        st.plotly_chart(fig_cum, use_container_width=True)
        st.caption(
            "Not: Ağırlıklar statik olup tum dönem kovaryansina dayanir. "
            "Gerçekci bir backtest için kayan pencere yeniden dengeleme gereklidir."
        )
