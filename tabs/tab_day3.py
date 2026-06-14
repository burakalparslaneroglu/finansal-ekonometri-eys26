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
COLORS = [
    "#a78bfa", "#34d399", "#f472b6", "#60a5fa",
    "#fbbf24", "#fb923c", "#38bdf8", "#4ade80",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _asset_cols(df: pd.DataFrame):
    """Return only return columns (exclude _RV and _BPV suffixes)."""
    return [c for c in df.columns if not c.endswith("_RV") and not c.endswith("_BPV")]


def _df_hash() -> str:
    """Return a short, stable string key derived from the active DataFrame."""
    return str(hash(st.session_state.returns_df.to_csv()))[:8]


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
    dict with keys: stats, rho_series, index, sample_mean_corr, dcc_rho_series
    """
    from dcc_garch import DCCGarch

    df = st.session_state.returns_df
    cols = list(asset_tuple)
    returns = df[cols]

    deco = DCCGarch(model_type="DECO")
    std_resid = deco.fit_univariate_garch(returns)
    deco.fit_dcc(std_resid)
    stats = deco.get_summary_stats()
    rho_series = deco.get_equicorrelation_series()

    corr_mat = returns.corr().values
    n = len(cols)
    idx_upper = np.triu_indices(n, k=1)
    sample_mean_corr = float(np.mean(corr_mat[idx_upper]))

    dcc = DCCGarch(model_type="DCC")
    dcc.fit_univariate_garch(returns)
    dcc.fit_dcc(std_resid)
    dcc_pairs = np.triu_indices(n, k=1)
    dcc_mean_corr = np.array([
        np.mean(dcc.R_seq[t][dcc_pairs]) for t in range(dcc.R_seq.shape[0])
    ])

    return {
        "stats": stats,
        "rho_series": rho_series,
        "index": returns.index,
        "sample_mean_corr": sample_mean_corr,
        "dcc_rho_series": dcc_mean_corr,
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

    for mtype in ("DCC", "cDCC", "DECO"):
        dcc = DCCGarch(model_type=mtype)
        std_resid = dcc.fit_univariate_garch(returns)
        dcc.fit_dcc(std_resid)
        results[mtype] = dcc.get_summary_stats()

    return results


@st.cache_data(show_spinner=False)
def _run_mvp(asset_tuple: tuple, model_type: str, df_hash: str) -> dict:
    """
    Fit a DCC-family model and compute MVP weights.

    Parameters
    ----------
    asset_tuple : tuple of str
        Column names to include.
    model_type : str
        One of "DCC", "cDCC", "ADCC", "DECO".
    df_hash : str
        Short string hash of the DataFrame (cache key discriminator).

    Returns
    -------
    dict with keys: weights, H_seq, sigmas, index, cols, stats, returns
    """
    from dcc_garch import DCCGarch

    df = st.session_state.returns_df
    cols = list(asset_tuple)
    returns = df[cols]

    dcc = DCCGarch(model_type=model_type)
    std_resid = dcc.fit_univariate_garch(returns)
    dcc.fit_dcc(std_resid)
    weights = dcc.compute_mvp_weights()
    stats = dcc.get_summary_stats()

    return {
        "weights": weights,
        "H_seq": dcc.H_seq,
        "sigmas": dcc.sigmas,
        "index": returns.index,
        "cols": cols,
        "stats": stats,
        "returns": returns.values,
    }


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

    tab_theory, tab_model, tab_deco, tab_compare, tab_portfolio = st.tabs([
        "\U0001f4d6 Teori",
        "⚙️ Model Tahmini",
        "\U0001f504 DECO Modeli",
        "\U0001f4ca Model Karşılaştırması",
        "\U0001f4bc MVP Portföy",
    ])

    # =========================================================================
    # TAB 1 -- THEORY
    # =========================================================================
    with tab_theory:
        st.markdown("### DCC-GARCH Ailesi: Teorik Özet")

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("DCC (Engle 2002)", expanded=True):
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
            with st.expander("cDCC (Aielli 2013)", expanded=True):
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
            with st.expander("ADCC (Cappiello et al. 2006)", expanded=True):
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
            with st.expander("DECO (Engle-Kelly 2012)", expanded=True):
                st.markdown(r"""
**Ekikorelasyon (equicorrelation):**
$$\rho_t = \frac{2}{N(N-1)}\sum_{i<j} R_{ij,t}^{\text{DCC}}$$

**DECO korelasyon matrisi:**
$$R_t^{\text{DECO}} = (1-\rho_t)I_N + \rho_t\,\mathbf{1}_N\mathbf{1}_N^\top$$

**Algoritma:**
1. DCC ile $R_{ij,t}$ dizisini tahmin et
2. Her $t$ için $\rho_t$ hesapla
3. $R_t^{\text{DECO}}$ oluştur

**Avantaj:** $N>50$'de $O(1)$ parametre.
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
| Model | Param. | Avantajlar | Dezavantajlar | Büyük N |
|-------|--------|------------|---------------|---------|
| **DCC** | a, b | Basit, yorumlanabilir | $\bar{Q}$ tutarsız | Orta ($N<50$) |
| **cDCC** | a, b | Tutarlı $\bar{Q}$ | Daha yavaş | Orta ($N<50$) |
| **ADCC** | a, b, c | Asimetriyi yakalar | Kısıt karmasık | Orta ($N<30$) |
| **DECO** | a, b | $O(1)$ param., hızlı | Bilgi kaybi | Büyük ($N>50$) |
""")

    # =========================================================================
    # TAB 2 -- MODEL ESTIMATION
    # =========================================================================
    with tab_model:
        st.markdown("### DCC / cDCC / ADCC Model Tahmini")

        col_a, col_b = st.columns(2)
        with col_a:
            selected_assets = st.multiselect(
                "Varlıklar",
                asset_cols,
                default=asset_cols[:3],
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
            st.stop()

        pairs = [
            f"{selected_assets[i]} vs {selected_assets[j]}"
            for i in range(len(selected_assets))
            for j in range(i + 1, len(selected_assets))
        ]
        selected_pair = st.selectbox("Korelasyon Çifti Grafigi", pairs, key="t2_pair")

        with st.spinner(f"{model_type} modeli tahmin ediliyor..."):
            try:
                result = _run_dcc_model(tuple(selected_assets), model_type, dh)
            except Exception as exc:
                st.error(f"Model tahmini başarısız oldu. Farklı varlık sayisi deneyin.\n\n`{exc}`")
                st.stop()

        stats = result["stats"]
        params = result["params"]
        corr_series = result["corr_series"]
        sigmas = result["sigmas"]
        idx = result["index"]
        cols = result["cols"]

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
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(
            x=idx, y=pair_data,
            mode="lines",
            name=selected_pair,
            line=dict(color=COLORS[0], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(167,139,250,0.10)",
        ))
        fig_corr.add_hline(
            y=float(np.mean(pair_data)) if len(pair_data) > 0 else 0,
            line_dash="dot",
            line_color="#fbbf24",
            annotation_text="Zaman ortalaması",
            annotation_position="bottom right",
        )
        fig_corr.update_layout(
            template=PLOT_TEMPLATE,
            title=f"Dinamik Koşullu Korelasyon -- {selected_pair} ({model_type})",
            xaxis_title="Tarih",
            yaxis_title="Korelasyon rho",
            height=380,
            margin=dict(l=20, r=20, t=50, b=30),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        n_sel = len(cols)
        fig_vol = make_subplots(rows=1, cols=n_sel, subplot_titles=cols, shared_yaxes=False)
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

        with st.expander("Tahmin Detayları"):
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

    # =========================================================================
    # TAB 3 -- DECO MODEL
    # =========================================================================
    with tab_deco:
        st.markdown("### DECO -- Dinamik Ekikorelasyon Modeli")

        st.info(
            "DECO, N>50 portföylerde DCC'yi asan hız avantajı sunar: "
            "O(1) vs O(N^2) parametre. Tum çiftlerin ortalaması olan "
            "tek bir rho_t serisi tahmin edilir."
        )

        col_da, col_db = st.columns(2)
        with col_da:
            deco_assets = st.multiselect(
                "Varlıklar (DECO)",
                asset_cols,
                default=asset_cols,
                key="t3_assets",
            )
        with col_db:
            show_dcc_compare = st.checkbox(
                "DCC ortalama korelasyonuyla karşılaştır",
                value=True,
                key="t3_compare",
            )

        if len(deco_assets) < 2:
            st.warning("En az 2 varlık seçiniz.")
            st.stop()

        with st.spinner("DECO modeli tahmin ediliyor..."):
            try:
                deco_res = _run_deco_model(tuple(deco_assets), dh)
            except Exception as exc:
                st.error(f"Model tahmini başarısız oldu. Farklı varlık sayisi deneyin.\n\n`{exc}`")
                st.stop()

        rho = deco_res["rho_series"]
        deco_idx = deco_res["index"]
        sample_mean = deco_res["sample_mean_corr"]
        dcc_mean_corr = deco_res["dcc_rho_series"]
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
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(
                x=deco_idx, y=rho,
                mode="lines",
                name="DECO rho_t",
                line=dict(color=COLORS[0], width=1.4),
            ))
            fig_cmp.add_trace(go.Scatter(
                x=deco_idx, y=dcc_mean_corr,
                mode="lines",
                name="DCC ortalama rho_t",
                line=dict(color=COLORS[2], width=1.4, dash="dash"),
            ))
            fig_cmp.update_layout(
                template=PLOT_TEMPLATE,
                title="DECO vs DCC Ortalama Korelasyon Karşılaştırması",
                xaxis_title="Tarih",
                yaxis_title="Ortalama Korelasyon",
                height=280,
                margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)

        st.markdown("#### DECO Parametre Özeti")
        deco_a = deco_stats["alpha"]
        deco_b = deco_stats["beta"]
        deco_p = deco_stats["persistence"]
        deco_hl = deco_stats["half_life_days"]
        deco_mc = deco_stats["mean_corr"]

        deco_table = pd.DataFrame([{
            "Model": "DECO",
            "alpha": f"{deco_a:.4f}",
            "beta": f"{deco_b:.4f}",
            "alpha+beta": f"{deco_p:.4f}",
            "Yarı-Ömür (gün)": f"{deco_hl:.1f}",
            "Ort. Ekikorelasyon": f"{deco_mc:.4f}",
            "Ornek Ort. Korelasyon": f"{sample_mean:.4f}",
        }])
        st.dataframe(deco_table, use_container_width=True, hide_index=True)

        st.markdown("""
**Temel Bulgular:**
- N buyudukce DECO'nun hesaplama avantajı belirginlesir.
- rho_t'nin yüksek oldugu dönemler piyasa krizlerine işaret eder.
- DECO, tek bir skaler korelasyon varsayımı yaptıgından portföy
  çeşitlendirme stratejilerinde muhafazakar bir alt sınır sunar.
""")

    # =========================================================================
    # TAB 4 -- MODEL COMPARISON
    # =========================================================================
    with tab_compare:
        st.markdown("### Model Karşılaştırması: DCC, cDCC, DECO")
        st.caption("Hız için sabit 3 varlık önerilir; aşağıdan değiştirebilirsiniz.")

        cmp_assets = st.multiselect(
            "Karşılaştırma İçin Varlıklar (önerilen: 3)",
            asset_cols,
            default=asset_cols[:3],
            key="t4_assets",
        )

        if len(cmp_assets) < 2:
            st.warning("En az 2 varlık seçiniz.")
            st.stop()

        with st.spinner("DCC, cDCC ve DECO tahmin ediliyor..."):
            try:
                cmp_results = _run_comparison(tuple(cmp_assets), dh)
            except Exception as exc:
                st.error(f"Model tahmini başarısız oldu. Farklı varlık sayisi deneyin.\n\n`{exc}`")
                st.stop()

        rows = []
        for mtype, s in cmp_results.items():
            rows.append({
                "Model": mtype,
                "alpha": round(s["alpha"], 4),
                "beta": round(s["beta"], 4),
                "alpha+beta": round(s["persistence"], 4),
                "Yarı-Ömür (gün)": round(s["half_life_days"], 1),
                "Ort. Korelasyon": round(s["mean_corr"], 4),
            })

        df_cmp = pd.DataFrame(rows)
        st.markdown("#### Parametre Karşılaştırması")

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

    # =========================================================================
    # TAB 5 -- MVP PORTFOLIO
    # =========================================================================
    with tab_portfolio:
        st.markdown("### Minimum Varyans Portföy (MVP) -- DCC Tabanlı")
        st.markdown(r"""
$$w_t = \frac{H_t^{-1}\mathbf{1}}{\mathbf{1}^\top H_t^{-1}\mathbf{1}}$$
""")

        col_pa, col_pb = st.columns(2)
        with col_pa:
            mvp_assets = st.multiselect(
                "Varlıklar (2-8)",
                asset_cols,
                default=asset_cols[:4],
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
            st.stop()

        with st.spinner(f"MVP agırlıkları hesaplanıyor ({mvp_model})..."):
            try:
                mvp_res = _run_mvp(tuple(mvp_assets), mvp_model, dh)
            except Exception as exc:
                st.error(f"Model tahmini başarısız oldu. Farklı varlık sayisi deneyin.\n\n`{exc}`")
                st.stop()

        weights = mvp_res["weights"]
        H_seq = mvp_res["H_seq"]
        mvp_idx = mvp_res["index"]
        mvp_cols = mvp_res["cols"]
        mvp_ret = mvp_res["returns"]
        T_mvp, N_mvp = weights.shape

        fig_w = go.Figure()
        for i, col in enumerate(mvp_cols):
            fig_w.add_trace(go.Scatter(
                x=mvp_idx,
                y=weights[:, i],
                mode="lines",
                name=col,
                stackgroup="one",
                line=dict(color=COLORS[i % len(COLORS)], width=0.5),
                fillcolor=COLORS[i % len(COLORS)],
            ))
        fig_w.update_layout(
            template=PLOT_TEMPLATE,
            title=f"Dinamik MVP Agırlıkları -- {mvp_model}",
            xaxis_title="Tarih",
            yaxis_title="Ağırlık",
            height=380,
            margin=dict(l=20, r=20, t=50, b=30),
            yaxis=dict(tickformat=".0%"),
        )
        st.plotly_chart(fig_w, use_container_width=True)

        port_vol_mvp = np.array([
            np.sqrt(float(weights[t] @ H_seq[t] @ weights[t]))
            for t in range(T_mvp)
        ])
        ew = np.full(N_mvp, 1.0 / N_mvp)
        port_vol_ew = np.array([
            np.sqrt(float(ew @ H_seq[t] @ ew))
            for t in range(T_mvp)
        ])

        fig_pv = go.Figure()
        fig_pv.add_trace(go.Scatter(
            x=mvp_idx,
            y=port_vol_mvp * 100,
            mode="lines",
            name="MVP",
            line=dict(color=COLORS[0], width=1.5),
        ))
        fig_pv.add_trace(go.Scatter(
            x=mvp_idx,
            y=port_vol_ew * 100,
            mode="lines",
            name="Esit Ağırlık",
            line=dict(color=COLORS[1], width=1.5, dash="dash"),
        ))
        fig_pv.update_layout(
            template=PLOT_TEMPLATE,
            title=f"Portföy Oynaklığı (%/gun) -- MVP vs Esit Ağırlık ({mvp_model})",
            xaxis_title="Tarih",
            yaxis_title="Portföy Volatilitesi (%)",
            height=280,
            margin=dict(l=20, r=20, t=50, b=30),
        )
        st.plotly_chart(fig_pv, use_container_width=True)

        ann = np.sqrt(252)
        mvp_ann_vol = float(np.mean(port_vol_mvp)) * ann * 100
        ew_ann_vol = float(np.mean(port_vol_ew)) * ann * 100
        max_weight = float(np.max(np.mean(weights, axis=0)))
        turnover = float(np.mean(np.sum(np.abs(np.diff(weights, axis=0)), axis=1)))

        mvp_daily_ret = np.einsum("tn,tn->t", weights[:-1], mvp_ret[1:])
        ew_daily_ret = mvp_ret[1:] @ ew

        mvp_sharpe = (
            float(np.mean(mvp_daily_ret) / np.std(mvp_daily_ret)) * ann
            if np.std(mvp_daily_ret) > 0 else float("nan")
        )
        ew_sharpe = (
            float(np.mean(ew_daily_ret) / np.std(ew_daily_ret)) * ann
            if np.std(ew_daily_ret) > 0 else float("nan")
        )

        metric_defs_mvp = [
            ("MVP Yıllık Vol. (%)", f"{mvp_ann_vol:.2f}"),
            ("EW Yıllık Vol. (%)", f"{ew_ann_vol:.2f}"),
            ("Maks. Ort. Ağırlık", f"{max_weight:.3f}"),
            ("Günlük Devir", f"{turnover:.4f}"),
        ]
        m_cols = st.columns(len(metric_defs_mvp))
        for mc, (lbl, val) in zip(m_cols, metric_defs_mvp):
            with mc:
                st.markdown(_metric_card(lbl, val), unsafe_allow_html=True)

        st.markdown("")

        st.markdown("#### MVP vs Esit Ağırlık Portföy Karşılaştırması")
        cmp_df = pd.DataFrame([
            {
                "Portföy": "Minimum Varyans (MVP)",
                "Yıllık Vol. (%)": f"{mvp_ann_vol:.2f}",
                "Sharpe Orani": f"{mvp_sharpe:.3f}" if not np.isnan(mvp_sharpe) else "---",
            },
            {
                "Portföy": "Esit Ağırlık (1/N)",
                "Yıllık Vol. (%)": f"{ew_ann_vol:.2f}",
                "Sharpe Orani": f"{ew_sharpe:.3f}" if not np.isnan(ew_sharpe) else "---",
            },
        ])
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        with st.expander("Ağırlık Özet İstatistikleri"):
            df_w_desc = pd.DataFrame(weights, index=mvp_idx, columns=mvp_cols)
            st.dataframe(
                df_w_desc.describe().round(4).style.background_gradient(cmap="RdPu"),
                use_container_width=True,
            )
