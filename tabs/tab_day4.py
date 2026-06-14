"""
4. Gün — Risk Ölçütleri: VaR, ES, PELVE, EVT, Sistemik Risk, Tahmin Kombinasyonu & Geriye Dönük Test
EYS'26 — Pamukkale Üniversitesi
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, genpareto
import warnings

PLOT_TEMPLATE = "plotly_dark"
COLORS = ["#a78bfa", "#34d399", "#f472b6", "#60a5fa", "#fbbf24", "#fb923c"]
COLORS_METHOD = {
    "Normal":         "#a78bfa",
    "Student-t":      "#34d399",
    "Hist.Sim.":      "#f472b6",
    "Cornish-Fisher": "#fbbf24",
}


def _asset_cols(df):
    return [c for c in df.columns if not c.endswith("_RV") and not c.endswith("_BPV")]


def _cf_es(r_win, alpha):
    """Approximate ES for Cornish-Fisher: Normal ES same mu/sigma."""
    z_n = norm.ppf(1 - alpha)
    return float(np.mean(-r_win) + np.std(r_win, ddof=1) * (norm.pdf(z_n) / alpha))


# ─── CACHED COMPUTATIONS ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_rolling_risk(asset, alpha, window, df_key):
    from risk_metrics import (calculate_var_es, calculate_pelve_single,
                               calculate_cornish_fisher_var)
    df = st.session_state.returns_df
    returns = df[asset].dropna().values
    n = len(returns)
    T = n - window
    var_n  = np.empty(T); es_n  = np.empty(T)
    var_t  = np.empty(T); es_t  = np.empty(T)
    var_hs = np.empty(T); es_hs = np.empty(T)
    var_cf = np.empty(T); es_cf = np.empty(T)
    pelve  = np.empty(T)
    for i in range(T):
        r_win = returns[i: i + window]
        var_n[i],  es_n[i]  = calculate_var_es(r_win, alpha, "parametric_normal")
        var_t[i],  es_t[i]  = calculate_var_es(r_win, alpha, "parametric_student_t")
        var_hs[i], es_hs[i] = calculate_var_es(r_win, alpha, "historical")
        cf_v, _, _, _       = calculate_cornish_fisher_var(r_win, alpha)
        var_cf[i] = cf_v
        es_cf[i]  = _cf_es(r_win, alpha)
        pelve[i]  = calculate_pelve_single(-r_win, alpha)
    idx = df.index[window: window + T]
    return {
        "idx": idx,
        "returns": returns[window: window + T],
        "var": {"Normal": var_n, "Student-t": var_t, "Hist.Sim.": var_hs, "Cornish-Fisher": var_cf},
        "es":  {"Normal": es_n,  "Student-t": es_t,  "Hist.Sim.": es_hs,  "Cornish-Fisher": es_cf},
        "pelve": pelve,
    }


@st.cache_data(show_spinner=False)
def compute_static_risk(asset, alpha, df_key):
    from risk_metrics import (calculate_var_es, calculate_pelve_single,
                               calculate_cornish_fisher_var)
    df = st.session_state.returns_df
    returns = df[asset].dropna().values
    rows = []
    for label, method in [("Normal", "parametric_normal"),
                           ("Student-t", "parametric_student_t"),
                           ("Hist.Sim.", "historical")]:
        v, e = calculate_var_es(returns, alpha, method)
        rows.append({"Yontem": label, "VaR": v, "ES": e})
    cf_v, _, _, _ = calculate_cornish_fisher_var(returns, alpha)
    cf_e = _cf_es(returns, alpha)
    rows.append({"Yontem": "Cornish-Fisher", "VaR": cf_v, "ES": cf_e})
    pelve_val = calculate_pelve_single(-returns, alpha)
    return pd.DataFrame(rows), pelve_val


@st.cache_data(show_spinner=False)
def compute_evt(asset, alpha, thresh_q, df_key):
    from risk_metrics import calculate_evt_var_es, calculate_var_es
    df = st.session_state.returns_df
    returns = df[asset].dropna().values
    losses = -returns
    evt = calculate_evt_var_es(losses, alpha=alpha, threshold_quantile=thresh_q)
    v_norm, _ = calculate_var_es(returns, alpha, "parametric_normal")
    # Mean excess plot data (60th to 98th quantile)
    u_quantiles = np.linspace(0.60, 0.98, 40)
    u_vals = np.quantile(losses, u_quantiles)
    me_vals = []
    for uu in u_vals:
        exc = losses[losses > uu] - uu
        me_vals.append(float(np.mean(exc)) if len(exc) >= 5 else np.nan)
    # GPD fit data
    exc_data = evt.get("exceedances", np.array([]))
    gpd_x = gpd_emp = gpd_theo = np.array([])
    if len(exc_data) > 0:
        exc_sorted = np.sort(exc_data)
        n_exc = len(exc_sorted)
        gpd_emp = np.arange(1, n_exc + 1) / n_exc
        xi_fit = float(evt["xi"]) if not np.isnan(evt["xi"]) else 0.0
        sig_fit = float(evt["sigma"]) if not np.isnan(evt["sigma"]) else 1e-6
        gpd_theo = genpareto.cdf(exc_sorted, xi_fit, scale=sig_fit, loc=0)
        gpd_x = exc_sorted
    return evt, v_norm, u_vals, me_vals, gpd_x, gpd_emp, gpd_theo


@st.cache_data(show_spinner=False)
def compute_systemic(asset_i, market_proxy, alpha, df_key):
    from risk_metrics import calculate_covar, calculate_mes
    df = st.session_state.returns_df
    ret_i = df[asset_i].dropna().values
    ret_m = df[market_proxy].dropna().values
    n = min(len(ret_i), len(ret_m))
    ret_i = ret_i[-n:]; ret_m = ret_m[-n:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        covar_res = calculate_covar(ret_i, ret_m, alpha=alpha)
        mes_res   = calculate_mes(ret_i, ret_m, alpha=alpha)
    return ret_i, ret_m, covar_res, mes_res


@st.cache_data(show_spinner=False)
def compute_rolling_covar(asset_i, market_proxy, alpha, roll_win, df_key):
    from risk_metrics import calculate_covar
    df = st.session_state.returns_df
    ret_i = df[asset_i].dropna().values
    ret_m = df[market_proxy].dropna().values
    n = min(len(ret_i), len(ret_m))
    ret_i = ret_i[-n:]; ret_m = ret_m[-n:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rolling_df = calculate_covar(ret_i, ret_m, alpha=alpha, rolling_window=roll_win)
    idx = df.index[-n + roll_win - 1:]
    rolling_df.index = idx[:len(rolling_df)]
    return rolling_df


@st.cache_data(show_spinner=False)
def compute_fz_comparison(asset, alpha, n_oos, df_key):
    from risk_metrics import (calculate_var_es, calculate_cornish_fisher_var,
                               fissler_ziegel_loss)
    df = st.session_state.returns_df
    returns = df[asset].dropna().values
    n = len(returns)
    if n_oos >= n:
        n_oos = n // 2
    oos_returns = returns[-n_oos:]
    train_win = max(60, min(250, n - n_oos))
    start_idx = n - n_oos
    labels4 = ["Normal", "Student-t", "Hist.Sim.", "Cornish-Fisher"]
    var_fcs = {m: [] for m in labels4}
    es_fcs  = {m: [] for m in labels4}
    for i in range(n_oos):
        cur = start_idx + i
        lo  = max(0, cur - train_win)
        r_win = returns[lo:cur] if cur > 0 else returns[:1]
        for label, method in [("Normal", "parametric_normal"),
                               ("Student-t", "parametric_student_t"),
                               ("Hist.Sim.", "historical")]:
            v, e = calculate_var_es(r_win, alpha, method)
            var_fcs[label].append(v)
            es_fcs[label].append(e)
        cf_v, _, _, _ = calculate_cornish_fisher_var(r_win, alpha)
        var_fcs["Cornish-Fisher"].append(cf_v)
        es_fcs["Cornish-Fisher"].append(_cf_es(r_win, alpha))
    for m in labels4:
        var_fcs[m] = np.array(var_fcs[m])
        es_fcs[m]  = np.array(es_fcs[m])
    # GARCH-Normal
    garch_var = np.full(n_oos, np.nan)
    garch_es  = np.full(n_oos, np.nan)
    try:
        from arch import arch_model as _arch_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm = _arch_model(returns * 100, vol="Garch", p=1, q=1, dist="normal")
            gr = gm.fit(disp="off")
        cond_vol = gr.conditional_volatility / 100
        garch_var = cond_vol[-n_oos:] * norm.ppf(1 - alpha)
        garch_es  = cond_vol[-n_oos:] * norm.pdf(norm.ppf(1 - alpha)) / alpha
    except Exception:
        pass
    var_fcs["GARCH-Normal"] = garch_var
    es_fcs["GARCH-Normal"]  = garch_es
    fz_losses = {}; var_means = {}; es_means = {}
    for m in var_fcs:
        v_arr = var_fcs[m]; e_arr = es_fcs[m]
        mask = ~np.isnan(v_arr) & ~np.isnan(e_arr)
        if mask.sum() < 10:
            fz_losses[m] = np.nan; var_means[m] = np.nan; es_means[m] = np.nan
        else:
            fz_losses[m] = fissler_ziegel_loss(oos_returns[mask], v_arr[mask], e_arr[mask], alpha)
            var_means[m] = float(np.mean(v_arr[mask]))
            es_means[m]  = float(np.mean(e_arr[mask]))
    return {
        "fz": fz_losses, "var_mean": var_means, "es_mean": es_means,
        "var_fcs": var_fcs, "es_fcs": es_fcs,
        "oos_returns": oos_returns, "oos_idx": df.index[-n_oos:],
    }


@st.cache_data(show_spinner=False)
def compute_backtest(asset, alpha, method, window, n_obs, df_key):
    from risk_metrics import (calculate_var_es, calculate_cornish_fisher_var,
                               backtest_var, backtest_es_acerbi_szekely,
                               berkowitz_pit_test, basel_traffic_light)
    df = st.session_state.returns_df
    returns = df[asset].dropna().values
    n = len(returns)
    if n_obs > n - window:
        n_obs = n - window
    T = n - window
    var_arr = np.empty(T)
    es_arr  = np.empty(T)
    for i in range(T):
        r_win = returns[i: i + window]
        if method == "Cornish-Fisher":
            cf_v, _, _, _ = calculate_cornish_fisher_var(r_win, alpha)
            var_arr[i] = cf_v
            es_arr[i]  = _cf_es(r_win, alpha)
        else:
            meth_key = {"Normal": "parametric_normal",
                        "Student-t": "parametric_student_t",
                        "Hist.Sim.": "historical"}[method]
            v, e = calculate_var_es(r_win, alpha, meth_key)
            var_arr[i] = v; es_arr[i] = e
    ret_aligned = returns[window:]
    idx_aligned = df.index[window:]
    ret_oos  = ret_aligned[-n_obs:]
    var_oos  = var_arr[-n_obs:]
    es_oos   = es_arr[-n_obs:]
    idx_oos  = idx_aligned[-n_obs:]
    bt     = backtest_var(ret_oos, var_oos, alpha)
    es_bt  = backtest_es_acerbi_szekely(ret_oos, var_oos, es_oos, alpha)
    berk   = berkowitz_pit_test(ret_oos, var_oos)
    n_250  = min(n_obs, 250)
    viol_250 = int(np.sum(-ret_oos[-n_250:] > var_oos[-n_250:]))
    traffic  = basel_traffic_light(viol_250, n_250, 1 - alpha)
    return {
        "ret_oos": ret_oos, "var_oos": var_oos, "es_oos": es_oos,
        "idx_oos": idx_oos, "bt": bt, "es_bt": es_bt, "berk": berk,
        "traffic": traffic, "viol_250": viol_250, "n_250": n_250,
    }


# ─── RENDER ─────────────────────────────────────────────────────────────────

def render():
    (tab_theory, tab_risk, tab_evt,
     tab_systemic, tab_fz, tab_backtest) = st.tabs([
        "📖 Teori",
        "⚙️ Risk Paneli",
        "🌊 EVT & Kuyruk Analizi",
        "🔗 Sistemik Risk (CoVaR & MES)",
        "📊 Tahmin Karşılaştırması (FZ Kaybı)",
        "🔬 Geriye Dönük Test",
    ])

    # =========================================================================
    # TAB 1 — TEORİ
    # =========================================================================
    with tab_theory:
        st.markdown("### Risk Ölçütleri & Geriye Dönük Test: Teorik Özet")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### Risk Ölçütleri")
            st.markdown(r"""
**Value at Risk (VaR):**
$$\mathrm{VaR}_\alpha(L) = F_L^{-1}(1-\alpha)$$
Kayıp dağılımının $(1-\alpha)$. kantili.
⚠️ **Alt-toplayıcı değildir** → portföy çeşitlendirmesini ödüllendirmez.

**Beklenen Kayıp (ES):**
$$\mathrm{ES}_\alpha(L) = \frac{1}{\alpha}\int_0^\alpha \mathrm{VaR}_u(L)\,du = E[L \mid L > \mathrm{VaR}_\alpha]$$

**PELVE (Li & Wang, 2023):**
$c_\alpha \in [1, 1/\alpha]$ öyle ki:
$$\mathrm{ES}_{1 - c_\alpha \cdot \alpha}(L) = \mathrm{VaR}_{1-\alpha}(L)$$
Normal dağılımda $\alpha \to 0$ iken $c \to e \approx 2.718$.
""")
            st.markdown("**Artzner vd. (1999) — Tutarlı Risk Aksiyomları:**")
            st.markdown("""
| Aksiyom | VaR | ES |
|---------|:---:|:--:|
| Monotonluk | ✓ | ✓ |
| Alt-toplayıcılık | ✗ | ✓ |
| Pozitif homojenlik | ✓ | ✓ |
| Öteleme değişmezliği | ✓ | ✓ |

ES dört aksiyomu da sağlar; **VaR alt-toplayıcılığı sağlamaz.**
""")

        with col2:
            st.markdown("#### Tahmin Yöntemleri")
            st.markdown(r"""
| Yöntem | Formül | Avantaj | Dezavantaj |
|--------|--------|---------|------------|
| **Parametrik Normal** | $\mu+\sigma z_{1-\alpha}$ | Basit | Kalın kuyruğu kaçırır |
| **Student-t** | $\mu+s_t\,t_{1-\alpha,\nu}$ | Kuyrukta iyi | $\nu$ seçimi |
| **Tarihsel Sim.** | $Q_{1-\alpha}(\{L_t\})$ | Varsayımsız | Pencereye duyarlı |
| **Cornish-Fisher** | $\mu+\sigma z_{\mathrm{CF}}$ | Çarpıklık/basıklık | Yüksek basıklıkta sapma |
| **EVT / GPD** | $u+\tfrac{\sigma}{\xi}[(\tfrac{n}{N_u}\alpha)^{-\xi}-1]$ | Kuyruk odaklı | Eşik seçimi |

**Cornish-Fisher açılımı:**
$$z_{\mathrm{CF}} = z + \tfrac{1}{6}(z^2-1)S + \tfrac{1}{24}(z^3-3z)K - \tfrac{1}{36}(2z^3-5z)S^2$$
$S$ = çarpıklık, $K$ = fazla basıklık.

**EVT / POT:**
$$\mathrm{ES}_\alpha = \frac{\mathrm{VaR}_\alpha + \sigma - \xi u}{1 - \xi}$$
""")

        with col3:
            st.markdown("#### Elicitability & Backtest")
            st.markdown(r"""
**Fissler-Ziegel (2016):**
ES tek başına elicitable **değil**.
(VaR, ES) çifti **birlikte elicitable** → model karşılaştırması mümkün.

**FZ Ortak Kaybı:**
$$S(v,e;\ell) = (\mathbf{1}_{\ell>v}-\alpha)(-v) - \mathbf{1}_{\ell>v}\ell$$
$$\quad + \frac{-1}{e}\!\left(e + \frac{\ell-v}{\alpha}\mathbf{1}_{\ell>v}\right) + \log e$$

Düşük FZ kaybı → daha iyi model.

**Basel Trafik Işığı (250 gün, %99 VaR):**

| İhlal | Bölge | Çarpan |
|-------|-------|--------|
| 0–4 | 🟢 Yeşil | 3.00 |
| 5–9 | 🟡 Sarı | 3.20–4.00 |
| ≥10 | 🔴 Kırmızı | 4.00 |
""")
            st.info(
                "**Kupiec (1995) POF:** İhlal oranını test eder.\n\n"
                "**Christoffersen (1998):** İhlallerin bağımsızlığını test eder.\n\n"
                "**Acerbi-Szekely (2014) Z₁:** ES modelini test eder.\n\n"
                "**Berkowitz (2001) PIT:** Tam dağılımı test eder."
            )


    # =========================================================================
    # TAB 2 — RISK PANELİ
    # =========================================================================
    with tab_risk:
        st.markdown("### Risk Paneli")
        df = st.session_state.returns_df
        all_cols = _asset_cols(df)
        df_key = id(df)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            asset = st.selectbox("Varlık", all_cols, key="rp2_asset")
        with col_b:
            alpha = st.select_slider(
                "Anlamlılık Düzeyi α",
                options=[0.01, 0.025, 0.05, 0.10],
                value=0.05, key="rp2_alpha",
            )
        with col_c:
            window = st.slider("Kayan Pencere (gün)", 60, 500, 250, 10, key="rp2_window")

        n_data = len(df[asset].dropna())
        if window > n_data // 2:
            st.warning(f"Pencere ({window}) verinin yarısından büyük — tahminler güvenilir olmayabilir.")

        with st.spinner("Risk ölçütleri hesaplanıyor..."):
            res = compute_rolling_risk(asset, alpha, window, df_key)
            static_df, static_pelve = compute_static_risk(asset, alpha, df_key)

        idx   = res["idx"]
        rets  = res["returns"]
        var_d = res["var"]
        pelve = res["pelve"]

        # Chart 1: Returns + VaR
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=idx, y=rets, mode="lines", name="Günlük Getiri",
            line=dict(color="rgba(96,165,250,0.45)", width=0.8),
        ))
        for meth, color in COLORS_METHOD.items():
            fig1.add_trace(go.Scatter(
                x=idx, y=-var_d[meth],
                mode="lines", name=f"VaR — {meth}",
                line=dict(color=color, width=1.4, dash="dash"),
            ))
        fig1.update_layout(
            template=PLOT_TEMPLATE,
            title=f"Kayan VaR Tahminleri — {asset} (α={alpha})",
            xaxis_title="Tarih", yaxis_title="Getiri / VaR",
            height=380, margin=dict(l=20, r=20, t=50, b=30),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Rolling PELVE
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=idx, y=pelve, mode="lines", name="PELVE (c)",
            line=dict(color="#fbbf24", width=1.6),
        ))
        fig2.add_hline(y=np.e, line_dash="dot", line_color="#a78bfa",
                       annotation_text="Normal referans: e ≈ 2.718",
                       annotation_position="bottom right")
        fig2.add_hline(y=2.5, line_dash="dot", line_color="#f472b6",
                       annotation_text="Basel FRTB: c = 2.5",
                       annotation_position="top right")
        fig2.update_layout(
            template=PLOT_TEMPLATE,
            title=f"Kayan PELVE — {asset}",
            xaxis_title="Tarih", yaxis_title="c (PELVE)",
            height=280, margin=dict(l=20, r=20, t=50, b=30),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Metric cards
        st.markdown("#### Güncel Risk Ölçütleri (Son Pencere)")
        mc = st.columns(6)
        card_data = [
            ("VaR — Normal",      f"{var_d['Normal'][-1]*100:.3f}%",         "#a78bfa"),
            ("VaR — Student-t",   f"{var_d['Student-t'][-1]*100:.3f}%",      "#34d399"),
            ("VaR — Hist.Sim.",   f"{var_d['Hist.Sim.'][-1]*100:.3f}%",      "#f472b6"),
            ("VaR — CF",          f"{var_d['Cornish-Fisher'][-1]*100:.3f}%", "#fbbf24"),
            ("PELVE (son)",       f"{pelve[-1]:.3f}",                         "#60a5fa"),
            ("İhlal (Normal)",    str(int(np.sum(-rets > var_d['Normal']))),   "#fb923c"),
        ]
        for col_m, (lbl, val, clr) in zip(mc, card_data):
            with col_m:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">{lbl}</div>'
                    f'<div class="value" style="color:{clr};font-size:1.2rem">{val}</div>'
                    f'</div>', unsafe_allow_html=True,
                )
        st.markdown("")

        # Static table
        st.markdown("#### Tam Örneklem VaR / ES / PELVE Karşılaştırması")
        disp = static_df.copy()
        disp["VaR (%)"] = (disp["VaR"] * 100).round(4)
        disp["ES (%)"]  = (disp["ES"]  * 100).round(4)
        disp["PELVE"]   = round(static_pelve, 4)
        disp = disp.rename(columns={"Yontem": "Yöntem"})[["Yöntem", "VaR (%)", "ES (%)", "PELVE"]]
        st.dataframe(
            disp.style.background_gradient(subset=["VaR (%)", "ES (%)"], cmap="RdPu"),
            use_container_width=True, hide_index=True,
        )


    # =========================================================================
    # TAB 3 — EVT & KUYRUK ANALİZİ
    # =========================================================================
    with tab_evt:
        st.markdown("### EVT — Aşım Eşiği Yöntemi (POT / GPD)")
        df = st.session_state.returns_df
        all_cols = _asset_cols(df)
        df_key = id(df)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            asset_e = st.selectbox("Varlık", all_cols, key="evt_asset")
        with col_b:
            thresh_q = st.slider("Eşik Kantil (u)", 0.80, 0.98, 0.90, 0.01,
                                 key="evt_thresh")
        with col_c:
            alpha_e = st.slider("α (VaR/ES düzeyi)", 0.01, 0.10, 0.05, 0.005,
                                key="evt_alpha", format="%.3f")

        with st.spinner("EVT hesaplanıyor..."):
            evt, v_norm, u_vals, me_vals, gpd_x, gpd_emp, gpd_theo = (
                compute_evt(asset_e, alpha_e, thresh_q, df_key)
            )

        if evt.get("warning"):
            st.warning(evt["warning"])

        xi_val = evt["xi"]
        if not np.isnan(xi_val):
            if xi_val > 0.05:
                st.error(f"ξ = {xi_val:.4f} > 0 → **Ağır kuyruk (Fréchet)** — sonsuz momentler mümkün")
            elif xi_val < -0.05:
                st.success(f"ξ = {xi_val:.4f} < 0 → **Sınırlı kuyruk (Weibull)** — kayıplar sınırlı")
            else:
                st.info(f"ξ = {xi_val:.4f} ≈ 0 → **Hafif kuyruk (Gumbel)** — Normal/Log-normal benzeri")

        col_left, col_right = st.columns(2)
        losses_full = -df[asset_e].dropna().values
        u_thresh = evt["threshold"]

        with col_left:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=losses_full, nbinsx=60, name="Tüm Kayıplar",
                marker_color="rgba(96,165,250,0.5)",
            ))
            exc_mask = losses_full > u_thresh
            fig_hist.add_trace(go.Histogram(
                x=losses_full[exc_mask], nbinsx=30, name="Aşımlar (L > u)",
                marker_color="rgba(251,146,60,0.75)",
            ))
            fig_hist.add_vline(x=u_thresh, line_color="#f472b6", line_dash="dash",
                               annotation_text=f"u = {u_thresh:.4f}",
                               annotation_font_color="#f472b6")
            fig_hist.update_layout(
                template=PLOT_TEMPLATE,
                title=f"Kayıp Histogramı — u = {u_thresh:.4f}",
                xaxis_title="Kayıp", yaxis_title="Frekans",
                barmode="overlay", height=320,
                margin=dict(l=20, r=20, t=50, b=30),
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_right:
            if len(gpd_x) > 0:
                fig_gpd = go.Figure()
                fig_gpd.add_trace(go.Scatter(
                    x=gpd_x, y=gpd_emp, mode="markers", name="Ampirik ECDF",
                    marker=dict(color="#60a5fa", size=5),
                ))
                fig_gpd.add_trace(go.Scatter(
                    x=gpd_x, y=gpd_theo, mode="lines", name="GPD CDF (teorik)",
                    line=dict(color="#f472b6", width=2),
                ))
                fig_gpd.update_layout(
                    template=PLOT_TEMPLATE,
                    title=f"GPD Uyumu: ξ={xi_val:.4f}, σ={evt['sigma']:.4f}",
                    xaxis_title="Aşım Y = L − u",
                    yaxis_title="Kümülatif Olasılık",
                    height=320, margin=dict(l=20, r=20, t=50, b=30),
                    legend=dict(orientation="h", y=-0.3),
                )
                st.plotly_chart(fig_gpd, use_container_width=True)
            else:
                st.info("GPD grafiği için yeterli aşım verisi yok.")

        # Mean Excess Plot
        valid = [(u_vals[i], me_vals[i]) for i in range(len(me_vals)) if not np.isnan(me_vals[i])]
        if valid:
            u_v, me_v = zip(*valid)
            fig_me = go.Figure()
            fig_me.add_trace(go.Scatter(
                x=list(u_v), y=list(me_v), mode="lines+markers",
                name="E[L−u | L>u]",
                line=dict(color="#34d399", width=2), marker=dict(size=5),
            ))
            fig_me.add_vline(x=u_thresh, line_color="#f472b6", line_dash="dash",
                             annotation_text="Seçili u", annotation_font_color="#f472b6")
            fig_me.update_layout(
                template=PLOT_TEMPLATE,
                title="Ortalama Aşım Grafiği (Mean Excess Plot)",
                xaxis_title="Eşik u", yaxis_title="Ort. Aşım E[L−u | L>u]",
                height=280, margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_me, use_container_width=True)
            st.caption("Doğrusal artış → GPD geçerli. Azalma → sınırlı kuyruk. Hızlı artış → ağır kuyruk.")

        st.markdown("#### EVT Sonuçları")
        result_rows = {
            "Eşik u":              f"{evt['threshold']:.6f}",
            "Aşım Sayısı Nᵤ":     str(evt["n_exceedances"]),
            "Şekil Parametresi ξ": f"{evt['xi']:.6f}"    if not np.isnan(evt['xi'])    else "—",
            "Ölçek Parametresi σ": f"{evt['sigma']:.6f}"  if not np.isnan(evt['sigma']) else "—",
            f"EVT VaR (α={alpha_e})": f"{evt['var']:.6f}" if not np.isnan(evt['var'])   else "—",
            f"EVT ES  (α={alpha_e})": f"{evt['es']:.6f}"  if not np.isnan(evt['es'])    else "—",
            "Normal VaR (karş.)":  f"{v_norm:.6f}",
        }
        st.table(pd.DataFrame(result_rows.items(), columns=["Parametre", "Değer"]))


    # =========================================================================
    # TAB 4 — SİSTEMİK RİSK (CoVaR & MES)
    # =========================================================================
    with tab_systemic:
        st.markdown("### Sistemik Risk Ölçütleri: CoVaR & MES")
        df = st.session_state.returns_df
        all_cols = _asset_cols(df)
        df_key = id(df)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            asset_i = st.selectbox("Varlık i (odak)", all_cols, index=0, key="sys_asset_i")
        with col_b:
            default_j = 1 if len(all_cols) > 1 else 0
            market_proxy = st.selectbox("Piyasa/Endeks j", all_cols,
                                        index=default_j, key="sys_market")
        with col_c:
            alpha_s = st.slider("α", 0.01, 0.10, 0.05, 0.005,
                                key="sys_alpha", format="%.3f")

        roll_win_s = st.slider("Kayan Pencere — CoVaR (gün)", 60, 500, 250, 10, key="sys_roll")

        with st.spinner("CoVaR / MES hesaplanıyor..."):
            ret_i, ret_m, covar_res, mes_res = compute_systemic(
                asset_i, market_proxy, alpha_s, df_key
            )

        col_sc, col_mc = st.columns([2, 1])
        with col_sc:
            var_j = covar_res["var_j"]
            beta0 = covar_res["beta0"]
            beta1 = covar_res["beta1"]
            x_line = np.linspace(ret_m.min(), ret_m.max(), 100)
            y_line = beta0 + beta1 * x_line
            crisis_mask = ret_m <= var_j

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=ret_m[crisis_mask], y=ret_i[crisis_mask],
                mode="markers", name="Kriz Günleri",
                marker=dict(color="#fb923c", size=5, opacity=0.75),
            ))
            fig_sc.add_trace(go.Scatter(
                x=ret_m[~crisis_mask], y=ret_i[~crisis_mask],
                mode="markers", name="Normal Günler",
                marker=dict(color="rgba(96,165,250,0.3)", size=3),
            ))
            fig_sc.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                name=f"Kantil Regresyon (α={alpha_s})",
                line=dict(color="#a78bfa", width=2),
            ))
            fig_sc.add_vline(x=var_j, line_color="#f472b6", line_dash="dash",
                             annotation_text=f"VaR_j={var_j:.4f}",
                             annotation_font_color="#f472b6")
            fig_sc.update_layout(
                template=PLOT_TEMPLATE,
                title=f"Saçılım: {asset_i} vs {market_proxy}",
                xaxis_title=f"{market_proxy} Getiri",
                yaxis_title=f"{asset_i} Getiri",
                height=380, margin=dict(l=20, r=20, t=50, b=30),
                legend=dict(orientation="h", y=-0.28),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        with col_mc:
            covar_val = covar_res["covar_alpha"]
            delta_val = covar_res["delta_covar"]
            mes_val   = mes_res["mes"]
            beta_tail = mes_res["beta_tail"]
            cards_s = [
                ("CoVaR (i|j krizde)", f"{covar_val:.5f}", "#f472b6"),
                ("ΔCoVaR (yayılım)",   f"{delta_val:.5f}", "#fb923c"),
                ("MES", f"{mes_val:.5f}" if not np.isnan(mes_val) else "—", "#34d399"),
                ("Tail Beta", f"{beta_tail:.4f}" if not np.isnan(beta_tail) else "—", "#60a5fa"),
                ("Kriz Gün Sayısı", str(mes_res["n_crisis_days"]), "#a78bfa"),
            ]
            st.markdown("<br>", unsafe_allow_html=True)
            for lbl, val, clr in cards_s:
                st.markdown(
                    f'<div class="metric-card" style="margin-bottom:0.5rem">'
                    f'<div class="label">{lbl}</div>'
                    f'<div class="value" style="color:{clr};font-size:1.15rem">{val}</div>'
                    f'</div>', unsafe_allow_html=True,
                )

        st.markdown("#### Kayan CoVaR ve ΔCoVaR")
        with st.spinner("Kayan CoVaR hesaplanıyor (quantile regression — biraz sürebilir)..."):
            try:
                roll_cov_df = compute_rolling_covar(
                    asset_i, market_proxy, alpha_s, roll_win_s, df_key
                )
                fig_rcov = go.Figure()
                fig_rcov.add_trace(go.Scatter(
                    x=roll_cov_df.index, y=roll_cov_df["covar"],
                    mode="lines", name="CoVaR",
                    line=dict(color="#a78bfa", width=1.4),
                ))
                fig_rcov.add_trace(go.Scatter(
                    x=roll_cov_df.index, y=roll_cov_df["delta_covar"],
                    mode="lines", name="ΔCoVaR",
                    line=dict(color="#fb923c", width=1.4, dash="dash"),
                ))
                fig_rcov.update_layout(
                    template=PLOT_TEMPLATE,
                    title=f"Kayan CoVaR — {asset_i}|{market_proxy} (pencere={roll_win_s})",
                    xaxis_title="Tarih", yaxis_title="CoVaR",
                    height=300, margin=dict(l=20, r=20, t=50, b=30),
                    legend=dict(orientation="h", y=-0.3),
                )
                st.plotly_chart(fig_rcov, use_container_width=True)
            except Exception as exc:
                st.error(f"Kayan CoVaR hesaplanamadı: {exc}")

        st.info(
            "**ΔCoVaR** = CoVaR(i|j krizde) − CoVaR(i|j medyanda): "
            "j varlığının kriz anında i üzerine yarattığı yayılım etkisi (spillover)."
        )
        st.info(
            "**SRISK** (Acharya vd., 2017) = max(0, k·(Borç + MES·Özsermaye)): "
            "MES + kaldıraç bilgisini birleştirir; sistemik risk katkısının parasal ölçütü."
        )


    # =========================================================================
    # TAB 5 — TAHMIN KARŞILAŞTIRMASI (FZ KAYBI)
    # =========================================================================
    with tab_fz:
        st.markdown("### Fissler-Ziegel Kaybı ile Model Karşılaştırması")
        df = st.session_state.returns_df
        all_cols = _asset_cols(df)
        df_key = id(df)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            asset_f = st.selectbox("Varlık", all_cols, key="fz_asset")
        with col_b:
            alpha_f = st.slider("α", 0.01, 0.10, 0.05, 0.005,
                                key="fz_alpha", format="%.3f")
        with col_c:
            n_oos = st.slider("Örneklem-Dışı Pencere (gün)", 100, 1000, 500, 50,
                              key="fz_noos")

        with st.spinner("FZ kayıpları hesaplanıyor..."):
            fz_res = compute_fz_comparison(asset_f, alpha_f, n_oos, df_key)

        fz  = fz_res["fz"]
        vm  = fz_res["var_mean"]
        em  = fz_res["es_mean"]
        methods_ok = [m for m in fz if not np.isnan(fz[m])]

        if not methods_ok:
            st.error("FZ kaybı hesaplanamadı — veri yetersiz olabilir.")
        else:
            best = min(methods_ok, key=lambda m: fz[m])
            bar_colors = [
                "#34d399" if m == best else COLORS[i % len(COLORS)]
                for i, m in enumerate(methods_ok)
            ]
            fig_fz = go.Figure(go.Bar(
                x=methods_ok,
                y=[fz[m] for m in methods_ok],
                marker_color=bar_colors,
                text=[f"{fz[m]:.4f}" for m in methods_ok],
                textposition="auto",
            ))
            fig_fz.update_layout(
                template=PLOT_TEMPLATE,
                title=f"FZ Ortak Kaybı — {asset_f} (α={alpha_f}, son {n_oos} gün)",
                yaxis_title="Ortalama FZ Kaybı (düşük = iyi)",
                height=360, margin=dict(l=20, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_fz, use_container_width=True)

            rows_fz = []
            for rank, m in enumerate(sorted(methods_ok, key=lambda x: fz[x]), 1):
                rows_fz.append({
                    "Sıra": rank, "Yöntem": m,
                    "Ort. VaR (%)": f"{vm[m]*100:.4f}" if not np.isnan(vm[m]) else "—",
                    "Ort. ES (%)":  f"{em[m]*100:.4f}" if not np.isnan(em[m]) else "—",
                    "FZ Kaybı": f"{fz[m]:.6f}",
                    "En İyi": "✓" if m == best else "",
                })
            st.dataframe(pd.DataFrame(rows_fz), use_container_width=True, hide_index=True)

            top2 = sorted(methods_ok, key=lambda x: fz[x])[:2]
            oos_idx = fz_res["oos_idx"]
            oos_ret = fz_res["oos_returns"]
            var_fcs = fz_res["var_fcs"]
            fig_ov = go.Figure()
            fig_ov.add_trace(go.Scatter(
                x=oos_idx, y=oos_ret, mode="lines", name="Getiri",
                line=dict(color="rgba(96,165,250,0.4)", width=0.8),
            ))
            clrs_top = ["#34d399", "#f472b6"]
            for ci, m in enumerate(top2):
                v_arr = var_fcs[m]
                n_plot = min(len(oos_idx), len(v_arr))
                fig_ov.add_trace(go.Scatter(
                    x=oos_idx[-n_plot:], y=-v_arr[-n_plot:],
                    mode="lines", name=f"VaR — {m}",
                    line=dict(color=clrs_top[ci], width=1.6, dash="dash"),
                ))
            fig_ov.update_layout(
                template=PLOT_TEMPLATE,
                title=f"En İyi 2 Model VaR Örtüşmesi — {asset_f}",
                xaxis_title="Tarih", yaxis_title="Getiri / VaR",
                height=300, margin=dict(l=20, r=20, t=50, b=30),
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(fig_ov, use_container_width=True)
            st.success(f"**En iyi model:** {best} (FZ Kaybı = {fz[best]:.6f})")


    # =========================================================================
    # TAB 6 — GERİYE DÖNÜK TEST
    # =========================================================================
    with tab_backtest:
        st.markdown("### Geriye Dönük Test (Backtesting)")
        df = st.session_state.returns_df
        all_cols = _asset_cols(df)
        df_key = id(df)

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            asset_b = st.selectbox("Varlık", all_cols, key="bt2_asset")
        with col_b:
            alpha_b = st.slider("α", 0.01, 0.10, 0.05, 0.005,
                                key="bt2_alpha", format="%.3f")
        with col_c:
            method_b = st.selectbox("Yöntem",
                                    ["Normal", "Student-t", "Hist.Sim.", "Cornish-Fisher"],
                                    key="bt2_method")
        with col_d:
            window_b = st.slider("Tahmin Penceresi (gün)", 60, 500, 250, 10,
                                 key="bt2_window")

        n_obs_b = st.slider("Test Penceresi (gün)", 100, 1000, 500, 50, key="bt2_nobs")

        with st.spinner("Backtest yapılıyor..."):
            try:
                bt_res = compute_backtest(asset_b, alpha_b, method_b, window_b, n_obs_b, df_key)
            except Exception as exc:
                st.error(f"Backtest hatası: {exc}")
                st.stop()

        ret_oos = bt_res["ret_oos"]
        var_oos = bt_res["var_oos"]
        idx_oos = bt_res["idx_oos"]
        bt      = bt_res["bt"]
        es_bt   = bt_res["es_bt"]
        berk    = bt_res["berk"]
        traffic = bt_res["traffic"]
        hits    = (-ret_oos > var_oos)
        viol_idx = idx_oos[hits]

        # İhlal grafiği
        fig_viol = go.Figure()
        fig_viol.add_trace(go.Scatter(
            x=idx_oos, y=ret_oos, mode="lines", name="Günlük Getiri",
            line=dict(color="rgba(96,165,250,0.5)", width=0.8),
        ))
        fig_viol.add_trace(go.Scatter(
            x=idx_oos, y=-var_oos, mode="lines",
            name=f"VaR — {method_b}",
            line=dict(color="#a78bfa", width=1.6, dash="dash"),
        ))
        fig_viol.add_trace(go.Scatter(
            x=viol_idx, y=ret_oos[hits], mode="markers", name="İhlal",
            marker=dict(color="#f87171", size=7, symbol="circle"),
        ))
        fig_viol.update_layout(
            template=PLOT_TEMPLATE,
            title=f"VaR İhlal Grafiği — {asset_b} ({method_b}, α={alpha_b})",
            xaxis_title="Tarih", yaxis_title="Getiri / VaR",
            height=380, margin=dict(l=20, r=20, t=50, b=30),
            legend=dict(orientation="h", y=-0.28),
        )
        st.plotly_chart(fig_viol, use_container_width=True)

        # Basel Trafik Işığı
        zone        = traffic["zone"]
        exp_viol    = traffic["expected_violations"]
        multiplier  = traffic["multiplier"]
        n_250       = bt_res["n_250"]
        viol_250    = bt_res["viol_250"]
        zone_cfg = {
            "green":  {"emoji": "🟢", "label": "YEŞİL BÖLGE",   "color": "#22c55e",
                       "desc": "Model kabul edilir"},
            "yellow": {"emoji": "🟡", "label": "SARI BÖLGE",    "color": "#eab308",
                       "desc": "Denetim altında izlenir"},
            "red":    {"emoji": "🔴", "label": "KIRMIZI BÖLGE", "color": "#ef4444",
                       "desc": "Model reddedilir"},
        }
        zc = zone_cfg[zone]
        st.markdown(
            f'<div style="background:#1e1e2e;border:2px solid {zc["color"]};'
            f'border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:1rem;'
            f'display:flex;align-items:center;gap:1.5rem;">'
            f'<div style="font-size:3rem;line-height:1">{zc["emoji"]}</div>'
            f'<div>'
            f'<div style="font-size:1.1rem;font-weight:700;color:{zc["color"]}">{zc["label"]}</div>'
            f'<div style="color:#ccc;font-size:0.88rem">{zc["desc"]}</div>'
            f'<div style="color:#aaa;font-size:0.82rem;margin-top:0.3rem">'
            f'Son {n_250} gün: Beklenen ihlal = <b>{exp_viol:.1f}</b> &nbsp;|&nbsp; '
            f'Gözlenen = <b>{viol_250}</b> &nbsp;|&nbsp; '
            f'Sermaye çarpanı = <b>{multiplier:.2f}</b>'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )

        # Backtest sonuçları tablosu
        st.markdown("#### Backtest Sonuçları")

        def _fs(v):
            try:
                return "—" if (v is None or np.isnan(float(v))) else f"{float(v):.4f}"
            except Exception:
                return "—"

        def _pr(pval):
            try:
                return "✗ Reddedilir" if float(pval) < 0.05 else "✓ Kabul edilir"
            except Exception:
                return "—"

        bt_table = pd.DataFrame([
            {"Test": "Kupiec POF (Koşulsuz Kapsam)",
             "İstatistik": _fs(bt["kupiec_stat"]),
             "p-değeri": _fs(bt["kupiec_pvalue"]),
             "Sonuç": _pr(bt["kupiec_pvalue"])},
            {"Test": "Christoffersen Bağımsızlık",
             "İstatistik": _fs(bt["independence_stat"]),
             "p-değeri": _fs(bt["independence_pvalue"]),
             "Sonuç": _pr(bt["independence_pvalue"])},
            {"Test": "Acerbi-Szekely Z₁ (ES)",
             "İstatistik": _fs(es_bt["z1_stat"]),
             "p-değeri": _fs(es_bt["pvalue"]),
             "Sonuç": _pr(es_bt["pvalue"])},
            {"Test": "Berkowitz PIT (Ljung-Box)",
             "İstatistik": _fs(berk["pit_acf1"]),
             "p-değeri": _fs(berk["lb_pvalue_level"]),
             "Sonuç": _pr(berk["lb_pvalue_level"])},
        ])

        def _clr_pval(val):
            try:
                return "color: #f87171" if float(val) < 0.05 else "color: #34d399"
            except Exception:
                return ""

        def _clr_res(val):
            if "Kabul" in str(val):   return "color: #34d399"
            if "Reddedilir" in str(val): return "color: #f87171"
            return ""

        st.dataframe(
            bt_table.style
            .map(_clr_pval, subset=["p-değeri"])
            .map(_clr_res, subset=["Sonuç"]),
            use_container_width=True, hide_index=True,
        )

        # Stats cards
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        stat_cards = [
            ("İhlal Sayısı (toplam)", str(bt["violations"]),                        "#fb923c"),
            ("İhlal Oranı",           f"{bt['violation_rate']*100:.2f}%",           "#fbbf24"),
            ("PIT Ortalama (≈ 0)",    f"{berk['pit_mean']:.4f}",                    "#60a5fa"),
            ("PIT Std.Sapma (≈ 1)",   f"{berk['pit_std']:.4f}",                     "#a78bfa"),
        ]
        for col_s, (lbl, val, clr) in zip([col_s1, col_s2, col_s3, col_s4], stat_cards):
            with col_s:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="label">{lbl}</div>'
                    f'<div class="value" style="color:{clr};font-size:1.2rem">{val}</div>'
                    f'</div>', unsafe_allow_html=True,
                )

        st.markdown("")
        st.markdown(
            "> **p > 0.05** → Model reddedilemez (uyumlu) &nbsp;|&nbsp; "
            "**p < 0.05** → Model başarısız"
        )
        st.markdown(
            "**Kupiec POF:** İhlal oranının beklenen α düzeyine eşit olup olmadığını test eder.  \n"
            "**Christoffersen:** İhlallerin zaman içinde bağımsız dağılıp dağılmadığını test eder.  \n"
            "**Acerbi-Szekely Z₁:** ES modelinin kuyruk kayıplarını doğru tahmin edip etmediğini test eder.  \n"
            "**Berkowitz PIT:** Tam dağılımın doğruluğunu Olasılık İntegral Dönüşümü (PIT) ile test eder."
        )
