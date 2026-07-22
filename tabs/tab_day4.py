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
COLORS = ["#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7","#999999"]

COLORS_METHOD = {
    "Normal":             "#E69F00",
    "Student-t":          "#56B4E9",
    "Hist.Sim.":          "#CC79A7",
    "Cornish-Fisher":     "#009E73",
    "GARCH-Normal":       "#0072B2",
    "GARCH-t":            "#D55E00",
    "GJR-N":              "#F0E442",
    "EGARCH-t":           "#999999",
    "GARCH-FHS":          "#fbbf24",
    "GJR-FHS":            "#a78bfa",
    "FZ-GAS":             "#34d399",
    "FZ-GARCH-N":         "#fb923c",
    "FZ-GARCH-t":         "#60a5fa",
    "CaViaR-AsL(simple)": "#f472b6",
    "CaViaR-AsL(dyn)":    "#4ade80",
}
_COMBO_COLORS = {
    "Min-skor":    "#34d399",
    "Göreli-skor": "#f87171",
    "Basit ort.":  "#a78bfa",
}

_BASELINE_METHODS = ["Normal", "Student-t", "Hist.Sim.", "Cornish-Fisher"]
_ALL_METHODS = [
    "Normal", "Student-t", "Hist.Sim.", "Cornish-Fisher", "GARCH-Normal",
    "GARCH-t", "GJR-N", "EGARCH-t", "GARCH-FHS", "GJR-FHS",
    "FZ-GAS", "FZ-GARCH-N", "FZ-GARCH-t", "CaViaR-AsL(simple)", "CaViaR-AsL(dyn)",
]
_DEFAULT_COMBO = [
    "Normal", "Student-t", "Cornish-Fisher", "GARCH-Normal",
    "GARCH-FHS", "GJR-FHS", "FZ-GAS", "FZ-GARCH-t", "CaViaR-AsL(dyn)",
]


# ─── HELPERS ────────────────────────────────────────────────────────────────

def _stress_window(index, returns_values):
    cw = st.session_state.get("crisis_window", None)
    if cw is not None:
        return cw
    v = np.asarray(returns_values)
    m = np.abs(v).mean(axis=1) if v.ndim == 2 else np.abs(v)
    k = 5
    ms = np.convolve(m, np.ones(k) / k, mode="same")
    pk = int(np.argmax(ms))
    half = 18
    a = max(0, pk - half); b = min(len(index) - 1, pk + half)
    return (index[a], index[b])


def _asset_cols(df):
    return [c for c in df.columns if not c.endswith("_RV") and not c.endswith("_BPV")]


def _cf_es(r_win, alpha):
    z_n = norm.ppf(1 - alpha)
    return float(np.mean(-r_win) + np.std(r_win, ddof=1) * (norm.pdf(z_n) / alpha))


# ─── CACHED COMPUTATIONS ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_rolling_risk(asset, alpha, window, df_key):
    _CACHE_VERSION = 3
    from risk_metrics import (calculate_var_es, calculate_pelve_single,
                               calculate_cornish_fisher_var)
    df = st.session_state.returns_df
    returns = df[asset].dropna().values
    n = len(returns); T = n - window
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
        cf_v, _, _, _        = calculate_cornish_fisher_var(r_win, alpha)
        var_cf[i] = cf_v; es_cf[i] = _cf_es(r_win, alpha)
        pelve[i]  = calculate_pelve_single(-r_win, alpha)
    idx = df.index[window: window + T]
    return {
        "idx": idx, "returns": returns[window: window + T],
        "var":  {"Normal": var_n, "Student-t": var_t, "Hist.Sim.": var_hs, "Cornish-Fisher": var_cf},
        "es":   {"Normal": es_n,  "Student-t": es_t,  "Hist.Sim.": es_hs,  "Cornish-Fisher": es_cf},
        "pelve": pelve,
    }


@st.cache_data(show_spinner=False)
def compute_static_risk(asset, alpha, df_key):
    from risk_metrics import (calculate_var_es, calculate_pelve_single,
                               calculate_cornish_fisher_var)
    df = st.session_state.returns_df
    returns = df[asset].dropna().values
    rows = []
    for label, method in [("Normal","parametric_normal"),
                           ("Student-t","parametric_student_t"),
                           ("Hist.Sim.","historical")]:
        v, e = calculate_var_es(returns, alpha, method)
        rows.append({"Yontem": label, "VaR": v, "ES": e})
    cf_v, _, _, _ = calculate_cornish_fisher_var(returns, alpha)
    rows.append({"Yontem": "Cornish-Fisher", "VaR": cf_v, "ES": _cf_es(returns, alpha)})
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
    u_quantiles = np.linspace(0.60, 0.98, 40)
    u_vals = np.quantile(losses, u_quantiles)
    me_vals = []
    for uu in u_vals:
        exc = losses[losses > uu] - uu
        me_vals.append(float(np.mean(exc)) if len(exc) >= 5 else np.nan)
    exc_data = evt.get("exceedances", np.array([]))
    gpd_x = gpd_emp = gpd_theo = np.array([])
    if len(exc_data) > 0:
        exc_sorted = np.sort(exc_data); n_exc = len(exc_sorted)
        gpd_emp = np.arange(1, n_exc + 1) / n_exc
        xi_fit  = float(evt["xi"])    if not np.isnan(evt["xi"])    else 0.0
        sig_fit = float(evt["sigma"]) if not np.isnan(evt["sigma"]) else 1e-6
        gpd_theo = genpareto.cdf(exc_sorted, xi_fit, scale=sig_fit, loc=0)
        gpd_x    = exc_sorted
    return evt, v_norm, u_vals, me_vals, gpd_x, gpd_emp, gpd_theo


@st.cache_data(show_spinner=False)
def compute_systemic(asset_i, market_proxy, alpha, df_key):
    from risk_metrics import calculate_covar, calculate_mes
    df = st.session_state.returns_df
    ret_i = df[asset_i].dropna().values; ret_m = df[market_proxy].dropna().values
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
    ret_i = df[asset_i].dropna().values; ret_m = df[market_proxy].dropna().values
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
    """Individual model OOS VaR/ES forecasts + FZ losses. No combination logic."""
    from risk_metrics import (calculate_var_es, calculate_cornish_fisher_var,
                               fissler_ziegel_loss)
    df = st.session_state.returns_df
    returns = df[asset].dropna().values
    n = len(returns)
    if n_oos >= n: n_oos = n // 2
    oos_returns = returns[-n_oos:]
    train_win   = max(60, min(250, n - n_oos))
    start_idx   = n - n_oos

    # ---- Baseline 4 (rolling window) ----
    var_fcs = {m: [] for m in _BASELINE_METHODS}
    es_fcs  = {m: [] for m in _BASELINE_METHODS}
    for i in range(n_oos):
        cur = start_idx + i; lo = max(0, cur - train_win)
        r_win = returns[lo:cur] if cur > 0 else returns[:1]
        for label, method in [("Normal","parametric_normal"),
                               ("Student-t","parametric_student_t"),
                               ("Hist.Sim.","historical")]:
            v, e = calculate_var_es(r_win, alpha, method)
            var_fcs[label].append(v); es_fcs[label].append(e)
        cf_v, _, _, _ = calculate_cornish_fisher_var(r_win, alpha)
        var_fcs["Cornish-Fisher"].append(cf_v)
        es_fcs["Cornish-Fisher"].append(_cf_es(r_win, alpha))
    for m in _BASELINE_METHODS:
        var_fcs[m] = np.array(var_fcs[m]); es_fcs[m] = np.array(es_fcs[m])

    # ---- GARCH-Normal (inline) ----
    garch_var = np.full(n_oos, np.nan); garch_es = np.full(n_oos, np.nan)
    try:
        from arch import arch_model as _arch_model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm = _arch_model(returns * 100, vol="Garch", p=1, q=1, dist="normal")
            gr = gm.fit(disp="off")
        cond_vol  = gr.conditional_volatility / 100
        garch_var = cond_vol[-n_oos:] * norm.ppf(1 - alpha)
        garch_es  = cond_vol[-n_oos:] * norm.pdf(norm.ppf(1 - alpha)) / alpha
    except Exception:
        pass
    var_fcs["GARCH-Normal"] = garch_var; es_fcs["GARCH-Normal"] = garch_es

    # ---- Semi-parametric (FZ-GAS, FZ-GARCH, CaViaR-AsL) ----
    try:
        from sp_estimators import compute_sp_estimators
        sp_res = compute_sp_estimators(returns, alpha, n_oos)
        for _nm, (_v, _e) in sp_res.items():
            var_fcs[_nm] = _v; es_fcs[_nm] = _e
    except Exception:
        pass

    # ---- GARCH universe (GJR, EGARCH, FHS) ----
    try:
        from sp_estimators import compute_garch_universe
        gu_res = compute_garch_universe(returns, alpha, n_oos)
        for _nm, (_v, _e) in gu_res.items():
            var_fcs[_nm] = _v; es_fcs[_nm] = _e
    except Exception:
        pass

    # ---- FZ losses ----
    fz_losses = {}; var_means = {}; es_means = {}
    for m in var_fcs:
        v_arr = var_fcs[m]; e_arr = es_fcs[m]
        mask  = ~np.isnan(v_arr) & ~np.isnan(e_arr)
        if mask.sum() < 10:
            fz_losses[m] = np.nan; var_means[m] = np.nan; es_means[m] = np.nan
        else:
            fz_losses[m] = fissler_ziegel_loss(oos_returns[mask], v_arr[mask], e_arr[mask], alpha)
            var_means[m] = float(np.mean(v_arr[mask])); es_means[m] = float(np.mean(e_arr[mask]))

    return {
        "fz": fz_losses, "var_mean": var_means, "es_mean": es_means,
        "var_fcs": var_fcs, "es_fcs": es_fcs,
        "oos_returns": oos_returns, "oos_idx": df.index[-n_oos:],
    }


@st.cache_data(show_spinner=False)
def compute_combination_tab(asset, alpha, n_oos, selected_models_tuple, df_key):
    """Combination analysis for the selected subset of models."""
    from risk_metrics import fissler_ziegel_loss, min_score_combine, relative_score_combine

    fz_res      = compute_fz_comparison(asset, alpha, n_oos, df_key)
    var_fcs_all = fz_res["var_fcs"]; es_fcs_all = fz_res["es_fcs"]
    oos_returns = fz_res["oos_returns"]; oos_idx = fz_res["oos_idx"]
    fz_all      = fz_res["fz"]
    n_actual    = len(oos_returns)

    valid = [m for m in selected_models_tuple
             if m in var_fcs_all
             and np.isfinite(var_fcs_all[m]).sum() >= max(10, 0.5 * n_actual)]

    _base = {"ok": False, "valid": valid, "oos_idx": oos_idx,
             "oos_returns": oos_returns, "all_fz": fz_all,
             "var_fcs": {m: var_fcs_all[m] for m in valid if m in var_fcs_all}}
    if len(valid) < 2:
        return _base

    mask = np.ones(n_actual, dtype=bool)
    for m in valid:
        mask &= np.isfinite(var_fcs_all[m]) & np.isfinite(es_fcs_all[m])
    if mask.sum() < 30:
        return _base

    y  = oos_returns[mask]
    Vm = np.column_stack([var_fcs_all[m][mask] for m in valid])
    Em = np.column_stack([es_fcs_all[m][mask]  for m in valid])

    vc, ec, wQ, wS     = min_score_combine(y, Vm, Em, alpha)
    vr, er, w_rel, lam = relative_score_combine(y, Vm, Em, alpha)
    vs_ = Vm.mean(1); es_ = Em.mean(1)

    indiv_fz = {m: fissler_ziegel_loss(y, var_fcs_all[m][mask], es_fcs_all[m][mask], alpha)
                for m in valid}
    combo_fz = {
        "Min-skor":    fissler_ziegel_loss(y, vc, ec, alpha),
        "Göreli-skor": fissler_ziegel_loss(y, vr, er, alpha),
        "Basit ort.":  fissler_ziegel_loss(y, vs_, es_, alpha),
    }

    return {
        "ok": True, "valid": valid, "mask": mask, "n_valid": int(mask.sum()),
        "oos_idx": oos_idx, "idx_valid": oos_idx[mask],
        "oos_returns": oos_returns, "oos_returns_valid": y,
        "var_fcs":       {m: var_fcs_all[m] for m in valid},
        "es_fcs":        {m: es_fcs_all[m]  for m in valid},
        "var_fcs_valid": {m: var_fcs_all[m][mask] for m in valid},
        "es_fcs_valid":  {m: es_fcs_all[m][mask]  for m in valid},
        "combo_var": {"Min-skor": vc, "Göreli-skor": vr, "Basit ort.": vs_},
        "combo_es":  {"Min-skor": ec, "Göreli-skor": er, "Basit ort.": es_},
        "indiv_fz": indiv_fz, "combo_fz": combo_fz,
        "weights": {
            "wQ":    np.asarray(wQ), "wS":    np.asarray(wS),
            "w_rel": np.asarray(w_rel), "lam": float(lam),
        },
        "all_fz": fz_all,
    }


@st.cache_data(show_spinner=False)
def compute_backtest(asset, alpha, method, window, n_obs, df_key):
    """Rolling-window backtest for baseline models."""
    from risk_metrics import (calculate_var_es, calculate_cornish_fisher_var,
                               backtest_var, backtest_es_acerbi_szekely,
                               berkowitz_pit_test, basel_traffic_light)
    df = st.session_state.returns_df
    returns = df[asset].dropna().values; n = len(returns)
    if n_obs > n - window: n_obs = n - window
    T = n - window
    var_arr = np.empty(T); es_arr = np.empty(T)
    for i in range(T):
        r_win = returns[i: i + window]
        if method == "Cornish-Fisher":
            cf_v, _, _, _ = calculate_cornish_fisher_var(r_win, alpha)
            var_arr[i] = cf_v; es_arr[i] = _cf_es(r_win, alpha)
        else:
            meth_key = {"Normal":"parametric_normal","Student-t":"parametric_student_t",
                        "Hist.Sim.":"historical"}[method]
            var_arr[i], es_arr[i] = calculate_var_es(r_win, alpha, meth_key)
    ret_aligned  = returns[window:]; idx_aligned = df.index[window:]
    ret_oos = ret_aligned[-n_obs:]; var_oos = var_arr[-n_obs:]
    es_oos  = es_arr[-n_obs:];     idx_oos = idx_aligned[-n_obs:]
    bt     = backtest_var(ret_oos, var_oos, alpha)
    es_bt  = backtest_es_acerbi_szekely(ret_oos, var_oos, es_oos, alpha)
    berk   = berkowitz_pit_test(ret_oos, var_oos)
    n_250  = min(n_obs, 250); viol_250 = int(np.sum(-ret_oos[-n_250:] > var_oos[-n_250:]))
    traffic = basel_traffic_light(viol_250, n_250, 1 - alpha)
    return {"ret_oos": ret_oos, "var_oos": var_oos, "es_oos": es_oos, "idx_oos": idx_oos,
            "bt": bt, "es_bt": es_bt, "berk": berk, "traffic": traffic,
            "viol_250": viol_250, "n_250": n_250}


def _run_backtest_tests(ret_oos, var_oos, es_oos, alpha):
    """Run backtest tests on pre-computed arrays. Returns (bt, es_bt, berk, traffic, viol_250, n_250)."""
    from risk_metrics import (backtest_var, backtest_es_acerbi_szekely,
                               berkowitz_pit_test, basel_traffic_light)
    n_250    = min(len(ret_oos), 250)
    bt       = backtest_var(ret_oos, var_oos, alpha)
    es_bt    = backtest_es_acerbi_szekely(ret_oos, var_oos, es_oos, alpha)
    berk     = berkowitz_pit_test(ret_oos, var_oos)
    viol_250 = int(np.sum(-ret_oos[-n_250:] > var_oos[-n_250:]))
    traffic  = basel_traffic_light(viol_250, n_250, 1 - alpha)
    return bt, es_bt, berk, traffic, viol_250, n_250


# ─── SHARED BACKTEST DISPLAY ─────────────────────────────────────────────────

def _display_backtest_results(ret_oos, var_oos, es_oos, idx_oos, alpha,
                               model_label, df, all_cols):
    """Render violation plot + Basel light + test table + stat cards."""
    bt, es_bt, berk, traffic, viol_250, n_250 = _run_backtest_tests(
        ret_oos, var_oos, es_oos, alpha)

    hits      = (-ret_oos > var_oos)
    viol_idx  = idx_oos[hits]
    stress    = _stress_window(df.index, df[all_cols].values)

    fig_viol = go.Figure()
    fig_viol.add_vrect(x0=stress[0], x1=stress[1], fillcolor="gray",
                       opacity=0.15, line_width=0,
                       annotation_text="stres", annotation_position="top left")
    fig_viol.add_trace(go.Scatter(
        x=idx_oos, y=ret_oos, mode="lines", name="Günlük Getiri",
        line=dict(color="rgba(150,150,150,0.55)", width=0.8)))
    fig_viol.add_trace(go.Scatter(
        x=idx_oos, y=-var_oos, mode="lines", name=f"VaR — {model_label}",
        line=dict(color=COLORS_METHOD.get(model_label, "#E69F00"), width=1.6, dash="dash")))
    fig_viol.add_trace(go.Scatter(
        x=viol_idx, y=ret_oos[hits], mode="markers", name="İhlal",
        marker=dict(color="#D55E00", size=7)))
    fig_viol.update_layout(
        template=PLOT_TEMPLATE,
        title=f"VaR İhlal Grafiği — {model_label} (α={alpha})",
        xaxis_title="Tarih", yaxis_title="Getiri / VaR",
        height=360, margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig_viol, use_container_width=True)

    # Basel Trafik Işığı
    zone_cfg = {
        "green":  ("🟢", "YEŞİL BÖLGE",   "#22c55e", "Model kabul edilir"),
        "yellow": ("🟡", "SARI BÖLGE",    "#eab308", "Denetim altında izlenir"),
        "red":    ("🔴", "KIRMIZI BÖLGE", "#ef4444", "Model reddedilir"),
    }
    em, lbl, clr, desc = zone_cfg[traffic["zone"]]
    st.markdown(
        f'<div style="background:#1e1e2e;border:2px solid {clr};border-radius:14px;'
        f'padding:1.2rem 1.5rem;margin-bottom:1rem;display:flex;align-items:center;gap:1.5rem;">'
        f'<div style="font-size:3rem;line-height:1">{em}</div>'
        f'<div><div style="font-size:1.1rem;font-weight:700;color:{clr}">{lbl}</div>'
        f'<div style="color:#ccc;font-size:0.88rem">{desc}</div>'
        f'<div style="color:#aaa;font-size:0.82rem;margin-top:0.3rem">'
        f'Son {n_250} gün: Beklenen = <b>{traffic["expected_violations"]:.1f}</b> '
        f'| Gözlenen = <b>{viol_250}</b> | Çarpan = <b>{traffic["multiplier"]:.2f}</b>'
        f'</div></div></div>', unsafe_allow_html=True)

    def _fs(v):
        try:   return "—" if (v is None or np.isnan(float(v))) else f"{float(v):.4f}"
        except: return "—"

    def _pr(pval):
        try:   return "✗ Reddedilir" if float(pval) < 0.05 else "✓ Kabul edilir"
        except: return "—"

    def _cpr(val):
        if "Kabul" in str(val):       return "color:#34d399"
        if "Reddedilir" in str(val):  return "color:#f87171"
        return ""

    tbl = pd.DataFrame([
        {"Test": "Kupiec POF (Koşulsuz Kapsam)",
         "İstatistik": _fs(bt["kupiec_stat"]),   "p-değeri": _fs(bt["kupiec_pvalue"]),
         "Sonuç": _pr(bt["kupiec_pvalue"])},
        {"Test": "Christoffersen Bağımsızlık",
         "İstatistik": _fs(bt["independence_stat"]), "p-değeri": _fs(bt["independence_pvalue"]),
         "Sonuç": _pr(bt["independence_pvalue"])},
        {"Test": "Acerbi-Szekely Z₁ (koşullu magnitüd, N_T)",
         "İstatistik": _fs(es_bt["z1_stat"]),    "p-değeri": _fs(es_bt["z1_pvalue"]),
         "Sonuç": _pr(es_bt["z1_pvalue"])},
        {"Test": "Acerbi-Szekely Z₂ (sıklık + magnitüd, Nα)",
         "İstatistik": _fs(es_bt["z2_stat"]),    "p-değeri": _fs(es_bt["z2_pvalue"]),
         "Sonuç": _pr(es_bt["z2_pvalue"])},
        {"Test": "Berkowitz PIT (Ljung-Box)",
         "İstatistik": _fs(berk["pit_acf1"]),    "p-değeri": _fs(berk["lb_pvalue_level"]),
         "Sonuç": _pr(berk["lb_pvalue_level"])},
    ])
    st.dataframe(
        tbl.style
           .map(lambda v: "color:#f87171" if (v and "." in str(v) and float(v if v != "—" else 1) < 0.05) else "",
                subset=["p-değeri"])
           .map(_cpr, subset=["Sonuç"]),
        use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    for col_s, (lbl2, val, cl) in zip([c1, c2, c3, c4], [
        ("İhlal Sayısı (toplam)", str(bt["violations"]),                "#fb923c"),
        ("İhlal Oranı",           f"{bt['violation_rate']*100:.2f}%",  "#fbbf24"),
        ("PIT Ortalama (≈ 0)",    f"{berk['pit_mean']:.4f}",           "#60a5fa"),
        ("PIT Std.Sapma (≈ 1)",   f"{berk['pit_std']:.4f}",            "#a78bfa"),
    ]):
        with col_s:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="label">{lbl2}</div>'
                f'<div class="value" style="color:{cl};font-size:1.2rem">{val}</div>'
                f'</div>', unsafe_allow_html=True)


# ─── SENTINEL ────────────────────────────────────────────────────────────────

class _NoCompute(Exception):
    pass


# ─── RENDER ─────────────────────────────────────────────────────────────────

def render():
    try:
        _render_impl()
    except Exception as _e:
        import traceback as _tb
        st.error(f"**4. Gün hata:** `{_e!r}`\n\n```\n{_tb.format_exc()}\n```")


def _render_impl():
    (tab_theory, tab_risk, tab_evt, tab_systemic,
     tab_fz, tab_combo, tab_backtest) = st.tabs([
        "📖 Teori",
        "⚙️ Temel Risk Paneli",
        "🌊 EVT & Kuyruk Analizi",
        "🔗 Sistemik Risk (CoVaR & MES)",
        "📊 Tahmin Karşılaştırması (FZ Kaybı)",
        "🔀 Öngörü Kombinasyonu",
        "🔬 Geriye Dönük Test",
    ])

    # =========================================================================
    # TAB 1 — TEORİ
    # =========================================================================
    with tab_theory:
        st.markdown("### Risk Ölçütleri & Geriye Dönük Test: Teorik Özet")

        with st.expander("📐 Risk Ölçütleri: VaR, ES & PELVE", expanded=False):
            st.markdown(r"""
**Value at Risk (VaR):**
$$\mathrm{VaR}_\alpha(L) = F_L^{-1}(1-\alpha)$$
Kayıp dağılımının $(1-\alpha)$. kantili. ⚠️ **Alt-toplayıcı değildir.**

**Beklenen Kayıp (ES):**
$$\mathrm{ES}_\alpha(L) = \frac{1}{\alpha}\int_0^\alpha \mathrm{VaR}_u(L)\,du = E[L \mid L > \mathrm{VaR}_\alpha]$$

**PELVE (Li & Wang, 2023):** $c_\alpha \in [1, 1/\alpha]$ öyle ki
$\mathrm{ES}_{1 - c_\alpha \alpha}(L) = \mathrm{VaR}_{1-\alpha}(L)$.
Normal'de $\alpha\to 0$ iken $c\to e\approx 2.718$.
""")
            st.markdown("""
| Aksiyom | VaR | ES |
|---------|:---:|:--:|
| Monotonluk | ✓ | ✓ |
| Alt-toplayıcılık | ✗ | ✓ |
| Pozitif homojenlik | ✓ | ✓ |
| Öteleme değişmezliği | ✓ | ✓ |

ES dört aksiyomu da sağlar; VaR sağlamaz (Artzner vd. 1999).
""")

        with st.expander("📊 Tahmin Yöntemleri", expanded=False):
            st.markdown("#### Klasik / Yarı-parametrik yöntemler")
            st.markdown(r"""
| Yöntem | VaR formülü | ES formülü | Not |
|--------|------------|------------|-----|
| **Normal** | $\mu + \sigma z_{1-\alpha}$ | $\mu + \sigma \phi(z_{1-\alpha})/\alpha$ | Hızlı; kalın kuyruğu ıskalamaz |
| **Student-t** | $\mu + s_t\, t_{1-\alpha,\nu}$ | $\mu + s_t\,f(\nu,\alpha)$ | $\nu$ seçimine duyarlı |
| **Tarihsel Sim.** | $Q_{1-\alpha}(\{L_t\})$ | $E[L \mid L > \mathrm{VaR}]$ | Varsayımsız; pencereye duyarlı |
| **Cornish-Fisher** | $\mu + \sigma z_{\mathrm{CF}}$ | Normal ES ($\mu,\sigma$) ile | Çarpıklık/basıklık düzeltmesi |
""")
            st.markdown(r"""
Cornish-Fisher açılımı: $z_{\mathrm{CF}} = z + \tfrac{1}{6}(z^2-1)S + \tfrac{1}{24}(z^3-3z)K - \tfrac{1}{36}(2z^3-5z)S^2$
""")
            st.divider()
            st.markdown("#### GARCH Ailesi")
            st.markdown(r"""
| Yöntem | Volatilite modeli | İnovasyon / Kuyruk | Temel özellik |
|--------|-------------------|--------------------|---------------|
| **GARCH-Normal** | GARCH(1,1) | Normal | Koşullu varyans; basit |
| **GARCH-t** | GARCH(1,1) | Student-t | Kalın kuyruk + volatilite kümelenmesi |
| **GJR-N** | GJR-GARCH(1,1,1) | Normal | Kaldıraç etkisi: $\gamma > 0$ ise negatif şok daha büyük |
| **EGARCH-t** | EGARCH(1,1,1) | Student-t | Log-spec → pozitiflik kısıtı yok |
| **GARCH-FHS** | GARCH(1,1) | Filtrelenmiş Tarihsel | Standart artıklar üzerinden ampirik kuyruk |
| **GJR-FHS** | GJR-GARCH | Filtrelenmiş Tarihsel | Kaldıraç + ampirik kuyruk |

VaR: $\widehat{\sigma}_t q_\alpha^z$ &nbsp;|&nbsp; ES: $\widehat{\sigma}_t e_\alpha^z$ &nbsp;(FHS için $q,e$: standart artık ampirik kantil/ortalaması)
""")
            st.divider()
            st.markdown("#### Yarı-parametrik: FZ-GAS, FZ-GARCH, CaViaR-AsL")
            st.markdown(r"""
| Yöntem | Dinamik | ES | Referans |
|--------|---------|-----|---------|
| **FZ-GAS** | GAS filtresi: $k_{t+1} = \omega + \beta k_t + A\cdot s_t^{\mathrm{FZ}}$ | $e_t = b\exp(k_t)$ | Patton, Ziegel, Chen (2019) |
| **FZ-GARCH-N** | GARCH vol; FZ0 kalibrasyonu; Normal inovasyon | $e_t = e_\alpha^N \sigma_t$ | — |
| **FZ-GARCH-t** | GARCH vol; FZ0 kalibrasyonu; Student-t | $e_t = e_\alpha^t \sigma_t$ | — |
| **CaViaR-AsL(simple)** | AS-CaViaR kantil özyinelemesi + sabit oran | $\xi_t = \lambda_0 / |q_t|$ | Taylor (2019) |
| **CaViaR-AsL(dyn)** | AS-CaViaR + dinamik spread | $\xi_t = \lambda_0 + \lambda_1 q_t$ | Taylor (2019) |

**FZ0 kaybı (içsel):**
$\ell(v,e;y) = -\tfrac{1}{\alpha e}\mathbf{1}_{y < v}(v-y) + \tfrac{v}{e} + \log(-e) - 1$
&nbsp;(kayıp sözleşmesi: $v,e < 0$; dışarıya pozitif büyüklük döndürülür)
""")

        with st.expander("🔀 Öngörü Kombinasyonu", expanded=False):
            st.markdown(r"""
**Motivasyon:** Hiçbir tek model veri üretme sürecini tam olarak bilmez. Birleştirme
(combination/pooling) modele özgü hatayı çeşitlendirerek ortalama kaybı düşürür (Timmermann 2006).
ES için tek başına *elicitability* olmadığından, kombinasyonun teorik dayanağı
**Fissler-Ziegel (2016)** ortak kaybıdır.

---
**FZ0 Ortak Kaybı** (dış pozitif sözleşme; burada kayıp $\ell_t = -y_t$):

$$S(v,e;\ell_t) = \bigl(\mathbf{1}_{\ell_t>v}-\alpha\bigr)(-v)
 - \mathbf{1}_{\ell_t>v}\ell_t
 + \tfrac{-1}{e}\!\left(e + \tfrac{\ell_t - v}{\alpha}\mathbf{1}_{\ell_t>v}\right)
 + \log e - 1$$

Düşük $\bar{S}$ → daha iyi (VaR, ES) çifti.

---
**Min-skor kombinasyonu** (Taylor 2020):

$$\min_{w^Q,\,w^S \geq 0,\;\mathbf{1}^\top w = 1}
  \sum_t S\!\left(\sum_j w^Q_j v_{jt},\; \sum_j w^S_j e_{jt};\; \ell_t\right)$$

$w^Q$ VaR kantilini, $w^S$ ES'yi ayrı ağırlıklar ile birleştirir (spacing trick ile dışbükey).

---
**Göreli-skor kombinasyonu** (softmax):

$$w_j^{\mathrm{rel}} = \frac{\exp(-\lambda \bar{S}_j)}{\sum_k \exp(-\lambda \bar{S}_k)}$$

Daha düşük bireysel FZ kaybına daha yüksek ağırlık. $\lambda$ arama ile seçilir.

---
**Beceri Skoru** (referans modele göre):

$$\mathrm{Skill}(M,\,\mathrm{ref}) = \left(\frac{\bar{S}_M}{\bar{S}_{\mathrm{ref}}} - 1\right)\times 100\%$$

Pozitif → referanstan daha iyi; negatif → daha kötü.
""")
            st.info(
                "**Taylor (2020) bulgusu:** Tek bir seriye bakıldığında en iyi bireysel model "
                "kombinasyonu geçebilir. Ancak çok-varlık ortalamasında kombinasyon avantajı "
                "tutarlı biçimde belirginleşir.")

        with st.expander("🔬 Elicitability & Geriye Dönük Test", expanded=False):
            st.markdown(r"""
**Fissler-Ziegel (2016):** ES tek başına elicitable **değil**; (VaR, ES) çifti **birlikte** elicitable.

**Basel Trafik Işığı (250 gün, %99 VaR):**

| İhlal | Bölge | Çarpan |
|-------|-------|--------|
| 0–4 | 🟢 Yeşil | 3.00 |
| 5–9 | 🟡 Sarı | 3.20–4.00 |
| ≥10 | 🔴 Kırmızı | 4.00 |
""")
            st.info(
                "**Kupiec (1995) POF:** İhlal oranı testi.  \n"
                "**Christoffersen (1998):** İhlal bağımsızlığı.  \n"
                "**Acerbi-Szekely (2014) Z₁/Z₂:** ES testi.  \n"
                "**Berkowitz (2001) PIT:** Tam dağılım testi.")

        st.divider()
        try:
            from pathlib import Path as _Path
            _nb = _Path(__file__).parent.parent / "notebooks" / "gun4_risk.ipynb"
            st.download_button("📥 Jupyter Not Defteri İndir (gun4_risk.ipynb)",
                               _nb.read_bytes(), "gun4_risk.ipynb",
                               "application/json", use_container_width=True)
        except FileNotFoundError:
            st.caption("📓 `notebooks/gun4_risk.ipynb` bulunamadı.")
        except Exception as _e:
            st.caption(f"Notebook hatası: {_e}")


    # =========================================================================
    # TAB 2 — TEMEL RİSK PANELİ
    # =========================================================================
    with tab_risk:
        try:
            st.markdown("### Temel Risk Paneli")
            df = st.session_state.returns_df
            all_cols = _asset_cols(df); df_key = id(df)

            col_a, col_b, col_c = st.columns(3)
            with col_a: asset   = st.selectbox("Varlık", all_cols, key="rp2_asset")
            with col_b: alpha   = st.select_slider("Anlamlılık Düzeyi α",
                                                    [0.01,0.025,0.05,0.10],
                                                    value=0.05, key="rp2_alpha")
            with col_c: window  = st.slider("Kayan Pencere (gün)", 60, 500, 250, 10, key="rp2_window")

            if window > len(df[asset].dropna()) // 2:
                st.warning("Pencere veri uzunluğunun yarısından büyük — tahminler güvenilir olmayabilir.")

            _key = f"d4_risk_{asset}_{alpha}_{window}_{df_key}"
            if st.button("🔄 Hesapla", key="d4_risk_run", type="primary"):
                with st.spinner("Risk ölçütleri hesaplanıyor..."):
                    try:
                        _r  = compute_rolling_risk(asset, alpha, window, df_key)
                        _sdf, _spelve = compute_static_risk(asset, alpha, df_key)
                        st.session_state[_key] = (_r, _sdf, _spelve)
                    except Exception as _exc:
                        st.error(f"Hesaplama hatası: `{_exc}`"); raise _NoCompute

            if _key not in st.session_state:
                st.info("⬆️ Parametreleri seçip **🔄 Hesapla**'ya tıklayın."); raise _NoCompute

            res, static_df, static_pelve = st.session_state[_key]
            idx   = res["idx"]; rets = res["returns"]
            var_d = res["var"]; pelve = res.get("pelve", np.full(len(idx), np.nan))

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=idx, y=rets, mode="lines", name="Günlük Getiri",
                                      line=dict(color="rgba(96,165,250,0.45)", width=0.8)))
            for meth, color in list(COLORS_METHOD.items())[:4]:
                if meth not in var_d: continue
                fig1.add_trace(go.Scatter(x=idx, y=-var_d[meth], mode="lines",
                                          name=f"VaR — {meth}",
                                          line=dict(color=color, width=1.4, dash="dash")))
            fig1.update_layout(template=PLOT_TEMPLATE,
                               title=f"Kayan VaR Tahminleri — {asset} (α={alpha})",
                               xaxis_title="Tarih", yaxis_title="Getiri / VaR",
                               height=380, margin=dict(l=20,r=20,t=50,b=30),
                               legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=idx, y=pelve, mode="lines", name="PELVE (c)",
                                      line=dict(color="#fbbf24", width=1.6)))
            fig2.add_hline(y=np.e, line_dash="dot", line_color="#a78bfa",
                           annotation_text="Normal: e ≈ 2.718", annotation_position="bottom right")
            fig2.add_hline(y=2.5, line_dash="dot", line_color="#f472b6",
                           annotation_text="Basel FRTB: c = 2.5", annotation_position="top right")
            fig2.update_layout(template=PLOT_TEMPLATE, title=f"Kayan PELVE — {asset}",
                               xaxis_title="Tarih", yaxis_title="c (PELVE)",
                               height=270, margin=dict(l=20,r=20,t=50,b=30))
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("#### Güncel Risk Ölçütleri (Son Pencere)")
            mc = st.columns(6)
            card_data = [
                ("VaR — Normal",    f"{var_d['Normal'][-1]*100:.3f}%",         "#a78bfa"),
                ("VaR — Student-t", f"{var_d['Student-t'][-1]*100:.3f}%",      "#34d399"),
                ("VaR — Hist.Sim.", f"{var_d['Hist.Sim.'][-1]*100:.3f}%",      "#f472b6"),
                ("VaR — CF",        f"{var_d['Cornish-Fisher'][-1]*100:.3f}%", "#fbbf24"),
                ("PELVE (son)",     f"{pelve[-1]:.3f}",                         "#60a5fa"),
                ("İhlal (Normal)",  str(int(np.sum(-rets > var_d['Normal']))),  "#fb923c"),
            ]
            for col_m, (lbl, val, clr) in zip(mc, card_data):
                with col_m:
                    st.markdown(f'<div class="metric-card">'
                                f'<div class="label">{lbl}</div>'
                                f'<div class="value" style="color:{clr};font-size:1.2rem">{val}</div>'
                                f'</div>', unsafe_allow_html=True)
            st.markdown("")
            st.markdown("#### Tam Örneklem VaR / ES / PELVE Karşılaştırması")
            disp = static_df.copy()
            disp["VaR (%)"] = (disp["VaR"]*100).round(4)
            disp["ES (%)"]  = (disp["ES"]*100).round(4)
            disp["PELVE"]   = round(static_pelve, 4)
            disp = disp.rename(columns={"Yontem":"Yöntem"})[["Yöntem","VaR (%)","ES (%)","PELVE"]]
            st.dataframe(disp.style.background_gradient(subset=["VaR (%)","ES (%)"], cmap="RdPu"),
                         use_container_width=True, hide_index=True)
        except _NoCompute:
            pass


    # =========================================================================
    # TAB 3 — EVT & KUYRUK ANALİZİ
    # =========================================================================
    with tab_evt:
        try:
            st.markdown("### EVT — Aşım Eşiği Yöntemi (POT / GPD)")
            df = st.session_state.returns_df
            all_cols = _asset_cols(df); df_key = id(df)

            col_a, col_b, col_c = st.columns(3)
            with col_a: asset_e  = st.selectbox("Varlık", all_cols, key="evt_asset")
            with col_b: thresh_q = st.slider("Eşik Kantil (u)", 0.80, 0.98, 0.90, 0.01, key="evt_thresh")
            with col_c: alpha_e  = st.slider("α (VaR/ES düzeyi)", 0.01, 0.10, 0.05, 0.005,
                                              key="evt_alpha", format="%.3f")

            _key = f"d4_evt_{asset_e}_{alpha_e}_{thresh_q}_{df_key}"
            if st.button("🔄 Hesapla", key="d4_evt_run", type="primary"):
                with st.spinner("EVT hesaplanıyor..."):
                    try:
                        st.session_state[_key] = compute_evt(asset_e, alpha_e, thresh_q, df_key)
                    except Exception as _exc:
                        st.error(f"Hesaplama hatası: `{_exc}`"); raise _NoCompute

            if _key not in st.session_state:
                st.info("⬆️ Parametreleri seçip **🔄 Hesapla**'ya tıklayın."); raise _NoCompute

            evt, v_norm, u_vals, me_vals, gpd_x, gpd_emp, gpd_theo = st.session_state[_key]
            if evt.get("warning"): st.warning(evt["warning"])

            xi_val = evt["xi"]
            if not np.isnan(xi_val):
                if   xi_val >  0.05: st.error(f"ξ = {xi_val:.4f} > 0 → **Ağır kuyruk (Fréchet)**")
                elif xi_val < -0.05: st.success(f"ξ = {xi_val:.4f} < 0 → **Sınırlı kuyruk (Weibull)**")
                else:                st.info(f"ξ = {xi_val:.4f} ≈ 0 → **Hafif kuyruk (Gumbel)**")

            losses_full = -df[asset_e].dropna().values
            u_thresh    = evt["threshold"]
            col_left, col_right = st.columns(2)

            with col_left:
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(x=losses_full, nbinsx=60, name="Tüm Kayıplar",
                                             marker_color="rgba(96,165,250,0.5)"))
                exc_mask = losses_full > u_thresh
                fig_h.add_trace(go.Histogram(x=losses_full[exc_mask], nbinsx=30,
                                             name="Aşımlar (L>u)",
                                             marker_color="rgba(251,146,60,0.75)"))
                fig_h.add_vline(x=u_thresh, line_color="#f472b6", line_dash="dash",
                                annotation_text=f"u={u_thresh:.4f}",
                                annotation_font_color="#f472b6")
                fig_h.update_layout(template=PLOT_TEMPLATE,
                                    title=f"Kayıp Histogramı — u={u_thresh:.4f}",
                                    barmode="overlay", height=300,
                                    margin=dict(l=20,r=20,t=50,b=30),
                                    legend=dict(orientation="h", y=-0.3))
                st.plotly_chart(fig_h, use_container_width=True)

            with col_right:
                if len(gpd_x) > 0:
                    fig_gpd = go.Figure()
                    fig_gpd.add_trace(go.Scatter(x=gpd_x, y=gpd_emp, mode="markers",
                                                 name="Ampirik ECDF",
                                                 marker=dict(color="#60a5fa", size=5)))
                    fig_gpd.add_trace(go.Scatter(x=gpd_x, y=gpd_theo, mode="lines",
                                                 name="GPD CDF (teorik)",
                                                 line=dict(color="#f472b6", width=2)))
                    fig_gpd.update_layout(template=PLOT_TEMPLATE,
                                          title=f"GPD Uyumu: ξ={xi_val:.4f}, σ={evt['sigma']:.4f}",
                                          height=300, margin=dict(l=20,r=20,t=50,b=30),
                                          legend=dict(orientation="h", y=-0.3))
                    st.plotly_chart(fig_gpd, use_container_width=True)
                else:
                    st.info("GPD grafiği için yeterli aşım yok.")

            valid_me = [(u_vals[i], me_vals[i]) for i in range(len(me_vals))
                        if not np.isnan(me_vals[i])]
            if valid_me:
                u_v, me_v = zip(*valid_me)
                fig_me = go.Figure()
                fig_me.add_trace(go.Scatter(x=list(u_v), y=list(me_v), mode="lines+markers",
                                            line=dict(color="#34d399", width=2),
                                            marker=dict(size=5)))
                fig_me.add_vline(x=u_thresh, line_color="#f472b6", line_dash="dash",
                                 annotation_text="Seçili u", annotation_font_color="#f472b6")
                fig_me.update_layout(template=PLOT_TEMPLATE,
                                     title="Ortalama Aşım Grafiği",
                                     xaxis_title="Eşik u", yaxis_title="E[L−u | L>u]",
                                     height=260, margin=dict(l=20,r=20,t=50,b=30))
                st.plotly_chart(fig_me, use_container_width=True)
                st.caption("Doğrusal artış → GPD geçerli. Azalma → sınırlı kuyruk.")

            st.markdown("#### EVT Sonuçları")
            st.table(pd.DataFrame({
                "Parametre": ["Eşik u","Aşım Sayısı Nᵤ","ξ","σ",
                              f"EVT VaR (α={alpha_e})",f"EVT ES (α={alpha_e})","Normal VaR (karş.)"],
                "Değer": [f"{evt['threshold']:.6f}", str(evt['n_exceedances']),
                          f"{evt['xi']:.6f}" if not np.isnan(evt['xi']) else "—",
                          f"{evt['sigma']:.6f}" if not np.isnan(evt['sigma']) else "—",
                          f"{evt['var']:.6f}" if not np.isnan(evt['var']) else "—",
                          f"{evt['es']:.6f}"  if not np.isnan(evt['es'])  else "—",
                          f"{v_norm:.6f}"],
            }))
        except _NoCompute:
            pass


    # =========================================================================
    # TAB 4 — SİSTEMİK RİSK (CoVaR & MES)
    # =========================================================================
    with tab_systemic:
        try:
            st.markdown("### Sistemik Risk Ölçütleri: CoVaR & MES")
            df = st.session_state.returns_df
            all_cols = _asset_cols(df); df_key = id(df)

            col_a, col_b, col_c = st.columns(3)
            with col_a: asset_i      = st.selectbox("Varlık i (odak)", all_cols, index=0, key="sys_asset_i")
            with col_b: market_proxy = st.selectbox("Piyasa/Endeks j", all_cols,
                                                     index=min(1,len(all_cols)-1), key="sys_market")
            with col_c: alpha_s = st.slider("α", 0.01, 0.10, 0.05, 0.005,
                                             key="sys_alpha", format="%.3f")
            roll_win_s = st.slider("Kayan Pencere — CoVaR (gün)", 60, 500, 250, 10, key="sys_roll")

            _key = f"d4_sys_{asset_i}_{market_proxy}_{alpha_s}_{roll_win_s}_{df_key}"
            if st.button("🔄 Hesapla", key="d4_sys_run", type="primary"):
                with st.spinner("CoVaR / MES hesaplanıyor..."):
                    try:
                        _ri, _rm, _cov, _mes = compute_systemic(asset_i, market_proxy, alpha_s, df_key)
                        try:    _rcov = compute_rolling_covar(asset_i, market_proxy, alpha_s, roll_win_s, df_key)
                        except: _rcov = None
                        st.session_state[_key] = (_ri, _rm, _cov, _mes, _rcov)
                    except Exception as _exc:
                        st.error(f"Hesaplama hatası: `{_exc}`"); raise _NoCompute

            if _key not in st.session_state:
                st.info("⬆️ Parametreleri seçip **🔄 Hesapla**'ya tıklayın."); raise _NoCompute

            ret_i, ret_m, covar_res, mes_res, roll_cov_df = st.session_state[_key]

            col_sc, col_mc = st.columns([2, 1])
            with col_sc:
                var_j = covar_res["var_j"]; beta0 = covar_res["beta0"]; beta1 = covar_res["beta1"]
                x_line = np.linspace(ret_m.min(), ret_m.max(), 100)
                y_line = beta0 + beta1 * x_line
                crisis_mask = ret_m <= var_j
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(x=ret_m[crisis_mask], y=ret_i[crisis_mask],
                                            mode="markers", name="Kriz Günleri",
                                            marker=dict(color="#fb923c", size=5, opacity=0.75)))
                fig_sc.add_trace(go.Scatter(x=ret_m[~crisis_mask], y=ret_i[~crisis_mask],
                                            mode="markers", name="Normal Günler",
                                            marker=dict(color="rgba(96,165,250,0.3)", size=3)))
                fig_sc.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                            name=f"Kantil Regresyon (α={alpha_s})",
                                            line=dict(color="#a78bfa", width=2)))
                fig_sc.add_vline(x=var_j, line_color="#f472b6", line_dash="dash",
                                 annotation_text=f"VaR_j={var_j:.4f}",
                                 annotation_font_color="#f472b6")
                fig_sc.update_layout(template=PLOT_TEMPLATE,
                                     title=f"Saçılım: {asset_i} vs {market_proxy}",
                                     xaxis_title=f"{market_proxy} Getiri",
                                     yaxis_title=f"{asset_i} Getiri",
                                     height=360, margin=dict(l=20,r=20,t=50,b=30),
                                     legend=dict(orientation="h", y=-0.28))
                st.plotly_chart(fig_sc, use_container_width=True)

            with col_mc:
                covar_val = covar_res["covar_alpha"]; delta_val = covar_res["delta_covar"]
                mes_val   = mes_res["mes"];            beta_tail = mes_res["beta_tail"]
                st.markdown("<br>", unsafe_allow_html=True)
                for lbl, val, clr in [
                    ("CoVaR (i|j krizde)", f"{covar_val:.5f}", "#f472b6"),
                    ("ΔCoVaR (yayılım)",   f"{delta_val:.5f}", "#fb923c"),
                    ("MES", f"{mes_val:.5f}" if not np.isnan(mes_val) else "—", "#34d399"),
                    ("Tail Beta", f"{beta_tail:.4f}" if not np.isnan(beta_tail) else "—", "#60a5fa"),
                    ("Kriz Gün Sayısı", str(mes_res["n_crisis_days"]), "#a78bfa"),
                ]:
                    st.markdown(
                        f'<div class="metric-card" style="margin-bottom:0.5rem">'
                        f'<div class="label">{lbl}</div>'
                        f'<div class="value" style="color:{clr};font-size:1.15rem">{val}</div>'
                        f'</div>', unsafe_allow_html=True)

            st.markdown("#### Kayan CoVaR ve ΔCoVaR")
            if roll_cov_df is not None:
                fig_rcov = go.Figure()
                fig_rcov.add_trace(go.Scatter(x=roll_cov_df.index, y=roll_cov_df["covar"],
                                              mode="lines", name="CoVaR",
                                              line=dict(color="#a78bfa", width=1.4)))
                fig_rcov.add_trace(go.Scatter(x=roll_cov_df.index, y=roll_cov_df["delta_covar"],
                                              mode="lines", name="ΔCoVaR",
                                              line=dict(color="#fb923c", width=1.4, dash="dash")))
                fig_rcov.update_layout(template=PLOT_TEMPLATE,
                                       title=f"Kayan CoVaR — {asset_i}|{market_proxy}",
                                       height=280, margin=dict(l=20,r=20,t=50,b=30),
                                       legend=dict(orientation="h", y=-0.3))
                st.plotly_chart(fig_rcov, use_container_width=True)
            else:
                st.error("Kayan CoVaR hesaplanamadı.")
            st.info("**ΔCoVaR** = CoVaR(i|j krizde) − CoVaR(i|j medyanda): j'nin kriz anında i'ye yayılım etkisi.")
        except _NoCompute:
            pass


    # =========================================================================
    # TAB 5 — TAHMIN KARŞILAŞTIRMASI (FZ KAYBI) — bireysel modeller
    # =========================================================================
    with tab_fz:
        try:
            st.markdown("### Fissler-Ziegel Kaybı ile Bireysel Model Karşılaştırması")
            st.caption("15 modelin OOS FZ kaybı. Kombinasyon analizi için **🔀 Öngörü Kombinasyonu** sekmesine bakın.")
            df = st.session_state.returns_df
            all_cols = _asset_cols(df); df_key = id(df)

            col_a, col_b, col_c = st.columns(3)
            with col_a: asset_f  = st.selectbox("Varlık", all_cols, key="fz_asset")
            with col_b: alpha_f  = st.slider("α", 0.01, 0.10, 0.05, 0.005,
                                              key="fz_alpha", format="%.3f")
            with col_c: n_oos    = st.slider("Örneklem-Dışı Pencere (gün)", 100, 1000, 500, 50,
                                             key="fz_noos")

            _key = f"d4_fz_{asset_f}_{alpha_f}_{n_oos}_{df_key}"
            if st.button("🔄 Hesapla", key="d4_fz_run", type="primary"):
                with st.spinner("FZ kayıpları hesaplanıyor..."):
                    try:
                        st.session_state[_key] = compute_fz_comparison(asset_f, alpha_f, n_oos, df_key)
                    except Exception as _exc:
                        st.error(f"Hesaplama hatası: `{_exc}`"); raise _NoCompute

            if _key not in st.session_state:
                st.info("⬆️ Parametreleri seçip **🔄 Hesapla**'ya tıklayın."); raise _NoCompute

            fz_res = st.session_state[_key]
            fz = fz_res["fz"]; vm = fz_res["var_mean"]; em = fz_res["es_mean"]
            methods_ok = [m for m in fz if not np.isnan(fz[m])]

            if not methods_ok:
                st.error("FZ kaybı hesaplanamadı — veri yetersiz."); raise _NoCompute

            best = min(methods_ok, key=lambda m: fz[m])
            bar_colors = ["#34d399" if m == best else COLORS_METHOD.get(m, "#999999")
                          for m in methods_ok]
            fig_fz = go.Figure(go.Bar(
                x=methods_ok, y=[fz[m] for m in methods_ok],
                marker_color=bar_colors,
                text=[f"{fz[m]:.4f}" for m in methods_ok], textposition="auto"))
            fig_fz.update_layout(template=PLOT_TEMPLATE,
                                 title=f"FZ Ortak Kaybı — {asset_f} (α={alpha_f}, son {n_oos} gün)",
                                 yaxis_title="Ort. FZ Kaybı (düşük = iyi)",
                                 height=380, margin=dict(l=20,r=20,t=50,b=30))
            st.plotly_chart(fig_fz, use_container_width=True)

            rows_fz = []
            for rank, m in enumerate(sorted(methods_ok, key=lambda x: fz[x]), 1):
                rows_fz.append({"Sıra": rank, "Yöntem": m,
                                 "Ort. VaR (%)": f"{vm[m]*100:.4f}" if not np.isnan(vm[m]) else "—",
                                 "Ort. ES (%)":  f"{em[m]*100:.4f}" if not np.isnan(em[m]) else "—",
                                 "FZ Kaybı": f"{fz[m]:.6f}",
                                 "En İyi": "✓" if m == best else ""})
            st.dataframe(pd.DataFrame(rows_fz), use_container_width=True, hide_index=True)

            # Best-2 overlay
            top2 = sorted(methods_ok, key=lambda x: fz[x])[:2]
            oos_idx = fz_res["oos_idx"]; oos_ret = fz_res["oos_returns"]
            var_fcs = fz_res["var_fcs"]
            stress  = _stress_window(df.index, df[all_cols].values)
            fig_ov  = go.Figure()
            fig_ov.add_vrect(x0=stress[0], x1=stress[1], fillcolor="gray",
                             opacity=0.15, line_width=0,
                             annotation_text="stres", annotation_position="top left")
            fig_ov.add_trace(go.Scatter(x=oos_idx, y=oos_ret, mode="lines",
                                        name="Getiri", line=dict(color="rgba(150,150,150,0.5)", width=0.8)))
            for ci, m in enumerate(top2):
                v_arr  = var_fcs[m]; n_plot = min(len(oos_idx), len(v_arr))
                fig_ov.add_trace(go.Scatter(
                    x=oos_idx[-n_plot:], y=-v_arr[-n_plot:], mode="lines",
                    name=f"VaR — {m}",
                    line=dict(color=["#009E73","#CC79A7"][ci], width=1.6, dash="dash")))
            fig_ov.update_layout(template=PLOT_TEMPLATE,
                                 title=f"En İyi 2 Model VaR — {asset_f}",
                                 height=290, margin=dict(l=20,r=20,t=50,b=30),
                                 legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig_ov, use_container_width=True)
            st.success(f"**En iyi model:** {best} (FZ = {fz[best]:.6f})")
        except _NoCompute:
            pass


    # =========================================================================
    # TAB 6 — ÖNGÖRÜ KOMBİNASYONU
    # =========================================================================
    with tab_combo:
        try:
            st.markdown("### Öngörü Kombinasyonu — Model Seçimi, Ağırlıklar & Yollar")
            df = st.session_state.returns_df
            all_cols = _asset_cols(df); df_key = id(df)

            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a: asset_c  = st.selectbox("Varlık", all_cols, key="combo_asset")
            with col_b: alpha_c  = st.slider("α", 0.01, 0.10, 0.05, 0.005,
                                              key="combo_alpha", format="%.3f")
            with col_c: n_oos_c  = st.slider("OOS Penceresi (gün)", 100, 1000, 500, 50,
                                             key="combo_noos")
            with col_d:
                ref_opts  = [m for m in _ALL_METHODS]
                ref_def   = ref_opts.index("Hist.Sim.") if "Hist.Sim." in ref_opts else 0
                ref_model = st.selectbox("Referans modeli (beceri skoru)",
                                         ref_opts, index=ref_def, key="combo_ref")

            selected = st.multiselect(
                "Kombine edilecek modeller (en az 2 gerekli)",
                _ALL_METHODS, default=_DEFAULT_COMBO, key="combo_sel")

            _key = f"d4_combo_{asset_c}_{alpha_c}_{n_oos_c}_{tuple(sorted(selected))}_{df_key}"
            if st.button("🔄 Hesapla", key="d4_combo_run", type="primary"):
                if len(selected) < 2:
                    st.error("En az 2 model seçin."); raise _NoCompute
                with st.spinner("Kombinasyon hesaplanıyor..."):
                    try:
                        st.session_state[_key] = compute_combination_tab(
                            asset_c, alpha_c, n_oos_c, tuple(sorted(selected)), df_key)
                    except Exception as _exc:
                        st.error(f"Hesaplama hatası: `{_exc}`"); raise _NoCompute

            if _key not in st.session_state:
                st.info("⬆️ Modelleri seçip **🔄 Hesapla**'ya tıklayın."); raise _NoCompute

            cres = st.session_state[_key]
            if not cres["ok"]:
                st.warning(f"Kombinasyon için yeterli geçerli model/gözlem yok. "
                           f"Geçerli: {cres['valid']}"); raise _NoCompute

            valid    = cres["valid"]; n_valid = cres["n_valid"]
            idx_v    = cres["idx_valid"]; y_v = cres["oos_returns_valid"]
            all_fz   = cres["all_fz"]

            # ── Beceri Skoru hesapla ─────────────────────────────────────────
            ref_fz_val = all_fz.get(ref_model, np.nan)
            def _skill(fz_val):
                if np.isnan(ref_fz_val) or ref_fz_val == 0: return np.nan
                return (fz_val / ref_fz_val - 1) * 100.0

            # ── FZ Kaybı & Beceri Skoru barplot ──────────────────────────────
            all_labels = list(cres["indiv_fz"].keys()) + list(cres["combo_fz"].keys())
            all_skills = [_skill(cres["indiv_fz"][m]) for m in cres["indiv_fz"]] + \
                         [_skill(v) for v in cres["combo_fz"].values()]
            is_combo   = [False]*len(cres["indiv_fz"]) + [True]*len(cres["combo_fz"])
            bar_clrs   = [_COMBO_COLORS.get(lb, "#0072B2") if ic
                          else COLORS_METHOD.get(lb, "#999999")
                          for lb, ic in zip(all_labels, is_combo)]
            fig_sk = go.Figure(go.Bar(
                x=all_labels, y=[s if not np.isnan(s) else 0 for s in all_skills],
                marker_color=bar_clrs,
                text=[f"{s:.2f}%" if not np.isnan(s) else "—" for s in all_skills],
                textposition="auto"))
            fig_sk.add_hline(y=0, line_dash="dot", line_color="white", line_width=1)
            fig_sk.update_layout(
                template=PLOT_TEMPLATE,
                title=f"AL Beceri Skoru vs {ref_model} — {asset_c} "
                      f"(α={alpha_c}, {n_valid} geçerli gün)",
                yaxis_title=f"Beceri Skoru (%, {ref_model}'e göre; pozitif = daha iyi)",
                height=380, margin=dict(l=20,r=20,t=60,b=30))
            st.plotly_chart(fig_sk, use_container_width=True)

            # ── Ağırlık Tablosu ───────────────────────────────────────────────
            st.markdown("#### Kombinasyon Ağırlıkları")
            w   = cres["weights"]
            wdf = pd.DataFrame({
                "Yöntem":            valid,
                "Min-skor w^Q":      [f"{x:.3f}" for x in w["wQ"]],
                "Min-skor w^S":      [f"{x:.3f}" for x in w["wS"]],
                "Göreli-skor w":     [f"{x:.3f}" for x in w["w_rel"]],
            })
            st.dataframe(wdf, use_container_width=True, hide_index=True)
            st.caption(f"Göreli-skor λ = {w['lam']:.4f}. "
                       "w^Q: VaR kantil ağırlığı; w^S: ES spacing ağırlığı.")

            # ── VaR Yolları ───────────────────────────────────────────────────
            st.markdown("#### VaR Yolları — Bireysel Modeller (ince) + Kombinasyon (kalın)")
            fig_var = go.Figure()
            fig_var.add_trace(go.Scatter(
                x=idx_v, y=y_v, mode="lines", name="Getiri (OOS)",
                line=dict(color="rgba(180,180,180,0.3)", width=0.7)))
            for m in valid:
                v_arr = cres["var_fcs_valid"][m]
                fig_var.add_trace(go.Scatter(
                    x=idx_v, y=v_arr, mode="lines", name=m,
                    line=dict(color=COLORS_METHOD.get(m,"#999999"),
                              width=0.9, dash="dot"), opacity=0.65))
            for cm, cv in cres["combo_var"].items():
                fig_var.add_trace(go.Scatter(
                    x=idx_v, y=cv, mode="lines", name=f"[{cm}]",
                    line=dict(color=_COMBO_COLORS[cm], width=2.5)))
            fig_var.update_layout(template=PLOT_TEMPLATE,
                                  title=f"OOS VaR — {asset_c}",
                                  xaxis_title="Tarih", yaxis_title="VaR (kayıp büyüklüğü)",
                                  height=380, margin=dict(l=20,r=20,t=50,b=30),
                                  legend=dict(orientation="h", y=-0.35, font_size=10))
            st.plotly_chart(fig_var, use_container_width=True)

            # ── ES Yolları ────────────────────────────────────────────────────
            st.markdown("#### ES Yolları — Bireysel Modeller (ince) + Kombinasyon (kalın)")
            fig_es = go.Figure()
            fig_es.add_trace(go.Scatter(
                x=idx_v, y=y_v, mode="lines", name="Getiri (OOS)",
                line=dict(color="rgba(180,180,180,0.3)", width=0.7)))
            for m in valid:
                e_arr = cres["es_fcs_valid"][m]
                fig_es.add_trace(go.Scatter(
                    x=idx_v, y=e_arr, mode="lines", name=m,
                    line=dict(color=COLORS_METHOD.get(m,"#999999"),
                              width=0.9, dash="dot"), opacity=0.65))
            for cm, ce in cres["combo_es"].items():
                fig_es.add_trace(go.Scatter(
                    x=idx_v, y=ce, mode="lines", name=f"[{cm}]",
                    line=dict(color=_COMBO_COLORS[cm], width=2.5)))
            fig_es.update_layout(template=PLOT_TEMPLATE,
                                 title=f"OOS ES — {asset_c}",
                                 xaxis_title="Tarih", yaxis_title="ES (kayıp büyüklüğü)",
                                 height=380, margin=dict(l=20,r=20,t=50,b=30),
                                 legend=dict(orientation="h", y=-0.35, font_size=10))
            st.plotly_chart(fig_es, use_container_width=True)

            # ── Özet ─────────────────────────────────────────────────────────
            best_combo  = min(cres["combo_fz"], key=cres["combo_fz"].get)
            best_indiv  = min(cres["indiv_fz"], key=cres["indiv_fz"].get)
            best_c_skill = _skill(cres["combo_fz"][best_combo])
            best_i_skill = _skill(cres["indiv_fz"][best_indiv])
            if not np.isnan(best_c_skill) and not np.isnan(best_i_skill):
                if best_c_skill > best_i_skill:
                    st.success(f"En iyi kombinasyon **[{best_combo}]** (beceri={best_c_skill:.2f}%) "
                               f"en iyi bireyseli **{best_indiv}** ({best_i_skill:.2f}%) geçiyor.")
                else:
                    st.info(f"Bu seride en iyi bireysel **{best_indiv}** ({best_i_skill:.2f}%) "
                            f"en iyi kombinasyonu **[{best_combo}]** ({best_c_skill:.2f}%) geçiyor. "
                            "Kombinasyon avantajı çok-seri ortalamasında belirginleşir (Taylor 2020).")
        except _NoCompute:
            pass


    # =========================================================================
    # TAB 7 — GERİYE DÖNÜK TEST (2 KATMAN: BİREYSEL & KOMBİNASYON)
    # =========================================================================
    with tab_backtest:
        try:
            st.markdown("### Geriye Dönük Test (Backtesting)")
            df = st.session_state.returns_df
            all_cols = _asset_cols(df); df_key = id(df)

            bt_mode = st.radio("Test Modu", ["Bireysel Model", "Kombinasyon"],
                               horizontal=True, key="bt_mode")
            st.divider()

            # ── Bireysel Model ─────────────────────────────────────────────────
            if bt_mode == "Bireysel Model":

                col_a, col_b, col_c = st.columns(3)
                with col_a: asset_b  = st.selectbox("Varlık", all_cols, key="bt2_asset")
                with col_b: alpha_b  = st.slider("α", 0.01, 0.10, 0.05, 0.005,
                                                  key="bt2_alpha", format="%.3f")
                with col_c: model_b  = st.selectbox("Model", _ALL_METHODS, key="bt2_model")

                is_baseline = model_b in _BASELINE_METHODS

                if is_baseline:
                    col_d, col_e = st.columns(2)
                    with col_d: window_b = st.slider("Tahmin Penceresi (gün)", 60, 500, 250, 10, key="bt2_win")
                    with col_e: n_obs_b  = st.slider("Test Penceresi (gün)", 100, 1000, 500, 50, key="bt2_nobs")
                    _key = f"d4_bt_{asset_b}_{alpha_b}_{model_b}_{window_b}_{n_obs_b}_{df_key}"
                    if st.button("🔄 Hesapla", key="d4_bt_run", type="primary"):
                        with st.spinner("Backtest yapılıyor..."):
                            try:
                                st.session_state[_key] = compute_backtest(
                                    asset_b, alpha_b, model_b, window_b, n_obs_b, df_key)
                            except Exception as _exc:
                                st.error(f"Hesaplama hatası: `{_exc}`"); raise _NoCompute
                    if _key not in st.session_state:
                        st.info("⬆️ Parametreleri seçip **🔄 Hesapla**'ya tıklayın."); raise _NoCompute
                    btr = st.session_state[_key]
                    _display_backtest_results(btr["ret_oos"], btr["var_oos"], btr["es_oos"],
                                              btr["idx_oos"], alpha_b, model_b, df, all_cols)

                else:
                    # GARCH / SP model — OOS forecasts from compute_fz_comparison
                    col_d, = [st.columns(1)[0]]
                    n_oos_b = st.slider("OOS Penceresi (gün)", 100, 1000, 500, 50, key="bt2_noos2")
                    _key = f"d4_bt_sp_{asset_b}_{alpha_b}_{model_b}_{n_oos_b}_{df_key}"
                    if st.button("🔄 Hesapla", key="d4_bt_run2", type="primary"):
                        with st.spinner(f"{model_b} backtest yapılıyor..."):
                            try:
                                _fzr = compute_fz_comparison(asset_b, alpha_b, n_oos_b, df_key)
                                st.session_state[_key] = _fzr
                            except Exception as _exc:
                                st.error(f"Hesaplama hatası: `{_exc}`"); raise _NoCompute
                    if _key not in st.session_state:
                        st.info("⬆️ Parametreleri seçip **🔄 Hesapla**'ya tıklayın."); raise _NoCompute

                    _fzr = st.session_state[_key]
                    if model_b not in _fzr["var_fcs"]:
                        st.error(f"**{model_b}** tahmin edilemedi — OOS penceresini küçültün veya farklı bir varlık seçin.")
                        raise _NoCompute
                    var_oos_b  = _fzr["var_fcs"][model_b]
                    es_oos_b   = _fzr["es_fcs"][model_b]
                    ret_oos_b  = _fzr["oos_returns"]
                    idx_oos_b  = _fzr["oos_idx"]
                    mask_b     = np.isfinite(var_oos_b) & np.isfinite(es_oos_b)
                    if mask_b.sum() < 30:
                        st.error("Yeterli geçerli gözlem yok — OOS penceresini değiştirin.")
                        raise _NoCompute
                    _display_backtest_results(
                        ret_oos_b[mask_b], var_oos_b[mask_b], es_oos_b[mask_b],
                        idx_oos_b[mask_b], alpha_b, model_b, df, all_cols)

            # ── Kombinasyon ────────────────────────────────────────────────────
            else:
                col_a, col_b, col_c = st.columns(3)
                with col_a: asset_b   = st.selectbox("Varlık", all_cols, key="bt_combo_asset")
                with col_b: alpha_b   = st.slider("α", 0.01, 0.10, 0.05, 0.005,
                                                   key="bt_combo_alpha", format="%.3f")
                with col_c: n_oos_b   = st.slider("OOS Penceresi (gün)", 100, 1000, 500, 50,
                                                   key="bt_combo_noos")

                sel_combo = st.multiselect(
                    "Kombine edilecek modeller (en az 2)",
                    _ALL_METHODS, default=_DEFAULT_COMBO, key="bt_combo_sel")
                combo_meth = st.selectbox("Kombinasyon yöntemi",
                                          list(_COMBO_COLORS.keys()), key="bt_combo_meth")

                _key = (f"d4_btcombo_{asset_b}_{alpha_b}_{n_oos_b}_"
                        f"{tuple(sorted(sel_combo))}_{combo_meth}_{df_key}")
                if st.button("🔄 Hesapla", key="d4_bt_combo_run", type="primary"):
                    if len(sel_combo) < 2:
                        st.error("En az 2 model seçin."); raise _NoCompute
                    with st.spinner("Kombinasyon + backtest hesaplanıyor..."):
                        try:
                            _cr = compute_combination_tab(
                                asset_b, alpha_b, n_oos_b, tuple(sorted(sel_combo)), df_key)
                            st.session_state[_key] = _cr
                        except Exception as _exc:
                            st.error(f"Hesaplama hatası: `{_exc}`"); raise _NoCompute

                if _key not in st.session_state:
                    st.info("⬆️ Parametreleri seçip **🔄 Hesapla**'ya tıklayın."); raise _NoCompute

                _cr = st.session_state[_key]
                if not _cr["ok"]:
                    st.warning(f"Kombinasyon için yeterli geçerli model/gözlem yok. "
                               f"Geçerli: {_cr['valid']}"); raise _NoCompute

                var_oos_c = _cr["combo_var"][combo_meth]
                es_oos_c  = _cr["combo_es"][combo_meth]
                ret_oos_c = _cr["oos_returns_valid"]
                idx_oos_c = _cr["idx_valid"]

                st.info(f"**{combo_meth}** kombinasyonu | Modeller: {', '.join(_cr['valid'])} "
                        f"| {_cr['n_valid']} geçerli gün")
                _display_backtest_results(ret_oos_c, var_oos_c, es_oos_c, idx_oos_c,
                                          alpha_b, f"Kombinasyon [{combo_meth}]", df, all_cols)

            st.markdown("")
            st.markdown(
                "> **p > 0.05** → Model reddedilemez (uyumlu) | **p < 0.05** → Model başarısız")
            st.markdown(
                "**Kupiec POF:** İhlal oranı testi. "
                "**Christoffersen:** İhlal bağımsızlığı. "
                "**Acerbi-Szekely Z₁/Z₂:** ES doğruluğu. "
                "**Berkowitz PIT:** Tam dağılım.")
        except _NoCompute:
            pass
