"""
har_fit.py -- Part B of the Day-5 empirical layer.

On a controlled daily realized-volatility series (produced by the hf_sim.py DGP,
so the TRUE integrated variance IV_t is known) we:
  1. fit HAR-RV (Corsi 2009), HAR-RV-J and HAR-RV-CJ (Andersen-Bollerslev-
     Diebold 2007) by OLS with Newey-West (1987) HAC standard errors,
  2. fit a log-linear Realized GARCH (Hansen-Huang-Shek 2012) by joint MLE,
  3. fit a GARCH(1,1) baseline, and
  4. compare 1-step-ahead OOS variance forecasts by QLIKE and MSE against BOTH
     the RV proxy and the true IV, with a Diebold-Mariano (1995) test.

The HAR-RV-J vs HAR-RV-CJ distinction here is exactly the one corrected in the
lecture notes: HAR-RV-J keeps *total-RV* lags and adds a jump term; HAR-RV-CJ
splits the lags into a continuous (BPV) part and a jump part, nesting HAR-RV-J.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from numba import njit
from scipy.optimize import minimize
import statsmodels.api as sm
import hf_sim

SEED = 20260731
T_TOTAL = 1500
T_TRAIN = 1000            # in-sample; remainder is OOS


# ----------------------------------------------------------------------
# HAR regressor construction
# ----------------------------------------------------------------------
def _har_lags(x, w=5, m=22):
    """Return daily (t-1), weekly-avg, monthly-avg lag vectors, aligned so that
    row t uses information through t-1. Leading max(w,m) rows are NaN."""
    T = len(x)
    d = np.full(T, np.nan)
    wk = np.full(T, np.nan)
    mo = np.full(T, np.nan)
    for t in range(1, T):
        d[t] = x[t-1]
        if t >= w:
            wk[t] = x[t-w:t].mean()
        if t >= m:
            mo[t] = x[t-m:t].mean()
    return d, wk, mo


def fit_har_family(df):
    RV = df["RV_5min"].values
    C  = df["BPV_5min"].values                 # continuous part proxy
    J  = df["Jump"].values                     # jump part = max(0, RV-BPV)

    rv_d, rv_w, rv_m = _har_lags(RV)
    c_d,  c_w,  c_m  = _har_lags(C)
    j_d,  j_w,  j_m  = _har_lags(J)

    y = RV
    specs = {
        "HAR-RV":    np.column_stack([rv_d, rv_w, rv_m]),
        "HAR-RV-J":  np.column_stack([rv_d, rv_w, rv_m, j_d]),
        "HAR-RV-CJ": np.column_stack([c_d, c_w, c_m, j_d, j_w, j_m]),
    }
    names = {
        "HAR-RV":    ["beta_d", "beta_w", "beta_m"],
        "HAR-RV-J":  ["beta_d", "beta_w", "beta_m", "beta_j"],
        "HAR-RV-CJ": ["beta_cd", "beta_cw", "beta_cm", "beta_jd", "beta_jw", "beta_jm"],
    }
    results = {}
    for key, X in specs.items():
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        Xt = sm.add_constant(X[mask])
        model = sm.OLS(y[mask], Xt).fit(cov_type="HAC", cov_kwds={"maxlags": 10})
        results[key] = {
            "params": model.params, "tvalues": model.tvalues,
            "r2": model.rsquared, "r2_adj": model.rsquared_adj,
            "names": ["const"] + names[key], "model": model,
        }
    return results


# ----------------------------------------------------------------------
# Realized GARCH (log-linear, Hansen-Huang-Shek 2012), joint MLE
#   return eq:      r_t = sqrt(h_t) z_t,           z_t ~ N(0,1)
#   GARCH eq:       log h_t = omega + beta log h_{t-1} + gamma log x_{t-1}
#   measurement eq: log x_t = xi + phi log h_t + tau1 z_t + tau2 (z_t^2 - 1) + u_t
# ----------------------------------------------------------------------
@njit(cache=True)
def _rgarch_recursion(r, x, omega, beta, gamma, xi, phi, tau1, tau2, sig_u, logh0):
    n = r.shape[0]
    logh = np.empty(n)
    ll = 0.0
    LOG2PI = np.log(2.0 * np.pi)
    logh[0] = logh0
    for t in range(n):
        if t > 0:
            logh[t] = omega + beta * logh[t-1] + gamma * np.log(x[t-1])
        h = np.exp(logh[t])
        z = r[t] / np.sqrt(h)
        u = np.log(x[t]) - xi - phi * logh[t] - tau1 * z - tau2 * (z*z - 1.0)
        ll += -0.5 * (LOG2PI + logh[t] + r[t]*r[t] / h)          # return density
        ll += -0.5 * (LOG2PI + 2.0*np.log(sig_u) + u*u / (sig_u*sig_u))  # measurement
    return ll, logh


def fit_realized_garch(r, x):
    logx_mean = np.log(x).mean()
    logh0 = logx_mean

    def negll(theta):
        omega, beta, gamma, xi, phi, tau1, tau2, log_sig_u = theta
        sig_u = np.exp(log_sig_u)
        if beta <= -0.999 or beta >= 0.999:
            return 1e10
        ll, _ = _rgarch_recursion(r, x, omega, beta, gamma, xi, phi,
                                  tau1, tau2, sig_u, logh0)
        if not np.isfinite(ll):
            return 1e10
        return -ll

    # start values: persistence split beta+gamma~0.98
    theta0 = np.array([logx_mean*0.04, 0.55, 0.42, 0.0, 1.0, -0.05, 0.05, np.log(0.4)])
    res = minimize(negll, theta0, method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-7})
    omega, beta, gamma, xi, phi, tau1, tau2, log_sig_u = res.x
    return {
        "omega": omega, "beta": beta, "gamma": gamma, "xi": xi, "phi": phi,
        "tau1": tau1, "tau2": tau2, "sig_u": np.exp(log_sig_u),
        "persistence": beta + gamma, "loglik": -res.fun, "logh0": logh0,
        "success": res.success,
    }


@njit(cache=True)
def _rgarch_filter_h(r, x, omega, beta, gamma, logh0):
    """Filter log h_t through the sample (uses observed r, x lags)."""
    n = r.shape[0]
    logh = np.empty(n)
    logh[0] = logh0
    for t in range(1, n):
        logh[t] = omega + beta * logh[t-1] + gamma * np.log(x[t-1])
    return np.exp(logh)


# ----------------------------------------------------------------------
# GARCH(1,1) baseline via arch
# ----------------------------------------------------------------------
def fit_garch11(r_train_pct):
    from arch import arch_model
    am = arch_model(r_train_pct, vol="GARCH", p=1, q=1, mean="Zero", dist="normal")
    return am.fit(disp="off")


# ----------------------------------------------------------------------
# OOS forecast comparison
# ----------------------------------------------------------------------
def qlike(proxy, fcast):
    z = proxy / fcast
    return z - np.log(z) - 1.0

def diebold_mariano(l1, l2, h=1):
    """DM statistic for equal predictive accuracy (loss1 - loss2)."""
    d = l1 - l2
    n = len(d)
    dbar = d.mean()
    # Newey-West long-run variance of d (h-1 lags; h=1 -> just variance)
    gamma0 = np.mean((d - dbar)**2)
    lrv = gamma0
    for k in range(1, h):
        w = 1.0 - k/h
        cov = np.mean((d[k:] - dbar)*(d[:-k] - dbar))
        lrv += 2*w*cov
    dm = dbar / np.sqrt(lrv / n)
    return dm


def run_partB(T=T_TOTAL, T_train=T_TRAIN):
    df, _ = hf_sim.run(T=T)
    RV = df["RV_5min"].values
    IV = df["IV"].values
    r  = df["ret"].values

    # ---- in-sample HAR family ----
    har = fit_har_family(df.iloc[:T_train].reset_index(drop=True))

    # ---- in-sample Realized GARCH & GARCH(1,1) ----
    x = RV.copy()
    x[x <= 0] = np.nanmin(x[x > 0])
    rg = fit_realized_garch(r[:T_train], x[:T_train])
    g11 = fit_garch11(r[:T_train] * 100.0)

    # ---- OOS 1-step forecasts (fixed-parameter scheme) ----
    idx = np.arange(T_train, T)
    # HAR-RV and HAR-RV-CJ forecasts: apply coeffs to observed lags
    rv_d, rv_w, rv_m = _har_lags(RV)
    c_d, c_w, c_m = _har_lags(df["BPV_5min"].values)
    j_d, j_w, j_m = _har_lags(df["Jump"].values)

    p_rv = har["HAR-RV"]["params"]
    f_har = p_rv[0] + p_rv[1]*rv_d + p_rv[2]*rv_w + p_rv[3]*rv_m
    p_cj = har["HAR-RV-CJ"]["params"]
    f_cj = (p_cj[0] + p_cj[1]*c_d + p_cj[2]*c_w + p_cj[3]*c_m
            + p_cj[4]*j_d + p_cj[5]*j_w + p_cj[6]*j_m)

    # Realized GARCH OOS: filter h through full sample with in-sample params
    h_rg = _rgarch_filter_h(r, x, rg["omega"], rg["beta"], rg["gamma"], rg["logh0"])
    f_rg = h_rg

    # GARCH(1,1) OOS: recursion with in-sample params on returns (pct), then /1e4
    p = g11.params
    om, al, be = p["omega"], p["alpha[1]"], p["beta[1]"]
    rpct = r * 100.0
    h_g = np.empty(T); h_g[0] = np.var(rpct[:T_train])
    for t in range(1, T):
        h_g[t] = om + al * rpct[t-1]**2 + be * h_g[t-1]
    f_g11 = h_g / 1e4

    # ---- losses on OOS block, vs RV proxy and vs true IV ----
    res = {}
    for target_name, target in [("RVproxy", RV), ("trueIV", IV)]:
        rows = {}
        for name, f in [("HAR-RV", f_har), ("HAR-RV-CJ", f_cj),
                        ("Realized-GARCH", f_rg), ("GARCH(1,1)", f_g11)]:
            fo = f[idx]; to = target[idx]
            ok = np.isfinite(fo) & (fo > 0) & np.isfinite(to) & (to > 0)
            ql = qlike(to[ok], fo[ok]).mean()
            ms = ((to[ok] - fo[ok])**2).mean()
            rows[name] = {"QLIKE": ql, "MSE": ms, "q_series": qlike(to[ok], fo[ok]),
                          "mask": ok}
        res[target_name] = rows

    return df, har, rg, g11, res, (f_har, f_cj, f_rg, f_g11), idx


if __name__ == "__main__":
    df, har, rg, g11, res, fc, idx = run_partB()
    print(f"=== HAR ailesi (in-sample, T_train={T_TRAIN}; Newey-West HAC lag=10) ===")
    for key in ["HAR-RV", "HAR-RV-J", "HAR-RV-CJ"]:
        r_ = har[key]
        print(f"\n{key}:  R2={r_['r2']:.3f}  R2_adj={r_['r2_adj']:.3f}")
        for nm, pv, tv in zip(r_["names"], r_["params"], r_["tvalues"]):
            print(f"    {nm:<9} = {pv:>10.4f}  (t={tv:>6.2f})")
        if key in ("HAR-RV", "HAR-RV-J"):
            persist = r_["params"][1] + r_["params"][2] + r_["params"][3]
            print(f"    persistence (b_d+b_w+b_m) = {persist:.3f}")

    print(f"\n=== Realized GARCH (log-linear, joint MLE) ===")
    for k in ["omega","beta","gamma","xi","phi","tau1","tau2","sig_u","persistence","loglik"]:
        print(f"    {k:<12} = {rg[k]:>10.4f}")

    print(f"\n=== GARCH(1,1) baseline (returns x100) ===")
    print(f"    omega={g11.params['omega']:.4f}  alpha={g11.params['alpha[1]']:.4f}  "
          f"beta={g11.params['beta[1]']:.4f}  "
          f"persist={g11.params['alpha[1]']+g11.params['beta[1]']:.4f}")

    print(f"\n=== OOS ongoru karsilastirmasi (T_oos={len(idx)}) ===")
    for target in ["RVproxy", "trueIV"]:
        print(f"\n  Hedef = {target}:")
        print(f"    {'Model':<16}{'QLIKE':>10}{'MSE(x1e8)':>12}")
        for name in ["HAR-RV", "HAR-RV-CJ", "Realized-GARCH", "GARCH(1,1)"]:
            row = res[target][name]
            print(f"    {name:<16}{row['QLIKE']:>10.4f}{row['MSE']*1e8:>12.3f}")

    # DM tests vs GARCH(1,1) on QLIKE (trueIV)
    print(f"\n  Diebold-Mariano (QLIKE, hedef=trueIV; negatif => satir modeli GARCH'tan iyi):")
    base = res["trueIV"]["GARCH(1,1)"]
    for name in ["HAR-RV", "HAR-RV-CJ", "Realized-GARCH"]:
        m = res["trueIV"][name]
        # align on common mask
        cm = m["mask"] & base["mask"]
        l1 = qlike(df["IV"].values[idx][cm], fc[["HAR-RV","HAR-RV-CJ","Realized-GARCH"].index(name)*0 + 0][idx][cm]) if False else None
        # simpler: recompute aligned losses
        pass
    # recompute cleanly
    IVo = df["IV"].values[idx]
    fmap = {"HAR-RV": fc[0][idx], "HAR-RV-CJ": fc[1][idx],
            "Realized-GARCH": fc[2][idx], "GARCH(1,1)": fc[3][idx]}
    okb = np.isfinite(fmap["GARCH(1,1)"]) & (fmap["GARCH(1,1)"]>0) & np.isfinite(IVo) & (IVo>0)
    for name in ["HAR-RV", "HAR-RV-CJ", "Realized-GARCH"]:
        ok = okb & np.isfinite(fmap[name]) & (fmap[name]>0)
        dm = diebold_mariano(qlike(IVo[ok], fmap[name][ok]),
                             qlike(IVo[ok], fmap["GARCH(1,1)"][ok]))
        print(f"    {name:<16} vs GARCH(1,1): DM={dm:>7.2f}")
