"""
Yari-parametrik VaR-ES tahmincileri (Gun-4, Task 2).

Tumu 1-adim-ileri OOS tahmin uretir: parametreler egitim bolumunde kestirilir,
ozyineleme OOS boyunca ILERI filtrelenir (her VaR_t, t-1'e kadarki bilgiye
dayanir -> gercek 1-adim-ileri, look-ahead yok).

Konvansiyon: iceride getiri olcegi (v,e < 0); DISARI POZITIF kayip buyuklugu
dondurulur (app'teki GARCH kalibiyla uyumlu: var/es >= 0).

Modeller:
  - FZ-GAS      : bir-faktorlu GAS (Patton, Ziegel, Chen 2019), FZ0 skoru
  - FZ-GARCH    : GARCH(1,1) oynakligi FZ kaybiyla kestirilir (Normal yenilik)
  - CaViaR-AsL  : CaViaR kantil ozyinelemesi (SAV/AS) + AL ile ES (Taylor 2019)

HAR-Range app'e dahil DEGILDIR (gun-ici yuksek-dusuk 'range' verisi gerektirir;
app yalnizca getiri tutar). Notlarda kalir.
"""
import numpy as np
from scipy import optimize, stats

_TINY = 1e-8


# ============================================================================
# FZ0 kayip (getiri konvansiyonu: v,e < 0)
# ============================================================================
def _fz0_loss(y, v, e, alpha):
    e = np.minimum(e, -_TINY)                 # e < 0 guvence
    hit = (y <= v).astype(float)
    return np.mean(-(1.0 / (alpha * e)) * hit * (v - y) + v / e + np.log(-e) - 1.0)


# ============================================================================
# FZ-GAS : bir-faktorlu GAS (Patton vd. 2019)
#   v_t = a e^{k_t},  xi_t = b e^{k_t},  b < a < 0
#   k_{t+1} = w + beta k_t + gamma lambda_t
#   lambda_t = (1/(alpha e_t)) 1{y_t<=v_t} y_t - 1     (FZ0 skoru)
# ============================================================================
def _gas_filter(returns, w, beta, gamma, a, b, alpha):
    T = len(returns)
    k = np.empty(T); v = np.empty(T); e = np.empty(T)
    k0 = w / max(1.0 - beta, 1e-3)            # kosulsuz ortalama
    kt = k0
    for t in range(T):
        k[t] = kt
        ek = np.exp(np.clip(kt, -50, 50))
        v[t] = a * ek
        e[t] = b * ek
        et = min(e[t], -_TINY)
        hit = 1.0 if returns[t] <= v[t] else 0.0
        lam = (1.0 / (alpha * et)) * hit * returns[t] - 1.0
        lam = np.clip(lam, -50, 50)
        kt = w + beta * kt + gamma * lam
    return v, e


def _gas_unpack(theta):
    w = theta[0]
    beta = 1.0 / (1.0 + np.exp(-theta[1]))    # (0,1)
    gamma = theta[2]
    a = -np.exp(np.clip(theta[3], -20, 5))    # a < 0
    b = a - np.exp(np.clip(theta[4], -20, 5)) # b < a < 0
    return w, beta, gamma, a, b


def fz_gas_fit_filter(returns, alpha, start_idx):
    r = np.asarray(returns, float)
    r_tr = r[:start_idx]
    q = np.quantile(r_tr, alpha)              # baslangic VaR ~ empirik kantil (<0)
    es0 = r_tr[r_tr <= q].mean() if (r_tr <= q).any() else 1.5 * q
    a0, b0 = q, min(es0, q - _TINY)
    # theta baslangici
    th0 = np.array([np.log(abs(a0)) * 0.0 + 0.0, 2.0, 0.05,
                    np.log(abs(a0)), np.log(max(abs(b0 - a0), 1e-3))])

    def obj(theta):
        w, beta, gamma, a, b = _gas_unpack(theta)
        if not (b < a < 0):
            return 1e10
        v, e = _gas_filter(r_tr, w, beta, gamma, a, b, alpha)
        L = _fz0_loss(r_tr, v, e, alpha)
        return L if np.isfinite(L) else 1e10

    best = None
    for g0 in (0.02, 0.05, 0.1):
        th = th0.copy(); th[2] = g0
        res = optimize.minimize(obj, th, method="Nelder-Mead",
                                options={"maxiter": 3000, "xatol": 1e-6, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    w, beta, gamma, a, b = _gas_unpack(best.x)
    v_all, e_all = _gas_filter(r, w, beta, gamma, a, b, alpha)   # tum seri filtrele
    n_oos = len(r) - start_idx
    return -v_all[-n_oos:], -e_all[-n_oos:]                       # POZITIF


# ============================================================================
# CaViaR + Asimetrik Laplace (Taylor 2019)  -- SAV ve AS spesifikasyonlari
#   kantil q_t (getiri konvansiyonu, <0); pinball kaybiyla kestirim
#   ES: AL 'basit' surum -> xi_t = q_t * (1 + 1/nu_hat)  yerine
#       tail-mean orani ile: xi_t = q_t * c,  c = mean(z|z<=Q_a)/Q_a  (>1)
# ============================================================================
def _caviar_filter(returns, beta, spec):
    T = len(returns); q = np.empty(T)
    q0 = np.quantile(returns[:max(20, T // 10)], 0.05)
    qt = q0
    for t in range(T):
        q[t] = qt
        r1 = returns[t]
        if spec == "SAV":
            qt = beta[0] + beta[1] * qt + beta[2] * abs(r1)
        elif spec == "AS":
            qt = beta[0] + beta[1] * qt + beta[2] * max(r1, 0.0) + beta[3] * min(r1, 0.0)
        qt = min(qt, -_TINY)                  # kantil negatif kalsin
    return q


def _pinball(returns, q, alpha):
    d = returns - q
    return np.mean(np.where(d >= 0, alpha * d, (alpha - 1.0) * d))


def _caviar_es_dyn(returns, q, omega, gamma, beta):
    """Dinamik-toplamsal ES (Taylor 2019): ES_t=q_t-x_t; x ihlalde guncellenir,
    aksi halde tutulur. omega,gamma,beta>=0 -> x>0 -> VaR-ES kesismez."""
    T = len(returns); x = np.empty(T)
    xt = max(abs(np.mean(q[:min(200, T)])) * 0.3, 1e-3)
    for t in range(T):
        x[t] = xt
        if returns[t] <= q[t]:                       # ihlal -> x'i guncelle
            exc = q[t] - returns[t]                   # >= 0
            xt = omega + gamma * exc + beta * (-q[t]) # hepsi >=0
            xt = max(xt, 1e-4)
    return q - x                                      # ES < q (getiri konv.)


def caviar_asl_fit_filter(returns, alpha, start_idx, spec="AS", es_type="simple"):
    r = np.asarray(returns, float)
    r_tr = r[:start_idx]
    q_emp = np.quantile(r_tr, alpha)
    if spec == "SAV":
        b0 = np.array([q_emp * 0.1, 0.8, 0.1])
    else:  # AS
        b0 = np.array([q_emp * 0.1, 0.8, 0.1, -0.1])

    def obj(beta):
        if not (0.0 <= beta[1] < 1.0):
            return 1e10
        q = _caviar_filter(r_tr, beta, spec)
        return _pinball(r_tr, q, alpha)

    res = optimize.minimize(obj, b0, method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9})
    beta = res.x
    q_all = _caviar_filter(r, beta, spec)

    if es_type == "dynamic":
        q_tr = q_all[:start_idx]

        def obj_es(p):
            om, ga, be = np.exp(np.clip(p, -20, 5))
            e_tr = _caviar_es_dyn(r_tr, q_tr, om, ga, be)
            return _fz0_loss(r_tr, q_tr, e_tr, alpha)

        rese = optimize.minimize(obj_es, np.log([0.5, 0.1, 0.1]),
                                 method="Nelder-Mead", options={"maxiter": 3000})
        om, ga, be = np.exp(np.clip(rese.x, -20, 5))
        xi_all = _caviar_es_dyn(r, q_all, om, ga, be)
    else:  # 'simple' : sabit carpansal oran (egitim tail-mean/kantil)
        z = r_tr / r_tr.std()
        za = np.quantile(z, alpha)
        tail = z[z <= za]
        c = (tail.mean() / za) if (len(tail) > 3 and za < 0) else 1.25
        c = float(np.clip(c, 1.02, 2.0))
        xi_all = q_all * c

    n_oos = len(r) - start_idx
    return -q_all[-n_oos:], -xi_all[-n_oos:]                          # POZITIF


# ============================================================================
# FZ-GARCH : GARCH(1,1) oynakligi FZ kaybiyla kestirilir (Normal yenilik)
#   sigma_t^2 = w + alpha_g r_{t-1}^2 + beta_g sigma_{t-1}^2
#   v_t = -za * sigma_t,  xi_t = -ea * sigma_t   (za,ea Normal'den, <0)
# ============================================================================
def _garch_vol(returns, wg, ag, bg):
    T = len(returns); s2 = np.empty(T)
    s2[0] = np.var(returns)
    for t in range(1, T):
        s2[t] = wg + ag * returns[t - 1] ** 2 + bg * s2[t - 1]
        if not np.isfinite(s2[t]) or s2[t] <= 0:
            s2[t] = wg + bg * s2[t - 1] + _TINY
    return np.sqrt(s2)


def _std_t_var_es(alpha, nu):
    """Birim-varyansa olceklenmis Student-t icin (za, ea) -- getiri konv. (<0)."""
    s = np.sqrt((nu - 2.0) / nu)
    q = stats.t.ppf(alpha, nu)
    za = s * q
    # ES: tail-integral (kapali form ile ayni), getiri olceginde negatif
    us = np.linspace(alpha / 200.0, alpha, 200)
    ea = s * float(_trap(stats.t.ppf(us, nu), us) / alpha)
    return za, ea


def fz_garch_fit_filter(returns, alpha, start_idx, dist="normal"):
    r = np.asarray(returns, float)
    r_tr = r[:start_idx]
    var_tr = np.var(r_tr)
    use_t = (dist == "t")

    def za_ea(theta_nu):
        if use_t:
            nu = 2.05 + np.exp(np.clip(theta_nu, -10, 4))   # nu > 2
            return _std_t_var_es(alpha, nu)
        za = stats.norm.ppf(alpha); ea = -stats.norm.pdf(za) / alpha
        return za, ea

    def unpack(theta):
        wg = np.exp(np.clip(theta[0], -30, 5))
        ag = 1.0 / (1.0 + np.exp(-theta[1])) * 0.3
        bg = 1.0 / (1.0 + np.exp(-theta[2])) * 0.98
        return wg, ag, bg

    def obj(theta):
        wg, ag, bg = unpack(theta)
        if ag + bg >= 0.999:
            return 1e10
        za, ea = za_ea(theta[3] if use_t else 0.0)
        s = _garch_vol(r_tr, wg, ag, bg)
        v = za * s; e = ea * s
        L = _fz0_loss(r_tr, v, e, alpha)
        return L if np.isfinite(L) else 1e10

    th0 = [np.log(var_tr * 0.05), -2.0, 2.5]
    if use_t:
        th0 = th0 + [np.log(6.0 - 2.05)]
    res = optimize.minimize(obj, np.array(th0), method="Nelder-Mead",
                            options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9})
    wg, ag, bg = unpack(res.x)
    za, ea = za_ea(res.x[3] if use_t else 0.0)
    s_all = _garch_vol(r, wg, ag, bg)
    n_oos = len(r) - start_idx
    v = za * s_all[-n_oos:]; e = ea * s_all[-n_oos:]
    return -v, -e                                     # POZITIF


# ============================================================================
# Toplu arayuz: app entegrasyonu icin
# ============================================================================
def compute_sp_estimators(returns, alpha, n_oos, train_win=None):
    """returns: tam getiri serisi. Doner: {ad: (var_arr, es_arr)} (pozitif)."""
    r = np.asarray(returns, float)
    n = len(r)
    start_idx = n - n_oos
    out = {}
    for name, fn in [("FZ-GAS", lambda: fz_gas_fit_filter(r, alpha, start_idx)),
                     ("FZ-GARCH-N", lambda: fz_garch_fit_filter(r, alpha, start_idx, "normal")),
                     ("FZ-GARCH-t", lambda: fz_garch_fit_filter(r, alpha, start_idx, "t")),
                     ("CaViaR-AsL(simple)", lambda: caviar_asl_fit_filter(r, alpha, start_idx, "AS", "simple")),
                     ("CaViaR-AsL(dyn)", lambda: caviar_asl_fit_filter(r, alpha, start_idx, "AS", "dynamic"))]:
        try:
            v, e = fn()
            v = np.where(np.isfinite(v), v, np.nan)
            e = np.where(np.isfinite(e), e, np.nan)
            out[name] = (v, e)
        except Exception as exc:
            out[name] = (np.full(n_oos, np.nan), np.full(n_oos, np.nan))
    return out


# ============================================================================
# GARCH EVRENI (R kodundaki 'GARCH' ailesi) -- parametrik VE FHS
#   arch ile GARCH/GJR/EGARCH/APARCH x Normal/t/skewt/GED
#   Parametrik: v_t=-sigma_t*Q_a,  xi_t=-sigma_t*(1/a)int_0^a Q_u du (dagilimdan)
#   FHS: ayni sigma_t, ama Q_a ve tail-mean egitim std-artiklarindan (ampirik)
# ============================================================================
_trap = getattr(np, "trapezoid", None) or np.trapz


def garch_family_var_es(returns, alpha, n_oos, vol_type="Garch", o=0,
                        dist="normal", use_fhs=False):
    from arch import arch_model
    import warnings as _w
    r = np.asarray(returns, float)
    start_idx = len(r) - n_oos
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        am = arch_model(r * 100.0, mean="Constant", vol=vol_type, p=1, o=o, q=1, dist=dist)
        res = am.fit(disp="off")
    sigma = np.asarray(res.conditional_volatility) / 100.0
    mu = float(res.params.get("mu", 0.0)) / 100.0
    sig_oos = sigma[-n_oos:]
    if use_fhs:
        sr = np.asarray(res.std_resid)
        sr_tr = sr[:start_idx]; sr_tr = sr_tr[np.isfinite(sr_tr)]
        qa = float(np.quantile(sr_tr, alpha))
        tail = sr_tr[sr_tr <= qa]
        ea = float(tail.mean()) if len(tail) > 3 else 1.3 * qa
    else:
        d = res.model.distribution
        pnames = list(d.parameter_names())
        dp = np.array([res.params[p] for p in pnames]) if pnames else np.array([])
        qa = float(d.ppf(alpha, dp)) if len(dp) else float(d.ppf(alpha))
        us = np.linspace(alpha / 200.0, alpha, 200)
        ppfs = d.ppf(us, dp) if len(dp) else d.ppf(us)
        ea = float(_trap(ppfs, us) / alpha)
    var_pos = -(mu + sig_oos * qa)
    es_pos = -(mu + sig_oos * ea)
    return var_pos, es_pos


# Kurulu GARCH-evreni alt-kumesi (feasible; vol-tipi + dagilim + FHS cesitliligi)
_GARCH_SET = [
    ("GARCH-t",   dict(vol_type="Garch",  o=0, dist="t",      use_fhs=False)),
    ("GJR-N",     dict(vol_type="Garch",  o=1, dist="normal", use_fhs=False)),
    ("EGARCH-t",  dict(vol_type="EGARCH", o=1, dist="t",      use_fhs=False)),
    ("GARCH-FHS", dict(vol_type="Garch",  o=0, dist="normal", use_fhs=True)),
    ("GJR-FHS",   dict(vol_type="Garch",  o=1, dist="normal", use_fhs=True)),
]


def compute_garch_universe(returns, alpha, n_oos):
    """Kurulu GARCH-evreni alt-kumesi. Doner: {ad: (var_arr, es_arr)} (pozitif).
    Not: 'GARCH-Normal' compute_fz_comparison'da zaten inline; burada tekrar yok."""
    r = np.asarray(returns, float); out = {}
    for name, kw in _GARCH_SET:
        try:
            v, e = garch_family_var_es(r, alpha, n_oos, **kw)
            out[name] = (np.where(np.isfinite(v), v, np.nan),
                         np.where(np.isfinite(e), e, np.nan))
        except Exception:
            out[name] = (np.full(n_oos, np.nan), np.full(n_oos, np.nan))
    return out
