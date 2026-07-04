"""
hf_sim.py -- Part A of the Day-5 empirical layer.

Controlled high-frequency DGP with a KNOWN integrated variance (IV) path, plus
microstructure noise and jumps, so that realized-measure estimators can be
checked against ground truth (impossible with real data).

Efficient log-price:   dX(s) = sqrt(v(s)) dB(s) + jumps
Spot variance:         v(s) = exp(h_d + g(s)),  h_d daily log-var (AR(1) across
                       days, persistence phi), g(s) intraday mean-zero log-OU.
Observed price:        Y_i = X_i + eps_i,  eps ~ iid N(0, omega^2)  (i.i.d. MS noise)

Estimators (per day): true IV, RV(Delta) at a grid of Delta, TSRV (Zhang-Mykland-
Ait-Sahalia 2005), Realized Kernel (Parzen; Barndorff-Nielsen-Hansen-Lunde-
Shephard 2008), BPV and the jump component.

All heavy loops are numba-njit + prange. On a workstation, raise T and N_FINE;
the sandbox run below uses a moderate configuration for verification.
"""
from __future__ import annotations
import numpy as np, pandas as pd, time
from numba import njit, prange

SEED = 20260731
SEC_PER_DAY = 23400          # 6.5h trading day in seconds
N_FINE = 4680                # efficient-price grid step = 5 seconds (23400/4680)
DT = 1.0 / N_FINE            # day normalized to [0,1]

# ---- structural parameters ----
PHI_H   = 0.98               # daily log-variance persistence (=> HAR long-memory look)
MU_H    = np.log(0.010**2)   # daily mean variance ~ (1.0% daily vol)^2
SIG_H   = 0.08               # daily log-var innovation sd (stationary sd(h)~0.40)
KAPPA_G = 40.0               # intraday log-vol mean reversion (per unit day)
XI_G    = 1.2                # intraday log-vol vol
NOISE_RATIO = 0.30           # MS-noise var as fraction of per-5s efficient var
P_JUMP  = 0.10               # prob. of a jump on a given day
JUMP_SD = 0.007              # jump size sd (in log-price)

# sampling scales (fine steps per observed return): 5s*k
K_1MIN, K_5MIN, K_15MIN = 12, 60, 180
SIG_GRID = np.array([1, 2, 3, 6, 12, 24, 60, 120, 180, 360], dtype=np.int64)  # signature


@njit(cache=True)
def _intraday_logvar(dWg, kappa, xi, dt):
    """Mean-zero intraday log-vol via Euler OU: dg = -kappa g dt + xi dWg."""
    n = dWg.shape[0]
    g = np.empty(n)
    g[0] = 0.0
    for i in range(1, n):
        g[i] = g[i-1] - kappa * g[i-1] * dt + xi * dWg[i]
    return g


@njit(cache=True)
def _rv(logY, k):
    """Realized variance from log-price sampled every k fine steps."""
    n = logY.shape[0] - 1
    m = n // k
    s = 0.0
    for j in range(1, m + 1):
        r = logY[j*k] - logY[(j-1)*k]
        s += r * r
    return s


@njit(cache=True)
def _bpv(logY, k):
    """Bipower variation (pi/2 scaling) at step k."""
    n = logY.shape[0] - 1
    m = n // k
    s = 0.0
    prev = abs(logY[k] - logY[0])
    for j in range(2, m + 1):
        cur = abs(logY[j*k] - logY[(j-1)*k])
        s += prev * cur
        prev = cur
    return (np.pi / 2.0) * s


@njit(cache=True)
def _tsrv(logY, K):
    """Two-scale RV (Zhang-Mykland-Ait-Sahalia 2005) with small-sample adjustment.
    Fast scale = every fine step; slow scale = every K-th, averaged over K grids."""
    n = logY.shape[0] - 1
    # fast RV (all fine returns)
    rv_fast = 0.0
    for i in range(1, n + 1):
        r = logY[i] - logY[i-1]
        rv_fast += r * r
    # slow RV averaged over K subgrids
    rv_slow_sum = 0.0
    cnt = 0
    for start in range(K):
        j = start
        while j + K <= n:
            r = logY[j+K] - logY[j]
            rv_slow_sum += r * r
            cnt += 1
            j += K
    rv_slow_avg = rv_slow_sum / K
    n_bar = cnt / K                      # avg # obs per subgrid
    adj = 1.0 / (1.0 - n_bar / n)
    return adj * (rv_slow_avg - (n_bar / n) * rv_fast)


@njit(cache=True)
def _parzen(x):
    ax = abs(x)
    if ax <= 0.5:
        return 1.0 - 6.0*ax*ax + 6.0*ax*ax*ax
    elif ax <= 1.0:
        t = 1.0 - ax
        return 2.0 * t*t*t
    else:
        return 0.0


@njit(cache=True)
def _realized_kernel(logY, k, H):
    """Parzen realized kernel on returns sampled every k fine steps.
    K = sum_{h=-H}^{H} parzen(h/(H+1)) * gamma_h ,  gamma_h realized autocov."""
    n = logY.shape[0] - 1
    m = n // k
    r = np.empty(m)
    for j in range(m):
        r[j] = logY[(j+1)*k] - logY[j*k]
    out = 0.0
    for h in range(0, H + 1):
        g = 0.0
        for j in range(h, m):
            g += r[j] * r[j-h]
        w = _parzen(h / (H + 1.0))
        if h == 0:
            out += w * g
        else:
            out += w * 2.0 * g          # symmetric: gamma_h + gamma_{-h} ~ 2 gamma_h
    return out


@njit(parallel=True, cache=True)
def simulate(T, h0_seed_ok, dW_price, dW_vol, eps_all, jump_mask, jump_val,
             sig_grid, k1, k5, k15, noise_ratio):
    """Simulate T days; return per-day realized measures + true IV.
    Random arrays are passed in (generated in numpy for reproducibility)."""
    ncols = 9 + sig_grid.shape[0]
    out = np.empty((T, ncols))
    # daily log-var AR(1) is sequential -> precompute outside; here dW_vol carries h too.
    for d in prange(T):
        # intraday mean-zero log-vol
        g = _intraday_logvar(dW_vol[d], KAPPA_G, XI_G, DT)
        h_d = h0_seed_ok[d]
        v = np.exp(h_d + g)                      # spot variance path (per unit day)
        IV = 0.0
        for i in range(N_FINE):
            IV += v[i] * DT
        # efficient log-price increments
        X = np.empty(N_FINE + 1)
        X[0] = 0.0
        for i in range(N_FINE):
            X[i+1] = X[i] + np.sqrt(v[i] * DT) * dW_price[d, i]
        # jump: single jump at mid-day if present
        if jump_mask[d]:
            jloc = int(0.5 * N_FINE)
            for i in range(jloc, N_FINE + 1):
                X[i] += jump_val[d]
        # microstructure noise
        omega2 = noise_ratio * (IV / N_FINE)     # noise var ~ fraction of per-5s eff var
        omega = np.sqrt(omega2)
        Y = np.empty(N_FINE + 1)
        for i in range(N_FINE + 1):
            Y[i] = X[i] + omega * eps_all[d, i]
        # --- realized measures ---
        rv1  = _rv(Y, k1)
        rv5  = _rv(Y, k5)
        rv15 = _rv(Y, k15)
        bpv5 = _bpv(Y, k5)
        jump = rv5 - bpv5
        if jump < 0.0:
            jump = 0.0
        tsrv = _tsrv(Y, k5)                        # slow scale = 5-min
        # RK bandwidth H* = c * xi^{4/5} * n^{3/5}, c_Parzen ~ 3.51, xi^2=omega2/sqrt(IV)
        n_rk = N_FINE // k1
        xi2 = omega2 / np.sqrt(IV) if IV > 0 else 0.0
        H = int(3.5134 * (xi2**0.4) * (n_rk**0.6)) + 1
        if H < 1:
            H = 1
        if H > n_rk // 2:
            H = n_rk // 2
        rk = _realized_kernel(Y, k1, H)
        out[d, 0] = IV
        out[d, 1] = rv1
        out[d, 2] = rv5
        out[d, 3] = rv15
        out[d, 4] = bpv5
        out[d, 5] = jump
        out[d, 6] = tsrv
        out[d, 7] = rk
        out[d, 8] = Y[N_FINE] - Y[0]              # daily close-to-close return
        for s in range(sig_grid.shape[0]):
            out[d, 9 + s] = _rv(Y, sig_grid[s])
    return out


def run(T=300):
    rng = np.random.default_rng(SEED)
    # daily log-variance AR(1) across days (sequential, cheap in numpy)
    h = np.empty(T)
    h[0] = MU_H + SIG_H * rng.standard_normal()
    for d in range(1, T):
        h[d] = MU_H + PHI_H * (h[d-1] - MU_H) + SIG_H * rng.standard_normal()
    dW_price = rng.standard_normal((T, N_FINE))
    dW_vol   = rng.standard_normal((T, N_FINE)) * np.sqrt(DT)
    eps_all  = rng.standard_normal((T, N_FINE + 1))
    jump_mask = rng.random(T) < P_JUMP
    jump_val  = rng.standard_normal(T) * JUMP_SD * jump_mask
    t0 = time.time()
    out = simulate(T, h, dW_price, dW_vol, eps_all, jump_mask, jump_val,
                   SIG_GRID, K_1MIN, K_5MIN, K_15MIN, NOISE_RATIO)
    el = time.time() - t0
    cols = (["IV", "RV_1min", "RV_5min", "RV_15min", "BPV_5min", "Jump", "TSRV", "RK", "ret"]
            + [f"RV_sig_{k}" for k in SIG_GRID])
    df = pd.DataFrame(out, columns=cols)
    df["jump_mask"] = jump_mask
    return df, el


if __name__ == "__main__":
    df, el = run(T=300)
    IV = df["IV"].values
    jm = df["jump_mask"].values
    nj = ~jm                                     # non-jump days
    print(f"Simulated T={len(df)} days in {el:.1f}s  (grid {N_FINE}=5s; {jm.sum()} jump days)")
    print(f"Mean daily IV = {IV.mean():.2e} (~{np.sqrt(IV.mean())*100:.2f}% vol); "
          f"sd(log IV)={np.std(np.log(IV)):.2f}\n")

    print("== Realized-measure recovery of IV on NON-JUMP days (temiz gurultu) ==")
    print(f"{'Estimator':<12}{'RelBias%':>10}{'RMSE/IV':>10}{'Corr(IV)':>10}")
    for est in ["RV_1min", "RV_5min", "RV_15min", "TSRV", "RK", "BPV_5min"]:
        x = df[est].values[nj]; iv = IV[nj]
        rb = (x/iv - 1).mean()*100
        rm = np.sqrt(((x-iv)**2).mean())/iv.mean()
        cr = np.corrcoef(x, iv)[0, 1]
        print(f"{est:<12}{rb:>10.2f}{rm:>10.3f}{cr:>10.3f}")

    print("\n== Volatilite imza (RV(Delta)/IV, NON-JUMP gun ort.; en yuksek frekansta gurultu baskin) ==")
    for k in SIG_GRID:
        ratio = (df[f"RV_sig_{k}"].values[nj] / IV[nj]).mean()
        secs = 5*k; lab = f"{secs}s" if secs < 60 else f"{secs//60}min"
        bar = "#" * int((ratio-1)*40)
        print(f"  Delta={lab:<6} (n={N_FINE//k:>4}): RV/IV={ratio:.3f} {bar}")

    print("\n== Sicrama teshisi (TUM gunler): RV-BPV ayrimi ==")
    rvmb = (df["RV_5min"].values - df["BPV_5min"].values) / IV
    print(f"  (RV-BPV)/IV: sicramali gun ort.={rvmb[jm].mean():.3f}, "
          f"sicramasiz={rvmb[nj].mean():.3f}")
    # simple ratio-jump flag: J/RV > threshold
    jr = df["Jump"].values / df["RV_5min"].values
    thr = 0.20
    tp = (jr[jm] > thr).mean()*100; fp = (jr[nj] > thr).mean()*100
    print(f"  J/RV>{thr}: gercek sicrama gunlerinde yakalama={tp:.0f}%, "
          f"sicramasiz gunlerde yanlis-alarm={fp:.0f}%")
