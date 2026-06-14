"""
Generates realistic synthetic financial data for the EYS teaching app.
Run once: python data/generate_sample.py
Produces: data/sample_returns.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# Parameters
T = 1500         # trading days (~6 years)
N_ASSETS = 8     # assets
N_INTRADAY = 78  # 5-min intervals in a day (6.5 hour US session)

# Asset names
ASSETS = ["BANKA", "SANAYI", "HOLD", "GAYRIM", "TEKNOLOJI", "ENERJI", "PERAK", "OTOMOT"]

# GARCH(1,1) parameters
omega = 0.00001
alpha = 0.05
beta  = 0.90

# Realistic base correlation (financials, industrials, etc.)
rho = 0.35  # moderate base correlation
base_corr = rho * np.ones((N_ASSETS, N_ASSETS)) + (1 - rho) * np.eye(N_ASSETS)
# Add sector structure
base_corr[0, 1] = base_corr[1, 0] = 0.75  # BANKA-SANAYI
base_corr[2, 3] = base_corr[3, 2] = 0.65  # HOLD-GAYRIM
base_corr[4, 5] = base_corr[5, 4] = 0.60  # TEKNOLOJI-ENERJI
base_corr[6, 7] = base_corr[7, 6] = 0.62  # PERAK-OTOMOT

# Ensure positive-definiteness via eigenvalue clipping
eigvals, eigvecs = np.linalg.eigh(base_corr)
eigvals = np.clip(eigvals, 1e-6, None)
base_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
# Re-normalise to correlation matrix
d = np.sqrt(np.diag(base_corr))
base_corr = base_corr / np.outer(d, d)

L = np.linalg.cholesky(base_corr)

# --- GARCH simulation ---
eps_raw = np.random.randn(T, N_ASSETS)
sigmas  = np.zeros((T, N_ASSETS))
returns = np.zeros((T, N_ASSETS))

sigma2 = omega / (1 - alpha - beta) * np.ones(N_ASSETS)  # unconditional variance

for t in range(T):
    eps_corr = (L @ eps_raw[t]).flatten()
    r_t = np.sqrt(sigma2) * eps_corr
    returns[t] = r_t
    sigmas[t]  = np.sqrt(sigma2)
    sigma2 = omega + alpha * r_t**2 + beta * sigma2

# Scale to realistic daily returns (annualized vol ~25%)
target_vol = 0.25 / np.sqrt(252)
current_vol = np.std(returns, axis=0)
returns = returns * (target_vol / current_vol)

# Build date index (business days)
dates = pd.bdate_range(start="2018-01-02", periods=T)
df_returns = pd.DataFrame(returns, index=dates, columns=ASSETS)

# --- Synthetic Realized Variance (5-min) ---
rv_base  = returns**2 * N_INTRADAY
rv_noise = np.abs(np.random.normal(loc=0, scale=rv_base * 0.15))
rv       = rv_base + rv_noise
bpv      = rv * (np.pi / 2) / (np.pi / 2 + 0.08)  # BPV slightly less than RV

df_rv  = pd.DataFrame(rv,  index=dates, columns=[f"{a}_RV"  for a in ASSETS])
df_bpv = pd.DataFrame(bpv, index=dates, columns=[f"{a}_BPV" for a in ASSETS])

# Combine all columns
df_all = pd.concat([df_returns, df_rv, df_bpv], axis=1)

out_path = Path(__file__).parent / "sample_returns.csv"
df_all.to_csv(out_path)
print(f"Saved {len(df_all)} rows to {out_path}")
print(f"Columns: {list(df_all.columns)}")
print(f"Shape: {df_all.shape}")
print(f"Date range: {df_all.index[0].date()} to {df_all.index[-1].date()}")
