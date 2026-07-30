"""
data_io.py
==========
External data ingestion for the app: file upload, column mapping, validation,
and an optional yfinance download.

Design rules
------------
* Nothing is written to disk.  Uploaded data lives in ``st.session_state`` only.
* ``.pkl`` is NOT an accepted upload format — unpickling arbitrary uploads is
  remote code execution.
* Cache keys are content hashes (``df_hash``), never ``hash()`` (randomised per
  process by PYTHONHASHSEED) and never ``id()`` (a memory address).
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pandas as pd

ACCEPTED_SUFFIXES = ("csv", "xlsx", "parquet")

# Validation thresholds (course notes 1.10.4)
MIN_OBS = 250
MIN_OBS_PER_ASSET = 5          # T < 5N -> Q_bar is noisy
MANY_ASSETS = 25
OUTLIER_Z = 10.0


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def df_hash(df: pd.DataFrame, n: int = 12) -> str:
    """
    Stable content hash of a DataFrame.

    ``str(hash(df.to_csv()))`` is stable only WITHIN a process: Python randomises
    string hashing per interpreter start (PYTHONHASHSEED), so cached results
    silently missed across sessions.  ``id(df)`` is worse still — a memory
    address that says nothing about content.
    """
    payload = df.to_csv().encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:n]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_uploaded(name: str, data: bytes) -> pd.DataFrame:
    """
    Parse an uploaded file into a raw DataFrame (no index handling yet).

    Parameters
    ----------
    name : str    original filename (used only for the extension)
    data : bytes  file content

    Raises
    ------
    ValueError on an unsupported extension.
    """
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    buf = io.BytesIO(data)

    if suffix == "csv":
        try:
            return pd.read_csv(buf)
        except UnicodeDecodeError:
            buf.seek(0)
            return pd.read_csv(buf, encoding="latin-1")
    if suffix == "xlsx":
        return pd.read_excel(buf)
    if suffix == "parquet":
        return pd.read_parquet(buf)

    raise ValueError(
        f"Desteklenmeyen dosya türü: '.{suffix}'. "
        f"Kabul edilenler: {', '.join('.' + s for s in ACCEPTED_SUFFIXES)}. "
        "(.pkl kabul edilmez — güvenlik.)"
    )


def preview(df: pd.DataFrame, rows: int = 10) -> dict:
    """Small summary used by the upload preview panel."""
    return {
        "head": df.head(rows),
        "shape": df.shape,
        "dtypes": pd.DataFrame({
            "Sütun": df.columns.astype(str),
            "Tip": [str(t) for t in df.dtypes],
            "Eksik": df.isna().sum().values,
        }),
    }


def guess_date_column(df: pd.DataFrame):
    """Best guess at which column holds dates; None if nothing looks like one."""
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ("date", "tarih", "datetime", "time", "index", "unnamed: 0"):
            return c
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
        if df[c].dtype == object:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.9:
                return c
    return None


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------

def build_returns(df: pd.DataFrame, date_col, asset_cols, series_type: str = "getiri",
                  return_type: str = "log", na_policy: str = "ortak",
                  winsorize: bool = False, winsor_z: float = OUTLIER_Z):
    """
    Turn a raw uploaded table into a return matrix indexed by date.

    Parameters
    ----------
    series_type : "fiyat" | "getiri"
    return_type : "log" | "basit"     (only used when series_type == "fiyat")
    na_policy   : "ortak" (drop rows with any NaN) | "ffill" | "yok"
    winsorize   : clip |z| > winsor_z.  OFF by default — the tails are the
                  information in a volatility course, not noise to be removed.

    Returns
    -------
    (returns_df, report_dict)
    """
    out = df.copy()

    if date_col is not None:
        idx = pd.to_datetime(out[date_col], errors="coerce")
        out = out.loc[idx.notna()].copy()
        out.index = pd.DatetimeIndex(idx[idx.notna()])
        out = out.sort_index()
    else:
        out.index = pd.RangeIndex(len(out))

    X = out[list(asset_cols)].apply(pd.to_numeric, errors="coerce")

    if series_type == "fiyat":
        X = X.where(X > 0)
        X = np.log(X / X.shift(1)) if return_type == "log" else X / X.shift(1) - 1.0
        X = X.iloc[1:]

    n_missing = int(X.isna().sum().sum())
    total = int(X.size)
    if na_policy == "ffill":
        X = X.ffill().dropna()
    elif na_policy == "ortak":
        X = X.dropna()

    report = {
        "missing_cells": n_missing,
        "missing_share": (n_missing / total) if total else 0.0,
        "n_winsorised": 0,
    }

    if winsorize and len(X) > 1:
        sd = X.std(ddof=0).replace(0.0, np.nan)
        z = (X - X.mean()) / sd
        mask = z.abs() > winsor_z
        report["n_winsorised"] = int(mask.sum().sum())
        lo = X.mean() - winsor_z * sd
        hi = X.mean() + winsor_z * sd
        X = X.clip(lower=lo, upper=hi, axis=1)

    return X, report


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_returns(X: pd.DataFrame) -> dict:
    """
    Check an assembled return matrix.

    Returns
    -------
    dict with keys:
        blocking : list[str]   — must be resolved before the data can be used
        warnings : list[str]
        info     : list[str]
        outliers : DataFrame   — observations with |z| > OUTLIER_Z
        T, N
    """
    blocking, warnings_, info = [], [], []
    T, N = X.shape

    if N < 2:
        blocking.append("En az 2 varlık sütunu gerekir.")
    if T < 30:
        blocking.append(f"Yalnızca {T} gözlem var; GARCH tahmini için yetersiz.")

    sd = X.std(ddof=0)
    constant = list(sd[sd <= 0].index.astype(str))
    if constant:
        blocking.append(
            "Sabit (sıfır varyanslı) seri(ler): " + ", ".join(constant)
            + ". GARCH tahmini tanımsızdır; bu sütunları çıkarın."
        )

    if T < MIN_OBS:
        warnings_.append(
            f"T = {T} < {MIN_OBS}. GARCH ve DCC parametreleri geniş güven "
            "aralıklarıyla tahmin edilir."
        )
    if T < MIN_OBS_PER_ASSET * N:
        warnings_.append(
            f"**T = {T} < 5N = {MIN_OBS_PER_ASSET * N}.** DCC hedeflemesi "
            f"N(N−1)/2 = {N*(N-1)//2} koşulsuz korelasyonu momentle sabitler; "
            "N/T büyüdükçe Q̄ gürültülüdür (§1.10.4). Bu boyutta DECO veya "
            "Faktör-DCC tercih edin."
        )
    if N > MANY_ASSETS:
        warnings_.append(
            f"N = {N} > {MANY_ASSETS}: tam DCC tahmini uzun sürer. DECO / "
            "Faktör-DCC ya da daha az varlık önerilir."
        )

    sd_safe = sd.replace(0.0, np.nan)
    z = (X - X.mean()) / sd_safe
    mask = z.abs() > OUTLIER_Z
    rows = []
    if mask.any().any():
        stacked = z.where(mask).stack()
        for (dt, col), val in stacked.items():
            rows.append({"Tarih": dt, "Varlık": col,
                         "Getiri": float(X.loc[dt, col]), "z": float(val)})
    outliers = pd.DataFrame(rows)
    if len(outliers):
        info.append(
            f"|z| > {OUTLIER_Z:.0f} olan {len(outliers)} uç gözlem var. "
            "Winsorize varsayılan olarak KAPALIDIR: kuyruk, oynaklık "
            "modellemesinde gürültü değil bilgidir."
        )

    return {
        "blocking": blocking,
        "warnings": warnings_,
        "info": info,
        "outliers": outliers,
        "T": int(T),
        "N": int(N),
    }


# ---------------------------------------------------------------------------
# yfinance
# ---------------------------------------------------------------------------

def available_tickers() -> dict:
    """Ticker dictionaries reused from ``data_downloader``."""
    from data_downloader import GLOBAL_TICKERS, TURKEY_TICKERS
    return {"Türkiye": dict(TURKEY_TICKERS), "Dünya": dict(GLOBAL_TICKERS)}


def download_returns(tickers: dict, start: str, end: str | None = None,
                     return_type: str = "log"):
    """
    Download adjusted closes with yfinance and convert them to returns.

    Network failures are surfaced as a RuntimeError with a readable message
    rather than a raw yfinance traceback.

    Returns
    -------
    (returns_df, prices_df)
    """
    try:
        import yfinance as yf
    except ImportError as exc:                      # pragma: no cover
        raise RuntimeError(
            "yfinance kurulu değil. `pip install yfinance` ile kurun ya da "
            "'Dosya yükle' seçeneğini kullanın."
        ) from exc

    symbols = list(tickers.keys())
    try:
        raw = yf.download(symbols, start=start, end=end, auto_adjust=True,
                          progress=False)
    except Exception as exc:
        raise RuntimeError(f"yfinance indirmesi başarısız: {exc}") from exc

    if raw is None or len(raw) == 0:
        raise RuntimeError(
            "yfinance boş sonuç döndürdü (ağ erişimi yok ya da semboller geçersiz)."
        )

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(raw.columns, pd.MultiIndex):
        prices.columns = symbols
    prices = prices.rename(columns=dict(tickers)).dropna(how="all")

    if return_type == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices / prices.shift(1) - 1.0

    return rets.dropna(), prices
