"""
Gerçek Piyasa Verisi İndirme Modülü — yfinance Entegrasyonu
Türkiye + Dünya Borsaları

Kullanım:
  python data_downloader.py          # Varsayılan tüm varlıkları indir
  python data_downloader.py --group turkey   # Sadece Türkiye
  python data_downloader.py --group global   # Sadece Dünya
"""

import os
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

# Drive letters differ across machines, and a hardcoded absolute path also
# contradicts app.py's "no hardcoded paths" promise.  Default to the repo's own
# data/ directory; override with the EYS_DATA_DIR environment variable.
DATA_DIR = Path(os.environ.get("EYS_DATA_DIR", Path(__file__).parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Varlık Tanımları ─────────────────────────────────────────────────────────

TURKEY_TICKERS = {
    # BIST Endeksleri ve Hisseler
    "XU100.IS":   "BIST 100",
    "XU030.IS":   "BIST 30",
    "THYAO.IS":   "THY",
    "GARAN.IS":   "Garanti BBVA",
    "ASELS.IS":   "ASELSAN",
    "KCHOL.IS":   "Koç Holding",
    "SAHOL.IS":   "Sabancı Holding",
    "EREGL.IS":   "Ereğli Demir Çelik",
    "BIMAS.IS":   "BİM Mağazalar",
    "AKBNK.IS":   "Akbank",
    # Döviz Kurları
    "USDTRY=X":   "USD/TRY",
    "EURTRY=X":   "EUR/TRY",
}

GLOBAL_TICKERS = {
    # ABD / Dünya Endeksleri
    "^GSPC":      "S&P 500",
    "^DJI":       "Dow Jones",
    "^IXIC":      "NASDAQ",
    "^FTSE":      "FTSE 100",
    "^GDAXI":     "DAX",
    "^N225":      "Nikkei 225",
    # Emtialar
    "GC=F":       "Altın (Gold)",
    "CL=F":       "Brent Petrol",
    # Risk Göstergeleri
    "^VIX":       "VIX",
    # Küresel ETF'ler (sektör proxy)
    "SPY":        "S&P 500 ETF",
    "EEM":        "Gelişmekte Olan Piyasalar ETF",
    "TLT":        "ABD 20+ Yıl Tahvil ETF",
}

def download_data(tickers_dict, start="2015-01-01", end=None):
    """
    yfinance ile veri indirir, getiri ve tanımlayıcı istatistik hesaplar.
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    symbols = list(tickers_dict.keys())
    names = list(tickers_dict.values())

    print(f"Indiriliyor: {len(symbols)} varlik ({start} -> {end})...")

    # Toplu indirme
    raw = yf.download(symbols, start=start, end=end,
                      auto_adjust=True, progress=True)

    # Kapanış fiyatları
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].copy()
        prices.columns = symbols

    # Sütun isimlerini oku-kolay isimlere çevir
    rename_map = dict(zip(symbols, names))
    prices = prices.rename(columns=rename_map)

    # Log-getiriler
    returns = np.log(prices / prices.shift(1)).dropna()

    return prices, returns

def compute_descriptive_stats(returns):
    """
    Tanımlayıcı istatistikleri hesaplar.
    """
    stats = pd.DataFrame({
        "Ortalama (%)":     returns.mean() * 252 * 100,
        "Std Sapma (%)":    returns.std() * np.sqrt(252) * 100,
        "Çarpıklık":        returns.skew(),
        "Basıklık":         returns.kurtosis(),
        "Min (%)":          returns.min() * 100,
        "Max (%)":          returns.max() * 100,
        "Gözlem":           returns.count(),
    }).round(4)
    return stats

def compute_realized_variance_proxy(returns, window=22):
    """
    Günlük getiri karelerinin aylık toplamı ile gerçekleşen varyans vekili oluşturur.
    (Gerçek yüksek frekanslı veri yoksa bu yaklaşım kullanılır)
    """
    rv_proxy = returns.pow(2).rolling(window=window).sum()
    return rv_proxy

def main():
    parser = argparse.ArgumentParser(description="EYS Finansal Veri İndirici")
    parser.add_argument("--group", choices=["turkey", "global", "all"],
                        default="all", help="İndirilecek varlık grubu")
    parser.add_argument("--start", default="2015-01-01", help="Başlangıç tarihi")
    parser.add_argument("--end", default=None, help="Bitiş tarihi")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.group in ("turkey", "all"):
        print("\n" + "="*60)
        print("TÜRKİYE BORSASI VERİLERİ")
        print("="*60)
        prices_tr, returns_tr = download_data(TURKEY_TICKERS, args.start, args.end)

        # Kaydet
        prices_tr.to_csv(os.path.join(DATA_DIR, "turkey_prices.csv"))
        returns_tr.to_csv(os.path.join(DATA_DIR, "turkey_returns.csv"))

        # RV vekili
        rv_tr = compute_realized_variance_proxy(returns_tr)
        rv_tr.to_csv(os.path.join(DATA_DIR, "turkey_rv_proxy.csv"))

        # Tanımlayıcı istatistikler
        stats_tr = compute_descriptive_stats(returns_tr)
        stats_tr.to_csv(os.path.join(DATA_DIR, "turkey_descriptive_stats.csv"))
        print("\nTürkiye Tanımlayıcı İstatistikler:")
        print(stats_tr.to_string())

    if args.group in ("global", "all"):
        print("\n" + "="*60)
        print("DÜNYA BORSALARI VERİLERİ")
        print("="*60)
        prices_gl, returns_gl = download_data(GLOBAL_TICKERS, args.start, args.end)

        # Kaydet
        prices_gl.to_csv(os.path.join(DATA_DIR, "global_prices.csv"))
        returns_gl.to_csv(os.path.join(DATA_DIR, "global_returns.csv"))

        # RV vekili
        rv_gl = compute_realized_variance_proxy(returns_gl)
        rv_gl.to_csv(os.path.join(DATA_DIR, "global_rv_proxy.csv"))

        # Tanımlayıcı istatistikler
        stats_gl = compute_descriptive_stats(returns_gl)
        stats_gl.to_csv(os.path.join(DATA_DIR, "global_descriptive_stats.csv"))
        print("\nDünya Tanımlayıcı İstatistikler:")
        print(stats_gl.to_string())

    # Birleşik dosya (seçilmiş varlıklar)
    if args.group == "all":
        # Türkiye + Dünya birleşik getiri matrisi
        combined = pd.concat([returns_tr, returns_gl], axis=1).dropna()
        combined.to_csv(os.path.join(DATA_DIR, "combined_returns.csv"))
        print(f"\nBirleşik veri: {combined.shape[0]} gözlem × {combined.shape[1]} varlık")
        print(f"Tüm dosyalar kaydedildi: {DATA_DIR}")

if __name__ == "__main__":
    main()
