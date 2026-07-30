# Uygulamalı Finansal Ekonometri — İnteraktif Öğretim Uygulaması

EYS'26 · Pamukkale Üniversitesi

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

## Kapsam

| Gün | Konular |
|-----|---------|
| **3. Gün** | DCC-GARCH, cDCC, ADCC, DECO, GO-GARCH, Faktör-DCC, MVP Portföy, tanı testleri |
| **4. Gün** | VaR, ES, PELVE, Cornish-Fisher, EVT/GPD, CoVaR, MES, Backtest |
| **5. Gün** | HAR-RV, HEAVY, GARCH-X, Realized GARCH, Marchenko-Pastur, POET, Ledoit-Wolf |

## Yerel Çalıştırma

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Testler (repo kökünde, yani `python/` içinde):

```bash
pytest
```

Uzun süren kabul testlerini (N=200 fit, bootstrap) atlamak için:

```bash
pytest -m "not slow"
```

## Streamlit Community Cloud Deploy

1. Bu repo'yu GitHub'a push edin (`main` branch)
2. [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Repository: `<kullanici>/<repo-adi>`
4. Branch: `main`
5. Main file path: `app.py`
6. **Python version: 3.13 veya üstü** — `requirements.txt` sabit sürümlere
   (`==`) pinlenmiştir ve bu wheel'ler daha eski Python'larda çözülmez.
7. **Deploy!**

## Veri

Üç kaynak sol menüden seçilir:

- **Örnek veri** — `data/sample_returns.csv`: 8 varlıklı sentetik Borsa İstanbul
  paneli (BANKA, SANAYİ, HOLDİNG, GAYRİMENKUL, TEKNOLOJİ, ENERJİ, PERAKENDE,
  OTOMOTİV — 1500 günlük GARCH(1,1) simülasyonu), kriz penceresi meta-verisiyle.
- **Dosya yükle** — `.csv`, `.xlsx`, `.parquet`. Sütun eşleme (tarih / varlık /
  opsiyonel faktör), fiyat→getiri dönüşümü, eksik değer politikası ve doğrulama
  adımlarıyla. `.pkl` bilinçli olarak kabul edilmez.
- **yfinance ile indir** — Türkiye ve dünya sembol listeleri
  `data_downloader.py`'den gelir.

Yüklenen veri yalnızca oturum belleğinde tutulur, diske yazılmaz.
Doğrulama sabit (sıfır varyanslı) serileri engeller; `T < 250`, `T < 5N` ve
`N > 25` durumlarında uyarır; `|z| > 10` uç gözlemleri listeler (winsorize
varsayılan olarak **kapalıdır** — kuyruk, oynaklık modellemesinde bilgidir).

## Modül Haritası

| Dosya | İçerik |
|---|---|
| `dcc_garch.py` | DCC / cDCC (Aielli hedeflemesi) / ADCC / DECO (kendi kapalı-biçim olabilirliği) |
| `factor_dcc.py` | `H_t = BΛ_tB' + Ω_t`, Woodbury + determinant lemması |
| `go_garch.py` | GO-GARCH, PCA veya FastICA döndürmesiyle |
| `factor_selection.py` | Bai-Ng ICp1, Onatski ED, Marchenko-Pastur (Gün 3 ve Gün 5 ortak) |
| `mgarch_diagnostics.py` | Engle-Sheppard, çift-bazlı LB + BH-FDR, işaret yanlılığı, Hosking-Li-McLeod, Nyblom, parametrik bootstrap |
| `data_io.py` | Yükleme, sütun eşleme, doğrulama, içerik-adresli önbellek anahtarı |

## Ortam Notları

- `requirements.txt` **kesin sürümlere pinlenmiştir**. Üç makine arasında NumPy
  sürümlerinin ayrışması `np.trapz` yamasına yol açmıştı; gevşek sınırlar bunun
  tekrarlanmasına izin verir. Yükseltirken önce yerelde `pytest` çalıştırın,
  sonra yeniden pinleyin.
- Veri dizini `EYS_DATA_DIR` ortam değişkeniyle değiştirilebilir; varsayılan
  `python/data`. Mutlak, makineye özgü yol yoktur.
- numba sade `@njit` ile kullanılır (`cache=True` yok), dolayısıyla diske JIT
  artefaktı yazılmaz. İleride `cache=True` eklenirse `NUMBA_CACHE_DIR`
  OneDrive dışına alınmalıdır.

## Teknik Yığın

- **Python 3.13+**
- **Streamlit** — UI
- **Plotly** — interaktif grafikler
- **arch** — tek değişkenli GARCH
- **statsmodels** — QuantReg (CoVaR), HAR, Ljung-Box, ARCH-LM, BH-FDR
- **scipy** — GPD, optimizasyon, Cholesky
- **scikit-learn** — PCA (POET), FastICA (GO-GARCH), Ledoit-Wolf
- **numba** — DCC/DECO/HEAVY döngü hızlandırma (opsiyonel)
- **pytest** — `tests/` altında kabul testleri
