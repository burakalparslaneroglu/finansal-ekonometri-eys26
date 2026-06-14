# Uygulamalı Finansal Ekonometri — İnteraktif Öğretim Uygulaması

EYS'26 · Pamukkale Üniversitesi

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

## Kapsam

| Gün | Konular |
|-----|---------|
| **3. Gün** | DCC-GARCH, cDCC, ADCC, DECO, MVP Portföy |
| **4. Gün** | VaR, ES, PELVE, Cornish-Fisher, EVT/GPD, CoVaR, MES, Backtest |
| **5. Gün** | HAR-RV, HEAVY, GARCH-X, Realized GARCH, Marchenko-Pastur, POET, Ledoit-Wolf |

## Yerel Çalıştırma

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud Deploy

1. Bu repo'yu GitHub'a push edin (`main` branch)
2. [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Repository: `<kullanici>/<repo-adi>`
4. Branch: `main`
5. Main file path: `app.py`
6. **Deploy!**

## Veri

Uygulama, `data/sample_returns.csv` içindeki 8 varlıklı sentetik Borsa İstanbul verisiyle çalışır (BANKA, SANAYİ, HOLDİNG, GAYRİMENKUL, TEKNOLOJİ, ENERJİ, PERAKENDE, OTOMOTİV — 1500 günlük GARCH(1,1) simülasyonu).

Kendi verinizi sol menüden CSV olarak yükleyebilirsiniz.

## Teknik Yığın

- **Python 3.10+**
- **Streamlit** — UI
- **Plotly** — interaktif grafikler
- **arch** — tek değişkenli GARCH
- **statsmodels** — QuantReg (CoVaR), HAR
- **scipy** — GPD, optimizasyon
- **scikit-learn** — PCA (POET)
- **numba** — DCC/HEAVY döngü hızlandırma (opsiyonel)
