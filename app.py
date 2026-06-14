"""
Uygulamalı Finansal Ekonometri — İnteraktif Öğretim Uygulaması
EYS'26 — Pamukkale Üniversitesi
Deploy: streamlit run app.py
GitHub: Streamlit Community Cloud compatible (no hardcoded paths)
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Uygulamalı Finansal Ekonometri",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Global CSS (dark purple gradient theme — unchanged from original)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .hero {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
  }
  .hero h1 { font-size: 2rem; font-weight: 700; margin: 0; }
  .hero p  { font-size: 1rem; opacity: 0.8; margin-top: 0.5rem; }

  .day-badge {
    display: inline-block;
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }
  .metric-card {
    background: #1e1e2e;
    border: 1px solid #333;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    color: white;
  }
  .metric-card .label { font-size: 0.75rem; opacity: 0.65; }
  .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #a78bfa; }

  .stTabs [data-baseweb="tab-list"] { gap: 8px; }
  .stTabs [data-baseweb="tab"] {
    background: #1e1e2e; border-radius: 8px 8px 0 0;
    color: #ccc; border: 1px solid #333; border-bottom: none;
    padding: 8px 18px; font-weight: 500;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
  }

  .stSidebar { background: #0f0c29; }
  .stSidebar * { color: white !important; }
  .stSidebar .stSelectbox label,
  .stSidebar .stSlider label { color: #c4b5fd !important; }

  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading — path-independent
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "sample_returns.csv"


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    """Load the bundled sample dataset. Cached so it is read only once."""
    return pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 Menü")
    day = st.selectbox(
        "Ders Günü Seçin",
        [
            "3. Gün — Çok Değişkenli Oynaklık (DCC)",
            "4. Gün — Risk Ölçütleri & Backtest",
            "5. Gün — Gerçekleşen Oynaklık & Büyük Boyut",
        ],
        key="day_select"
    )

    st.markdown("---")
    st.markdown("**Veri**")

    use_custom = st.checkbox("Kendi verinizi yükleyin", value=False)
    if use_custom:
        uploaded = st.file_uploader(
            "CSV yükle (tarih index, getiri sütunlar)",
            type=["csv"],
            key="csv_uploader"
        )
        if uploaded is not None:
            try:
                df_uploaded = pd.read_csv(uploaded, index_col=0, parse_dates=True)
                st.session_state.returns_df = df_uploaded
                st.success(f"Yuklendi: {df_uploaded.shape[0]} satir, {df_uploaded.shape[1]} sutun")
            except Exception as exc:
                st.error(f"CSV okunamadi: {exc}")
                st.session_state.returns_df = load_default_data()
        else:
            st.session_state.returns_df = load_default_data()
    else:
        st.session_state.returns_df = load_default_data()

    # Data info
    df_info = st.session_state.returns_df
    asset_cols_info = [
        c for c in df_info.columns
        if not c.endswith("_RV") and not c.endswith("_BPV")
    ]
    st.caption(f"Varlik sayisi: {len(asset_cols_info)}")
    st.caption(f"Gozlem: {len(df_info)}")
    if not df_info.empty:
        st.caption(
            f"Tarih: {df_info.index[0].strftime('%Y-%m-%d')} — "
            f"{df_info.index[-1].strftime('%Y-%m-%d')}"
        )

    st.markdown("---")
    st.markdown("**Yazilim:** Python · Streamlit · Plotly")
    st.markdown("**Kurs:** EYS'26 — PAU")

# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
day_label_map = {
    "3. Gün — Çok Değişkenli Oynaklık (DCC)":
        ("3. Gün · 29 Temmuz", "DCC-GARCH, cDCC, ADCC, GO-GARCH, DECO & MVP Portföy"),
    "4. Gün — Risk Ölçütleri & Backtest":
        ("4. Gün · 30 Temmuz", "VaR, ES, PELVE, Öngörü Kombinasyonu & Geriye Dönük Test"),
    "5. Gün — Gerçekleşen Oynaklık & Büyük Boyut":
        ("5. Gün · 31 Temmuz", "HAR-RV, HEAVY, GARCH-X, POET & Ledoit-Wolf Büzülme"),
}
badge, subtitle = day_label_map[day]

st.markdown(f"""
<div class="hero">
  <span class="day-badge">{badge}</span>
  <h1>Uygulamalı Finansal Ekonometri</h1>
  <p>{subtitle}</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Route to day module
# Tabs read data from st.session_state.returns_df (a DataFrame).
# ---------------------------------------------------------------------------
if day == "3. Gün — Çok Değişkenli Oynaklık (DCC)":
    from tabs import tab_day3
    tab_day3.render()

elif day == "4. Gün — Risk Ölçütleri & Backtest":
    from tabs import tab_day4
    tab_day4.render()

elif day == "5. Gün — Gerçekleşen Oynaklık & Büyük Boyut":
    from tabs import tab_day5
    tab_day5.render()
