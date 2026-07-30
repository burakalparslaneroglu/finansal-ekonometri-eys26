"""
Data-source wizard (Task D).

Three sources: the bundled sample, an uploaded file, or a yfinance download.
Whatever the source, the assembled return matrix is validated before it is
allowed into ``st.session_state`` — and it is never written to disk.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st

import data_io

SOURCES = ("Örnek veri", "Dosya yükle", "yfinance ile indir")


# ---------------------------------------------------------------------------

def _commit(df: pd.DataFrame, label: str, crisis_window=None):
    """Install a validated return matrix as the active dataset."""
    for k in [k for k in list(st.session_state.keys())
              if str(k).startswith(("d3_", "d4_", "d5_"))]:
        del st.session_state[k]
    st.session_state.returns_df = df
    st.session_state["crisis_window"] = crisis_window
    st.session_state["data_source"] = label
    st.session_state["data_loaded"] = True
    st.rerun()


def _show_validation(report: dict) -> bool:
    """Render validation output; return True when the data may be used."""
    for msg in report["blocking"]:
        st.error(f"⛔ {msg}")
    for msg in report["warnings"]:
        st.warning(msg)
    for msg in report["info"]:
        st.info(msg)

    if len(report["outliers"]):
        with st.expander(f"Uç gözlemler (|z| > {data_io.OUTLIER_Z:.0f}) — "
                         f"{len(report['outliers'])} adet", expanded=False):
            st.dataframe(report["outliers"], use_container_width=True,
                         hide_index=True)

    return not report["blocking"]


# ---------------------------------------------------------------------------

def _render_sample(load_default_data, load_crisis_window):
    st.markdown("#### Paketlenmiş örnek veri")
    st.caption("Derste kullanılan sentetik panel: kriz penceresi meta-verisiyle birlikte.")
    if st.button("📂 Örnek Veriyi Yükle", use_container_width=True, type="primary"):
        with st.spinner("Veri yükleniyor…"):
            df = load_default_data()
            cw = load_crisis_window()
        _commit(df, "Örnek veri", crisis_window=cw)


def _render_upload():
    st.markdown("#### Dosya yükle")
    st.caption(
        f"Kabul edilen türler: {', '.join('.' + s for s in data_io.ACCEPTED_SUFFIXES)}. "
        "`.pkl` kabul edilmez (rastgele pickle açmak kod çalıştırmaktır). "
        "Veri yalnızca bellekte tutulur, diske yazılmaz."
    )

    up = st.file_uploader("Veri dosyası", type=list(data_io.ACCEPTED_SUFFIXES),
                          key="d_upload")
    if up is None:
        return

    try:
        raw = data_io.read_uploaded(up.name, up.getvalue())
    except Exception as exc:
        st.error(f"Dosya okunamadı: `{exc}`")
        return

    pv = data_io.preview(raw)
    st.markdown(f"**Önizleme** — {pv['shape'][0]} satır × {pv['shape'][1]} sütun")
    st.dataframe(pv["head"], use_container_width=True)
    with st.expander("Sütun tipleri ve eksik değerler", expanded=False):
        st.dataframe(pv["dtypes"], use_container_width=True, hide_index=True)

    st.markdown("#### Sütun eşleme")
    cols = list(raw.columns.astype(str))
    guess = data_io.guess_date_column(raw)

    c1, c2 = st.columns(2)
    with c1:
        date_col = st.selectbox(
            "Tarih sütunu", ["(yok — sıra numarası kullan)"] + cols,
            index=(cols.index(guess) + 1) if guess in cols else 0,
            key="d_datecol",
        )
        date_col = None if date_col.startswith("(yok") else date_col
    with c2:
        default_assets = [c for c in cols if c != date_col][:8]
        asset_sel = st.multiselect("Varlık sütunları", cols, default=default_assets,
                                   key="d_assets")

    factor_sel = st.multiselect(
        "Faktör sütunları (opsiyonel — Faktör-DCC'de 'gözlenen faktör' seçeneği için)",
        [c for c in cols if c not in asset_sel and c != date_col],
        default=[], key="d_factors",
    )

    c3, c4 = st.columns(2)
    with c3:
        series_type = st.radio("Seri tipi", ["getiri", "fiyat"], horizontal=True,
                               key="d_stype")
    with c4:
        return_type = st.radio("Fiyattan getiriye", ["log", "basit"], horizontal=True,
                               key="d_rtype", disabled=(series_type != "fiyat"))

    c5, c6 = st.columns(2)
    with c5:
        na_policy = st.radio(
            "Eksik değer politikası",
            ["ortak", "ffill", "yok"], horizontal=True, key="d_na",
            help="ortak = yalnızca tüm serilerde gözlem olan tarihler; "
                 "ffill = son değeri taşı; yok = dokunma",
        )
    with c6:
        winsor = st.checkbox(
            f"Winsorize (|z| > {data_io.OUTLIER_Z:.0f})", value=False, key="d_wins",
            help="Varsayılan KAPALI: kuyruk, oynaklık modellemesinde bilgidir.",
        )

    if not asset_sel or len(asset_sel) < 2:
        st.warning("En az 2 varlık sütunu seçiniz.")
        return

    try:
        X, rep = data_io.build_returns(
            raw, date_col, asset_sel, series_type=series_type,
            return_type=return_type, na_policy=na_policy, winsorize=winsor,
        )
    except Exception as exc:
        st.error(f"Dönüştürme başarısız: `{exc}`")
        return

    if rep["missing_share"] > 0:
        st.caption(f"Eksik hücre oranı: {rep['missing_share']:.2%} "
                   f"({rep['missing_cells']} hücre)")
    if rep["n_winsorised"]:
        st.caption(f"Winsorize edilen gözlem: {rep['n_winsorised']}")

    if factor_sel:
        try:
            F, _ = data_io.build_returns(
                raw, date_col, factor_sel, series_type=series_type,
                return_type=return_type, na_policy=na_policy, winsorize=False,
            )
            F = F.reindex(X.index).dropna()
            st.session_state["_pending_factors"] = F
            st.caption(f"Faktör sütunları hazır: {', '.join(factor_sel)}")
        except Exception as exc:
            st.warning(f"Faktör sütunları hazırlanamadı: `{exc}`")

    report = data_io.validate_returns(X)
    st.markdown(f"#### Doğrulama — T = {report['T']}, N = {report['N']}")
    ok = _show_validation(report)

    if ok and st.button("✅ Bu veriyi kullan", type="primary",
                        use_container_width=True):
        st.session_state["factors_df"] = st.session_state.pop("_pending_factors", None)
        _commit(X, f"Yüklenen dosya: {up.name}")


def _render_yfinance():
    st.markdown("#### yfinance ile indir")
    st.caption("Ağ erişimi gerektirir. Streamlit Community Cloud'da yfinance "
               "istekleri zaman zaman engellenebilir; hata alırsanız dosya "
               "yükleme seçeneğini kullanın.")

    try:
        groups = data_io.available_tickers()
    except Exception as exc:
        st.error(f"Sembol listeleri okunamadı: `{exc}`")
        return

    gname = st.radio("Grup", list(groups.keys()), horizontal=True, key="d_yf_group")
    tickers = groups[gname]
    labels = {v: k for k, v in tickers.items()}

    picked = st.multiselect("Varlıklar", list(labels.keys()),
                            default=list(labels.keys())[:5], key="d_yf_pick")

    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.date_input("Başlangıç", pd.Timestamp("2018-01-01"), key="d_yf_start")
    with c2:
        end = st.date_input("Bitiş", pd.Timestamp.today().normalize(), key="d_yf_end")
    with c3:
        rtype = st.radio("Getiri", ["log", "basit"], horizontal=True, key="d_yf_rt")

    if len(picked) < 2:
        st.warning("En az 2 varlık seçiniz.")
        return

    if st.button("⬇️ İndir", type="primary", use_container_width=True):
        sel = {labels[p]: p for p in picked}
        with st.spinner("yfinance'ten indiriliyor…"):
            try:
                X, _prices = data_io.download_returns(
                    sel, str(start), str(end), return_type=rtype)
            except Exception as exc:
                st.error(f"İndirme başarısız: `{exc}`")
                return
        st.session_state["_yf_data"] = X

    X = st.session_state.get("_yf_data")
    if X is None:
        return

    st.dataframe(X.tail(5), use_container_width=True)
    report = data_io.validate_returns(X)
    st.markdown(f"#### Doğrulama — T = {report['T']}, N = {report['N']}")
    ok = _show_validation(report)

    if ok and st.button("✅ Bu veriyi kullan", type="primary",
                        use_container_width=True, key="d_yf_commit"):
        st.session_state.pop("_yf_data", None)
        _commit(X, f"yfinance ({gname})")


# ---------------------------------------------------------------------------

def render(source: str, load_default_data, load_crisis_window):
    """Render the wizard for the selected source."""
    if source == "Örnek veri":
        _render_sample(load_default_data, load_crisis_window)
    elif source == "Dosya yükle":
        _render_upload()
    else:
        _render_yfinance()
