"""
End-to-end smoke test of the Day-3 page.

Drives the real Streamlit app through ``AppTest``: loads the bundled data,
fills in each tab's widgets and presses each "Hesapla" button, asserting that
no tab raises.  This is the automated version of "click every Day-3 tab before
delivery".

Marked slow: every button press fits real GARCH models.
"""

import os

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import conftest

pytestmark = pytest.mark.slow

APP = os.path.join(conftest.ROOT, "app.py")
ASSETS = None          # filled by the fixture


@pytest.fixture(scope="module")
def loaded_app_factory(sample_returns):
    """Return a factory producing an AppTest with data already loaded."""
    cols = list(sample_returns.columns)[:4]
    df = sample_returns[cols]

    def _make():
        at = AppTest.from_file(APP, default_timeout=600)
        at.session_state["returns_df"] = df
        at.session_state["data_loaded"] = True
        at.session_state["data_source"] = "Örnek veri (test)"
        at.session_state["crisis_window"] = None
        at.run()
        assert not at.exception, at.exception
        return at

    _make.cols = cols
    return _make


def _press(at, assets_key, assets, button_key, **widgets):
    """Set a tab's widgets, press its compute button, assert no exception."""
    at.multiselect(key=assets_key).set_value(assets)
    for k, v in widgets.items():
        for accessor in ("selectbox", "radio", "checkbox", "number_input"):
            try:
                getattr(at, accessor)(key=k).set_value(v)
                break
            except (KeyError, ValueError):
                continue
    at.run()
    assert not at.exception, at.exception
    at.button(key=button_key).click().run()
    assert not at.exception, at.exception
    return at


def test_page_renders_all_tabs(loaded_app_factory):
    at = loaded_app_factory()
    labels = [t.label for t in at.tabs] if hasattr(at, "tabs") else []
    text = " ".join(m.value for m in at.markdown)
    for expected in ("DCC-GARCH Ailesi", "GO-GARCH", "Faktör-DCC"):
        assert expected in text, expected


def test_model_estimation_tab(loaded_app_factory):
    at = loaded_app_factory()
    _press(at, "t2_assets", loaded_app_factory.cols[:3], "d3_model_run",
           t2_model="cDCC")


def test_deco_tab(loaded_app_factory):
    at = loaded_app_factory()
    _press(at, "t3_assets", loaded_app_factory.cols[:3], "d3_deco_run")
    text = " ".join(m.value for m in at.markdown)
    assert "Eşkorelasyon Kısıtının Maliyeti" in text


def test_factor_dcc_tab(loaded_app_factory):
    at = loaded_app_factory()
    _press(at, "tf_assets", loaded_app_factory.cols[:4], "d3_fdcc_run")
    text = " ".join(m.value for m in at.markdown)
    assert "Woodbury" in text


def test_go_garch_tab(loaded_app_factory):
    at = loaded_app_factory()
    _press(at, "tg_assets", loaded_app_factory.cols[:3], "d3_go_run")


def test_comparison_tab(loaded_app_factory):
    at = loaded_app_factory()
    _press(at, "t4_assets", loaded_app_factory.cols[:3], "d3_cmp_run")


def test_mvp_tab(loaded_app_factory):
    at = loaded_app_factory()
    _press(at, "t5_assets", loaded_app_factory.cols[:3], "d3_mvp_run",
           t5_model="DECO")


def test_diagnostics_tab_has_no_pvalue_column(loaded_app_factory):
    at = loaded_app_factory()
    _press(at, "t6_assets", loaded_app_factory.cols[:3], "d3_diag_run")

    warnings_text = " ".join(w.value for w in at.warning)
    assert "1.12.1" in warnings_text
    assert "dördüncü moment yoktur" in warnings_text

    # the CCC table keeps its p-values; the filtered table must not have one
    frames = [el.value for el in at.dataframe]
    filtered = [f for f in frames
                if isinstance(f, pd.DataFrame) and "Nitel okuma" in f.columns]
    assert filtered, "filtered-residual diagnostics table not found"
    for f in filtered:
        assert not any("p-değeri" in str(c) for c in f.columns)
        assert not any("Karar" == str(c) for c in f.columns)


def test_day5_still_renders(sample_returns):
    """Day 5 shares the cache-key helper; make sure the swap did not break it."""
    at = AppTest.from_file(APP, default_timeout=600)
    at.session_state["returns_df"] = sample_returns
    at.session_state["data_loaded"] = True
    at.session_state["data_source"] = "Örnek veri (test)"
    at.session_state["crisis_window"] = None
    at.run()
    at.selectbox(key="day_select").set_value(
        "5. Gün — Gerçekleşen Oynaklık & Büyük Boyut").run()
    assert not at.exception, at.exception
