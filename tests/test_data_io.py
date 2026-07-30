"""
Task D — external data ingestion, validation and cache keys.
"""

import io
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

import data_io


def _panel(T=400, N=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=T)
    return pd.DataFrame(rng.standard_normal((T, N)) * 0.01, index=idx,
                        columns=[f"A{i+1}" for i in range(N)])


# --- D.3 cache keys --------------------------------------------------------

def test_df_hash_is_content_addressed():
    a = _panel(seed=1)
    b = a.copy()
    c = a.copy()
    c.iloc[0, 0] += 1e-9

    assert data_io.df_hash(a) == data_io.df_hash(b)
    assert data_io.df_hash(a) != data_io.df_hash(c)
    assert len(data_io.df_hash(a)) == 12


def test_df_hash_is_stable_across_processes():
    """
    The old key used str(hash(...)), which PYTHONHASHSEED randomises per
    process.  Run the hash in two fresh interpreters and require a match.
    """
    code = textwrap.dedent("""
        import sys
        sys.path.insert(0, sys.argv[1])
        import numpy as np, pandas as pd
        from data_io import df_hash
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2020-01-01", periods=50)
        df = pd.DataFrame(rng.standard_normal((50, 2)), index=idx, columns=["a", "b"])
        print(df_hash(df))
        print(str(hash(df.to_csv()))[:8])
    """)
    import conftest
    root = conftest.ROOT

    outs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", code, root],
                           stdin=subprocess.DEVNULL, capture_output=True,
                           text=True, check=True)
        outs.append(r.stdout.strip().splitlines())

    assert outs[0][0] == outs[1][0], "sha1 key drifted across processes"
    # and demonstrate that the old scheme really was unstable
    assert outs[0][1] != outs[1][1], (
        "builtin hash() happened to match — rerun; PYTHONHASHSEED may be fixed"
    )


# --- reading ---------------------------------------------------------------

def test_read_csv_roundtrip():
    df = _panel(T=20)
    raw = df.reset_index(names="Date").to_csv(index=False).encode()
    out = data_io.read_uploaded("x.csv", raw)
    assert list(out.columns) == ["Date", "A1", "A2", "A3"]


def test_pickle_upload_is_rejected():
    with pytest.raises(ValueError, match=r"\.pkl"):
        data_io.read_uploaded("evil.pkl", b"\x80\x04.")


def test_unknown_extension_rejected():
    with pytest.raises(ValueError):
        data_io.read_uploaded("data.txt", b"a,b\n1,2\n")


def test_guess_date_column():
    df = pd.DataFrame({"Tarih": ["2020-01-01", "2020-01-02"], "A": [1.0, 2.0]})
    assert data_io.guess_date_column(df) == "Tarih"
    assert data_io.guess_date_column(pd.DataFrame({"A": [1.0], "B": [2.0]})) is None


# --- transformation --------------------------------------------------------

def test_price_to_log_returns():
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"Date": idx, "P1": [100, 110, 121, 133.1, 146.41],
                           "P2": [50, 50, 50, 50, 50.0]})
    X, rep = data_io.build_returns(prices, "Date", ["P1", "P2"],
                                   series_type="fiyat", return_type="log")
    assert len(X) == 4
    assert np.allclose(X["P1"], np.log(1.1))
    assert np.allclose(X["P2"], 0.0)
    assert rep["missing_cells"] == 0


def test_simple_returns_option():
    idx = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame({"Date": idx, "P": [100.0, 110.0, 121.0]})
    X, _ = data_io.build_returns(prices, "Date", ["P"], series_type="fiyat",
                                 return_type="basit")
    assert np.allclose(X["P"], 0.1)


def test_winsorize_is_off_by_default_and_reports_when_on():
    df = _panel(T=300, seed=3).reset_index(names="Date")
    df.loc[5, "A1"] = 5.0          # ~500 sigma
    X_off, rep_off = data_io.build_returns(df, "Date", ["A1", "A2", "A3"])
    assert rep_off["n_winsorised"] == 0
    assert X_off["A1"].max() == pytest.approx(5.0)

    X_on, rep_on = data_io.build_returns(df, "Date", ["A1", "A2", "A3"],
                                         winsorize=True)
    assert rep_on["n_winsorised"] >= 1
    assert X_on["A1"].max() < 5.0


# --- validation ------------------------------------------------------------

def test_constant_series_is_blocking():
    X = _panel(T=300)
    X["A2"] = 0.0
    rep = data_io.validate_returns(X)
    assert rep["blocking"]
    assert "A2" in rep["blocking"][0]


def test_short_sample_warnings():
    rep = data_io.validate_returns(_panel(T=100, N=3))
    assert any("250" in w for w in rep["warnings"])


def test_t_less_than_5n_warning():
    rep = data_io.validate_returns(_panel(T=60, N=20))
    assert any("5N" in w for w in rep["warnings"])
    assert any("DECO" in w for w in rep["warnings"])


def test_many_assets_warning():
    rep = data_io.validate_returns(_panel(T=2000, N=30))
    assert any("uzun sürer" in w for w in rep["warnings"])


def test_outliers_are_listed_not_removed():
    X = _panel(T=500, seed=7)
    X.iloc[10, 0] = 50 * X.iloc[:, 0].std()
    rep = data_io.validate_returns(X)
    assert len(rep["outliers"]) >= 1
    assert set(rep["outliers"].columns) == {"Tarih", "Varlık", "Getiri", "z"}
    assert not rep["blocking"]


def test_clean_panel_passes():
    rep = data_io.validate_returns(_panel(T=1500, N=4))
    assert rep["blocking"] == []
    assert rep["T"] == 1500 and rep["N"] == 4


# --- yfinance wiring -------------------------------------------------------

def test_ticker_dictionaries_are_reachable():
    groups = data_io.available_tickers()
    assert set(groups) == {"Türkiye", "Dünya"}
    assert "XU100.IS" in groups["Türkiye"]
    assert "^GSPC" in groups["Dünya"]


def test_data_downloader_has_no_hardcoded_drive():
    import pathlib
    import conftest
    src = pathlib.Path(conftest.ROOT) / "data_downloader.py"
    text = src.read_text(encoding="utf-8")
    assert "e:\\OneDrive" not in text.lower().replace("E:", "e:")
    assert "EYS_DATA_DIR" in text


def test_data_downloader_default_dir_is_relative():
    import data_downloader
    import pathlib
    import conftest
    assert pathlib.Path(data_downloader.DATA_DIR).resolve() == \
        (pathlib.Path(conftest.ROOT) / "data").resolve()
