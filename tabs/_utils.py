"""
Helpers shared by the day tabs.

The cache-key helper lives here because Day 3 and Day 5 previously used two
different — and both broken — schemes: ``str(hash(df.to_csv()))[:8]`` (Python
string hashing is randomised per process by PYTHONHASHSEED, so keys were stable
within a session but not across sessions) and ``id(df)`` (a memory address,
which says nothing about content and can be recycled).  Both silently returned
stale or missed results once external data could be swapped in at runtime.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data_io import df_hash


def active_df() -> pd.DataFrame:
    """The return matrix currently loaded in the session."""
    return st.session_state.returns_df


def df_key(df: pd.DataFrame | None = None) -> str:
    """Content-addressed cache key for the active (or given) DataFrame."""
    return df_hash(active_df() if df is None else df)


def asset_cols(df: pd.DataFrame):
    """
    Return columns holding asset returns.

    The bundled sample dataset carries realised-measure columns alongside the
    returns, tagged with the ``_RV`` / ``_BPV`` suffixes; those are inputs to
    Day 5, not assets.  Uploaded data is expected to follow the same
    convention — the upload panel states it and lets the user pick the columns
    explicitly, so nothing is filtered behind their back.
    """
    return [c for c in df.columns if not c.endswith("_RV") and not c.endswith("_BPV")]
