"""
data/loader.py
==============
All file I/O for the dashboard lives here.

Storage strategy (requirement #3):
  1. If data/latest.parquet exists -> load it (fast, typed, ~10x quicker than xlsx).
  2. Otherwise fall back to data/DATA.xlsx (always shipped, keeps Excel
     compatibility so a fresh Bloomberg pull just works).

Run `python scripts/build_parquet.py` to (re)generate the parquet plus the
metadata.csv / ticker_map.csv sidecar files from a refreshed DATA.xlsx.

`load_data()` is Streamlit-cached; the underlying `_load_dataframe()` is a plain
function so it can be imported and tested headlessly (e.g. by the smoke test or
the parquet build script) without a running Streamlit session.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:  # Streamlit is only present when running the dashboard
    import streamlit as st
    _HAS_ST = True
except Exception:  # pragma: no cover - import guard for headless use
    _HAS_ST = False

from config.tickers import TICKERS

DATA_DIR = Path(__file__).resolve().parent
EXCEL_PATH = DATA_DIR / "DATA.xlsx"
PARQUET_PATH = DATA_DIR / "latest.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise the raw frame: Date index, sorted, upper-case columns.

    Column names are upper-cased so ticker look-ups are robust to the mixed
    casing seen across Bloomberg pulls ("Index" vs "INDEX").
    """
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    df.columns = [c.upper() if isinstance(c, str) else c for c in df.columns]
    # Drop accidental fully-empty duplicate columns, keep first occurrence
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def _load_dataframe() -> pd.DataFrame:
    """Load the price panel, preferring parquet and falling back to Excel."""
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        # Parquet round-trips the index, so it is already a clean DatetimeIndex.
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df.columns = [c.upper() if isinstance(c, str) else c for c in df.columns]
        return df
    if EXCEL_PATH.exists():
        raw = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1", header=0)
        return _normalize(raw)
    raise FileNotFoundError(
        f"No data file found. Expected {PARQUET_PATH.name} or {EXCEL_PATH.name} "
        f"in {DATA_DIR}."
    )


def data_source_label() -> str:
    """Human-readable description of which backing store is in use."""
    if PARQUET_PATH.exists():
        return "latest.parquet"
    if EXCEL_PATH.exists():
        return "DATA.xlsx"
    return "(none)"


if _HAS_ST:
    @st.cache_data(show_spinner="Loading market data...")
    def load_data() -> pd.DataFrame:
        return _load_dataframe()
else:  # pragma: no cover
    def load_data() -> pd.DataFrame:
        return _load_dataframe()


def get_series(df: pd.DataFrame, key: str) -> pd.Series:
    """Return a single ticker series by internal key, NaNs dropped.

    Returns an empty float Series when the key is unknown or its column is
    absent — callers can safely do ``len(s)`` checks and the dashboard degrades
    gracefully instead of crashing (requirement #14).
    """
    col = TICKERS.get(key)
    if col is None:
        return pd.Series(dtype=float)
    col_upper = col.upper()
    if col_upper not in df.columns:
        return pd.Series(dtype=float)
    return df[col_upper].dropna()


def date_filter(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Slice the frame to [start, end] inclusive."""
    return df.loc[(df.index >= start) & (df.index <= end)]
