"""
data/loader.py
==============
All file I/O for the dashboard.

Data model (manual-update workflow)
-----------------------------------
    DATA.xlsx                 = SOURCE OF TRUTH. The only file you edit; refresh
                                it by hand whenever you have new Bloomberg data.
    latest.parquet            = derived cache only. Rebuilt AUTOMATICALLY when the
                                Excel changes; never edited by hand; safe to delete.
    latest.parquet.meta.json  = records the SHA-256 of the Excel the parquet was
                                built from, so staleness can be detected.

On every cold start ``load_data()`` will:
    1. hash DATA.xlsx (SHA-256),
    2. compare it to the hash recorded in latest.parquet.meta.json,
    3. if the parquet is missing or its recorded hash differs (i.e. stale),
       rebuild the parquet + meta from the Excel automatically,
    4. load the (now fresh) parquet,
    5. if anything in the cache path fails (e.g. read-only filesystem, missing
       parquet engine, corrupt cache), fall back to reading DATA.xlsx directly so
       the dashboard still runs.

A *content hash* is used rather than a modified-timestamp because file mtimes are
unreliable after a git checkout / Streamlit Cloud redeploy. DATA.xlsx is only a
few MB, so hashing it on each run is cheap.

``load_data()`` is Streamlit-cached and keyed on the Excel's content hash, so a
manually updated DATA.xlsx is picked up automatically on the next run — you never
need to run scripts/build_parquet.py by hand.
"""

from __future__ import annotations

import hashlib
import json
import sys
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
META_PATH = DATA_DIR / "latest.parquet.meta.json"

# Records what the most recent load actually did, for the Data Quality panel.
# One of: "fresh", "rebuilt", "fallback", "unknown".
_LOAD_STATUS = "unknown"


# ---------------------------------------------------------------------------
# Excel reading / normalisation
# ---------------------------------------------------------------------------
def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise a raw frame: Date index, sorted, upper-cased + stripped cols.

    Column names are upper-cased and stripped so ticker look-ups are robust to
    mixed casing / stray whitespace across Bloomberg pulls.
    """
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    df.columns = [str(c).upper().strip() if not isinstance(c, str) else c.upper().strip()
                  for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]  # keep first of any duplicate columns
    return df


def read_excel_data(path: Path = EXCEL_PATH) -> pd.DataFrame:
    """Read + normalise the Bloomberg workbook (first sheet)."""
    raw = pd.read_excel(path, header=0)
    return _normalize(raw)


# ---------------------------------------------------------------------------
# Hash-based freshness + automatic rebuild
# ---------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_signature() -> str:
    """Short content signature of the source Excel, for cache keys.

    Returns a sentinel when the Excel is absent so callers still get a stable
    (if useless) key rather than raising here.
    """
    if EXCEL_PATH.exists():
        return file_sha256(EXCEL_PATH)
    return "no-excel"


def parquet_is_fresh() -> bool:
    """True only if parquet + meta exist and the meta's recorded source hash
    matches the current DATA.xlsx content hash."""
    if not (PARQUET_PATH.exists() and META_PATH.exists() and EXCEL_PATH.exists()):
        return False
    try:
        with META_PATH.open("r") as f:
            meta = json.load(f)
        return meta.get("source_sha256") == file_sha256(EXCEL_PATH)
    except Exception:
        return False


def rebuild_parquet_from_excel() -> pd.DataFrame:
    """Rebuild latest.parquet (+ meta.json) from DATA.xlsx. Returns the frame.

    This is the single shared rebuild routine reused by both the automatic
    in-app path and scripts/build_parquet.py.
    """
    df = read_excel_data(EXCEL_PATH)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, engine="pyarrow", index=True)
    meta = {
        "source_file": EXCEL_PATH.name,
        "source_sha256": file_sha256(EXCEL_PATH),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "start_date": str(df.index.min().date()) if len(df) else None,
        "end_date": str(df.index.max().date()) if len(df) else None,
    }
    with META_PATH.open("w") as f:
        json.dump(meta, f, indent=2)
    return df


def load_meta() -> dict:
    """Return the parquet meta sidecar (rows/cols/dates/hash), or {} if absent."""
    if META_PATH.exists():
        try:
            with META_PATH.open("r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------------------
# Core load logic (no Streamlit dependency — safe to call headlessly)
# ---------------------------------------------------------------------------
def _warn(msg: str) -> None:
    """Surface a warning via Streamlit when available, else stderr."""
    if _HAS_ST:
        try:
            st.warning(msg)
            return
        except Exception:
            pass
    print(f"[loader] {msg}", file=sys.stderr)


def _load_core() -> pd.DataFrame:
    """Freshness-aware load: rebuild parquet if stale, else read it; on any
    failure fall back to reading the Excel directly. Sets ``_LOAD_STATUS``."""
    global _LOAD_STATUS
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Missing source data file: {EXCEL_PATH}")
    try:
        if parquet_is_fresh():
            df = pd.read_parquet(PARQUET_PATH)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df.columns = [c.upper().strip() if isinstance(c, str) else c
                          for c in df.columns]
            _LOAD_STATUS = "fresh"
            return df
        # Missing or stale -> rebuild automatically from the source of truth.
        df = rebuild_parquet_from_excel()
        _LOAD_STATUS = "rebuilt"
        return df
    except Exception as exc:  # parquet engine / read-only FS / corrupt cache
        _LOAD_STATUS = "fallback"
        _warn(f"Could not use the parquet cache ({exc}). Reading DATA.xlsx directly.")
        return read_excel_data(EXCEL_PATH)


# Backwards-compatible alias used by the smoke test / build profiling.
def _load_dataframe() -> pd.DataFrame:
    return _load_core()


# ---------------------------------------------------------------------------
# Public, Streamlit-cached entry point
# ---------------------------------------------------------------------------
if _HAS_ST:
    @st.cache_data(show_spinner="Loading market data...")
    def _load_data_cached(_source_hash: str) -> pd.DataFrame:
        # Keyed on the Excel content hash: a manually edited DATA.xlsx changes the
        # hash, busts this cache, and triggers an automatic rebuild via _load_core.
        return _load_core()

    def load_data() -> pd.DataFrame:
        return _load_data_cached(source_signature())
else:  # pragma: no cover
    def load_data() -> pd.DataFrame:
        return _load_core()


# ---------------------------------------------------------------------------
# Status helpers for the UI
# ---------------------------------------------------------------------------
def get_load_status() -> str:
    """What the most recent load did: fresh | rebuilt | fallback | unknown."""
    return _LOAD_STATUS


def cache_status_label() -> str:
    """Human-readable cache status for the Data Quality panel."""
    return {
        "fresh": "Fresh — cache in sync with DATA.xlsx",
        "rebuilt": "Rebuilt automatically from DATA.xlsx",
        "fallback": "Fallback — reading DATA.xlsx directly",
    }.get(_LOAD_STATUS, "Unknown")


def data_source_label() -> str:
    """Short backing-store description used in the sidebar caption."""
    if _LOAD_STATUS == "fallback":
        return "DATA.xlsx (direct)"
    if PARQUET_PATH.exists():
        return "latest.parquet (auto-cache)"
    return "DATA.xlsx"


# ---------------------------------------------------------------------------
# Series access
# ---------------------------------------------------------------------------
def get_series(df: pd.DataFrame, key: str) -> pd.Series:
    """Return a single ticker series by internal key, NaNs dropped.

    Returns an empty float Series when the key is unknown or its column is
    absent, so callers can len()-check and degrade gracefully (requirement #14).
    """
    col = TICKERS.get(key)
    if col is None:
        return pd.Series(dtype=float)
    col_upper = col.upper().strip()
    if col_upper not in df.columns:
        return pd.Series(dtype=float)
    return df[col_upper].dropna()


def date_filter(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Slice the frame to [start, end] inclusive."""
    return df.loc[(df.index >= start) & (df.index <= end)]
