"""
scripts/build_parquet.py
========================
Offline data-prep step (requirement #3). Reads the raw Bloomberg pull from
``data/DATA.xlsx`` and writes three artefacts next to it:

    data/latest.parquet  — cleaned, typed, ~10x faster to load than xlsx
    data/metadata.csv     — per-column coverage report (obs, dates, missing %)
    data/ticker_map.csv   — internal key -> Bloomberg ticker (from config)

The dashboard prefers latest.parquet automatically; if it is absent it falls
back to DATA.xlsx, so running this script is optional but recommended before
deploying.

Usage:
    python scripts/build_parquet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make the package importable when run as a script from anywhere.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.loader import EXCEL_PATH, PARQUET_PATH, DATA_DIR, _normalize  # noqa: E402
from config.tickers import TICKERS  # noqa: E402

METADATA_PATH = DATA_DIR / "metadata.csv"
TICKER_MAP_PATH = DATA_DIR / "ticker_map.csv"


def build() -> None:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Source workbook not found: {EXCEL_PATH}")

    print(f"Reading {EXCEL_PATH.name} ...")
    raw = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1", header=0)
    df = _normalize(raw)
    print(f"  -> {df.shape[0]:,} rows x {df.shape[1]:,} columns, "
          f"{df.index.min().date()} to {df.index.max().date()}")

    # 1. Parquet -----------------------------------------------------------
    df.to_parquet(PARQUET_PATH, engine="pyarrow", index=True)
    print(f"Wrote {PARQUET_PATH.name} ({PARQUET_PATH.stat().st_size/1e6:.1f} MB)")

    # 2. Metadata (one row per column) -------------------------------------
    dataset_last = df.index.max()
    meta_rows = []
    for col in df.columns:
        valid = df[col].dropna()
        meta_rows.append({
            "column": col,
            "n_obs": int(valid.shape[0]),
            "first_date": valid.index.min() if len(valid) else pd.NaT,
            "last_date": valid.index.max() if len(valid) else pd.NaT,
            "missing_pct": round(float(df[col].isna().mean()) * 100, 2),
            "stale_vs_dataset": (bool(valid.index.max() < dataset_last
                                      - pd.tseries.offsets.BusinessDay(5))
                                 if len(valid) else True),
        })
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(METADATA_PATH, index=False)
    print(f"Wrote {METADATA_PATH.name} ({len(meta):,} columns profiled)")

    # 3. Ticker map (internal key -> Bloomberg ticker) ---------------------
    tmap = pd.DataFrame(
        [{"key": k, "ticker": v, "present": v.upper() in df.columns}
         for k, v in TICKERS.items()]
    )
    tmap.to_csv(TICKER_MAP_PATH, index=False)
    present = int(tmap["present"].sum())
    print(f"Wrote {TICKER_MAP_PATH.name} "
          f"({present}/{len(tmap)} mapped tickers present in data)")

    print("Done.")


if __name__ == "__main__":
    build()
