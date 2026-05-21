"""
scripts/build_parquet.py
========================
OPTIONAL local pre-build step. The dashboard now rebuilds the parquet cache
automatically whenever DATA.xlsx changes (see data/loader.py), so you never have
to run this by hand. It remains handy for warming the cache locally or for
producing the inspection CSVs below.

It reuses the shared rebuild routine from data/loader.py (no duplicated logic)
and additionally writes two human-readable sidecars:

    data/latest.parquet              — cleaned, typed cache (via loader)
    data/latest.parquet.meta.json    — source hash + shape (via loader)
    data/metadata.csv                — per-column coverage profile (extra)
    data/ticker_map.csv              — internal key -> Bloomberg ticker (extra)

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

from data.loader import (  # noqa: E402
    rebuild_parquet_from_excel, EXCEL_PATH, PARQUET_PATH, META_PATH, DATA_DIR,
)
from config.tickers import TICKERS  # noqa: E402

METADATA_PATH = DATA_DIR / "metadata.csv"
TICKER_MAP_PATH = DATA_DIR / "ticker_map.csv"


def build() -> None:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Source workbook not found: {EXCEL_PATH}")

    print(f"Reading {EXCEL_PATH.name} and rebuilding cache ...")
    # Shared routine: writes latest.parquet + latest.parquet.meta.json.
    df = rebuild_parquet_from_excel()
    print(f"  -> {df.shape[0]:,} rows x {df.shape[1]:,} columns, "
          f"{df.index.min().date()} to {df.index.max().date()}")
    print(f"Wrote {PARQUET_PATH.name} ({PARQUET_PATH.stat().st_size/1e6:.1f} MB) "
          f"+ {META_PATH.name}")

    # --- Optional inspection sidecars -------------------------------------
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
    pd.DataFrame(meta_rows).to_csv(METADATA_PATH, index=False)
    print(f"Wrote {METADATA_PATH.name} ({len(meta_rows):,} columns profiled)")

    tmap = pd.DataFrame(
        [{"key": k, "ticker": v, "present": v.upper().strip() in df.columns}
         for k, v in TICKERS.items()]
    )
    tmap.to_csv(TICKER_MAP_PATH, index=False)
    print(f"Wrote {TICKER_MAP_PATH.name} "
          f"({int(tmap['present'].sum())}/{len(tmap)} mapped tickers present)")

    print("Done. (Note: the app rebuilds the parquet automatically — this script "
          "is optional.)")


if __name__ == "__main__":
    build()
