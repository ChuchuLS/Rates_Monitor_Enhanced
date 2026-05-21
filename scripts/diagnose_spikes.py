"""
scripts/diagnose_spikes.py
==========================
Reproducible diagnostic for index instability (requirement #4 / #10). Identifies
which components and buckets drive the largest moves in a chosen window, flags
low-variation (spike-prone) series, and reports the coverage timeline and the
date from which the index is published.

Usage:
    python scripts/diagnose_spikes.py [START END]
    e.g. python scripts/diagnose_spikes.py 2016-01-01 2018-06-30
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pd.set_option("display.width", 160, "display.max_columns", 30)

from data.loader import _load_core            # noqa: E402
from index.components import build_components, BUCKET_OF, DIRECTION  # noqa: E402
from index.composite import compute_index     # noqa: E402

START, END = (sys.argv[1], sys.argv[2]) if len(sys.argv) >= 3 else ("2016-01-01", "2018-06-30")


def main() -> None:
    df = _load_core()
    res = compute_index(df)

    print(f"=== Coverage & publication ===")
    print(f"first computable date : {res.first_valid_date}")
    print(f"first PUBLISHED date  : {res.first_published_date}  <- reliable from here")
    print(f"latest index          : {res.latest:.2f} ({res.latest_regime})")

    print(f"\n=== Spike attribution {START}..{END} (pre-fix diagnostic window) ===")
    pz = res.z_scores.loc[START:END].dropna(how="all")
    pt = res.bucket_terms.loc[START:END].dropna(how="all")
    if pt.empty:
        print("No computable index values in this window.")
    else:
        print("\nTop spike-driving BUCKETS (largest |contribution| each day):")
        print(pt.abs().idxmax(axis=1).value_counts())
        print("\nTop spike-driving COMPONENTS (largest |z| each day):")
        print(pz.abs().idxmax(axis=1).value_counts())

    print(f"\n=== Low-variation flags (raw series, {START}..{END}) ===")
    raw, _ = build_components(df)
    rows = []
    for cid, s in raw.items():
        adj = (s * DIRECTION[cid]).sort_index()
        w = adj.loc[START:END]
        if w.dropna().empty:
            continue
        rstd = adj.rolling(1260, min_periods=504).std().loc[START:END]
        nun = adj.rolling(1260, min_periods=504).apply(
            lambda a: np.unique(a[~np.isnan(a)]).size, raw=True).loc[START:END]
        rows.append({
            "component": cid, "bucket": BUCKET_OF[cid],
            "frac_days_flat": round(float((w.diff() == 0).mean()), 3),
            "min_rolling_std": round(float(rstd.min()), 5) if rstd.notna().any() else np.nan,
            "min_rolling_nunique": int(nun.min()) if nun.notna().any() else -1,
        })
    flags = pd.DataFrame(rows).sort_values("frac_days_flat", ascending=False)
    print(flags.to_string(index=False))

    print("\n=== Live components per bucket at key dates ===")
    for d in ["2017-01-03", "2019-08-19", "2022-08-01"]:
        near = res.z_scores.index[res.z_scores.index.get_indexer(
            [pd.Timestamp(d)], method="nearest")[0]]
        per = res.components_by_bucket.loc[near].to_dict()
        print(f"   {d}: {per}  -> qualifying buckets="
              f"{int(res.available_bucket_count.loc[near])}, "
              f"published={bool(res.published_mask.loc[near])}")


if __name__ == "__main__":
    main()
