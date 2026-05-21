"""
data/transforms.py
==================
Pure, stateless numerical helpers. No Streamlit, no plotting — just pandas in,
pandas out. This keeps the z-score / normalisation methodology (requirement #7)
in one testable place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Z-score methodology (requirement #7)
# ---------------------------------------------------------------------------
# We standardise every indicator with a *rolling* z-score rather than a static
# full-sample one. A rolling window means "looser/tighter than normal" is judged
# against the recent regime, not against 2008, so the index stays interpretable
# as conditions evolve.
#
#     z = (x - rolling_mean) / rolling_std
#
# Defaults:
#   window      = 1260  (~5 years of trading days)
#   min_periods =  504  (~2 years; below this we emit NaN rather than a noisy z)
#   clip        = [-3, 3] (cap the influence of any single outlier print)
Z_WINDOW = 1260
Z_MIN_PERIODS = 504
Z_CLIP = 3.0


def rolling_zscore(
    s: pd.Series,
    window: int = Z_WINDOW,
    min_periods: int = Z_MIN_PERIODS,
    clip: float = Z_CLIP,
) -> pd.Series:
    """Rolling z-score of a series, clipped to +/- ``clip``.

    The series is forward-filled first so weekly/irregular indicators (e.g. Fed
    reserve balances) line up on the daily grid without injecting NaNs into the
    rolling statistics. We never back-fill, so no future information leaks in.
    """
    if s is None or s.empty:
        return pd.Series(dtype=float)
    s = s.sort_index().ffill()
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()
    z = (s - mean) / std.replace(0.0, np.nan)
    return z.clip(-clip, clip)


def align_frame(series_map: dict[str, pd.Series]) -> pd.DataFrame:
    """Combine named series into one daily-frequency frame, forward-filled.

    Empty series are dropped so a missing ticker never creates an all-NaN column
    that would drag a bucket average toward NaN.
    """
    clean = {k: v for k, v in series_map.items() if v is not None and not v.empty}
    if not clean:
        return pd.DataFrame()
    df = pd.DataFrame(clean).sort_index()
    # Reindex to a continuous business-day grid then forward-fill levels.
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(df.index.union(full_idx)).ffill()
    return df


def pct_missing(s: pd.Series) -> float:
    """Fraction of NaN observations over the span of the series."""
    if s is None or len(s) == 0:
        return float("nan")
    return float(s.isna().mean())
