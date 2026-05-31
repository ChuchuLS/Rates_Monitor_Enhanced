"""
data/transforms.py
==================
Pure, stateless numerical helpers. No Streamlit, no plotting — just pandas in,
pandas out. This keeps the z-score / normalisation methodology in one testable
place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Z-score methodology
# ---------------------------------------------------------------------------
# Every indicator is standardised with a *rolling* z-score rather than a static
# full-sample one, so "looser/tighter than normal" is judged against the recent
# regime rather than against 2008.
#
#     z = (x - rolling_mean) / rolling_std
#
# Defaults:
#   window      = 1260  (~5 years of trading days)
#   min_periods =  504  (~2 years; below this we emit NaN rather than a noisy z)
#   clip        = [-3, 3] (cap the influence of any single outlier print)
#
# Low-variation guard (fixes the 2016-2018 spike pathology)
# ---------------------------------------------------------
# A z-score divides by the rolling std regardless of whether the variation is
# economically meaningful. For a spread that sits in a 2-3bp range for years
# (e.g. EFFR-IORB pre-2019), the rolling std collapses toward zero and an
# otherwise-trivial 1bp move becomes a +/-3 sigma "spike". We therefore NaN the
# z-score whenever the trailing window contains too few DISTINCT values
# (``min_unique``) — a scale-free way of saying "this series has been too flat
# to standardise reliably". This is preferred over a fixed ``min_std`` floor,
# which is impossible to choose sensibly across reserves ($mn), spreads (bp) and
# indices (pts).
Z_WINDOW = 1260
Z_MIN_PERIODS = 504
Z_CLIP = 3.0
Z_MIN_UNIQUE = 20          # trailing window must contain >= this many distinct values
DEFAULT_MAX_FFILL = 5      # business days a value may persist before going stale


def capped_ffill(s: pd.Series, max_ffill: int | None = DEFAULT_MAX_FFILL) -> pd.Series:
    """Forward-fill, but only up to ``max_ffill`` rows since the last real obs.

    Prevents a series that stops updating from injecting a fake flat signal
    forever: once it has been stale for more than ``max_ffill`` steps it reverts
    to NaN until the next genuine observation. ``max_ffill=None`` disables the
    cap (unlimited ffill). Operates on the series' own index (one row per obs).
    """
    if s is None or s.empty:
        return s
    s = s.sort_index()
    if max_ffill is None:
        return s.ffill()
    filled = s.ffill(limit=max_ffill)
    return filled


def rolling_zscore(
    s: pd.Series,
    window: int = Z_WINDOW,
    min_periods: int = Z_MIN_PERIODS,
    clip: float = Z_CLIP,
    min_unique: int | None = Z_MIN_UNIQUE,
    max_ffill: int | None = DEFAULT_MAX_FFILL,
) -> pd.Series:
    """Rolling z-score of a series, clipped to +/- ``clip``.

    Steps:
      1. capped forward-fill (so a stale series becomes NaN, not a flat line),
      2. rolling mean/std over ``window`` (needs ``min_periods``),
      3. NaN out dates whose trailing window has < ``min_unique`` distinct values
         OR zero std (the low-variation guard described above),
      4. clip to +/- ``clip``.

    We never back-fill, so no future information leaks in.
    """
    if s is None or s.empty:
        return pd.Series(dtype=float)
    s = capped_ffill(s.sort_index(), max_ffill)
    mean = s.rolling(window, min_periods=min_periods).mean()
    std = s.rolling(window, min_periods=min_periods).std()

    z = (s - mean) / std.replace(0.0, np.nan)

    if min_unique is not None and min_unique > 1:
        # Distinct-value count in the trailing window; cheap np.unique per row.
        nuniq = s.rolling(window, min_periods=min_periods).apply(
            lambda a: np.unique(a[~np.isnan(a)]).size, raw=True
        )
        z = z.where(nuniq >= min_unique)

    return z.clip(-clip, clip)


# Per-frequency rolling window/min-periods in *observation* units, so a weekly
# series is judged over ~5y of weekly prints (not 1260 weeks). Keeps the
# "5-year regime" interpretation consistent across frequencies.
OBS_WINDOW_BY_FREQ = {
    "daily":     (1260, 504),
    "weekly":    (260, 104),
    "monthly":   (60, 24),
    "irregular": (60, 24),
}


def true_observations(s: pd.Series, mode: str, weekday: int | None = None) -> pd.Series:
    """Reduce a daily-dense series to its genuine observation dates.

    mode = "weekday"     : keep only the official observation weekday (e.g. Fed
                           H.4.1 reserves are Wednesday-dated) — safe even when
                           the value legitimately repeats week to week.
    mode = "change_dates": keep only rows where the value changed — a fallback
                           when the observation calendar is unknown (can wrongly
                           drop genuinely-unchanged prints, so not the default).
    mode = "daily"/other : every row is a real observation.
    """
    s = s.sort_index().dropna()
    if s.empty:
        return s
    if mode == "weekday" and weekday is not None:
        return s[s.index.weekday == weekday]
    if mode == "change_dates":
        keep = s.ne(s.shift())
        keep.iloc[0] = True
        return s[keep]
    return s


def lowfreq_zscore(
    adjusted_daily: pd.Series,
    daily_index: pd.Index,
    mode: str,
    weekday: int | None,
    window: int,
    min_periods: int,
    clip: float = Z_CLIP,
    min_unique: int | None = Z_MIN_UNIQUE,
    max_ffill: int | None = 10,
) -> pd.Series:
    """Z-score a weekly/low-frequency series on its TRUE observations, then map
    back to the daily grid by forward-filling the *z-score* (not the raw value)
    for at most ``max_ffill`` business days. Beyond that the component goes stale
    (NaN) until the next real observation — so a frozen weekly print can't stay
    live forever, and repeated daily values never count as fresh observations.
    """
    obs = true_observations(adjusted_daily, mode, weekday)
    if obs.empty:
        return pd.Series(index=daily_index, dtype=float)
    z_obs = rolling_zscore(obs, window=window, min_periods=min_periods,
                           clip=clip, min_unique=min_unique, max_ffill=None)
    grid = daily_index.union(z_obs.index)
    z_daily = z_obs.reindex(grid).sort_index().ffill(limit=max_ffill)
    return z_daily.reindex(daily_index)


def align_frame(series_map: dict[str, pd.Series], max_ffill: int | None = DEFAULT_MAX_FFILL) -> pd.DataFrame:
    """Combine named series into one daily-frequency frame, capped-forward-filled.

    Empty series are dropped so a missing ticker never creates an all-NaN column
    that would drag a bucket average toward NaN.
    """
    clean = {k: v for k, v in series_map.items() if v is not None and not v.empty}
    if not clean:
        return pd.DataFrame()
    df = pd.DataFrame(clean).sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(df.index.union(full_idx))
    if max_ffill is None:
        return df.ffill()
    return df.ffill(limit=max_ffill)


def pct_missing(s: pd.Series) -> float:
    """Fraction of NaN observations over the span of the series."""
    if s is None or len(s) == 0:
        return float("nan")
    return float(s.isna().mean())
