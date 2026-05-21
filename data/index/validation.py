"""
index/validation.py
===================
Benchmarks the Composite Liquidity Index against established gauges
(requirement #11). These are used ONLY for validation/comparison — they are
never inputs to the index itself.

Available benchmarks in this dataset:
    Bloomberg US FCI (BFCIUS) : higher = looser  -> same orientation as our index
    Chicago Fed NFCI          : higher = tighter -> flipped to a "looseness" view
The OFR Financial Stress Index is not in the data file; if a column for it is
added later it will be picked up automatically.

To compare series with different native units we standardise each to a z-score
for the overlay chart, while correlations are computed on the aligned levels and
on daily changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.loader import get_series

# label -> (ticker_key, sign) where sign maps the benchmark to "looseness"
# (+1 keeps as-is, -1 flips so that higher = looser, matching our index).
BENCHMARKS = {
    "Bloomberg US FCI": ("FCI_BBG", +1),
    "Chicago Fed NFCI": ("FCI_NFCI", -1),
    "OFR Financial Stress Index": ("OFR_FSI", -1),  # absent today; future-proof
}

# Named crisis windows for the regime behaviour check (requirement #11).
CRISIS_WINDOWS = {
    "Sep 2019 repo spike":   ("2019-09-01", "2019-10-15"),
    "COVID Mar 2020":        ("2020-02-20", "2020-04-30"),
    "2022 QT / rate shock":  ("2022-09-01", "2022-11-30"),
    "Mar 2023 bank stress":  ("2023-03-01", "2023-04-15"),
}


def _zscore_full(s: pd.Series) -> pd.Series:
    """Whole-sample standardisation, used only for the overlay comparison."""
    s = s.dropna()
    if s.std() == 0 or s.empty:
        return s * 0.0
    return (s - s.mean()) / s.std()


def benchmark_looseness(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return available benchmarks oriented so higher = looser."""
    out: dict[str, pd.Series] = {}
    for label, (key, sign) in BENCHMARKS.items():
        s = get_series(df, key)
        if not s.empty:
            out[label] = s * sign
    return out


def aligned_panel(index: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """Index + benchmarks (looseness-oriented) on a common daily index."""
    cols = {"Liquidity Index": index}
    cols.update(benchmark_looseness(df))
    panel = pd.DataFrame(cols).sort_index()
    return panel.dropna(how="all")


def correlation_table(index: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """Correlation of the index vs each benchmark, on levels and on changes."""
    bench = benchmark_looseness(df)
    rows = []
    for label, s in bench.items():
        joined = pd.concat([index, s], axis=1, join="inner").dropna()
        joined.columns = ["idx", "bench"]
        if len(joined) < 30:
            rows.append({"benchmark": label, "corr_levels": np.nan,
                         "corr_changes": np.nan, "n_obs": len(joined)})
            continue
        corr_lvl = joined["idx"].corr(joined["bench"])
        chg = joined.diff().dropna()
        corr_chg = chg["idx"].corr(chg["bench"])
        rows.append({
            "benchmark": label,
            "corr_levels": float(corr_lvl),
            "corr_changes": float(corr_chg),
            "n_obs": int(len(joined)),
        })
    return pd.DataFrame(rows)


def rolling_correlation(index: pd.Series, df: pd.DataFrame,
                        window: int = 252) -> pd.DataFrame:
    """Rolling (default 1y) correlation of the index vs each benchmark levels."""
    bench = benchmark_looseness(df)
    out = {}
    for label, s in bench.items():
        joined = pd.concat([index, s], axis=1, join="inner").dropna()
        joined.columns = ["idx", "bench"]
        if len(joined) > window:
            out[label] = joined["idx"].rolling(window).corr(joined["bench"])
    return pd.DataFrame(out).sort_index() if out else pd.DataFrame()


def standardized_overlay(index: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """Index + benchmarks each standardised to z, for a same-scale overlay.

    Restricted to the index's published (non-NaN) span so every series is
    standardised over the same period — otherwise the benchmarks (which run from
    2015) would be centred on a different sample than the index (published from
    ~2019), distorting the visual comparison.
    """
    valid = index.dropna()
    panel = aligned_panel(index, df)
    if len(valid):
        panel = panel.loc[(panel.index >= valid.index[0]) & (panel.index <= valid.index[-1])]
    return panel.apply(_zscore_full)


def crisis_behaviour(index: pd.Series) -> pd.DataFrame:
    """Min / mean index level inside each crisis window — should dip into Tight/Stress."""
    rows = []
    s = index.dropna()
    for name, (start, end) in CRISIS_WINDOWS.items():
        window = s.loc[(s.index >= start) & (s.index <= end)]
        if window.empty:
            rows.append({"crisis": name, "min": np.nan, "mean": np.nan,
                         "trough_date": pd.NaT})
            continue
        rows.append({
            "crisis": name,
            "min": float(window.min()),
            "mean": float(window.mean()),
            "trough_date": window.idxmin(),
        })
    return pd.DataFrame(rows)


def lead_lag(index: pd.Series, df: pd.DataFrame, benchmark: str,
             max_lag: int = 20) -> pd.Series:
    """Cross-correlation of daily index changes vs a benchmark's changes.

    Positive lag k = the index leads the benchmark by k business days
    (i.e. corr(index_t, benchmark_{t+k})). The peak location indicates which
    series tends to move first.
    """
    bench = benchmark_looseness(df).get(benchmark)
    if bench is None:
        return pd.Series(dtype=float)
    joined = pd.concat([index, bench], axis=1, join="inner").dropna()
    joined.columns = ["idx", "bench"]
    di = joined["idx"].diff()
    db = joined["bench"].diff()
    lags = range(-max_lag, max_lag + 1)
    vals = {k: di.corr(db.shift(-k)) for k in lags}
    return pd.Series(vals)
