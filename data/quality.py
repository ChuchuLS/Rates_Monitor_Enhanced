"""
data/quality.py
===============
Data-quality checks for every Bloomberg ticker the dashboard depends on
(requirement #2). The output drives the "Data Quality" dashboard section.

For each ticker we report: internal key, Bloomberg ticker, whether the column
exists, latest available date, missing-data %, and a staleness flag. A series is
flagged stale when its last observation is more than ``STALE_BDAYS`` business
days behind the most recent date present anywhere in the dataset.
"""

from __future__ import annotations

import pandas as pd

# A series is "stale" if its newest point is more than this many business days
# behind the dataset's overall latest date.
STALE_BDAYS = 5


def validate_data(
    df: pd.DataFrame,
    tickers: dict[str, str],
    stale_bdays: int = STALE_BDAYS,
) -> pd.DataFrame:
    """Return a tidy data-quality report, one row per ticker.

    Mirrors the reference logic in the brief and adds the staleness flag.
    """
    # Reference "today" = latest date with any data in the whole panel.
    dataset_last = df.index.max() if len(df.index) else None
    stale_cutoff = (
        dataset_last - pd.tseries.offsets.BusinessDay(stale_bdays)
        if dataset_last is not None
        else None
    )

    rows = []
    for key, ticker in tickers.items():
        col = ticker.upper()
        exists = col in df.columns

        if exists and df[col].notna().any():
            valid = df[col].dropna()
            last_date = valid.index.max()
            missing_pct = float(df[col].isna().mean())
            n_obs = int(valid.shape[0])
            is_stale = bool(stale_cutoff is not None and last_date < stale_cutoff)
        else:
            last_date = pd.NaT
            missing_pct = float("nan")
            n_obs = 0
            is_stale = bool(exists)  # column present but empty == effectively stale

        rows.append(
            {
                "key": key,
                "ticker": ticker,
                "exists": exists,
                "last_date": last_date,
                "missing_pct": missing_pct,
                "n_obs": n_obs,
                "stale": is_stale,
            }
        )

    report = pd.DataFrame(rows)
    return report


def quality_summary(report: pd.DataFrame) -> dict[str, int]:
    """Headline counts for the top of the Data Quality panel."""
    total = len(report)
    missing = int((~report["exists"]).sum())
    stale = int(report["stale"].sum())
    healthy = total - missing - stale
    return {"total": total, "healthy": healthy, "stale": stale, "missing": missing}
