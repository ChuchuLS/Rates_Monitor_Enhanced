"""
index/components.py
===================
Defines the *building blocks* of the Composite Liquidity Index: the five sub-
index buckets (requirement #6), the raw indicators in each, and — critically —
the DIRECTION of each indicator.

Sign convention (requirement #7)
--------------------------------
Before z-scoring, every indicator is multiplied by its ``direction`` so that:

    higher adjusted value  ==  LOOSER liquidity
    lower  adjusted value  ==  TIGHTER liquidity

Examples:
    HY OAS  : a *wider* spread means tighter liquidity -> direction = -1
    Reserves: *more* reserves means looser liquidity   -> direction = +1
    XCCY    : a *less negative* basis means looser USD  -> direction = +1

This single rule makes the whole index interpretable: a high final value is
always "loose", a low value always "tight", regardless of how the underlying
indicator is quoted.

Bucket weights (first-version, requirement #8)
    Money-market funding ........ 30%
    Dollar funding / XCCY ....... 20%
    Credit liquidity ............ 20%
    Central bank liquidity ...... 20%
    Market liquidity / vol ...... 10%
"""

from __future__ import annotations

import pandas as pd

from data.loader import get_series
from config.tickers import TICKERS

# ---------------------------------------------------------------------------
# Buckets: label, first-version weight, sort order
# ---------------------------------------------------------------------------
BUCKETS: dict[str, dict] = {
    "central_bank": {"label": "Central bank / reserves",   "weight": 0.20, "order": 4},
    "money_market": {"label": "Money-market funding",      "weight": 0.30, "order": 1},
    "xccy":         {"label": "Dollar funding / XCCY",     "weight": 0.20, "order": 2},
    "credit":       {"label": "Credit liquidity",          "weight": 0.20, "order": 3},
    "market_liq":   {"label": "Market liquidity / vol",    "weight": 0.10, "order": 5},
}

# ---------------------------------------------------------------------------
# Component metadata. ``direction`` encodes the looser=higher rule above.
# ``builder`` names the recipe used to construct the raw series from the data.
# ---------------------------------------------------------------------------
# Each entry: (component_id, label, bucket, direction, builder_spec)
#   builder_spec forms:
#     ("ticker", KEY)            -> raw ticker series
#     ("spread", KEY_A, KEY_B)   -> KEY_A - KEY_B (aligned)
#     ("mean", [KEY, ...])       -> row-wise mean of available tickers
#     ("mtg_spread", KEY_MTG, KEY_UST) -> (mortgage - UST) in same units
COMPONENTS: list[tuple] = [
    # A. Central bank / reserve liquidity --------------------------------
    ("cb_reserves",  "Fed reserve balances",      "central_bank", +1, ("ticker", "FED_RESERVES")),
    ("cb_repo",      "Fed repo / SRF usage",       "central_bank", -1, ("ticker", "FED_REPO")),
    # B. Money-market funding pressure (spreads vs IORB) -----------------
    ("mm_sofr_iorb", "SOFR - IORB",                "money_market", -1, ("spread", "SOFR", "IORB")),
    ("mm_effr_iorb", "EFFR - IORB",                "money_market", -1, ("spread", "EFFR", "IORB")),
    ("mm_sofr_effr", "SOFR - EFFR",                "money_market", -1, ("spread", "SOFR", "EFFR")),
    ("mm_tgcr_iorb", "TGCR - IORB",                "money_market", -1, ("spread", "TGCR", "IORB")),
    ("mm_bgcr_iorb", "BGCR - IORB",                "money_market", -1, ("spread", "BGCR", "IORB")),
    # C. Dollar funding / cross-currency basis ---------------------------
    ("xccy_eur", "EUR/USD 3M basis",               "xccy", +1, ("ticker", "XCCY_EUR")),
    ("xccy_jpy", "JPY/USD 3M basis",               "xccy", +1, ("ticker", "XCCY_JPY")),
    ("xccy_gbp", "GBP/USD 3M basis",               "xccy", +1, ("ticker", "XCCY_GBP")),
    ("xccy_aud", "AUD/USD 3M basis",               "xccy", +1, ("ticker", "XCCY_AUD")),
    ("xccy_cad", "CAD/USD 3M basis",               "xccy", +1, ("ticker", "XCCY_CAD")),
    # D. Credit liquidity / risk appetite --------------------------------
    ("cr_ig_oas",   "IG OAS",                      "credit", -1, ("ticker", "IG_OAS")),
    ("cr_hy_oas",   "HY OAS",                      "credit", -1, ("ticker", "HY_OAS")),
    ("cr_embi",     "EMBI sovereign spread",       "credit", -1, ("ticker", "EMBI")),
    ("cr_itrx_eu",  "iTraxx Europe Main",          "credit", -1, ("ticker", "ITRX_EUROPE")),
    ("cr_itrx_xo",  "iTraxx Crossover",            "credit", -1, ("ticker", "ITRX_XOVER")),
    ("cr_bank_cds", "Bank CDS (avg)",              "credit", -1, ("mean", ["CDS_BOFA", "CDS_JPM", "CDS_GS", "CDS_CITI"])),
    ("cr_mtg",      "Mortgage spread vs UST10Y",   "credit", -1, ("mtg_spread", "MTG_30Y", "US_10Y")),
    # E. Market liquidity / volatility -----------------------------------
    ("mkt_ust_liq", "UST liquidity index",         "market_liq", -1, ("ticker", "UST_LIQ")),
    ("mkt_swap",    "10Y swap spread",             "market_liq", +1, ("ticker", "SWAP_10Y")),
    ("mkt_move",    "MOVE (rate vol)",             "market_liq", -1, ("ticker", "MOVE")),
    ("mkt_vix",     "VIX (equity vol)",            "market_liq", -1, ("ticker", "VIX")),
]


# ---------------------------------------------------------------------------
# Update-frequency metadata (requirement #6).
# Several macro series (Fed reserve balances, Fed repo/SRF usage) are published
# weekly but arrive forward-filled onto a daily grid in the Bloomberg/Excel pull.
# We record the native frequency so the loader can cap how long a value may
# persist before it is treated as stale (NaN) rather than a live daily signal.
# ---------------------------------------------------------------------------
DEFAULT_FREQUENCY = "daily"
FREQUENCY: dict[str, str] = {
    "cb_reserves": "weekly",   # Fed H.4.1 reserve balances (Wednesday)
    "cb_repo":     "weekly",   # Fed repo / SRF usage (Wednesday)
    # everything else is daily
}

# Max business days a value may be carried forward before becoming NaN.
MAX_FFILL_BY_FREQ = {"daily": 5, "weekly": 10, "monthly": 35, "irregular": 10}


def frequency_of(comp_id: str) -> str:
    return FREQUENCY.get(comp_id, DEFAULT_FREQUENCY)


def max_ffill_of(comp_id: str) -> int:
    return MAX_FFILL_BY_FREQ.get(frequency_of(comp_id), 5)


# Observation mode controls how a daily-dense column is reduced to its genuine
# observations before z-scoring (requirement #4). Fed reserves/repo are reported
# weekly on Wednesdays even though the Bloomberg pull repeats the value daily, so
# we treat Wednesday as the true observation. Default for everything else is
# "daily" (every row is a real print).
OBSERVATION_MODE: dict[str, str] = {
    "cb_reserves": "weekday",
    "cb_repo":     "weekday",
}
OBSERVATION_WEEKDAY: dict[str, int] = {  # Monday=0 ... Sunday=6
    "cb_reserves": 2,   # Wednesday
    "cb_repo":     2,   # Wednesday
}


def observation_mode_of(comp_id: str) -> str:
    return OBSERVATION_MODE.get(comp_id, "daily")


def observation_weekday_of(comp_id: str) -> int | None:
    return OBSERVATION_WEEKDAY.get(comp_id)


# Map each component to its underlying Bloomberg ticker(s) for audit tables.
_SPEC_OF = {cid: spec for cid, _, _, _, spec in COMPONENTS}


def component_ticker(comp_id: str) -> str:
    """Human-readable Bloomberg ticker(s) behind a component's builder spec."""
    spec = _SPEC_OF.get(comp_id)
    if not spec:
        return ""
    kind = spec[0]
    if kind == "ticker":
        return TICKERS.get(spec[1], spec[1])
    if kind in ("spread", "mtg_spread"):
        return f"{TICKERS.get(spec[1], spec[1])} − {TICKERS.get(spec[2], spec[2])}"
    if kind == "mean":
        return " · ".join(TICKERS.get(k, k) for k in spec[1])
    return ""


def _build_raw(df: pd.DataFrame, spec: tuple) -> pd.Series:
    """Resolve a builder spec into a raw (un-adjusted) series."""
    kind = spec[0]
    if kind == "ticker":
        return get_series(df, spec[1])
    if kind == "spread":
        a = get_series(df, spec[1])
        b = get_series(df, spec[2])
        if a.empty or b.empty:
            return pd.Series(dtype=float)
        a, b = a.align(b, join="inner")
        return (a - b).dropna()
    if kind == "mean":
        cols = [get_series(df, k) for k in spec[1]]
        cols = [c for c in cols if not c.empty]
        if not cols:
            return pd.Series(dtype=float)
        frame = pd.concat(cols, axis=1)
        return frame.mean(axis=1).dropna()
    if kind == "mtg_spread":
        mtg = get_series(df, spec[1])
        ust = get_series(df, spec[2])
        if mtg.empty or ust.empty:
            return pd.Series(dtype=float)
        mtg, ust = mtg.align(ust, join="inner")
        # Both quoted in %, express the spread in bp.
        return ((mtg - ust) * 100).dropna()
    raise ValueError(f"Unknown builder spec: {spec!r}")


def build_components(df: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Build every component's raw series and an availability report.

    Returns
    -------
    raw : dict[component_id, Series]
        Only includes components whose underlying data is present.
    meta : DataFrame
        One row per defined component with label/bucket/direction/available
        so the UI can show exactly which indicators fed the index.
    """
    raw: dict[str, pd.Series] = {}
    meta_rows = []
    for comp_id, label, bucket, direction, spec in COMPONENTS:
        series = _build_raw(df, spec)
        available = not series.empty
        if available:
            raw[comp_id] = series
        meta_rows.append(
            {
                "component": comp_id,
                "label": label,
                "bucket": bucket,
                "bucket_label": BUCKETS[bucket]["label"],
                "direction": direction,
                "available": available,
                "n_obs": int(series.shape[0]),
                "frequency": frequency_of(comp_id),
                "max_ffill": max_ffill_of(comp_id),
                "observation_mode": observation_mode_of(comp_id),
                "ticker": component_ticker(comp_id),
            }
        )
    meta = pd.DataFrame(meta_rows)
    return raw, meta


# Quick lookup helpers used elsewhere
DIRECTION = {cid: d for cid, _, _, d, _ in COMPONENTS}
BUCKET_OF = {cid: b for cid, _, b, _, _ in COMPONENTS}
LABEL_OF = {cid: lab for cid, lab, _, _, _ in COMPONENTS}
