"""
index/composite.py
==================
Constructs the Composite Liquidity Index from the raw components.

Pipeline (requirements #7-#10)
------------------------------
1. For each component: multiply by its direction (looser = higher), then take a
   rolling z-score  ->  ``z_scores`` frame.
2. Sub-index per bucket = mean of that bucket's available component z-scores.
3. Composite z = weighted average of the five sub-indices. Weights are
   renormalised across whichever buckets have data on a given day, so a missing
   bucket never silently biases the index toward zero.
4. Rescale to the interpretable 0-100-ish scale:
        liquidity_index = 50 + 10 * composite_z
   => 50 neutral, >50 looser than normal, <50 tighter than normal.
5. Regime label, period changes, and an additive contribution decomposition so
   we can explain *why* liquidity is easing or tightening, not just the number.

The contribution maths is built so the per-bucket terms sum *exactly* to
(index - 50). That means the 1m change of the index equals the sum of the 1m
changes of the bucket contributions — the decomposition always reconciles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.transforms import rolling_zscore, Z_WINDOW, Z_MIN_PERIODS, Z_CLIP
from index.components import BUCKETS, DIRECTION, BUCKET_OF, build_components

# Index scaling constants (requirement #8)
INDEX_CENTER = 50.0
INDEX_SCALE = 10.0

# Regime thresholds (requirement #9). Higher = looser.
#   index >= 60  -> Loose
#   45 <= index < 60 -> Neutral
#   35 <= index < 45 -> Tight
#   index < 35  -> Stress
REGIME_THRESHOLDS = [(60.0, "Loose"), (45.0, "Neutral"), (35.0, "Tight")]

# Period look-backs in business days for the headline changes (requirement #9)
HORIZONS = {"1w": 5, "1m": 21, "3m": 63}


def regime_label(value: float) -> str:
    """Map an index level to a regime label."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    for cutoff, label in REGIME_THRESHOLDS:
        if value >= cutoff:
            return label
    return "Stress"


@dataclass
class IndexResult:
    """Everything the dashboard needs to render the index, in one object."""
    index: pd.Series             # final 0-100-ish liquidity index
    composite_z: pd.Series       # raw weighted-average z (index = 50 + 10*z)
    sub_indices: pd.DataFrame    # date x bucket sub-index z-scores
    z_scores: pd.DataFrame       # date x component direction-adjusted z-scores
    bucket_terms: pd.DataFrame   # date x bucket additive contributions to (index-50)
    weights: pd.Series           # bucket weights actually used
    meta: pd.DataFrame           # per-component availability metadata

    # ------- convenience accessors -------------------------------------
    @property
    def latest(self) -> float:
        return float(self.index.dropna().iloc[-1]) if self.index.notna().any() else float("nan")

    @property
    def latest_regime(self) -> str:
        return regime_label(self.latest)

    def changes(self) -> dict[str, float]:
        """Index point change over each headline horizon."""
        s = self.index.dropna()
        out = {}
        for name, n in HORIZONS.items():
            out[name] = float(s.iloc[-1] - s.iloc[-1 - n]) if len(s) > n else float("nan")
        return out

    def level_contributions(self) -> pd.Series:
        """Latest contribution of each bucket to (index - 50), in index points."""
        return self.bucket_terms.dropna(how="all").iloc[-1]

    def change_contributions(self, horizon: str = "1m") -> pd.Series:
        """Each bucket's contribution to the index change over ``horizon``.

        Sums exactly to the index change over the same window.
        """
        n = HORIZONS[horizon]
        terms = self.bucket_terms.dropna(how="all")
        if len(terms) <= n:
            return pd.Series(dtype=float)
        return terms.iloc[-1] - terms.iloc[-1 - n]

    def drivers(self, horizon: str = "1m") -> tuple[str, str]:
        """(main_easing_bucket_label, main_tightening_bucket_label) for horizon.

        Easing  = most positive contribution to the change (pushes index up).
        Tighten = most negative contribution to the change (pushes index down).
        """
        contrib = self.change_contributions(horizon)
        if contrib.empty or contrib.isna().all():
            return ("n/a", "n/a")
        easing = contrib.idxmax()
        tightening = contrib.idxmin()
        easing_lbl = BUCKETS.get(easing, {}).get("label", easing)
        tight_lbl = BUCKETS.get(tightening, {}).get("label", tightening)
        return (easing_lbl, tight_lbl)


def compute_index(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    z_window: int = Z_WINDOW,
    z_min_periods: int = Z_MIN_PERIODS,
    z_clip: float = Z_CLIP,
) -> IndexResult:
    """Build the Composite Liquidity Index from the price panel."""
    # 1. Raw components + availability metadata.
    raw, meta = build_components(df)

    # 2. Direction-adjust then z-score each available component.
    z_cols: dict[str, pd.Series] = {}
    for comp_id, series in raw.items():
        adjusted = series * DIRECTION[comp_id]   # looser = higher (requirement #7)
        z_cols[comp_id] = rolling_zscore(
            adjusted, window=z_window, min_periods=z_min_periods, clip=z_clip
        )
    z_scores = pd.DataFrame(z_cols).sort_index() if z_cols else pd.DataFrame()

    # 3. Sub-index per bucket = mean of available component z-scores.
    sub_data: dict[str, pd.Series] = {}
    for bucket in BUCKETS:
        members = [c for c in z_scores.columns if BUCKET_OF[c] == bucket]
        if members:
            sub_data[bucket] = z_scores[members].mean(axis=1, skipna=True)
    sub_indices = pd.DataFrame(sub_data).sort_index() if sub_data else pd.DataFrame()

    # 4. Weighted composite with per-row weight renormalisation.
    base_weights = {b: BUCKETS[b]["weight"] for b in BUCKETS}
    if weights:
        base_weights.update(weights)
    w = pd.Series(base_weights)
    # Keep only buckets that actually produced a sub-index.
    w = w[[b for b in w.index if b in sub_indices.columns]]
    w = w / w.sum() if w.sum() else w  # normalise the supplied weights to 1.0

    if sub_indices.empty:
        empty = pd.Series(dtype=float)
        return IndexResult(empty, empty, sub_indices, z_scores,
                           pd.DataFrame(), w, meta)

    avail = sub_indices[w.index].notna()
    # Per-row sum of weights over buckets that have data that day.
    row_w = avail.mul(w, axis=1).sum(axis=1).replace(0.0, np.nan)
    # Additive bucket terms: 10 * weight_b * sub_b / row_w. NaN where bucket
    # missing that day, and excluded from the renormalised denominator.
    weighted = sub_indices[w.index].mul(w, axis=1)
    bucket_terms = INDEX_SCALE * weighted.div(row_w, axis=0)

    # Composite z and final index.
    index = INDEX_CENTER + bucket_terms.sum(axis=1, min_count=1)
    composite_z = (index - INDEX_CENTER) / INDEX_SCALE

    return IndexResult(
        index=index,
        composite_z=composite_z,
        sub_indices=sub_indices,
        z_scores=z_scores,
        bucket_terms=bucket_terms,
        weights=w,
        meta=meta,
    )
