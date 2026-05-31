"""
index/composite.py
==================
Constructs the Composite Liquidity Index from the raw components.

Pipeline
--------
1. For each component: multiply by its direction (looser = higher), then take a
   rolling z-score (with a low-variation guard) -> ``z_scores`` frame.
2. Sub-index per bucket = mean of that bucket's available component z-scores,
   but ONLY on days the bucket has at least ``min_per_bucket`` live components.
   (A single fragile series must not *be* a bucket — that was the source of the
   2016-2018 spikes, when money-market = EFFR-IORB alone.)
3. Composite z = weighted average of the sub-indices, with weights renormalised
   across whichever buckets qualify that day. The renormalised (effective)
   weights are exposed so the concentration is transparent.
4. Rescale: ``liquidity_index = 50 + 10 * composite_z`` (50 neutral).
5. COVERAGE GATE: a date is only PUBLISHED if it has >= ``min_buckets``
   qualifying buckets and >= ``min_components`` contributing components, and is
   past the rolling-z warm-up. Dates failing the gate are set to NaN in the
   published ``index`` (the unmasked ``raw_index`` is retained for diagnostics).
6. Regime label, period changes, and an additive bucket contribution
   decomposition (terms sum exactly to index-50, so the decomposition always
   reconciles).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.transforms import (
    rolling_zscore, lowfreq_zscore, OBS_WINDOW_BY_FREQ,
    Z_WINDOW, Z_MIN_PERIODS, Z_CLIP, Z_MIN_UNIQUE,
)
from index.components import (
    BUCKETS, DIRECTION, BUCKET_OF, build_components, max_ffill_of,
    frequency_of, observation_mode_of, observation_weekday_of,
)

# Index scaling
INDEX_CENTER = 50.0
INDEX_SCALE = 10.0

# Regime thresholds. Higher = looser.
REGIME_THRESHOLDS = [(60.0, "Loose"), (45.0, "Neutral"), (35.0, "Tight")]

# Headline change horizons in business days
HORIZONS = {"1w": 5, "1m": 21, "3m": 63}

# -------------------- Coverage / reliability rules --------------------------
# A bucket needs at least this many live components before its sub-index counts.
# Stops a lone, possibly-flat series from carrying a full bucket weight.
MIN_COMPONENTS_PER_BUCKET = 2
# A date is only published if at least this many buckets qualify ...
MIN_AVAILABLE_BUCKETS = 3
# ... and at least this many components (within qualifying buckets) contribute.
MIN_AVAILABLE_COMPONENTS = 8
# After the index first becomes computable, skip this many business days so the
# rolling z-scores are past their warm-up before we publish.
WARMUP_DAYS_AFTER_FIRST_VALID = 126


def regime_label(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    for cutoff, label in REGIME_THRESHOLDS:
        if value >= cutoff:
            return label
    return "Stress"


@dataclass
class IndexResult:
    """Everything the dashboard needs to render the index, in one object."""
    index: pd.Series             # PUBLISHED 50-centred index (low-coverage -> NaN)
    raw_index: pd.Series         # index before the coverage/warm-up mask (diagnostics)
    composite_z: pd.Series       # weighted-average z (raw_index = 50 + 10*z)
    sub_indices: pd.DataFrame    # date x bucket sub-index z-scores (min_per_bucket applied)
    z_scores: pd.DataFrame       # date x component direction-adjusted z-scores
    bucket_terms: pd.DataFrame   # date x bucket additive contributions to (raw_index-50)
    component_terms: pd.DataFrame  # date x component additive contributions to (raw_index-50)
    weights: pd.Series           # base bucket weights (normalised over buckets seen)
    effective_weights: pd.DataFrame  # date x bucket renormalised weights actually used
    components_by_bucket: pd.DataFrame  # date x bucket live-component counts
    available_component_count: pd.Series  # contributing components per day
    available_bucket_count: pd.Series     # qualifying buckets per day
    coverage_ok: pd.Series       # bool: meets min bucket/component rules
    published_mask: pd.Series    # bool: coverage_ok AND past warm-up
    meta: pd.DataFrame           # per-component availability metadata
    first_valid_date: pd.Timestamp | None = None      # first computable date
    first_published_date: pd.Timestamp | None = None  # first reliable/published date

    # ------- convenience accessors (operate on the PUBLISHED index) ----------
    @property
    def latest(self) -> float:
        return float(self.index.dropna().iloc[-1]) if self.index.notna().any() else float("nan")

    @property
    def latest_regime(self) -> str:
        return regime_label(self.latest)

    def changes(self) -> dict[str, float]:
        s = self.index.dropna()
        out = {}
        for name, n in HORIZONS.items():
            out[name] = float(s.iloc[-1] - s.iloc[-1 - n]) if len(s) > n else float("nan")
        return out

    def _terms_on_published(self, terms: pd.DataFrame) -> pd.DataFrame:
        """Restrict a terms frame to the published index dates, so contribution
        differences reconcile exactly to the published index changes regardless of
        any unpublished/low-frequency rows elsewhere in the frame."""
        if terms is None or terms.empty:
            return pd.DataFrame()
        pub = self.index.dropna().index
        return terms.reindex(pub)

    def level_contributions(self) -> pd.Series:
        terms = self._terms_on_published(self.bucket_terms).dropna(how="all")
        return terms.iloc[-1] if len(terms) else pd.Series(dtype=float)

    def change_contributions(self, horizon: str = "1m") -> pd.Series:
        n = HORIZONS[horizon]
        terms = self._terms_on_published(self.bucket_terms).dropna(how="all")
        if len(terms) <= n:
            return pd.Series(dtype=float)
        return terms.iloc[-1] - terms.iloc[-1 - n]

    def drivers(self, horizon: str = "1m") -> tuple[str, str]:
        contrib = self.change_contributions(horizon)
        if contrib.empty or contrib.isna().all():
            return ("n/a", "n/a")
        easing_lbl = BUCKETS.get(contrib.idxmax(), {}).get("label", contrib.idxmax())
        tight_lbl = BUCKETS.get(contrib.idxmin(), {}).get("label", contrib.idxmin())
        return (easing_lbl, tight_lbl)

    def coverage_frame(self) -> pd.DataFrame:
        """Tidy frame for the coverage diagnostic chart."""
        return pd.DataFrame({
            "components": self.available_component_count,
            "buckets": self.available_bucket_count,
        })

    def component_level_contributions(self) -> pd.Series:
        """Latest contribution of each component to (index - 50), index points."""
        terms = self._terms_on_published(self.component_terms).dropna(how="all")
        return terms.iloc[-1] if len(terms) else pd.Series(dtype=float)

    def component_change_contributions(self, horizon: str = "1m") -> pd.Series:
        """Each component's contribution to the index change over ``horizon``."""
        n = HORIZONS[horizon]
        terms = self._terms_on_published(self.component_terms).dropna(how="all")
        if len(terms) <= n:
            return pd.Series(dtype=float)
        return terms.iloc[-1] - terms.iloc[-1 - n]


def compute_index(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    z_window: int = Z_WINDOW,
    z_min_periods: int = Z_MIN_PERIODS,
    z_clip: float = Z_CLIP,
    z_min_unique: int | None = Z_MIN_UNIQUE,
    min_per_bucket: int = MIN_COMPONENTS_PER_BUCKET,
    min_buckets: int = MIN_AVAILABLE_BUCKETS,
    min_components: int = MIN_AVAILABLE_COMPONENTS,
    warmup_days: int = WARMUP_DAYS_AFTER_FIRST_VALID,
    lowfreq_handling: bool = True,
    ffill_cap: str = "by_freq",
) -> IndexResult:
    """Build the Composite Liquidity Index from the price panel.

    ``lowfreq_handling`` (True) z-scores weekly/low-frequency components on their
    true observations; set False to treat every daily row as a fresh print (the
    legacy behaviour). ``ffill_cap`` is "by_freq" (per-component caps) or "none"
    (unlimited forward-fill, legacy).
    """
    # 1. Raw components + availability metadata.
    raw, meta = build_components(df)

    # 2. Direction-adjust then z-score each available component. Daily series use
    #    a daily rolling window; weekly/low-frequency series are standardised on
    #    their true observations (e.g. Wednesdays) and the z is forward-filled.
    daily_grid = None
    if raw:
        lo = min(s.index.min() for s in raw.values())
        hi = max(s.index.max() for s in raw.values())
        daily_grid = pd.date_range(lo, hi, freq="B")
    cap_daily = (lambda cid: max_ffill_of(cid)) if ffill_cap == "by_freq" else (lambda cid: None)

    z_cols: dict[str, pd.Series] = {}
    for comp_id, series in raw.items():
        adjusted = series * DIRECTION[comp_id]   # looser = higher
        mode = observation_mode_of(comp_id)
        if lowfreq_handling and mode != "daily":
            win, minp = OBS_WINDOW_BY_FREQ.get(frequency_of(comp_id), (Z_WINDOW, Z_MIN_PERIODS))
            z_cols[comp_id] = lowfreq_zscore(
                adjusted, daily_grid, mode, observation_weekday_of(comp_id),
                window=win, min_periods=minp, clip=z_clip, min_unique=z_min_unique,
                max_ffill=max_ffill_of(comp_id),
            )
        else:
            z_cols[comp_id] = rolling_zscore(
                adjusted, window=z_window, min_periods=z_min_periods, clip=z_clip,
                min_unique=z_min_unique, max_ffill=cap_daily(comp_id),
            )
    z_scores = pd.DataFrame(z_cols).sort_index() if z_cols else pd.DataFrame()
    # Pin the whole index to a business-day grid. Some raw Bloomberg series carry
    # stray weekend timestamps; without this, daily components would get z-scores
    # on Sat/Sun while weekly components (already on a B-day grid) are NaN there,
    # giving inconsistent bucket coverage on weekends and breaking the change
    # reconciliation (the row-counted "1w ago" could land on a weekend).
    if not z_scores.empty and daily_grid is not None:
        z_scores = z_scores.reindex(daily_grid)

    if z_scores.empty:
        empty = pd.Series(dtype=float)
        empty_df = pd.DataFrame()
        return IndexResult(empty, empty, empty, empty_df, z_scores, empty_df,
                           empty_df, pd.Series(dtype=float), empty_df, empty_df,
                           empty, empty, empty, empty, meta)

    # 3. Per-bucket live-component counts and sub-index (with min_per_bucket).
    sub_data: dict[str, pd.Series] = {}
    count_data: dict[str, pd.Series] = {}
    for bucket in BUCKETS:
        members = [c for c in z_scores.columns if BUCKET_OF[c] == bucket]
        if not members:
            count_data[bucket] = pd.Series(0, index=z_scores.index)
            continue
        member_z = z_scores[members]
        cnt = member_z.notna().sum(axis=1)
        count_data[bucket] = cnt
        # Sub-index only on days the bucket has >= min_per_bucket live components.
        sub_data[bucket] = member_z.mean(axis=1, skipna=True).where(cnt >= min_per_bucket)
    components_by_bucket = pd.DataFrame(count_data).sort_index()
    sub_indices = pd.DataFrame(sub_data).sort_index() if sub_data else pd.DataFrame()

    # 4. Weighted composite with per-row weight renormalisation over QUALIFYING
    #    buckets (those with a non-NaN sub-index that day).
    base_weights = {b: BUCKETS[b]["weight"] for b in BUCKETS}
    if weights:
        base_weights.update(weights)
    w = pd.Series(base_weights)
    w = w[[b for b in w.index if b in sub_indices.columns]]
    w = w / w.sum() if w.sum() else w

    avail = sub_indices[w.index].notna()                 # qualifying buckets per day
    row_w = avail.mul(w, axis=1).sum(axis=1).replace(0.0, np.nan)
    effective_weights = avail.mul(w, axis=1).div(row_w, axis=0)   # rows sum to 1.0

    # Additive bucket terms: 10 * eff_weight_b * sub_b. Sum -> raw_index - 50.
    bucket_terms = INDEX_SCALE * sub_indices[w.index].mul(effective_weights)
    raw_index = INDEX_CENTER + bucket_terms.sum(axis=1, min_count=1)
    composite_z = (raw_index - INDEX_CENTER) / INDEX_SCALE

    # Component-level additive terms (requirement #3):
    #   term_i = 10 * eff_weight_b * z_i / n_live_in_bucket_b
    # so terms sum within a bucket to bucket_term_b, and across all to raw_index-50.
    comp_cols: dict[str, pd.Series] = {}
    for comp_id in z_scores.columns:
        b = BUCKET_OF[comp_id]
        if b not in effective_weights.columns:
            continue
        n_live = components_by_bucket[b].replace(0, np.nan)
        comp_cols[comp_id] = INDEX_SCALE * effective_weights[b] * z_scores[comp_id] / n_live
    component_terms = pd.DataFrame(comp_cols).sort_index() if comp_cols else pd.DataFrame()

    # 5. Coverage diagnostics + publication gate.
    available_bucket_count = avail.sum(axis=1)
    # Components that actually contribute = those in qualifying buckets.
    contributing = pd.Series(0, index=z_scores.index)
    for bucket in avail.columns:
        members = [c for c in z_scores.columns if BUCKET_OF[c] == bucket]
        if members:
            contributing = contributing.add(
                z_scores[members].notna().sum(axis=1).where(avail[bucket], 0),
                fill_value=0)
    available_component_count = contributing.astype(int)

    coverage_ok = (available_bucket_count >= min_buckets) & \
                  (available_component_count >= min_components)

    # Warm-up: skip the first warmup_days business days after the index is first
    # computable, so the rolling z-scores are mature before we publish.
    computable = raw_index.notna()
    first_valid_date = raw_index.index[computable.argmax()] if computable.any() else None
    if first_valid_date is not None and warmup_days > 0:
        warmup_cutoff = first_valid_date + pd.tseries.offsets.BusinessDay(warmup_days)
        warmup_ok = pd.Series(raw_index.index >= warmup_cutoff, index=raw_index.index)
    else:
        warmup_ok = pd.Series(True, index=raw_index.index)

    published_mask = coverage_ok.reindex(raw_index.index, fill_value=False) & warmup_ok
    index = raw_index.where(published_mask)
    published = index.dropna()
    first_published_date = published.index[0] if len(published) else None

    return IndexResult(
        index=index,
        raw_index=raw_index,
        composite_z=composite_z,
        sub_indices=sub_indices,
        z_scores=z_scores,
        bucket_terms=bucket_terms,
        component_terms=component_terms,
        weights=w,
        effective_weights=effective_weights,
        components_by_bucket=components_by_bucket,
        available_component_count=available_component_count,
        available_bucket_count=available_bucket_count,
        coverage_ok=coverage_ok,
        published_mask=published_mask,
        meta=meta,
        first_valid_date=first_valid_date,
        first_published_date=first_published_date,
    )
