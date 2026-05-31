"""Headless smoke test — verifies the package builds the index without a
Streamlit server. Run: python smoke_test.py"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

print("1. Importing all modules ...")
from data.loader import _load_dataframe, data_source_label, get_series
from data.quality import validate_data, quality_summary
from data.transforms import rolling_zscore
from config.tickers import TICKERS
from index.components import build_components, BUCKETS
from index.composite import compute_index, regime_label
from index import validation as V
# chart modules import streamlit but must not call st.* at import time
import charts.common, charts.rates, charts.funding, charts.credit, charts.liquidity
print("   all imports OK")

print(f"2. Loading data (source: {data_source_label()}) ...")
t0 = time.time()
df = _load_dataframe()
print(f"   {df.shape[0]:,} rows x {df.shape[1]:,} cols in {time.time()-t0:.2f}s")
assert isinstance(df.index, pd.DatetimeIndex)

print("3. Building components ...")
raw, meta = build_components(df)
avail = int(meta["available"].sum())
print(f"   {avail}/{len(meta)} components available")
print(f"   missing: {list(meta.loc[~meta['available'],'component'])}")
assert avail >= 15, "expected most components present"

print("4. Computing index ...")
t0 = time.time()
res = compute_index(df)
dt = time.time() - t0
print(f"   computed in {dt:.2f}s")
assert dt < 5, "index computation too slow"

latest = res.latest
print(f"   latest index = {latest:.2f}  regime = {res.latest_regime}")
assert 0 < latest < 100, "index out of expected range"
assert res.latest_regime in ("Loose", "Neutral", "Tight", "Stress")

print("5. Changes:", {k: round(v, 2) for k, v in res.changes().items()})

print("6. Contribution reconciliation ...")
lvl = res.level_contributions()
print("   level contributions:", {k: round(v, 3) for k, v in lvl.items()})
recon = abs(lvl.sum() - (latest - 50.0))
print(f"   sum(level contrib) - (index-50) = {recon:.6f}")
assert recon < 1e-6, "level contributions must reconcile to index-50"

for h in ("1w", "1m", "3m"):
    cc = res.change_contributions(h)
    idx_chg = res.changes()[h]
    if not cc.empty and not np.isnan(idx_chg):
        r = abs(cc.sum() - idx_chg)
        assert r < 1e-6, f"{h} change contributions must reconcile ({r})"
        print(f"   {h} change contrib reconciles (Σ={cc.sum():+.3f} vs Δindex={idx_chg:+.3f})")

print("7. Drivers (1m):", res.drivers("1m"))

print("8. Validation / benchmarks ...")
bench = V.benchmark_looseness(df)
print("   benchmarks available:", list(bench))
corr = V.correlation_table(res.index, df)
print(corr.to_string(index=False))
crisis = V.crisis_behaviour(res.index)
print("   crisis troughs:")
for _, row in crisis.iterrows():
    print(f"     {row['crisis']:<24} min={row['min']:.1f}")
roll = V.rolling_correlation(res.index, df)
print(f"   rolling-corr frame: {roll.shape}")
if bench:
    ll = V.lead_lag(res.index, df, list(bench)[0])
    print(f"   lead-lag peak lag = {ll.idxmax()}d (corr {ll.max():.2f})")

print("9. Data quality ...")
rep = validate_data(df, TICKERS)
print("   summary:", quality_summary(rep))

# ---------------------------------------------------------------------------
# v0.3 audit / methodology / reconciliation checks (requirement #6)
# ---------------------------------------------------------------------------
from index.methodology import (
    compute_legacy_index, reconciliation, component_contribution_table,
    forward_fill_audit, methodology_audit, INDEX_METHODOLOGY,
)
from index.components import frequency_of, max_ffill_of

print("10. Methodology & reconciliation ...")
# (1) current index not NaN given sufficient coverage
assert not np.isnan(res.latest), "current index must not be NaN with coverage"
# (2) bucket contributions sum to index-50
assert abs(res.level_contributions().sum() - (latest - 50.0)) < 1e-6
# (3) component contributions sum to index-50
ct = component_contribution_table(res, df)
live = ct[ct["live"]]
assert abs(live["contribution"].sum() - (latest - 50.0)) < 1e-6, "component contribs must reconcile"
print(f"    component contribs reconcile (Σ={live['contribution'].sum():+.3f}, index-50={latest-50:.3f})")
# (4) reconciliation: Σ contrib diffs == current - legacy index
legacy = compute_legacy_index(df)
rec = reconciliation(res, legacy, df)
c = rec["checks"]
assert abs(c["sum_current_contrib"] - c["current_index_minus_50"]) < 1e-6
assert abs(c["sum_legacy_contrib"] - c["legacy_index_minus_50"]) < 1e-6
assert abs(c["sum_contrib_diff"] - c["current_minus_legacy"]) < 1e-6
print(f"    reconciliation holds: legacy={rec['legacy_index']:.2f} current={rec['current_index']:.2f} "
      f"(methodology Δ={rec['index_diff']:+.2f})")
# (5) coverage gate removes low-coverage dates (2016-2018 not published)
assert res.index.loc["2016-01-01":"2018-12-31"].notna().sum() == 0, "low-coverage era must be NaN"
print(f"    coverage gate: 2016-2018 excluded, reliable from {res.first_published_date.date()}")
# (6) weekly components don't stay live beyond max_ffill_days
ffa = forward_fill_audit(res, df)
wk = ffa[ffa["frequency"] != "daily"]
for _, r in wk.iterrows():
    if r["is_live"]:
        assert r["days_since_true_obs"] <= r["max_ffill_days"], \
            f"{r['component']} live beyond max_ffill"
print(f"    weekly freshness OK ({len(wk)} weekly components within ffill cap)")
# (7) forward-fill audit flags weekly series as substantially forward-filled
assert (wk["pct_ffilled_1y"] > 50).all(), "weekly series should be majority forward-filled"
print(f"    ffill audit: weekly %ffilled(1y) = {sorted(wk['pct_ffilled_1y'].round(0).tolist())}")
# (8) loader treats DATA.xlsx as source of truth, parquet as derived cache
from data.loader import EXCEL_PATH, source_signature, parquet_is_fresh
assert EXCEL_PATH.name == "DATA.xlsx" and EXCEL_PATH.exists()
assert isinstance(source_signature(), str) and len(source_signature()) > 0
print(f"    source of truth = {EXCEL_PATH.name}, parquet fresh = {parquet_is_fresh()}")
# methodology audit smoke
aud = methodology_audit(res, df, data_hash=source_signature())
assert aud["version"] == INDEX_METHODOLOGY["version"]
print(f"    methodology {aud['version']} · {aud['components_on_latest']} components, "
      f"{aud['buckets_on_latest']} buckets on latest date")

print("\nALL SMOKE TESTS PASSED ✓")
