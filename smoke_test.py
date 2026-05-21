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

print("\nALL SMOKE TESTS PASSED ✓")
