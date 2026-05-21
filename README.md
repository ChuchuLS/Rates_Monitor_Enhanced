# Rates & Liquidity Monitor

A clean, institutional dark-theme Streamlit dashboard for rates, funding, and
credit — now anchored by a **Composite Liquidity Index** built from raw market
indicators that tells you, at a glance, whether financial-market liquidity is
loose or tight, whether it is improving or deteriorating, and which part of the
market is driving the move.

---

## Quick start (run on your own computer)

You need Python 3.10+ installed. Then, from a terminal in this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

A browser tab opens with the dashboard. That's it.

## Updating the market data

`DATA.xlsx` is the **source of truth** — it's the only file you edit. The
`latest.parquet` next to it is a **derived cache** that the app rebuilds
**automatically** whenever the Excel changes, so the workflow is simply:

1. Replace `data/DATA.xlsx` with your refreshed Bloomberg pull.
2. Commit & push it (and let Streamlit Cloud redeploy), or just re-run locally.

On the next start the app hashes `DATA.xlsx` (SHA-256), compares it to the hash
recorded in `latest.parquet.meta.json`, and if they differ — or the parquet is
missing — it rebuilds the cache from the Excel on the spot, then loads it. If the
cache can't be built for any reason (e.g. a read-only filesystem), it falls back
to reading `DATA.xlsx` directly so the dashboard always runs. A content hash is
used rather than a file timestamp because mtimes are unreliable after a git
checkout or Cloud redeploy.

You can confirm what happened any time on the **Data Quality** page, which shows
the source file, the cache status (Fresh / Rebuilt automatically / Fallback to
Excel), the latest data date, and the row/column counts.

> Running `python scripts/build_parquet.py` is therefore **optional** — it just
> pre-warms the cache locally and also writes the `metadata.csv` / `ticker_map.csv`
> inspection sidecars. You never need it for the app to see new data.

> **Tip:** don't commit the derived files. The included `.gitignore` already
> excludes `latest.parquet`, its `.meta.json`, and the CSV sidecars so that only
> `DATA.xlsx` travels in version control.

### Deploying to share a link
Push the folder to GitHub and point [Streamlit Community Cloud](https://share.streamlit.io)
at `app.py`. To password-protect it, add a secret named `app_password` in the
app settings — the login gate turns on automatically. With no secret set, the
app is open (handy for local use).

---

## What's on each page

The sidebar switches between seven sections (and shows the live liquidity read-out):

1. **Composite Liquidity Index** — the homepage. Opens with a summary panel
   (level, regime, 1w/1m/3m change, main easing & tightening contributor), then
   the index line with regime bands, the five sub-indices, a contribution
   decomposition, and benchmark validation vs Bloomberg FCI / Chicago Fed NFCI.
2. **Money Market Plumbing** — the five OFR-style funding-spread panels and the
   layered overnight-rates chart.
3. **Dollar Funding / XCCY Basis** — 3M and 12M cross-currency basis grids.
4. **Credit Liquidity** — IG/HY OAS, bank CDS, EMBI, a credit-index explorer,
   and the mortgage-vs-UST spread.
5. **Rates / Real Rates / Curve Regime** — the curve-slope regime classifier and
   real-rate term-structure curves.
6. **Inflation Expectations** — TIPS breakevens, ZC inflation swaps, 5Y5Y forward.
7. **Data Quality** — per-ticker coverage: exists / last date / missing % / stale.

---

## The Composite Liquidity Index — methodology

**Reading it:** higher = looser, **50 = neutral**, ≥60 Loose, <45 Tight, <35 Stress.

It is built from *raw* indicators — Bloomberg FCI and Chicago Fed NFCI are used
**only as benchmarks**, never as inputs.

**1. Indicators, grouped into five buckets**

| Bucket | Weight | Example indicators |
|---|---|---|
| Money-market funding | 30% | SOFR−IORB, EFFR−IORB, SOFR−EFFR, TGCR/BGCR−IORB |
| Dollar funding / XCCY | 20% | EUR/JPY/GBP/AUD/CAD 3M basis |
| Credit liquidity | 20% | IG & HY OAS, EMBI, iTraxx, bank CDS, mortgage spread |
| Central bank / reserves | 20% | Fed reserve balances, Fed repo/SRF usage |
| Market liquidity / vol | 10% | UST liquidity index, swap spread, (MOVE, VIX if present) |

**2. Direction adjustment.** Each indicator is multiplied by ±1 *before*
z-scoring so that **higher always means looser** (e.g. HY OAS gets −1: a wider
spread is tighter; reserves get +1). This single rule makes the whole index
interpretable.

**3. Z-scoring.** Each adjusted indicator becomes a rolling z-score —
`window = 1260` (~5y), `min_periods = 504` (~2y), clipped to `[-3, 3]`.

**4. Sub-index & composite.** Each bucket's sub-index is the mean of its
component z-scores; the composite is the weighted average of the five
sub-indices. Weights renormalise across whichever buckets have data on a given
day, so a missing bucket never silently biases the index toward neutral.

**5. Scaling.** `liquidity_index = 50 + 10 × composite_z`.

**6. Contribution decomposition.** Each bucket's contribution is built so the
terms sum *exactly* to `index − 50`, and so each bucket's change over any
horizon sums exactly to the index change. That's what powers the "why is
liquidity moving" attribution on the homepage — a positive contribution eased
liquidity, a negative one tightened it.

**7. Validation.** The index is compared against Bloomberg FCI / Chicago Fed
NFCI via a correlation table, rolling 1y correlation, a crisis-window check
(Sep-2019 repo, COVID, 2022 QT, Mar-2023 banks — the index correctly dips into
Tight/Stress in each), and a lead-lag cross-correlation.

---

## Project structure

```
rates_monitor/
  app.py                 # layout + page routing only
  config/
    tickers.py           # internal-key -> Bloomberg-ticker map + tenor configs
    theme.py             # OFR dark palette, regime/bucket colours, CSS, layout
  data/
    loader.py            # hash-based auto-rebuild of the cache, Excel fallback
    transforms.py        # rolling z-score (window/min_periods/clip)
    quality.py           # validate_data / staleness report
    DATA.xlsx            # raw Bloomberg pull — SOURCE OF TRUTH (the file you edit)
    latest.parquet       # derived cache, rebuilt automatically when Excel changes
    latest.parquet.meta.json  # records source hash + shape for staleness checks
    metadata.csv         # optional per-column profile (build_parquet.py only)
    ticker_map.csv       # optional key -> ticker map (build_parquet.py only)
  charts/
    common.py            # auto-scaling y-axis helper + shared chart primitives
    rates.py, funding.py, credit.py, liquidity.py
  index/
    components.py        # buckets, indicators, directions, builders
    composite.py         # z-score -> sub-index -> weighted index + contributions
    validation.py        # benchmark correlations / crisis check / lead-lag
  scripts/
    build_parquet.py     # DATA.xlsx -> latest.parquet + metadata + ticker_map
  smoke_test.py          # headless check: imports, index, reconciliation
```

**Robustness:** every series is fetched through `get_series`, which returns an
empty series for any missing ticker. Charts and the index simply skip absent
inputs and surface a warning rather than crashing — e.g. MOVE and VIX are not in
the current dataset, so they're listed in Data Quality and excluded from the
market-liquidity bucket automatically.

**Y-axis:** the real-rate / breakeven curves auto-scale to the visible data
(`charts/common.py:autoscale_range`) and only pull zero into view when the
series actually crosses or hugs zero — no more hard-coded `[-1, 3]`.
