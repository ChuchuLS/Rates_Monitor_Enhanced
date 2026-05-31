"""
index/export.py
===============
Build a multi-sheet Excel workbook of the index, its decomposition, and the
audit tables. Used by the 'Export to Excel' button on the index page. Pure data
side (no Streamlit); the caller wraps this in @st.cache_data so the workbook is
built at most once per DATA.xlsx version.

Sheets
------
1. Index               full B-day series: published index, raw_index, composite-z,
                       regime, available buckets/components, published flag.
2. Buckets             per-bucket sub-z, effective weight, contribution by date.
3. Component_z         date x component z-scores.
4. Component_contrib   date x component contribution to (index-50);
                       row-sum equals raw_index-50 (published rows = index-50).
5. Latest_components   current snapshot: raw, adjusted, z, eff_w, contribution,
                       1w/1m/3m change contribution, live/status.
6. Reconciliation      legacy vs current at latest date + bucket decomposition.
7. Forward_fill_audit  freshness / staleness report per component.
8. Methodology         version + parameters + audit (key/value).
"""
from __future__ import annotations
import io
import pandas as pd

from index.components import BUCKETS
from index.composite import IndexResult, regime_label


def _index_sheet(result: IndexResult) -> pd.DataFrame:
    if result.raw_index.empty:
        return pd.DataFrame()
    dates = result.raw_index.index
    pub = result.index.reindex(dates)
    return pd.DataFrame({
        "date": dates,
        "index": pub.values,                                    # NaN if not published
        "raw_index": result.raw_index.values,                   # NaN if not computable
        "composite_z": result.composite_z.reindex(dates).values,
        "regime": [regime_label(v) if pd.notna(v) else "" for v in pub.values],
        "available_buckets": result.available_bucket_count
                                  .reindex(dates).fillna(0).astype(int).values,
        "available_components": result.available_component_count
                                  .reindex(dates).fillna(0).astype(int).values,
        "published": result.published_mask.reindex(dates).fillna(False)
                            .astype(bool).values,
    })


def _bucket_sheet(result: IndexResult) -> pd.DataFrame:
    if result.bucket_terms.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"date": result.bucket_terms.index})
    for b in BUCKETS:
        if b in result.sub_indices.columns:
            out[f"{b}_sub_z"] = result.sub_indices[b].values
        if b in result.effective_weights.columns:
            out[f"{b}_eff_weight"] = result.effective_weights[b].values
        if b in result.bucket_terms.columns:
            out[f"{b}_contribution"] = result.bucket_terms[b].values
    return out


def _flat_dated(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.insert(0, "date", out.index)
    out = out.reset_index(drop=True)
    return out


def _methodology_sheet(audit_methodology: dict) -> pd.DataFrame:
    rows = []
    for k, v in audit_methodology.items():
        if isinstance(v, dict):
            v = ", ".join(f"{kk}={vv}" for kk, vv in v.items())
        elif isinstance(v, pd.Timestamp):
            v = v.date().isoformat()
        rows.append({"parameter": k, "value": v if v is not None else ""})
    return pd.DataFrame(rows)


def _reconciliation_sheets(rec: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (summary, identities, bucket_table) for the reconciliation sheet."""
    if not rec:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    summary = pd.DataFrame([
        {"metric": "Date", "legacy": str(rec["date"].date()),
         "current": str(rec["date"].date())},
        {"metric": "Index level", "legacy": rec["legacy_index"],
         "current": rec["current_index"]},
        {"metric": "Index diff (current - legacy)", "legacy": "",
         "current": rec["index_diff"]},
        {"metric": "Composite z", "legacy": rec["legacy_z"],
         "current": rec["current_z"]},
        {"metric": "Composite z diff", "legacy": "",
         "current": rec["z_diff"]},
    ])
    checks = rec["checks"]
    identities = pd.DataFrame([
        {"identity": "sum_current_contrib  ==  current_index - 50",
         "lhs": checks["sum_current_contrib"], "rhs": checks["current_index_minus_50"]},
        {"identity": "sum_legacy_contrib   ==  legacy_index  - 50",
         "lhs": checks["sum_legacy_contrib"], "rhs": checks["legacy_index_minus_50"]},
        {"identity": "sum_contrib_diff     ==  current  - legacy",
         "lhs": checks["sum_contrib_diff"], "rhs": checks["current_minus_legacy"]},
    ])
    bucket_table = rec["table"].copy()
    return summary, identities, bucket_table


def build_index_workbook(result: IndexResult, audit: dict,
                         df: pd.DataFrame) -> bytes:
    """Assemble the multi-sheet workbook and return its bytes."""
    audit = audit or {}
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        # 1 - Index time series
        _index_sheet(result).to_excel(xw, sheet_name="Index", index=False)

        # 2 - Bucket decomposition
        _bucket_sheet(result).to_excel(xw, sheet_name="Buckets", index=False)

        # 3 - Component z
        _flat_dated(result.z_scores).to_excel(xw, sheet_name="Component_z", index=False)

        # 4 - Component contributions
        _flat_dated(result.component_terms).to_excel(
            xw, sheet_name="Component_contrib", index=False)

        # 5 - Latest-snapshot component table
        comps = audit.get("components")
        if comps is not None and not comps.empty:
            comps.to_excel(xw, sheet_name="Latest_components", index=False)

        # 6 - Reconciliation (summary + identities + per-bucket)
        rec = audit.get("reconciliation")
        if rec:
            summary, identities, bucket_table = _reconciliation_sheets(rec)
            summary.to_excel(xw, sheet_name="Reconciliation",
                             index=False, startrow=0)
            startrow = len(summary) + 2
            identities.to_excel(xw, sheet_name="Reconciliation",
                                index=False, startrow=startrow)
            startrow += len(identities) + 2
            bucket_table.to_excel(xw, sheet_name="Reconciliation",
                                  index=False, startrow=startrow)

        # 7 - Forward-fill audit
        ffa = audit.get("ffill_audit")
        if ffa is not None and not ffa.empty:
            ffa.to_excel(xw, sheet_name="Forward_fill_audit", index=False)

        # 8 - Methodology + audit trail (key/value)
        meth = audit.get("methodology")
        if meth:
            _methodology_sheet(meth).to_excel(xw, sheet_name="Methodology", index=False)

    return buf.getvalue()


def export_filename(audit: dict | None) -> str:
    """Stable, informative filename for the download."""
    version = (audit or {}).get("methodology", {}).get("version", "v0.3")
    latest = (audit or {}).get("methodology", {}).get("latest_data_date")
    stamp = latest.date().isoformat() if isinstance(latest, pd.Timestamp) else "latest"
    return f"liquidity_index_{version}_{stamp}.xlsx"
