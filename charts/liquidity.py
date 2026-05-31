"""
charts/liquidity.py
===================
All rendering for the Composite Liquidity Index (requirements #9-#12):

  * render_summary_panel()  -> the homepage KPI strip (level, regime,
    1w/1m/3m change, main easing + tightening contributor).
  * render_index_page()     -> the full "Composite Liquidity Index" section:
    index line with regime bands, sub-index lines, contribution decomposition,
    benchmark comparison (overlay + correlation table + rolling correlation +
    crisis check + lead-lag), and the component-availability table.

Interpretation reminder (so the charts read correctly):
    higher index  = looser liquidity   (>=60 Loose)
    50            = neutral
    lower index   = tighter liquidity   (<35 Stress)
A positive bucket contribution pushes the index UP (easing); a negative
contribution pushes it DOWN (tightening).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.theme import (
    BG, GRID, LINE_WHITE, TEXT_DIM, TEXT_VERY_DIM, DARK_LAYOUT,
    POS_GREEN, NEG_RED, REGIME_COLORS, BUCKET_COLORS,
    ACCENT_CYAN, ACCENT_AMBER, ACCENT_PURPLE,
)
from charts.common import autoscale_range, section_header
from index.components import BUCKETS
from index.composite import (
    IndexResult, HORIZONS, INDEX_CENTER, regime_label,
    MIN_AVAILABLE_BUCKETS, MIN_AVAILABLE_COMPONENTS, MIN_COMPONENTS_PER_BUCKET,
)
from index.validation import (
    BENCHMARKS, correlation_table, rolling_correlation,
    standardized_overlay, crisis_behaviour, lead_lag, benchmark_looseness,
)

# Regime band definitions for the index chart: (low, high, label).
# Mirrors REGIME_THRESHOLDS in index/composite.py.
_REGIME_BANDS = [
    (60.0, 100.0, "Loose"),
    (45.0, 60.0, "Neutral"),
    (35.0, 45.0, "Tight"),
    (0.0, 35.0, "Stress"),
]

# Stable bucket plotting order (money-market first, matching the weights).
_BUCKET_ORDER = sorted(BUCKETS, key=lambda b: BUCKETS[b]["order"])


# ===========================================================================
# Small formatting helpers
# ===========================================================================
def _fmt_change(v: float) -> tuple[str, str]:
    """(text, colour) for a signed index-point change. + = looser = green."""
    if v is None or pd.isna(v):
        return ("—", TEXT_DIM)
    colour = POS_GREEN if v >= 0 else NEG_RED
    return (f"{v:+.1f}", colour)


def _regime_colour(value: float) -> str:
    return REGIME_COLORS.get(regime_label(value), TEXT_DIM)


# ===========================================================================
# Homepage summary panel (requirement #12)
# ===========================================================================
def render_summary_panel(result: IndexResult) -> None:
    """Top-of-page KPI strip: level, regime, period changes, drivers."""
    level = result.latest
    regime = result.latest_regime
    changes = result.changes()
    easing_lbl, tight_lbl = result.drivers("1m")

    if pd.isna(level):
        st.warning(
            "The Composite Liquidity Index could not be computed — no component "
            "data is available. Check the Data Quality section."
        )
        return

    reg_colour = _regime_colour(level)
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1], gap="small")

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">Composite Liquidity Index</div>
              <div class="kpi-value" style="color:{reg_colour};">{level:.1f}</div>
              <div class="kpi-sub">
                <span style="color:{reg_colour};font-weight:700;
                       text-transform:uppercase;letter-spacing:0.08em;">
                  {regime}</span>
                &nbsp;·&nbsp; 50 = neutral
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for col, key, lbl in (
        (c2, "1w", "1-week change"),
        (c3, "1m", "1-month change"),
        (c4, "3m", "3-month change"),
    ):
        txt, colour = _fmt_change(changes.get(key))
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{lbl}</div>
                  <div class="kpi-value" style="color:{colour};font-size:26px;">
                    {txt}</div>
                  <div class="kpi-sub">index points</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    d1, d2 = st.columns(2, gap="small")
    with d1:
        st.markdown(
            f"""
            <div class="kpi-card" style="margin-top:0.6rem;">
              <div class="kpi-label">Main easing contributor (1m)</div>
              <div class="kpi-value" style="color:{POS_GREEN};font-size:18px;
                   line-height:1.3;">{easing_lbl}</div>
              <div class="kpi-sub">pushing liquidity looser</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with d2:
        st.markdown(
            f"""
            <div class="kpi-card" style="margin-top:0.6rem;">
              <div class="kpi-label">Main tightening contributor (1m)</div>
              <div class="kpi-value" style="color:{NEG_RED};font-size:18px;
                   line-height:1.3;">{tight_lbl}</div>
              <div class="kpi-sub">pushing liquidity tighter</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# Index line chart with regime bands
# ===========================================================================
def index_line_chart(index: pd.Series, height: int = 460) -> go.Figure:
    """Liquidity index over time with shaded Loose/Neutral/Tight/Stress bands."""
    s = index.dropna()
    fig = go.Figure()

    # Determine the visible y-range, always keeping the neutral 50 line in view.
    if len(s):
        lo = min(s.min(), 45.0)
        hi = max(s.max(), 60.0)
        pad = (hi - lo) * 0.08 or 1.0
        y_lo, y_hi = lo - pad, hi + pad
    else:
        y_lo, y_hi = 30.0, 70.0

    # Regime bands (drawn first so the line sits on top).
    for low, high, label in _REGIME_BANDS:
        band_lo = max(low, y_lo)
        band_hi = min(high, y_hi)
        if band_hi <= band_lo:
            continue
        fig.add_hrect(
            y0=band_lo, y1=band_hi, line_width=0,
            fillcolor=REGIME_COLORS[label], opacity=0.07, layer="below",
        )

    fig.add_hline(y=INDEX_CENTER, line=dict(color=TEXT_VERY_DIM, width=0.8, dash="dot"),
                  annotation_text="Neutral 50", annotation_position="right",
                  annotation_font=dict(size=9, color=TEXT_DIM))

    if len(s):
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            line=dict(color=LINE_WHITE, width=1.4),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[s.index[-1]], y=[s.iloc[-1]], mode="markers",
            marker=dict(color=_regime_colour(s.iloc[-1]), size=8,
                        line=dict(color=BG, width=1)),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        **DARK_LAYOUT, height=height,
        margin=dict(l=55, r=60, t=20, b=30),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#bbb"), linecolor="#222")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, range=[y_lo, y_hi],
                     tickfont=dict(size=10, color="#bbb"), linecolor="#222",
                     title=dict(text="Index (50 = neutral)", font=dict(size=10, color="#888")))
    return fig


def sub_index_chart(sub_indices: pd.DataFrame, height: int = 360) -> go.Figure:
    """One z-score line per bucket sub-index (higher = looser)."""
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=TEXT_VERY_DIM, width=0.6, dash="dot"))
    for bucket in _BUCKET_ORDER:
        if bucket not in sub_indices.columns:
            continue
        s = sub_indices[bucket].dropna()
        if not len(s):
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            line=dict(color=BUCKET_COLORS.get(bucket, LINE_WHITE), width=1.2),
            name=BUCKETS[bucket]["label"],
            hovertemplate=(f"{BUCKETS[bucket]['label']}<br>"
                           "%{x|%Y-%m-%d}: %{y:+.2f}σ<extra></extra>"),
        ))
    fig.update_layout(
        **{**DARK_LAYOUT, "showlegend": True}, height=height,
        margin=dict(l=55, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#ccc")),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#bbb"), linecolor="#222")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(size=10, color="#bbb"), linecolor="#222",
                     ticksuffix="σ",
                     title=dict(text="Sub-index (z-score)", font=dict(size=10, color="#888")))
    return fig


def contribution_chart(contrib: pd.Series, title: str, height: int = 300) -> go.Figure:
    """Horizontal bars of each bucket's contribution (green=easing, red=tightening)."""
    fig = go.Figure()
    contrib = contrib.reindex([b for b in _BUCKET_ORDER if b in contrib.index]).dropna()
    if len(contrib):
        labels = [BUCKETS[b]["label"] for b in contrib.index]
        colours = [POS_GREEN if v >= 0 else NEG_RED for v in contrib.values]
        fig.add_trace(go.Bar(
            x=contrib.values, y=labels, orientation="h",
            marker=dict(color=colours, line=dict(width=0)),
            text=[f"{v:+.2f}" for v in contrib.values],
            textposition="outside", textfont=dict(size=11, color="#ddd"),
            hovertemplate="%{y}: %{x:+.2f} pts<extra></extra>",
        ))
    fig.add_vline(x=0, line=dict(color=TEXT_VERY_DIM, width=0.8))
    fig.update_layout(
        **DARK_LAYOUT, height=height,
        margin=dict(l=10, r=30, t=40, b=30),
        title=dict(
            text=f"<span style='color:#fff;font-weight:700;letter-spacing:0.05em;"
                 f"font-size:12px;'>{title.upper()}</span>",
            font=dict(size=12), x=0, xanchor="left", y=0.97),
    )
    if len(contrib):
        span = max(abs(contrib.min()), abs(contrib.max())) or 1.0
        fig.update_xaxes(range=[-span * 1.45, span * 1.45])
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(size=9, color="#bbb"), linecolor="#222",
                     title=dict(text="index points", font=dict(size=9, color="#888")))
    fig.update_yaxes(showgrid=False, tickfont=dict(size=10, color="#ddd"),
                     linecolor="#222", automargin=True)
    return fig


def benchmark_overlay_chart(overlay: pd.DataFrame, height: int = 360) -> go.Figure:
    """Index vs benchmarks, each standardised to a full-sample z (same scale)."""
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=TEXT_VERY_DIM, width=0.6, dash="dot"))
    palette = {
        "Liquidity Index": LINE_WHITE,
        "Bloomberg US FCI": ACCENT_CYAN,
        "Chicago Fed NFCI": ACCENT_AMBER,
        "OFR Financial Stress Index": ACCENT_PURPLE,
    }
    for col in overlay.columns:
        s = overlay[col].dropna()
        if not len(s):
            continue
        is_idx = col == "Liquidity Index"
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            line=dict(color=palette.get(col, TEXT_DIM),
                      width=1.6 if is_idx else 1.1,
                      dash="solid" if is_idx else "dot"),
            name=col,
            hovertemplate=f"{col}<br>%{{x|%Y-%m-%d}}: %{{y:+.2f}}σ<extra></extra>",
        ))
    fig.update_layout(
        **{**DARK_LAYOUT, "showlegend": True}, height=height,
        margin=dict(l=55, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#ccc")),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#bbb"), linecolor="#222")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(size=10, color="#bbb"), linecolor="#222",
                     ticksuffix="σ",
                     title=dict(text="Standardised (higher = looser)",
                                font=dict(size=10, color="#888")))
    return fig


def rolling_corr_chart(roll: pd.DataFrame, height: int = 300) -> go.Figure:
    """Rolling 1y correlation of the index vs each benchmark."""
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=TEXT_VERY_DIM, width=0.6, dash="dot"))
    palette = {"Bloomberg US FCI": ACCENT_CYAN, "Chicago Fed NFCI": ACCENT_AMBER,
               "OFR Financial Stress Index": ACCENT_PURPLE}
    for col in roll.columns:
        s = roll[col].dropna()
        if not len(s):
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            line=dict(color=palette.get(col, LINE_WHITE), width=1.2),
            name=col,
            hovertemplate=f"{col}<br>%{{x|%Y-%m-%d}}: %{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(
        **{**DARK_LAYOUT, "showlegend": True}, height=height,
        margin=dict(l=55, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#ccc")),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#bbb"), linecolor="#222")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, range=[-1.05, 1.05],
                     tickfont=dict(size=10, color="#bbb"), linecolor="#222",
                     title=dict(text="Rolling 1y correlation", font=dict(size=10, color="#888")))
    return fig


def lead_lag_chart(ll: pd.Series, benchmark: str, height: int = 300) -> go.Figure:
    """Cross-correlation of daily changes by lag (peak = which series leads)."""
    fig = go.Figure()
    if len(ll):
        peak_lag = ll.idxmax()
        colours = [ACCENT_CYAN if k == peak_lag else "rgba(79,168,184,0.35)"
                   for k in ll.index]
        fig.add_trace(go.Bar(
            x=list(ll.index), y=ll.values,
            marker=dict(color=colours, line=dict(width=0)),
            hovertemplate="lag %{x}d: %{y:.2f}<extra></extra>",
        ))
        fig.add_vline(x=0, line=dict(color=TEXT_VERY_DIM, width=0.7, dash="dot"))
    fig.update_layout(
        **DARK_LAYOUT, height=height,
        margin=dict(l=50, r=20, t=20, b=40),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#bbb"), linecolor="#222",
                     title=dict(text=f"Index lead (−) / lag (+) vs {benchmark}, business days",
                                font=dict(size=10, color="#888")))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(size=10, color="#bbb"), linecolor="#222",
                     title=dict(text="corr of changes", font=dict(size=10, color="#888")))
    return fig


def coverage_chart(components: pd.Series, buckets: pd.Series,
                   first_published, height: int = 280) -> go.Figure:
    """Available components (left axis) and qualifying buckets (right axis) over
    time, with the reliable-from line marked."""
    fig = go.Figure()
    c = components.dropna()
    b = buckets.dropna()
    fig.add_trace(go.Scatter(
        x=c.index, y=c.values, mode="lines", name="Components",
        line=dict(color=ACCENT_CYAN, width=1.3),
        hovertemplate="%{x|%Y-%m-%d}: %{y} components<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=b.index, y=b.values, mode="lines", name="Qualifying buckets",
        line=dict(color=ACCENT_AMBER, width=1.3), yaxis="y2",
        hovertemplate="%{x|%Y-%m-%d}: %{y} buckets<extra></extra>"))
    if first_published is not None:
        fp_str = pd.Timestamp(first_published).strftime("%Y-%m-%d")
        fig.add_shape(type="line", x0=fp_str, x1=fp_str, y0=0, y1=1, yref="paper",
                      line=dict(color=POS_GREEN, width=1, dash="dash"))
        fig.add_annotation(x=fp_str, y=1.0, yref="paper", yanchor="bottom",
                           text="reliable from", showarrow=False,
                           font=dict(size=9, color=POS_GREEN))
    fig.update_layout(
        **{**DARK_LAYOUT, "showlegend": True}, height=height,
        margin=dict(l=45, r=45, t=20, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#ccc")),
        yaxis2=dict(overlaying="y", side="right", showgrid=False, range=[0, 5.5],
                    tickfont=dict(size=10, color=ACCENT_AMBER),
                    title=dict(text="buckets", font=dict(size=9, color=ACCENT_AMBER))),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#bbb"), linecolor="#222")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(size=10, color=ACCENT_CYAN), linecolor="#222",
                     title=dict(text="components", font=dict(size=9, color=ACCENT_CYAN)))
    return fig


def effective_weights_chart(eff: pd.DataFrame, height: int = 280) -> go.Figure:
    """Stacked area of the renormalised bucket weights actually used each day."""
    fig = go.Figure()
    eff = eff.dropna(how="all")
    for bucket in _BUCKET_ORDER:
        if bucket not in eff.columns:
            continue
        s = (eff[bucket].fillna(0) * 100)
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=BUCKETS[bucket]["label"],
            line=dict(width=0.5, color=BUCKET_COLORS.get(bucket, LINE_WHITE)),
            stackgroup="w", fillcolor=BUCKET_COLORS.get(bucket, LINE_WHITE),
            hovertemplate=(f"{BUCKETS[bucket]['label']}<br>"
                           "%{x|%Y-%m-%d}: %{y:.0f}%<extra></extra>")))
    fig.update_layout(
        **{**DARK_LAYOUT, "showlegend": True}, height=height,
        margin=dict(l=45, r=20, t=20, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#ccc")),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=10, color="#bbb"), linecolor="#222")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, range=[0, 100],
                     ticksuffix="%", tickfont=dict(size=10, color="#bbb"), linecolor="#222",
                     title=dict(text="effective weight", font=dict(size=9, color="#888")))
    return fig


def raw_index_chart(index: pd.Series, smooth: int = 1, height: int = 360) -> go.Figure:
    """The published 50-centred index itself (optionally smoothed for display)."""
    s = index.dropna()
    if smooth and smooth > 1:
        s = s.rolling(smooth, min_periods=1).mean()
    fig = index_line_chart(s, height=height)
    return fig


def _smooth(s: pd.Series, window: int) -> pd.Series:
    """Visual-only moving average. Never feeds back into the index calculation."""
    if window and window > 1:
        return s.rolling(window, min_periods=1).mean()
    return s


# ===========================================================================
# Full page renderer (requirement #9-#11)
# ===========================================================================
def render_index_page(df: pd.DataFrame, dff: pd.DataFrame, result: IndexResult,
                      audit: dict | None = None,
                      export_bytes: bytes | None = None,
                      export_name: str | None = None) -> None:
    """Render the entire Composite Liquidity Index section."""
    audit = audit or {}
    section_header(
        "Composite Liquidity Index",
        "Raw-indicator liquidity gauge · higher = looser · 50 = neutral · "
        "z-scored & weighted across five buckets",
    )

    # Excel export — one click, full multi-sheet workbook.
    if export_bytes:
        _, btn_col = st.columns([6, 2])
        with btn_col:
            st.download_button(
                label="⬇  Export to Excel",
                data=export_bytes,
                file_name=export_name or "liquidity_index.xlsx",
                mime="application/vnd.openxmlformats-officedocument."
                     "spreadsheetml.sheet",
                help="Multi-sheet workbook: index series, bucket & component "
                     "contributions, latest snapshot, legacy reconciliation, "
                     "forward-fill audit, and methodology / audit trail.",
                use_container_width=True,
                key="liq_export_xlsx",
            )

    if result.index.dropna().empty:
        st.warning(
            "No index could be built because none of the component indicators "
            "are present in the data. See the Data Quality section for details."
        )
        return

    # --- Index level + regime line ----------------------------------------
    start, end = (dff.index.min(), dff.index.max()) if len(dff) else (df.index.min(), df.index.max())

    # Warn if the chosen lookback reaches into the unpublished low-coverage era.
    fp = result.first_published_date
    if fp is not None and start < fp:
        st.warning(
            f"The selected lookback starts {start.date()}, but the index is only "
            f"published from **{fp.date()}** — earlier dates fail the minimum "
            f"coverage rules (too few buckets / single-component buckets) and are "
            f"left blank. See *Coverage & reliability* below."
        )

    idx_window = result.index.loc[(result.index.index >= start) & (result.index.index <= end)]
    st.plotly_chart(index_line_chart(idx_window), use_container_width=True,
                    key="liq_index_line", config={"displayModeBar": False})
    fp_txt = f" · reliable from {fp.date()}" if fp is not None else ""
    st.caption(
        "Bands: green = Loose (≥60) · grey = Neutral (45–60) · "
        "amber = Tight (35–45) · red = Stress (<35)." + fp_txt
    )

    # --- Sub-indices + contribution decomposition -------------------------
    left, right = st.columns([1.4, 1], gap="medium")
    with left:
        st.markdown(
            "<div style='color:#888;font-size:11px;letter-spacing:0.08em;"
            "text-transform:uppercase;margin:0.4rem 0;'>Sub-index by bucket "
            "(z-score, higher = looser)</div>", unsafe_allow_html=True)
        sub_window = result.sub_indices.loc[
            (result.sub_indices.index >= start) & (result.sub_indices.index <= end)]
        st.plotly_chart(sub_index_chart(sub_window), use_container_width=True,
                        key="liq_subindex", config={"displayModeBar": False})

    with right:
        horizon = st.selectbox(
            "CONTRIBUTION HORIZON", options=list(HORIZONS.keys()), index=1,
            key="liq_contrib_horizon",
            help="Which buckets drove the index change over this window.",
        )
        st.plotly_chart(
            contribution_chart(result.change_contributions(horizon),
                               f"{horizon} change decomposition"),
            use_container_width=True, key="liq_contrib_change",
            config={"displayModeBar": False})
        st.plotly_chart(
            contribution_chart(result.level_contributions(),
                               "current level vs neutral"),
            use_container_width=True, key="liq_contrib_level",
            config={"displayModeBar": False})

    # Explain the decomposition in words (requirement #10).
    _render_driver_note(result, horizon)

    # --- Coverage & reliability diagnostic (requirements #1, #2, #7) ------
    _render_coverage_block(result)

    # --- Benchmark comparison (requirement #11) ---------------------------
    _render_benchmark_block(df, result)

    # --- Component availability + methodology -----------------------------
    _render_components_table(result)
    _render_component_contributions(audit.get("components"))
    _render_reconciliation(audit.get("reconciliation"))
    _render_ffill_audit(audit.get("ffill_audit"))
    _render_methodology_audit(result, audit.get("methodology"))


def _render_coverage_block(result: IndexResult) -> None:
    """Show how many components/buckets feed the index over time + effective
    weights, making partial-coverage periods obvious (requirements #1, #7)."""
    st.markdown(
        "<div style='border-top:1px solid #1a1a1a;margin-top:1rem;padding-top:0.6rem;'>"
        "<div style='font-size:14px;font-weight:700;letter-spacing:0.06em;color:#fff;"
        "text-transform:uppercase;'>Coverage &amp; reliability</div>"
        "<div style='font-size:10px;color:#888;letter-spacing:0.08em;"
        "text-transform:uppercase;margin-top:2px;'>How much of the component "
        "universe actually feeds the index each day</div></div>",
        unsafe_allow_html=True,
    )

    fp = result.first_published_date
    fv = result.first_valid_date
    if fp is not None:
        st.markdown(
            f"""
            <div style="background:#0f0f0f;border:1px solid #1a1a1a;border-radius:6px;
                        padding:0.6rem 0.9rem;margin:0.3rem 0 0.6rem;font-size:12px;
                        color:#ccc;line-height:1.6;">
              The index is computable from <b>{fv.date() if fv is not None else '—'}</b>
              (once ~2y of history exists for the first indicators) but is only
              <b style="color:{POS_GREEN};">published from {fp.date()}</b>, when at
              least {MIN_AVAILABLE_BUCKETS} buckets — each with ≥
              {MIN_COMPONENTS_PER_BUCKET} live components — and ≥
              {MIN_AVAILABLE_COMPONENTS} components are available. Earlier dates are a
              <b>low-coverage / warm-up period</b> and are not shown as a valid signal.
            </div>
            """,
            unsafe_allow_html=True)

    c_left, c_right = st.columns(2, gap="medium")
    with c_left:
        st.markdown(
            "<div style='color:#888;font-size:11px;letter-spacing:0.08em;"
            "text-transform:uppercase;margin:0.3rem 0 0.2rem;'>Components &amp; "
            "buckets available over time</div>", unsafe_allow_html=True)
        st.plotly_chart(
            coverage_chart(result.available_component_count,
                           result.available_bucket_count, fp),
            use_container_width=True, key="liq_coverage",
            config={"displayModeBar": False})
    with c_right:
        st.markdown(
            "<div style='color:#888;font-size:11px;letter-spacing:0.08em;"
            "text-transform:uppercase;margin:0.3rem 0 0.2rem;'>Effective bucket "
            "weights over time (renormalised)</div>", unsafe_allow_html=True)
        st.plotly_chart(effective_weights_chart(result.effective_weights),
                        use_container_width=True, key="liq_effweights",
                        config={"displayModeBar": False})
    st.caption(
        "When a bucket is missing, its weight is spread across the remaining "
        "buckets — so before XCCY data begins (mid-2022) the other buckets carry "
        "more. The index is only published once coverage is broad enough that this "
        "renormalisation no longer rests on one or two fragile series.")


def _render_driver_note(result: IndexResult, horizon: str) -> None:
    """Plain-language summary of the bucket contributions to the move."""
    contrib = result.change_contributions(horizon)
    if contrib.empty or contrib.isna().all():
        return
    parts = []
    for bucket in _BUCKET_ORDER:
        if bucket not in contrib.index or pd.isna(contrib[bucket]):
            continue
        v = contrib[bucket]
        colour = POS_GREEN if v >= 0 else NEG_RED
        parts.append(
            f"<span style='color:#aaa;'>{BUCKETS[bucket]['label']}</span> "
            f"<span style='color:{colour};font-weight:700;'>{v:+.2f}</span>")
    total = contrib.sum()
    total_col = POS_GREEN if total >= 0 else NEG_RED
    body = " &nbsp;·&nbsp; ".join(parts)
    st.markdown(
        f"""
        <div style="background:#0f0f0f;border:1px solid #1a1a1a;border-radius:6px;
                    padding:0.7rem 0.9rem;margin-top:0.4rem;">
          <div style="font-size:10px;color:#888;letter-spacing:0.1em;
                      text-transform:uppercase;margin-bottom:6px;">
            {horizon} index change attribution &nbsp;·&nbsp; total
            <span style="color:{total_col};font-weight:700;">{total:+.2f}</span> pts</div>
          <div style="font-size:12px;line-height:1.9;">{body}</div>
          <div style="font-size:10px;color:#666;margin-top:6px;">
            Positive = bucket eased liquidity (index up) · negative = bucket
            tightened liquidity (index down). Contributions sum exactly to the
            index move.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_benchmark_block(df: pd.DataFrame, result: IndexResult) -> None:
    """Validation vs Bloomberg FCI / Chicago Fed NFCI / OFR FSI."""
    available = benchmark_looseness(df)
    st.markdown(
        "<div style='border-top:1px solid #1a1a1a;margin-top:1rem;padding-top:0.6rem;'>"
        "<div style='font-size:14px;font-weight:700;letter-spacing:0.06em;color:#fff;"
        "text-transform:uppercase;'>Benchmark validation</div>"
        "<div style='font-size:10px;color:#888;letter-spacing:0.08em;"
        "text-transform:uppercase;margin-top:2px;'>Comparison only — these gauges "
        "are never index inputs</div></div>",
        unsafe_allow_html=True,
    )

    if not available:
        st.info(
            "No benchmark indices (Bloomberg US FCI, Chicago Fed NFCI, OFR FSI) "
            "are present in the dataset, so validation charts are skipped."
        )
        return

    # Display controls: Raw vs Smoothed (visual only — never changes the index).
    ctrl, _sp = st.columns([1.1, 2], gap="small")
    with ctrl:
        mode = st.radio("DISPLAY", ["Smoothed (5d)", "Raw"], index=0, horizontal=True,
                        key="liq_overlay_mode",
                        help="Smoothing is for readability only. The underlying "
                             "index, correlations and crisis stats always use the raw "
                             "(unsmoothed) values.")
    smooth_win = 5 if mode.startswith("Smoothed") else 1

    # (a) The published 50-centred index itself, with regime bands.
    st.markdown(
        "<div style='color:#888;font-size:11px;letter-spacing:0.08em;"
        "text-transform:uppercase;margin:0.4rem 0 0.2rem;'>Liquidity index "
        "(50-centred, higher = looser)</div>", unsafe_allow_html=True)
    st.plotly_chart(raw_index_chart(result.index, smooth=smooth_win, height=320),
                    use_container_width=True, key="liq_raw50",
                    config={"displayModeBar": False})

    # (b) Standardised overlay vs benchmarks (index smoothed for display only).
    st.markdown(
        "<div style='color:#888;font-size:11px;letter-spacing:0.08em;"
        "text-transform:uppercase;margin:0.5rem 0 0.2rem;'>Standardised overlay "
        "(all oriented so higher = looser)</div>", unsafe_allow_html=True)
    overlay = standardized_overlay(result.index, df)
    if smooth_win > 1 and "Liquidity Index" in overlay.columns:
        overlay = overlay.copy()
        overlay["Liquidity Index"] = _smooth(overlay["Liquidity Index"], smooth_win)
    st.plotly_chart(benchmark_overlay_chart(overlay),
                    use_container_width=True, key="liq_overlay",
                    config={"displayModeBar": False})
    if result.first_published_date is not None:
        st.caption(
            f"Overlay covers the published period (from "
            f"{result.first_published_date.date()}); the low-coverage warm-up era "
            f"is excluded so it no longer distorts the standardisation. Smoothing "
            f"is visual only.")

    # Correlation table + crisis behaviour side by side.
    tbl_col, crisis_col = st.columns(2, gap="medium")
    with tbl_col:
        st.markdown(
            "<div style='color:#888;font-size:11px;letter-spacing:0.08em;"
            "text-transform:uppercase;margin:0.5rem 0 0.2rem;'>Correlation with "
            "benchmarks</div>", unsafe_allow_html=True)
        corr = correlation_table(result.index, df)
        if not corr.empty:
            disp = corr.rename(columns={
                "benchmark": "Benchmark", "corr_levels": "Corr (levels)",
                "corr_changes": "Corr (changes)", "n_obs": "Obs"})
            st.dataframe(
                disp.style.format({"Corr (levels)": "{:.2f}",
                                   "Corr (changes)": "{:.2f}", "Obs": "{:,}"}),
                hide_index=True, use_container_width=True)
            st.caption(
                "Positive correlation = our index agrees with the benchmark's "
                "looseness reading.")
    with crisis_col:
        st.markdown(
            "<div style='color:#888;font-size:11px;letter-spacing:0.08em;"
            "text-transform:uppercase;margin:0.5rem 0 0.2rem;'>Crisis-period "
            "behaviour</div>", unsafe_allow_html=True)
        crisis = crisis_behaviour(result.index)
        if not crisis.empty:
            crisis = crisis.copy()
            crisis["trough_date"] = pd.to_datetime(crisis["trough_date"]).dt.date.astype(str)
            disp = crisis.rename(columns={
                "crisis": "Episode", "min": "Index trough",
                "mean": "Mean level", "trough_date": "Trough date"})
            st.dataframe(
                disp.style.format({"Index trough": "{:.1f}", "Mean level": "{:.1f}"}),
                hide_index=True, use_container_width=True)
            st.caption("The index should dip into Tight/Stress during these episodes.")

    # Rolling correlation + lead-lag.
    roll = rolling_correlation(result.index, df)
    if not roll.empty:
        rc_col, ll_col = st.columns(2, gap="medium")
        with rc_col:
            st.markdown(
                "<div style='color:#888;font-size:11px;letter-spacing:0.08em;"
                "text-transform:uppercase;margin:0.5rem 0 0.2rem;'>Rolling 1y "
                "correlation</div>", unsafe_allow_html=True)
            st.plotly_chart(rolling_corr_chart(roll), use_container_width=True,
                            key="liq_rollcorr", config={"displayModeBar": False})
        with ll_col:
            primary = next((b for b in ("Bloomberg US FCI", "Chicago Fed NFCI")
                            if b in available), list(available)[0])
            st.markdown(
                "<div style='color:#888;font-size:11px;letter-spacing:0.08em;"
                f"text-transform:uppercase;margin:0.5rem 0 0.2rem;'>Lead-lag vs "
                f"{primary}</div>", unsafe_allow_html=True)
            ll = lead_lag(result.index, df, primary)
            if len(ll):
                st.plotly_chart(lead_lag_chart(ll, primary), use_container_width=True,
                                key="liq_leadlag", config={"displayModeBar": False})
                peak = ll.idxmax()
                lead_txt = ("move roughly together" if peak == 0 else
                            f"our index tends to LEAD by {-peak}d" if peak < 0 else
                            f"our index tends to LAG by {peak}d")
                st.caption(f"Peak cross-correlation at lag {peak:+d}d — {lead_txt}.")


def _render_components_table(result: IndexResult) -> None:
    """Show exactly which indicators fed the index, by bucket, with direction."""
    st.markdown(
        "<div style='border-top:1px solid #1a1a1a;margin-top:1rem;padding-top:0.6rem;'>"
        "<div style='font-size:14px;font-weight:700;letter-spacing:0.06em;color:#fff;"
        "text-transform:uppercase;'>Index components &amp; methodology</div></div>",
        unsafe_allow_html=True,
    )
    meta = result.meta.copy()
    meta["direction"] = meta["direction"].map({1: "↑ looser", -1: "↓ looser"})
    meta["available"] = meta["available"].map({True: "✓", False: "—"})
    meta["weight"] = meta["bucket"].map(lambda b: f"{result.weights.get(b, 0) * 100:.0f}%"
                                        if b in result.weights.index else "—")
    disp = meta[["bucket_label", "label", "direction", "frequency", "available", "n_obs", "weight"]]
    disp = disp.rename(columns={
        "bucket_label": "Bucket", "label": "Indicator", "direction": "Direction",
        "frequency": "Freq", "available": "In index", "n_obs": "Obs", "weight": "Bucket wt"})
    st.dataframe(
        disp.style.format({"Obs": "{:,}"}), hide_index=True, use_container_width=True,
        height=min(560, 42 + 36 * len(disp)))
    st.caption(
        "Each indicator is direction-adjusted so higher = looser, then converted "
        "to a rolling z-score (5y window, 2y min, clipped ±3, with a low-variation "
        "guard). A bucket sub-index is the mean of its z-scores, but only counts on "
        f"days it has ≥ {MIN_COMPONENTS_PER_BUCKET} live components; the composite is "
        "the weight-renormalised average. Weekly series (Fed reserves/repo) are "
        "forward-fill-capped so a stale value can't masquerade as a live daily print.")


def _contrib_table_style(df_disp: pd.DataFrame):
    """Shared dataframe formatting for component tables."""
    return df_disp.style.format({
        "Raw": "{:,.4g}", "Adjusted": "{:,.4g}", "Z": "{:+.2f}",
        "Eff wt": "{:.0%}", "Contribution": "{:+.3f}",
        "1w Δ": "{:+.3f}", "1m Δ": "{:+.3f}", "3m Δ": "{:+.3f}",
    })


def _component_disp(rows: pd.DataFrame) -> pd.DataFrame:
    cols = ["component", "name", "ticker", "bucket", "raw_latest", "adjusted_latest",
            "z", "eff_weight", "contribution", "chg_1w", "chg_1m", "chg_3m", "status"]
    out = rows[cols].rename(columns={
        "component": "ID", "name": "Component", "ticker": "Ticker", "bucket": "Bucket",
        "raw_latest": "Raw", "adjusted_latest": "Adjusted", "z": "Z",
        "eff_weight": "Eff wt", "contribution": "Contribution",
        "chg_1w": "1w Δ", "chg_1m": "1m Δ", "chg_3m": "3m Δ", "status": "Status"})
    return out


def _render_component_contributions(ct: pd.DataFrame | None) -> None:
    """Component-level explainability (requirement #3)."""
    if ct is None or ct.empty:
        return
    st.markdown(
        "<div style='border-top:1px solid #1a1a1a;margin-top:1rem;padding-top:0.6rem;'>"
        "<div style='font-size:14px;font-weight:700;letter-spacing:0.06em;color:#fff;"
        "text-transform:uppercase;'>Component contributions</div>"
        "<div style='font-size:10px;color:#888;letter-spacing:0.08em;"
        "text-transform:uppercase;margin-top:2px;'>Exactly which market variables "
        "drive the signal · live contributions sum to index − 50</div></div>",
        unsafe_allow_html=True,
    )
    live = ct[ct["live"]].copy()
    total = live["contribution"].sum()
    st.caption(f"{len(live)} live components · Σ contributions = {total:+.3f} = index − 50.")

    view = st.radio(
        "VIEW", ["Current level", "1-week change", "1-month change", "3-month change"],
        index=0, horizontal=True, key="liq_compcontrib_view")
    sort_col = {"Current level": "contribution", "1-week change": "chg_1w",
                "1-month change": "chg_1m", "3-month change": "chg_3m"}[view]

    ranked = live.dropna(subset=[sort_col])
    easing = ranked.nlargest(10, sort_col)
    tightening = ranked.nsmallest(10, sort_col)
    e_col, t_col = st.columns(2, gap="medium")
    with e_col:
        st.markdown("<div style='color:#3fb37f;font-size:11px;letter-spacing:0.08em;"
                    "text-transform:uppercase;margin:0.3rem 0 0.2rem;'>Top 10 easing "
                    "(pushes index up)</div>", unsafe_allow_html=True)
        st.dataframe(_contrib_table_style(_component_disp(easing)),
                     hide_index=True, use_container_width=True)
    with t_col:
        st.markdown("<div style='color:#e0564a;font-size:11px;letter-spacing:0.08em;"
                    "text-transform:uppercase;margin:0.3rem 0 0.2rem;'>Top 10 tightening "
                    "(pushes index down)</div>", unsafe_allow_html=True)
        st.dataframe(_contrib_table_style(_component_disp(tightening)),
                     hide_index=True, use_container_width=True)

    excluded = ct[~ct["live"]]
    if not excluded.empty:
        st.markdown("<div style='color:#888;font-size:11px;letter-spacing:0.08em;"
                    "text-transform:uppercase;margin:0.5rem 0 0.2rem;'>Excluded "
                    "components &amp; reason</div>", unsafe_allow_html=True)
        st.dataframe(
            excluded[["component", "name", "bucket", "status"]].rename(columns={
                "component": "ID", "name": "Component", "bucket": "Bucket",
                "status": "Reason excluded"}),
            hide_index=True, use_container_width=True)


def _render_reconciliation(rec: dict | None) -> None:
    """Legacy vs current methodology reconciliation (requirement #2)."""
    if not rec:
        return
    st.markdown(
        "<div style='border-top:1px solid #1a1a1a;margin-top:1rem;padding-top:0.6rem;'>"
        "<div style='font-size:14px;font-weight:700;letter-spacing:0.06em;color:#fff;"
        "text-transform:uppercase;'>Index methodology reconciliation</div>"
        "<div style='font-size:10px;color:#888;letter-spacing:0.08em;"
        "text-transform:uppercase;margin-top:2px;'>Legacy vs current rules on the "
        "same latest data — isolates the methodology effect</div></div>",
        unsafe_allow_html=True,
    )
    d = rec["date"].date()
    diff = rec["index_diff"]
    col = POS_GREEN if diff >= 0 else NEG_RED
    st.markdown(
        f"""
        <div style="background:#0f0f0f;border:1px solid #1a1a1a;border-radius:6px;
                    padding:0.7rem 0.9rem;margin:0.3rem 0 0.6rem;font-size:13px;
                    color:#ccc;line-height:1.8;">
          On <b>{d}</b>: legacy methodology = <b>{rec['legacy_index']:.2f}</b>,
          current (v0.3) = <b>{rec['current_index']:.2f}</b>,
          difference = <b style="color:{col};">{diff:+.2f}</b> index points
          (composite-z {rec['legacy_z']:.3f} → {rec['current_z']:.3f},
          Δ {rec['z_diff']:+.3f}). This is the change attributable to
          <b>methodology only</b> — market-data effects are separate.
        </div>
        """,
        unsafe_allow_html=True)

    tab = rec["table"].copy()
    disp = tab[["bucket_label", "legacy_sub", "current_sub", "legacy_eff_w",
                "current_eff_w", "legacy_contrib", "current_contrib", "contrib_diff"]]
    disp = disp.rename(columns={
        "bucket_label": "Bucket", "legacy_sub": "Legacy sub", "current_sub": "Current sub",
        "legacy_eff_w": "Legacy wt", "current_eff_w": "Current wt",
        "legacy_contrib": "Legacy contrib", "current_contrib": "Current contrib",
        "contrib_diff": "Δ contrib"})
    st.dataframe(
        disp.style.format({
            "Legacy sub": "{:+.3f}", "Current sub": "{:+.3f}",
            "Legacy wt": "{:.0%}", "Current wt": "{:.0%}",
            "Legacy contrib": "{:+.3f}", "Current contrib": "{:+.3f}",
            "Δ contrib": "{:+.3f}"}),
        hide_index=True, use_container_width=True)

    c = rec["checks"]
    ok = (abs(c["sum_current_contrib"] - c["current_index_minus_50"]) < 1e-6 and
          abs(c["sum_legacy_contrib"] - c["legacy_index_minus_50"]) < 1e-6 and
          abs(c["sum_contrib_diff"] - c["current_minus_legacy"]) < 1e-6)
    mark = "✓ reconciles" if ok else "✗ mismatch"
    st.caption(
        f"Σ current contribs {c['sum_current_contrib']:+.3f} = index−50 "
        f"{c['current_index_minus_50']:+.3f} · Σ legacy {c['sum_legacy_contrib']:+.3f} "
        f"= {c['legacy_index_minus_50']:+.3f} · Σ Δ {c['sum_contrib_diff']:+.3f} "
        f"= current−legacy {c['current_minus_legacy']:+.3f}  ({mark}).")


def _render_ffill_audit(ffa: pd.DataFrame | None) -> None:
    """Forward-fill / staleness audit (requirement #5)."""
    if ffa is None or ffa.empty:
        return
    with st.expander("Forward-fill audit — weekly / low-frequency freshness", expanded=False):
        disp = ffa.rename(columns={
            "component": "ID", "name": "Component", "frequency": "Freq",
            "max_ffill_days": "Max ffill", "latest_raw_obs": "Latest raw",
            "latest_true_obs": "Latest true obs", "days_since_true_obs": "Days since",
            "is_live": "Live", "reason": "Reason", "stale_days_1y": "Stale days (1y)",
            "pct_ffilled_1y": "% ffilled (1y)"})
        st.dataframe(
            disp.style.format({"% ffilled (1y)": "{:.0f}%"}),
            hide_index=True, use_container_width=True,
            height=min(560, 42 + 34 * len(disp)))
        st.caption(
            "Weekly series (Fed reserves/repo) are observed on Wednesdays; the z is "
            "computed on those true observations and forward-filled at most "
            "'Max ffill' business days. '% ffilled' near 80% for weekly series is "
            "expected (4 of 5 weekdays are fills); a component goes not-live once "
            "'Days since' exceeds 'Max ffill'.")


def _render_methodology_audit(result: IndexResult, audit: dict | None) -> None:
    """Methodology version, parameters, audit trail, and the math (req #1, #9)."""
    with st.expander("Methodology & audit trail", expanded=False):
        if audit:
            ver = audit.get("version", "?")
            st.markdown(
                f"<div style='font-size:13px;color:#fff;'>Composite Liquidity Index "
                f"Methodology Version: <b>{ver}</b></div>"
                f"<div style='font-size:11px;color:#888;margin-bottom:8px;'>"
                f"{audit.get('description','')}</div>", unsafe_allow_html=True)
            bw = audit.get("bucket_weights", {})
            params = {
                "Methodology version": ver,
                "Z-score window": audit.get("z_window"),
                "Minimum periods": audit.get("z_min_periods"),
                "Z-score clip": f"±{audit.get('z_clip')}",
                "Min unique observations": audit.get("z_min_unique"),
                "Min available buckets": audit.get("min_available_buckets"),
                "Min available components": audit.get("min_available_components"),
                "Min components per bucket": audit.get("min_components_per_bucket"),
                "Warm-up (business days)": audit.get("warmup_days_after_first_valid"),
                "Bucket weights": ", ".join(f"{BUCKETS[b]['label']} {w:.0%}"
                                            for b, w in bw.items()),
                "Latest DATA.xlsx hash": str(audit.get("data_hash"))[:16] + "…",
                "Latest data date": str(getattr(audit.get("latest_data_date"), "date",
                                                 lambda: audit.get("latest_data_date"))()),
                "Latest published date": str(getattr(audit.get("latest_published_date"),
                                             "date", lambda: "n/a")()),
                "Reliable from": str(getattr(audit.get("first_published_date"),
                                     "date", lambda: "n/a")()),
                "Components on latest date": audit.get("components_on_latest"),
                "Buckets on latest date": audit.get("buckets_on_latest"),
                "Latest index": f"{audit.get('latest_index'):.2f} "
                                f"({audit.get('latest_regime')})",
            }
            pdf = pd.DataFrame({"Parameter": list(params.keys()),
                                "Value": [str(v) for v in params.values()]})
            st.dataframe(pdf, hide_index=True, use_container_width=True,
                         height=min(640, 42 + 34 * len(pdf)))
            st.caption(
                "If the headline value changes after a methodology update, the "
                "version bump + the reconciliation table above tell you whether the "
                "move came from market data or from methodology.")

        st.markdown(
            r"""
**The math.**
**1. Direction** — $\text{adj}_{i,t} = \text{raw}_{i,t}\times \text{dir}_i$ (higher = looser).
**2. Rolling z** — $z_{i,t} = (\text{adj}_{i,t}-\mu_{i,t})/\sigma_{i,t}$, trailing 5y
(min 2y), clipped $[-3,3]$; NaN if the window holds < 20 distinct values
(low-variation guard). Weekly series are z-scored on their true (Wednesday)
observations and the z is forward-filled ≤ 10 business days.
**3. Bucket sub-index** — mean of live z's, only if the bucket has ≥ 2 live components.
**4. Composite** — $\sum_b \tilde w_{b,t}\,\text{bucket}_{b,t}$, weights renormalised over
qualifying buckets.
**5. Index** — $50 + 10\times\text{composite}$.
**6. Component contribution** — $10\,\tilde w_{b,t}\,z_{i,t}/n_{b,t}$, summing to index−50.
**7. Coverage gate** — published only with ≥ 3 buckets, ≥ 8 components, past warm-up.
**Benchmarks** (Bloomberg FCI, Chicago Fed NFCI) are validation only, never inputs.
            """
        )
