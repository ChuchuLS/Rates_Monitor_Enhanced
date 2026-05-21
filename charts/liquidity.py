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


# ===========================================================================
# Full page renderer (requirement #9-#11)
# ===========================================================================
def render_index_page(df: pd.DataFrame, dff: pd.DataFrame, result: IndexResult) -> None:
    """Render the entire Composite Liquidity Index section."""
    section_header(
        "Composite Liquidity Index",
        "Raw-indicator liquidity gauge · higher = looser · 50 = neutral · "
        "z-scored & weighted across five buckets",
    )

    if result.index.dropna().empty:
        st.warning(
            "No index could be built because none of the component indicators "
            "are present in the data. See the Data Quality section for details."
        )
        return

    # --- Index level + regime line ----------------------------------------
    start, end = (dff.index.min(), dff.index.max()) if len(dff) else (df.index.min(), df.index.max())
    idx_window = result.index.loc[(result.index.index >= start) & (result.index.index <= end)]
    st.plotly_chart(index_line_chart(idx_window), use_container_width=True,
                    key="liq_index_line", config={"displayModeBar": False})
    st.caption(
        "Bands: green = Loose (≥60) · grey = Neutral (45–60) · "
        "amber = Tight (35–45) · red = Stress (<35)."
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

    # --- Benchmark comparison (requirement #11) ---------------------------
    _render_benchmark_block(df, result)

    # --- Component availability + methodology -----------------------------
    _render_components_table(result)


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

    # Standardised overlay.
    st.markdown(
        "<div style='color:#888;font-size:11px;letter-spacing:0.08em;"
        "text-transform:uppercase;margin:0.5rem 0 0.2rem;'>Standardised overlay "
        "(all oriented so higher = looser)</div>", unsafe_allow_html=True)
    st.plotly_chart(benchmark_overlay_chart(standardized_overlay(result.index, df)),
                    use_container_width=True, key="liq_overlay",
                    config={"displayModeBar": False})

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
    disp = meta[["bucket_label", "label", "direction", "available", "n_obs", "weight"]]
    disp = disp.rename(columns={
        "bucket_label": "Bucket", "label": "Indicator", "direction": "Direction",
        "available": "In index", "n_obs": "Obs", "weight": "Bucket wt"})
    st.dataframe(
        disp.style.format({"Obs": "{:,}"}), hide_index=True, use_container_width=True,
        height=min(560, 42 + 36 * len(disp)))
    st.caption(
        "Each indicator is direction-adjusted so higher = looser, then converted "
        "to a rolling z-score (5y window, 2y min, clipped ±3). Bucket sub-index = "
        "mean of its z-scores; composite = weighted average (money-market 30%, "
        "XCCY 20%, credit 20%, central-bank 20%, market-liq 10%), rescaled as "
        "50 + 10 × composite-z. Weights renormalise across whichever buckets have "
        "data, so a missing bucket never biases the index.")
