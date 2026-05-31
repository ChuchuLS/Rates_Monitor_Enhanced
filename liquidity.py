"""
charts/common.py
================
Reusable, theme-aware chart primitives shared across every section. Each
builder returns a Plotly figure (pure) so the rendering layer in app.py just
calls st.plotly_chart.

Includes the y-axis auto-scaling helper that replaces hard-coded ranges
(requirement #4): axes follow the visible data and only pull zero into view when
the data crosses zero or sits close to it.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.theme import (
    BG, GRID, LINE_WHITE, TEXT_DIM, TEXT_VERY_DIM, DARK_LAYOUT,
    ACCENT_GREEN, ACCENT_RED, NOTE_RED_BG, NOTE_RED_BORDER, NOTE_RED_TEXT,
    NOTE_GREEN_BG, NOTE_GREEN_BORDER, NOTE_GREEN_TEXT,
)


# ---------------------------------------------------------------------------
# Y-axis auto-scaling (requirement #4)
# ---------------------------------------------------------------------------
def autoscale_range(values, pad_frac: float = 0.08,
                    zero_proximity: float = 0.30) -> list | None:
    """Compute a sensible [min, max] y-range from the data.

    Rules:
      * Pad above and below the data by ``pad_frac`` of its span.
      * Only force zero into view when the series crosses zero, or when it sits
        within ``zero_proximity`` of its own range from zero. Otherwise zoom to
        where the data actually lives — no dead vertical space.
      * Return None when there is nothing to scale, letting Plotly autorange.
    """
    vals = [v for v in values if v is not None and pd.notna(v)]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else (abs(hi) or 1.0)
    pad = span * pad_frac
    y_min, y_max = lo - pad, hi + pad

    if lo < 0 < hi:
        return [y_min, y_max]                 # crosses zero -> keep both sides
    if 0 < lo < span * zero_proximity:
        return [-span * 0.05, y_max]          # hugs zero from above -> show it
    if -span * zero_proximity < hi < 0:
        return [y_min, span * 0.05]           # hugs zero from below -> show it
    return [y_min, y_max]                     # zoom to data


def section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-header">
          <div class="section-title">{title}</div>
          <div class="section-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mini_dark(series: pd.Series, title: str, color: str = LINE_WHITE,
              height: int = 180, zero_line: bool = True,
              fmt: str = "{:+.1f}") -> go.Figure:
    """Compact dark line chart with a last-value badge in the title."""
    fig = go.Figure()
    s = series.dropna()
    if len(s):
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            line=dict(color=color, width=1.1),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[s.index[-1]], y=[s.iloc[-1]], mode="markers",
            marker=dict(color=color, size=5, line=dict(color=BG, width=1)),
            hoverinfo="skip", showlegend=False,
        ))
    if zero_line:
        fig.add_hline(y=0, line=dict(color=TEXT_VERY_DIM, width=0.5, dash="dot"))

    last_val = s.iloc[-1] if len(s) else float("nan")
    last_str = fmt.format(last_val) if pd.notna(last_val) else "—"
    last_color = ACCENT_GREEN if (pd.notna(last_val) and last_val >= 0) else ACCENT_RED
    title_html = (
        f"<span style='color:#fff;font-weight:700;letter-spacing:0.05em'>"
        f"{title.upper()}</span>  "
        f"<span style='color:{last_color};font-size:10px;font-weight:700'>"
        f"{last_str}</span>"
    )
    fig.update_layout(
        **DARK_LAYOUT, height=height,
        margin=dict(l=35, r=10, t=28, b=22),
        title=dict(text=title_html, font=dict(size=10), x=0, xanchor="left", y=0.97),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=8, color=TEXT_DIM), linecolor="#222")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(size=8, color=TEXT_DIM), linecolor="#222")
    return fig


def ofr_chart(series: pd.Series, top_note: str | None,
              bottom_note: str | None, height: int = 160) -> go.Figure:
    """OFR-style dark chart with optional red/green interpretation boxes."""
    fig = go.Figure()
    s = series.dropna()
    if len(s):
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            line=dict(color=LINE_WHITE, width=1),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[s.index[-1]], y=[s.iloc[-1]], mode="markers",
            marker=dict(color=LINE_WHITE, size=6, line=dict(color=BG, width=1)),
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_hline(y=0, line=dict(color=TEXT_VERY_DIM, width=0.5, dash="dot"))

    if top_note:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.92,
            text=f"<b>{top_note}</b>", showarrow=False,
            bgcolor=NOTE_RED_BG, bordercolor=NOTE_RED_BORDER, borderwidth=1,
            font=dict(color=NOTE_RED_TEXT, size=9, family="Inter, sans-serif"),
            align="center",
        )
    if bottom_note:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.08,
            text=f"<b>{bottom_note}</b>", showarrow=False,
            bgcolor=NOTE_GREEN_BG, bordercolor=NOTE_GREEN_BORDER, borderwidth=1,
            font=dict(color=NOTE_GREEN_TEXT, size=9, family="Inter, sans-serif"),
            align="center",
        )

    fig.update_layout(
        **DARK_LAYOUT, height=height,
        margin=dict(l=10, r=55, t=10, b=20),
        yaxis=dict(side="right", showgrid=True, gridcolor=GRID, zeroline=False,
                   tickfont=dict(color=TEXT_DIM, size=9),
                   title=dict(text="<i>spread</i>",
                              font=dict(size=9, color="#aaa"), standoff=2)),
        xaxis=dict(showgrid=False, tickfont=dict(color=TEXT_DIM, size=9)),
    )
    return fig
