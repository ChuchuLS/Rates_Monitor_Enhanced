"""
charts/rates.py
===============
Rates-section charts: curve-slope regime classification and the term-structure
("curve") panels used for real rates and inflation breakevens.

The real-rate / breakeven curve panel uses the shared ``autoscale_range`` helper
so the y-axis follows the data instead of a hard-coded [-1, 3] (requirement #4).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from config.theme import (
    BG, GRID, LINE_WHITE, TEXT_DIM, TEXT_VERY_DIM, DARK_LAYOUT,
    ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    CURVE_REGIME_COLORS as REGIME_COLORS,
    CURVE_REGIME_LABELS as REGIME_LABELS,
)
from charts.common import autoscale_range
from data.loader import get_series


def classify_regime(short: pd.Series, long_: pd.Series,
                    lookback: int) -> tuple[pd.Series, pd.Series]:
    """Classify curve regime from a lookback comparison (Bloomberg Studio logic).

    dc = curve change, ds = short-rate change, dl = long-rate change.
      Steepening (dc>0): both down=bull, both up=bear, mixed=twist
      Flattening (dc<0): both down=bull, both up=bear, mixed=twist
    """
    short, long_ = short.align(long_, join="inner")
    slope = (long_ - short) * 100.0  # bp

    ds = short - short.shift(lookback)
    dl = long_ - long_.shift(lookback)
    dc = slope - slope.shift(lookback)

    regime = pd.Series("none", index=slope.index, dtype="object")
    regime[(dc > 0) & (ds < 0) & (dl < 0)] = "bull_steepener"
    regime[(dc > 0) & (ds > 0) & (dl > 0)] = "bear_steepener"
    regime[(dc > 0) & (ds < 0) & (dl > 0)] = "steepener_twist"
    regime[(dc < 0) & (ds < 0) & (dl < 0)] = "bull_flattener"
    regime[(dc < 0) & (ds > 0) & (dl > 0)] = "bear_flattener"
    regime[(dc < 0) & (ds > 0) & (dl < 0)] = "flattener_twist"
    return slope, regime


def big_regime_panel(slope: pd.Series, regime: pd.Series,
                     title: str, height: int = 560) -> go.Figure:
    """Slope histogram coloured by regime with an amber slope line overlay."""
    fig = go.Figure()
    if len(slope):
        bar_colors = regime.map(REGIME_COLORS).fillna(REGIME_COLORS["none"])
        fig.add_trace(go.Bar(
            x=slope.index, y=slope.values,
            marker=dict(color=bar_colors.values, line=dict(width=0)),
            customdata=regime.map(REGIME_LABELS).fillna("—").values,
            hovertemplate=("%{x|%Y-%m-%d}<br>Slope: %{y:.1f} bp<br>"
                           "Regime: %{customdata}<extra></extra>"),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=slope.index, y=slope.values, mode="lines",
            line=dict(color=ACCENT_AMBER, width=1.4),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[slope.index[-1]], y=[slope.iloc[-1]], mode="markers",
            marker=dict(color=ACCENT_AMBER, size=8, line=dict(color=BG, width=1)),
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_hline(y=0, line=dict(color=TEXT_VERY_DIM, width=0.7, dash="dot"))

    last_val = slope.iloc[-1] if len(slope) else float("nan")
    last_str = f"{last_val:+.0f}bp" if pd.notna(last_val) else "—"
    last_color = ACCENT_GREEN if (pd.notna(last_val) and last_val >= 0) else ACCENT_RED
    title_html = (
        f"<span style='color:#fff;font-weight:700;letter-spacing:0.05em;font-size:14px;'>"
        f"{title.upper()}</span>  &nbsp;"
        f"<span style='color:{last_color};font-size:13px;font-weight:700'>{last_str}</span>"
    )
    fig.update_layout(
        **DARK_LAYOUT, height=height,
        margin=dict(l=70, r=30, t=50, b=50),
        title=dict(text=title_html, font=dict(size=13), x=0, xanchor="left", y=0.97),
        bargap=0.0,
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=11, color="#bbb"), linecolor="#222")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(size=11, color="#bbb"), linecolor="#222",
                     ticksuffix="bp",
                     title=dict(text="Slope (bp)", font=dict(size=11, color="#888")))
    return fig


def curve_at(df: pd.DataFrame, d, tenors):
    """Look up curve values at (or just before) date d. Skip missing tenors."""
    xs, ys, labels = [], [], []
    for label, t, key in tenors:
        s = get_series(df, key)
        if len(s) == 0:
            continue
        v = s.asof(d)
        if pd.notna(v):
            xs.append(t)
            ys.append(float(v))
            labels.append(label)
    return xs, ys, labels


def big_curve_panel(df: pd.DataFrame, title: str, tenors, anchor_date,
                    height: int = 560, y_title: str = "Real yield (%)",
                    ) -> go.Figure:
    """Term-structure panel: today vs 1w / 1m ago, with auto-scaled y-axis."""
    today = anchor_date
    week_ago = today - pd.Timedelta(days=7)
    month_ago = today - pd.Timedelta(days=30)
    fig = go.Figure()

    xs, ys, _ = curve_at(df, month_ago, tenors)
    if len(xs) >= 2:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color="rgba(255,255,255,0.30)", width=1.5, dash="dot"),
            marker=dict(color="rgba(255,255,255,0.30)", size=7),
            name="1 month ago",
            hovertemplate="1m ago · %{x}Y: %{y:.2f}%<extra></extra>",
        ))

    xs, ys, _ = curve_at(df, week_ago, tenors)
    if len(xs) >= 2:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color="rgba(217,152,48,0.70)", width=1.8, dash="dash"),
            marker=dict(color="rgba(217,152,48,0.70)", size=8),
            name="1 week ago",
            hovertemplate="1w ago · %{x}Y: %{y:.2f}%<extra></extra>",
        ))

    xs, ys, labels = curve_at(df, today, tenors)
    if len(xs) >= 2:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers+text",
            line=dict(color=LINE_WHITE, width=2.4),
            marker=dict(color=LINE_WHITE, size=11, line=dict(color=BG, width=1.5)),
            text=[f"{y:+.2f}%" for y in ys],
            textposition="top center", textfont=dict(size=12, color="#fff"),
            name="Today",
            hovertemplate="Today · %{x}Y: %{y:.2f}%<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color=TEXT_VERY_DIM, width=0.7, dash="dot"))

    # Auto-scale the y-axis to the visible curve data (requirement #4).
    all_y = [v for tr in fig.data if getattr(tr, "y", None) is not None
             for v in tr.y if v is not None and pd.notna(v)]
    y_range = autoscale_range(all_y, pad_frac=0.20, zero_proximity=0.30)

    title_html = (
        f"<span style='color:#fff;font-weight:700;letter-spacing:0.05em;font-size:14px;'>"
        f"{title.upper()}</span>"
    )
    fig.update_layout(
        **{**DARK_LAYOUT, "showlegend": True}, height=height,
        margin=dict(l=70, r=30, t=50, b=70),
        title=dict(text=title_html, font=dict(size=13), x=0, xanchor="left", y=0.97),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12, color="#ccc")),
    )
    if len(xs) >= 2:
        x_pad = (max(xs) - min(xs)) * 0.10
        fig.update_xaxes(tickvals=xs, ticktext=labels,
                         range=[min(xs) - x_pad, max(xs) + x_pad],
                         showgrid=False, tickfont=dict(size=13, color="#ddd"),
                         linecolor="#222", title=None)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(size=12, color="#bbb"), linecolor="#222",
                     ticksuffix="%", range=y_range, nticks=8,
                     title=dict(text=y_title, font=dict(size=11, color="#888")))
    return fig
