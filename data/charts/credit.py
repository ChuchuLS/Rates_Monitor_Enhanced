"""
charts/credit.py
================
Credit Liquidity section: the composite IG/HY/EMBI/CDS chart, the CDX/iTraxx
index explorer, and the 30Y mortgage vs UST10Y spread panel.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config.theme import (
    BG, GRID, LINE_WHITE, TEXT_DIM, TEXT_VERY_DIM, DARK_LAYOUT,
    ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER, ACCENT_CYAN, ACCENT_PURPLE,
)
from charts.common import section_header
from config.tickers import CREDIT_INDICES
from data.loader import get_series


def render_credit(dff: pd.DataFrame) -> None:
    section_header("Credit", "IG/HY OAS · Bank CDS · EMBI sovereign · basis points")

    credit_fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
    ig = get_series(dff, "IG_OAS") * 100   # OAS quoted in %, convert to bp
    hy = get_series(dff, "HY_OAS") * 100
    embi = get_series(dff, "EMBI")
    bofa = get_series(dff, "CDS_BOFA")
    jpm = get_series(dff, "CDS_JPM")
    db_sub = get_series(dff, "CDS_DB_SUB")

    def _add(series, name, color, width, sec):
        if len(series):
            credit_fig.add_trace(
                go.Scatter(x=series.index, y=series.values, name=name,
                           line=dict(color=color, width=width),
                           hovertemplate=f"{name}: %{{y:.0f}}bp<extra></extra>"),
                secondary_y=sec)

    _add(ig, "IG OAS", ACCENT_CYAN, 1.5, False)
    _add(embi, "EMBI sovereign", ACCENT_GREEN, 1.3, False)
    _add(bofa, "BofA 5Y CDS", TEXT_DIM, 1.0, False)
    _add(jpm, "JPM 5Y CDS", ACCENT_PURPLE, 1.0, False)
    _add(hy, "HY OAS", ACCENT_RED, 1.7, True)
    _add(db_sub, "DB sub CDS", ACCENT_AMBER, 1.0, True)

    credit_fig.update_layout(
        **{**DARK_LAYOUT, "showlegend": True}, height=440,
        margin=dict(l=50, r=50, t=20, b=30),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#ccc")))
    credit_fig.update_xaxes(showgrid=False, linecolor="#222",
                            tickfont=dict(color=TEXT_DIM, size=9))
    credit_fig.update_yaxes(title_text="IG / EMBI / Bank CDS (bp)", secondary_y=False,
                            showgrid=True, gridcolor=GRID, linecolor="#222",
                            title_font=dict(size=10, color="#aaa"),
                            tickfont=dict(color=TEXT_DIM, size=9))
    credit_fig.update_yaxes(title_text="HY / DB sub (bp)", secondary_y=True,
                            showgrid=False, linecolor="#222",
                            title_font=dict(size=10, color="#aaa"),
                            tickfont=dict(color=TEXT_DIM, size=9))
    st.plotly_chart(credit_fig, use_container_width=True, key="credit_panel",
                    config={"displayModeBar": False})

    _render_credit_explorer(dff)
    _render_mortgage(dff)


def _render_credit_explorer(dff: pd.DataFrame) -> None:
    st.markdown(
        """
        <div style="padding:0.6rem 0 0.25rem;margin-top:0.75rem;">
          <div style="font-size:13px;font-weight:700;letter-spacing:0.06em;
                      color:#ccc;text-transform:uppercase;">Credit Index Explorer</div>
          <div style="font-size:10px;color:#888;letter-spacing:0.08em;
                      text-transform:uppercase;margin-top:2px;">
            Pick any CDX or iTraxx series · daily change histogram below</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    credit_choice = st.selectbox("INDEX", options=list(CREDIT_INDICES.keys()),
                                 index=0, key="credit_idx_choice")
    credit_key, credit_unit = CREDIT_INDICES[credit_choice]
    cs = get_series(dff, credit_key).dropna()

    if cs.empty:
        st.warning(f"No data available for {credit_choice}.")
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04)
    line_color = LINE_WHITE if credit_unit == "price" else ACCENT_CYAN
    fig.add_trace(go.Scatter(
        x=cs.index, y=cs.values, mode="lines", line=dict(color=line_color, width=1.3),
        fill="tozeroy" if credit_unit == "spread" else None,
        fillcolor="rgba(79,168,184,0.08)" if credit_unit == "spread" else None,
        hovertemplate=("%{x|%Y-%m-%d}: $%{y:.2f}<extra></extra>" if credit_unit == "price"
                       else "%{x|%Y-%m-%d}: %{y:.1f}bp<extra></extra>"),
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[cs.index[-1]], y=[cs.iloc[-1]], mode="markers",
        marker=dict(color=line_color, size=7, line=dict(color=BG, width=1)),
        hoverinfo="skip", showlegend=False), row=1, col=1)
    chg = cs.diff().dropna()
    bar_colors = ["#67c757" if v >= 0 else "#e64545" for v in chg.values]
    fig.add_trace(go.Bar(
        x=chg.index, y=chg.values, marker=dict(color=bar_colors, line=dict(width=0)),
        hovertemplate=("%{x|%Y-%m-%d}: $%{y:+.2f}<extra></extra>" if credit_unit == "price"
                       else "%{x|%Y-%m-%d}: %{y:+.1f}bp<extra></extra>"),
        showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=TEXT_VERY_DIM, width=0.5, dash="dot"), row=2, col=1)

    last_val = cs.iloc[-1]
    last_str = f"${last_val:.2f}" if credit_unit == "price" else f"{last_val:.1f}bp"
    fig.update_layout(
        **DARK_LAYOUT, height=520, margin=dict(l=55, r=20, t=45, b=30),
        title=dict(text=(f"<span style='color:#fff;font-weight:700;letter-spacing:0.05em;"
                         f"font-size:14px;'>{credit_choice.upper()}</span>  "
                         f"&nbsp;<span style='color:#aaa;font-size:11px;'>{last_str}</span>"),
                   font=dict(size=12), x=0, xanchor="left", y=0.97),
        bargap=0.0)
    fig.update_xaxes(showgrid=False, linecolor="#222", tickfont=dict(size=10, color=TEXT_DIM))
    suffix = "" if credit_unit == "price" else "bp"
    y1_title = "Price ($)" if credit_unit == "price" else "Spread (bp)"
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(size=10, color=TEXT_DIM), linecolor="#222",
                     ticksuffix=suffix, row=1, col=1,
                     title=dict(text=y1_title, font=dict(size=10, color="#888")))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(size=10, color=TEXT_DIM), linecolor="#222",
                     ticksuffix=suffix, row=2, col=1,
                     title=dict(text="1d Δ", font=dict(size=10, color="#888")))
    st.plotly_chart(fig, use_container_width=True, key="credit_idx_explorer",
                    config={"displayModeBar": False})


def _render_mortgage(dff: pd.DataFrame) -> None:
    st.markdown(
        """
        <div style="padding:0.6rem 0 0.25rem;margin-top:0.75rem;">
          <div style="font-size:13px;font-weight:700;letter-spacing:0.06em;
                      color:#ccc;text-transform:uppercase;">Mortgage rate vs. 10Y Treasury</div>
          <div style="font-size:10px;color:#888;letter-spacing:0.08em;
                      text-transform:uppercase;margin-top:2px;">
            30Y fixed mortgage vs. UST 10Y · spread histogram below · bp</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mtg = get_series(dff, "MTG_30Y").dropna()
    ust10 = get_series(dff, "US_10Y").dropna()
    if mtg.empty or ust10.empty:
        st.warning("Mortgage or UST 10Y data unavailable.")
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=mtg.index, y=mtg.values, mode="lines",
                             line=dict(color=ACCENT_AMBER, width=1.3), name="30Y mortgage",
                             hovertemplate="30Y mortgage: %{y:.2f}%<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=ust10.index, y=ust10.values, mode="lines",
                             line=dict(color=LINE_WHITE, width=1.3), name="UST 10Y",
                             hovertemplate="UST 10Y: %{y:.2f}%<extra></extra>"),
                  row=1, col=1)
    aligned = pd.concat([mtg, ust10], axis=1, join="inner").dropna()
    aligned.columns = ["mtg", "ust"]
    spread_bp = (aligned["mtg"] - aligned["ust"]) * 100
    fig.add_trace(go.Scatter(x=spread_bp.index, y=spread_bp.values, mode="lines",
                             line=dict(color="#e64545", width=1.1),
                             fill="tozeroy", fillcolor="rgba(230,69,69,0.18)",
                             name="Mortgage – UST10Y spread",
                             hovertemplate="Spread: %{y:+.0f}bp<extra></extra>",
                             showlegend=False), row=2, col=1)
    avg_spread = float(spread_bp.mean())
    fig.add_hline(y=avg_spread, line=dict(color=TEXT_VERY_DIM, width=0.5, dash="dash"),
                  annotation_text=f"avg {avg_spread:.0f}bp", annotation_position="right",
                  annotation_font=dict(size=9, color="#888"), row=2, col=1)

    mtg_last, ust_last = mtg.iloc[-1], ust10.iloc[-1]
    spread_last = (mtg_last - ust_last) * 100
    fig.update_layout(
        **{**DARK_LAYOUT, "showlegend": True}, height=520, margin=dict(l=55, r=20, t=45, b=30),
        title=dict(text=(f"<span style='color:#fff;font-weight:700;letter-spacing:0.05em;"
                         f"font-size:14px;'>30Y MORTGAGE vs UST 10Y</span>  "
                         f"&nbsp;<span style='color:{ACCENT_AMBER};font-size:11px;font-weight:700'>"
                         f"Mtg {mtg_last:.2f}%</span>  "
                         f"<span style='color:#fff;font-size:11px;font-weight:700'>UST {ust_last:.2f}%</span>  "
                         f"<span style='color:#e64545;font-size:11px;font-weight:700'>"
                         f"Spread {spread_last:+.0f}bp</span>"),
                   font=dict(size=12), x=0, xanchor="left", y=0.97),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#ccc")))
    fig.update_xaxes(showgrid=False, linecolor="#222", tickfont=dict(size=10, color=TEXT_DIM))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(size=10, color=TEXT_DIM), linecolor="#222",
                     ticksuffix="%", row=1, col=1,
                     title=dict(text="Yield (%)", font=dict(size=10, color="#888")))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                     tickfont=dict(size=10, color=TEXT_DIM), linecolor="#222",
                     ticksuffix="bp", row=2, col=1,
                     title=dict(text="Spread (bp)", font=dict(size=10, color="#888")))
    st.plotly_chart(fig, use_container_width=True, key="mortgage_panel",
                    config={"displayModeBar": False})
