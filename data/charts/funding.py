"""
charts/funding.py
=================
Money-Market Plumbing section: the five OFR-style spread panels (with their
label/interpretation boxes) and the layered overnight-rates chart.

``render_money_market`` owns its Streamlit output so app.py only has to call it.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.theme import (
    BG, GRID, TEXT_DIM, DARK_LAYOUT, ACCENT_GREEN, ACCENT_RED,
)
from charts.common import ofr_chart
from data.loader import get_series


def _spread_defs(dff: pd.DataFrame) -> list[tuple]:
    """Definitions for the five money-market spread panels."""
    return [
        ("GCF − TPR", "DEALER BALANCE<br>SHEET CAPACITY",
         "The spread between interdealer and triparty repo rates, a proxy for "
         "funding demand and dealer balance sheet capacity.",
         get_series(dff, "GCF") - get_series(dff, "TPR"),
         "INFLEXIBLE BALANCE SHEETS ↑", "FLEXIBLE BALANCE SHEETS ↓"),
        ("TGCR − RRP", "PRIVATE REPO<br>DEMAND",
         "The spread between private repo &amp; Fed RRP rates, which measures "
         "demand for cash vs. collateral.",
         get_series(dff, "TGCR") - get_series(dff, "RRP"),
         "EXCESS COLLATERAL ↑", "EXCESS CASH ↓"),
        ("SOFR − IORB", "BANK REPOS",
         "A positive spread indicates banks are lending reserves in repo on a "
         "consistent basis, reducing liquidity elsewhere.",
         get_series(dff, "SOFR") - get_series(dff, "IORB"),
         "ABOVE ZERO, BANKS DEPLOY RESERVES<br>CONSISTENTLY INTO REPO MARKETS ↑", None),
        ("EFFR − IORB", "RESERVE<br>DEMAND",
         "A positive spread suggests scarce reserves on a historical basis.",
         get_series(dff, "EFFR") - get_series(dff, "IORB"),
         "SCARCITY ↑", "ABUNDANCE ↓"),
        ("SOFR − EFFR", "FHLB REPO<br>DEMAND",
         "The spread suggests where Federal Home Loan Banks might invest more "
         "of their liquidity portfolios.",
         get_series(dff, "SOFR") - get_series(dff, "EFFR"),
         "FHLBs DEPLOY MORE<br>CASH INTO REPOS ↑", None),
    ]


def render_money_market(dff: pd.DataFrame) -> None:
    st.markdown(
        """
        <div style="padding:0.6rem 0 0.5rem;border-bottom:1px solid #1a1a1a;">
          <div style="font-size:18px;font-weight:700;letter-spacing:0.06em;
                      color:#fff;text-transform:uppercase;">Money Market Spreads Monitor</div>
          <div style="font-size:10px;color:#888;letter-spacing:0.1em;
                      text-transform:uppercase;margin-top:2px;">
            Key money-market spreads and how to interpret them</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for name, category, explainer, s, top_note, bottom_note in _spread_defs(dff):
        sd = s.dropna()
        last_val = sd.iloc[-1] if len(sd) else float("nan")
        last_color = ACCENT_RED if (pd.notna(last_val) and last_val < 0) else ACCENT_GREEN
        last_str = f"{last_val:+.3f}" if pd.notna(last_val) else "—"
        parts = name.split(" − ")
        left_ticker, right_ticker = parts[0], (parts[1] if len(parts) > 1 else "")

        label_col, chart_col = st.columns([1, 4], gap="small")
        with label_col:
            st.markdown(
                f"""
                <div style="background:{BG};padding:1rem 0.9rem;height:160px;color:#fff;
                            display:flex;flex-direction:column;font-family:Inter,sans-serif;">
                  <div style="font-size:13px;font-weight:700;letter-spacing:0.06em;
                              line-height:1.15;margin-bottom:10px;color:#fff;">{category}</div>
                  <div style="background:#1a1a1a;padding:5px 10px;border:1px solid #2a2a2a;
                              display:inline-block;width:fit-content;margin-bottom:8px;">
                    <span style="color:{ACCENT_GREEN};font-weight:700;font-size:13px;
                                 letter-spacing:0.05em;">{left_ticker}</span>
                    <span style="color:#888;font-weight:700;font-size:13px;"> − </span>
                    <span style="color:{ACCENT_RED};font-weight:700;font-size:13px;
                                 letter-spacing:0.05em;">{right_ticker}</span>
                    <div style="font-size:8px;color:#666;letter-spacing:0.18em;
                                margin-top:1px;text-align:center;">SPREAD</div>
                  </div>
                  <div style="font-size:9px;color:#aaa;line-height:1.45;letter-spacing:0.04em;
                              text-transform:uppercase;">{explainer}</div>
                  <div style="margin-top:auto;font-size:9px;color:#666;letter-spacing:0.05em;
                              text-transform:uppercase;padding-top:6px;">
                    Latest: <span style="color:{last_color};font-weight:700;font-size:11px;
                                         letter-spacing:0.02em;">{last_str}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with chart_col:
            st.plotly_chart(
                ofr_chart(s, top_note, bottom_note), use_container_width=True,
                key=f"mm_{name.replace(' ', '_').replace('−', '_')}",
                config={"displayModeBar": False},
            )

    # --- Overnight rates layered -------------------------------------------
    st.markdown(
        """
        <div style="padding:0.4rem 0 0.25rem;margin-top:0.5rem;">
          <div style="font-size:13px;font-weight:700;letter-spacing:0.06em;
                      color:#ccc;text-transform:uppercase;">Overnight rates layered</div>
          <div style="font-size:10px;color:#888;letter-spacing:0.08em;
                      text-transform:uppercase;margin-top:2px;">
            Six US dollar overnight rates plotted together · %</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ovr_fig = go.Figure()
    ovr_series = [
        ("Fed funds target (lower)", "RRP", "#d062ff"),
        ("IORB", "IORB", "#9bd62a"),
        ("SOFR", "SOFR", "#ffd200"),
        ("TGCR", "TGCR", "#ff8a3d"),
        ("USD repo GC ON", "USRG_1T", "#5dd6e0"),
        ("RRP award rate", "TOMO_TCSO", "#ffffff"),
    ]
    for label, key, color in ovr_series:
        s = get_series(dff, key)
        if len(s):
            ovr_fig.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines",
                line=dict(color=color, width=1.2), name=label,
                hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}: %{{y:.3f}}%<extra></extra>",
            ))
    ovr_fig.update_layout(
        **{**DARK_LAYOUT, "showlegend": True}, height=380,
        margin=dict(l=60, r=20, t=20, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#ccc")),
    )
    ovr_fig.update_xaxes(showgrid=False, tickfont=dict(size=10, color=TEXT_DIM), linecolor="#222")
    ovr_fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                         tickfont=dict(size=10, color=TEXT_DIM), linecolor="#222", ticksuffix="%")
    st.plotly_chart(ovr_fig, use_container_width=True, key="overnight_rates",
                    config={"displayModeBar": False})


def render_xccy(dff: pd.DataFrame) -> None:
    """Dollar Funding / XCCY Basis section: 3M (top) and 12M (bottom) minis."""
    from charts.common import mini_dark, section_header
    from config.theme import ACCENT_AMBER, ACCENT_CYAN

    section_header("Cross-Currency Basis Swaps",
                   "Top: 3M · Bottom: 12M · negative = USD funding premium · bp")
    xccy_list = [("EUR", "EUR"), ("JPY", "JPY"), ("AUD", "AUD"),
                 ("GBP", "GBP"), ("CAD", "CAD")]

    cols = st.columns(5)
    for col, (ccy, label) in zip(cols, xccy_list):
        with col:
            s = get_series(dff, f"XCCY_{ccy}")
            if len(s):
                st.plotly_chart(
                    mini_dark(s, f"{label}/USD 3M basis", color=ACCENT_AMBER),
                    use_container_width=True, key=f"xccy3m_{ccy}",
                    config={"displayModeBar": False})
            else:
                st.warning(f"{label}/USD 3M basis unavailable")

    cols = st.columns(5)
    for col, (ccy, label) in zip(cols, xccy_list):
        with col:
            s = get_series(dff, f"XCCY12_{ccy}")
            if len(s):
                st.plotly_chart(
                    mini_dark(s, f"{label}/USD 12M basis", color=ACCENT_CYAN),
                    use_container_width=True, key=f"xccy12m_{ccy}",
                    config={"displayModeBar": False})
