"""
app.py — Rates, Funding & Liquidity Monitor
===========================================
Thin Streamlit entry point: layout, the password gate, the sidebar, and page
routing. All heavy lifting (data loading, chart construction, ticker
definitions, the Composite Liquidity Index) lives in the config / data / charts
/ index packages — see each module's docstring.

Page structure (requirement #12)
    1. Composite Liquidity Index   (homepage — leads with the summary panel)
    2. Money Market Plumbing
    3. Dollar Funding / XCCY Basis
    4. Credit Liquidity
    5. Rates / Real Rates / Curve Regime
    6. Inflation Expectations
    7. Data Quality
"""

from __future__ import annotations

import os
import sys

# Be defensive: ensure this script's directory is importable even if Streamlit's
# automatic sys.path insertion ever changes. Lets `from config... import ...` work.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.theme import (
    BG, GRID, LINE_WHITE, TEXT_DIM, TEXT_VERY_DIM, DARK_LAYOUT,
    ACCENT_GREEN, ACCENT_RED, ACCENT_AMBER,
    CURVE_REGIME_COLORS, CURVE_REGIME_LABELS, REGIME_COLORS, page_css,
)
from config.tickers import (
    TICKERS, REGIME_COUNTRIES, REAL_RATE_TENORS, TENOR_PAIRS,
    INFL_BE_TENORS, INFL_ZCIS_TENORS,
)
from data.loader import load_data, date_filter, get_series, data_source_label
from data.quality import validate_data, quality_summary, STALE_BDAYS
from charts.common import section_header
from charts.rates import classify_regime, big_regime_panel, big_curve_panel
from charts.funding import render_money_market, render_xccy
from charts.credit import render_credit
from charts.liquidity import render_summary_panel, render_index_page
from index.composite import compute_index

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Rates & Liquidity Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Password gate (ported unchanged: disabled unless `app_password` secret set)
# ---------------------------------------------------------------------------
def _check_password() -> bool:
    # Reading st.secrets raises StreamlitSecretNotFoundError when no
    # secrets.toml is present (e.g. local dev). Treat that as "no password
    # configured" so the gate stays disabled rather than crashing.
    try:
        expected = st.secrets.get("app_password")
    except Exception:
        expected = None
    if not expected:
        return True
    if st.session_state.get("password_correct"):
        return True
    st.markdown(
        """
        <div style="max-width:420px;margin:5rem auto 1rem;padding:2rem;
                    background:#0a0a0a;border:1px solid #1a1a1a;border-radius:6px;
                    font-family:Inter,system-ui,sans-serif;color:#fff;">
          <div style="font-size:18px;font-weight:700;letter-spacing:0.06em;
                      text-transform:uppercase;margin-bottom:6px;">
            Rates &amp; Liquidity Monitor</div>
          <div style="font-size:11px;color:#888;letter-spacing:0.08em;
                      text-transform:uppercase;margin-bottom:1.5rem;">
            Authentication required</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pwd = st.text_input("Password", type="password", key="password_input",
                        label_visibility="collapsed", placeholder="Enter password")
    if pwd:
        if pwd == expected:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not _check_password():
    st.stop()

# ---------------------------------------------------------------------------
# Global styling + data
# ---------------------------------------------------------------------------
st.markdown(page_css(), unsafe_allow_html=True)

df = load_data()


@st.cache_data(show_spinner="Building Composite Liquidity Index...")
def _build_index(_source: str, n_rows: int):
    """Compute the index once per data load. Cache key is the source + row count
    so it recomputes when a refreshed parquet/Excel is dropped in."""
    return compute_index(load_data())


index_result = _build_index(data_source_label(), len(df))


# ===========================================================================
# Page 5 — Rates / Real Rates / Curve Regime (ported Curve Explorer)
# ===========================================================================
def render_rates_page(df: pd.DataFrame, dff: pd.DataFrame) -> None:
    section_header(
        "Rates / Real Rates / Curve Regime",
        "Pick a country, chart type, tenor pair, and lookback to drill into "
        "rates structure",
    )

    cols = st.columns([1, 1.3, 1, 1, 2])
    with cols[0]:
        country = st.selectbox("COUNTRY", options=list(REGIME_COUNTRIES), index=0,
                               key="explorer_country")
    with cols[1]:
        chart_type = st.selectbox("CHART TYPE",
                                  options=["Curve slope (regime)", "Real rate curve"],
                                  index=0, key="explorer_chart")
    is_slope = chart_type == "Curve slope (regime)"
    with cols[2]:
        if is_slope:
            pair = st.selectbox("TENOR PAIR", options=list(TENOR_PAIRS.keys()),
                                index=0, key="explorer_pair")
        else:
            pair = None
            st.markdown("<div style='color:#444;font-size:11px;padding-top:1.7rem;'>"
                        "— n/a —</div>", unsafe_allow_html=True)
    with cols[3]:
        if is_slope:
            lb_choice = st.selectbox("LOOKBACK", options=["5d", "10d", "20d", "60d", "120d"],
                                     index=2, key="explorer_lookback")
            lookback = int(lb_choice.rstrip("d"))
        else:
            lookback = None
            st.markdown("<div style='color:#444;font-size:11px;padding-top:1.7rem;'>"
                        "— n/a —</div>", unsafe_allow_html=True)

    if is_slope:
        short_t, long_t = TENOR_PAIRS[pair]
        short = get_series(dff, f"{country}_{short_t}")
        long_ = get_series(dff, f"{country}_{long_t}")
        if len(short) == 0 or len(long_) == 0:
            st.warning(f"No nominal yield data for {country} {short_t}/{long_t}. "
                       "Try a different pair or country.")
            return
        slope, regime = classify_regime(short, long_, lookback)
        chips = " &nbsp;&nbsp; ".join(
            f"<span style='display:inline-block;width:11px;height:11px;"
            f"background:{CURVE_REGIME_COLORS[k]};vertical-align:middle;"
            f"margin-right:5px;'></span>"
            f"<span style='color:#bbb;font-size:10px;letter-spacing:0.05em;"
            f"text-transform:uppercase;'>{CURVE_REGIME_LABELS[k]}</span>"
            for k in ["bull_steepener", "bear_steepener", "steepener_twist",
                      "bull_flattener", "bear_flattener", "flattener_twist"])
        st.markdown(f"<div style='padding:0.5rem 0 0.25rem;'>{chips}</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(
            big_regime_panel(slope, regime, f"{country} {pair} (regime vs {lookback}d ago)"),
            use_container_width=True, key="explorer_slope",
            config={"displayModeBar": False})
    else:
        anchor = dff.index.max() if len(dff) else df.index.max()
        tenors = REAL_RATE_TENORS.get(country, [])
        if not tenors:
            st.warning(f"No real-rate curve configured for {country}.")
            return
        st.plotly_chart(
            big_curve_panel(df, f"{country} real rate curve", tenors, anchor,
                            y_title="Real yield (%)"),
            use_container_width=True, key="explorer_curve",
            config={"displayModeBar": False})


# ===========================================================================
# Page 6 — Inflation Expectations (ported)
# ===========================================================================
def render_inflation_page(df: pd.DataFrame, dff: pd.DataFrame) -> None:
    section_header(
        "Inflation Expectations",
        "TIPS breakevens · ZC inflation swaps · 5Y5Y forward",
    )
    choice = st.selectbox(
        "MEASURE",
        options=["TIPS breakeven curve", "ZC inflation swap curve",
                 "5Y5Y forward inflation swap"],
        index=0, key="infl_choice")
    anchor = dff.index.max() if len(dff) else df.index.max()

    if choice == "TIPS breakeven curve":
        st.plotly_chart(
            big_curve_panel(df, "US TIPS breakeven curve", INFL_BE_TENORS, anchor,
                            y_title="Breakeven (%)"),
            use_container_width=True, key="infl_be", config={"displayModeBar": False})
    elif choice == "ZC inflation swap curve":
        st.plotly_chart(
            big_curve_panel(df, "USD ZC inflation swap curve", INFL_ZCIS_TENORS, anchor,
                            y_title="Inflation swap (%)"),
            use_container_width=True, key="infl_zcis", config={"displayModeBar": False})
    else:
        s = get_series(dff, "INFL_5Y5Y")
        fig = go.Figure()
        if len(s):
            fig.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines",
                line=dict(color=LINE_WHITE, width=1.4),
                fill="tozeroy", fillcolor="rgba(255,255,255,0.05)",
                hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}%<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=[s.index[-1]], y=[s.iloc[-1]], mode="markers",
                marker=dict(color=LINE_WHITE, size=8, line=dict(color=BG, width=1)),
                hoverinfo="skip", showlegend=False))
            fig.add_hline(y=2.0, line=dict(color=ACCENT_AMBER, width=0.8, dash="dash"),
                          annotation_text="Fed target 2%", annotation_position="right",
                          annotation_font=dict(size=10, color=ACCENT_AMBER))
        last_val = s.iloc[-1] if len(s) else float("nan")
        last_str = f"{last_val:+.2f}%" if pd.notna(last_val) else "—"
        fig.update_layout(
            **DARK_LAYOUT, height=520, margin=dict(l=70, r=30, t=50, b=40),
            title=dict(text=(f"<span style='color:#fff;font-weight:700;"
                             f"letter-spacing:0.05em;font-size:14px;'>USD 5Y5Y FORWARD "
                             f"INFLATION SWAP</span> &nbsp;<span style='color:#aaa;"
                             f"font-size:12px;'>{last_str}</span>"),
                       font=dict(size=13), x=0, xanchor="left", y=0.97))
        fig.update_xaxes(showgrid=False, tickfont=dict(size=11, color="#bbb"),
                         linecolor="#222")
        fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False,
                         tickfont=dict(size=11, color="#bbb"), linecolor="#222",
                         ticksuffix="%",
                         title=dict(text="5Y5Y forward (%)", font=dict(size=11, color="#888")))
        st.plotly_chart(fig, use_container_width=True, key="infl_5y5y",
                        config={"displayModeBar": False})


# ===========================================================================
# Page 7 — Data Quality (requirement #2)
# ===========================================================================
def render_data_quality_page(df: pd.DataFrame) -> None:
    section_header(
        "Data Quality",
        f"Per-ticker coverage check · stale = no obs in last {STALE_BDAYS} business days",
    )
    report = validate_data(df, TICKERS)
    summary = quality_summary(report)

    k1, k2, k3, k4 = st.columns(4, gap="small")
    for col, label, value, colour in (
        (k1, "Tickers tracked", summary["total"], "#fff"),
        (k2, "Healthy", summary["healthy"], ACCENT_GREEN),
        (k3, "Stale", summary["stale"], ACCENT_AMBER),
        (k4, "Missing", summary["missing"], ACCENT_RED),
    ):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value" style="color:{colour};font-size:28px;">{value}</div>
                </div>
                """,
                unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    show = st.radio("FILTER", ["All", "Problems only (missing or stale)"],
                    index=0, horizontal=True, key="dq_filter")
    table = report.copy()
    if show.startswith("Problems"):
        table = table[(~table["exists"]) | (table["stale"])]

    table["last_date"] = pd.to_datetime(table["last_date"]).dt.date.astype("string")
    table["last_date"] = table["last_date"].fillna("—")
    table["missing_pct"] = (table["missing_pct"] * 100)
    disp = table.rename(columns={
        "key": "Key", "ticker": "Bloomberg ticker", "exists": "Exists",
        "last_date": "Last date", "missing_pct": "Missing %", "n_obs": "Obs",
        "stale": "Stale"})
    disp = disp[["Key", "Bloomberg ticker", "Exists", "Last date",
                 "Missing %", "Obs", "Stale"]]

    def _row_style(row):
        if not row["Exists"]:
            return ["background-color: rgba(208,72,72,0.12)"] * len(row)
        if row["Stale"]:
            return ["background-color: rgba(217,152,48,0.12)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        disp.style.apply(_row_style, axis=1).format(
            {"Missing %": "{:.1f}", "Obs": "{:,}"}, na_rep="—"),
        hide_index=True, use_container_width=True, height=620)
    st.caption(
        "Red rows = ticker column absent from the dataset · amber rows = stale "
        "(no recent observations). Missing tickers degrade gracefully — the index "
        "and charts simply skip them rather than crashing.")


# ===========================================================================
# Sidebar — navigation + lookback window
# ===========================================================================
PAGES = [
    "1 · Composite Liquidity Index",
    "2 · Money Market Plumbing",
    "3 · Dollar Funding / XCCY Basis",
    "4 · Credit Liquidity",
    "5 · Rates / Real Rates / Curve",
    "6 · Inflation Expectations",
    "7 · Data Quality",
]

with st.sidebar:
    st.markdown(
        """
        <div style="padding:0.5rem 0 0.25rem;">
          <div style="font-size:14px;font-weight:700;letter-spacing:0.08em;
                      color:#fff;text-transform:uppercase;">
            Rates &amp; Liquidity Monitor</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"{df.index.min().date()} → {df.index.max().date()}  ·  "
               f"src: {data_source_label()}")
    st.divider()

    page = st.radio("SECTION", PAGES, index=0, key="nav_page")
    st.divider()

    range_preset = st.radio("LOOKBACK", ["6M", "1Y", "3Y", "5Y", "10Y", "Max", "Custom"],
                            index=2, key="lookback_preset")
    end_date = df.index.max()
    if range_preset == "6M":
        start_date = end_date - pd.DateOffset(months=6)
    elif range_preset == "1Y":
        start_date = end_date - pd.DateOffset(years=1)
    elif range_preset == "3Y":
        start_date = end_date - pd.DateOffset(years=3)
    elif range_preset == "5Y":
        start_date = end_date - pd.DateOffset(years=5)
    elif range_preset == "10Y":
        start_date = end_date - pd.DateOffset(years=10)
    elif range_preset == "Max":
        start_date = df.index.min()
    else:
        custom = st.date_input("Range",
                               value=(end_date - pd.DateOffset(years=3), end_date),
                               min_value=df.index.min().date(),
                               max_value=df.index.max().date())
        if isinstance(custom, tuple) and len(custom) == 2:
            start_date, end_date = pd.Timestamp(custom[0]), pd.Timestamp(custom[1])
        else:
            start_date = end_date - pd.DateOffset(years=3)

    # Live liquidity read-out in the sidebar, always visible.
    if not pd.isna(index_result.latest):
        reg = index_result.latest_regime
        reg_colour = REGIME_COLORS.get(reg, TEXT_DIM)
        st.divider()
        st.markdown(
            f"""
            <div style="font-size:10px;color:#888;letter-spacing:0.1em;
                        text-transform:uppercase;">Liquidity now</div>
            <div style="font-size:26px;font-weight:700;color:{reg_colour};
                        line-height:1.1;">{index_result.latest:.1f}</div>
            <div style="font-size:11px;color:{reg_colour};font-weight:700;
                        text-transform:uppercase;letter-spacing:0.06em;">{reg}</div>
            """,
            unsafe_allow_html=True,
        )

dff = date_filter(df, start_date, end_date)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="padding:0 0 1rem 0;border-bottom:1px solid #1a1a1a;margin-bottom:1rem;">
      <div style="font-size:24px;font-weight:700;letter-spacing:0.06em;color:#fff;
                  text-transform:uppercase;">Rates &amp; Liquidity Monitor</div>
      <div style="font-size:10px;color:#888;letter-spacing:0.1em;text-transform:uppercase;
                  margin-top:4px;">
        Latest: <span style="color:#ccc;font-weight:700;">
        {df.index.max().strftime('%b %d, %Y').upper()}</span> &nbsp;·&nbsp;
        Viewing: {start_date.strftime('%b %Y').upper()} →
        {end_date.strftime('%b %Y').upper()} &nbsp;·&nbsp; {len(dff):,} obs
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
if page == PAGES[0]:
    # Homepage: high-level liquidity summary panel first (requirement #12).
    render_summary_panel(index_result)
    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    render_index_page(df, dff, index_result)
elif page == PAGES[1]:
    render_money_market(dff)
elif page == PAGES[2]:
    render_xccy(dff)
elif page == PAGES[3]:
    render_credit(dff)
elif page == PAGES[4]:
    render_rates_page(df, dff)
elif page == PAGES[5]:
    render_inflation_page(df, dff)
elif page == PAGES[6]:
    render_data_quality_page(df)
