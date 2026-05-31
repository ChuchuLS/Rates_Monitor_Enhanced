"""
config/theme.py
===============
Central visual configuration for the dashboard. Everything here is about
*look and feel* so that the rest of the codebase never hard-codes a colour or a
font. The palette is the original OFR-style institutional dark theme — clean,
low-saturation, suitable for a macro / rates research desk (requirement #13).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core palette — OFR dark theme
# ---------------------------------------------------------------------------
BG = "#0a0a0a"           # page / chart background
PANEL_BG = "#0f0f0f"     # slightly lighter panel background
LINE_WHITE = "#ffffff"   # primary series colour
GRID = "rgba(255,255,255,0.05)"
TEXT_DIM = "#888"
TEXT_VERY_DIM = "#666"

# Accent line colours for multi-series charts (muted, not retail-bright)
ACCENT_GREEN = "#5fb04f"
ACCENT_RED = "#d04848"
ACCENT_AMBER = "#d99830"
ACCENT_CYAN = "#4fa8b8"
ACCENT_PURPLE = "#9080d0"

# Brighter signal colours reserved for "good / bad" deltas
POS_GREEN = "#67c757"
NEG_RED = "#e64545"

# OFR-style interpretation note boxes
NOTE_RED_BG = "rgba(120,30,30,0.85)"
NOTE_RED_BORDER = "#C04040"
NOTE_RED_TEXT = "#FFB0B0"
NOTE_GREEN_BG = "rgba(30,80,40,0.85)"
NOTE_GREEN_BORDER = "#40A060"
NOTE_GREEN_TEXT = "#B0E8B8"

# ---------------------------------------------------------------------------
# Liquidity-regime colours (used by the Composite Liquidity Index section)
# Looser conditions are green, tighter conditions shade toward red.
# ---------------------------------------------------------------------------
REGIME_COLORS = {
    "Loose":   "#5fb04f",
    "Neutral": "#9aa0a6",
    "Tight":   "#d99830",
    "Stress":  "#d04848",
}

# Curve-regime colours (rates section) — matches the Bloomberg Studio look
CURVE_REGIME_COLORS = {
    "bull_steepener":  "#67c757",
    "bear_steepener":  "#e64545",
    "steepener_twist": "#f0a020",
    "bull_flattener":  "#9fc8e8",
    "bear_flattener":  "#5e95c2",
    "flattener_twist": "#f0e040",
    "none":            "#444444",
}
CURVE_REGIME_LABELS = {
    "bull_steepener":  "Bull steepener",
    "bear_steepener":  "Bear steepener",
    "steepener_twist": "Steepener twist",
    "bull_flattener":  "Bull flattener",
    "bear_flattener":  "Bear flattener",
    "flattener_twist": "Flattener twist",
}

# Bucket colours for the contribution chart (one stable colour per sub-index)
BUCKET_COLORS = {
    "central_bank": "#9bd62a",
    "money_market": "#4fa8b8",
    "xccy":         "#9080d0",
    "credit":       "#d99830",
    "market_liq":   "#d04848",
}

# ---------------------------------------------------------------------------
# Shared Plotly layout — applied to (almost) every chart for consistency
# ---------------------------------------------------------------------------
DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="Inter, system-ui, sans-serif", size=10, color=TEXT_DIM),
    hovermode="x unified",
    showlegend=False,
)


def page_css() -> str:
    """Return the global CSS block injected once at app start."""
    return """
    <style>
    .stApp {
        background-color: #0a0a0a;
        color: #e0e0e0;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #1a1a1a;
    }
    section[data-testid="stSidebar"] * { color: #ccc !important; }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        letter-spacing: 0.04em;
        font-family: 'Inter', system-ui, sans-serif !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #888 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-size: 11px !important;
    }
    .stMarkdown p { color: #ccc; }
    hr { border-color: #1a1a1a !important; margin: 0.75rem 0 !important; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        color: #ccc !important; font-size: 13px !important;
    }
    [data-testid="stPlotlyChart"] { background-color: transparent !important; }
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }
    /* Hide Streamlit Cloud chrome */
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    [data-testid="stDeployButton"] { display: none !important; }
    [data-testid="stActionButtonIcon"] { display: none !important; }
    [data-testid="stToolbarActions"] { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    .viewerBadge_container__1QSob { display: none !important; }
    .viewerBadge_link__1S137 { display: none !important; }
    /* Section header */
    .section-header {
        background: #0a0a0a; padding: 0.6rem 0; margin: 0.5rem 0 0.25rem 0;
        border-bottom: 1px solid #1a1a1a;
    }
    .section-title {
        font-size: 18px; font-weight: 700; letter-spacing: 0.06em;
        color: #ffffff; text-transform: uppercase;
    }
    .section-sub {
        font-size: 10px; color: #888; letter-spacing: 0.08em;
        text-transform: uppercase; margin-top: 2px;
    }
    /* KPI metric cards used on the liquidity summary panel */
    .kpi-card {
        background: #0f0f0f; border: 1px solid #1a1a1a; border-radius: 6px;
        padding: 0.9rem 1rem; height: 100%;
    }
    .kpi-label {
        font-size: 10px; color: #888; letter-spacing: 0.1em;
        text-transform: uppercase; margin-bottom: 6px;
    }
    .kpi-value { font-size: 30px; font-weight: 700; line-height: 1; }
    .kpi-sub { font-size: 11px; color: #aaa; margin-top: 6px; }
    </style>
    """
