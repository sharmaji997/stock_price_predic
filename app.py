"""
app.py
------
Streamlit UI for the Stock Buy/Sell/Hold predictor.

Layout is a sidebar-driven "trading terminal" style dashboard (custom CSS
theme + HTML card components) rather than the plain top-tabs layout, with
three pages reachable from the sidebar:
  1. Stock Analysis        - candlestick chart + indicators + prediction
  2. Watchlist Suggestions - scans a list of stocks and ranks them
  3. Model Insights         - accuracy, confusion matrix, feature importances

Run with:
    streamlit run app.py
"""

import pickle
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from config import WATCHLIST
from features import FEATURE_COLUMNS, LABEL_NAMES, add_technical_indicators

MODEL_DIR = Path(__file__).parent / "model"

st.set_page_config(page_title="Signal Desk", layout="wide", page_icon="◆",
                    initial_sidebar_state="expanded")

# ---------------------------------------------------------------- theme / css
PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#4fd1c5"
BUY_COLOR = "#22c55e"
HOLD_COLOR = "#eab308"
SELL_COLOR = "#ef4444"
BG = "#0e1117"
PANEL = "#161b26"
BORDER = "#232a3b"

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif;
}}
.stApp {{
    background: {BG};
}}
section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {BORDER};
}}
.desk-brand {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 700;
    color: {ACCENT};
    letter-spacing: 1px;
    padding: 0.4rem 0 0.2rem 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 0.8rem;
}}
.desk-sub {{
    color: #8b94a8;
    font-size: 0.8rem;
    margin-bottom: 1.2rem;
}}
.card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.9rem;
}}
.stat-card {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: left;
}}
.stat-label {{
    color: #8b94a8;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
}}
.stat-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8ebf2;
}}
.stat-delta-up {{ color: {BUY_COLOR}; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; }}
.stat-delta-down {{ color: {SELL_COLOR}; font-size: 0.85rem; font-family: 'JetBrains Mono', monospace; }}
.signal-pill {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.35rem 1.1rem;
    border-radius: 999px;
    letter-spacing: 0.04em;
}}
.pill-buy {{ background: rgba(34,197,94,0.15); color: {BUY_COLOR}; border: 1px solid {BUY_COLOR}; }}
.pill-hold {{ background: rgba(234,179,8,0.15); color: {HOLD_COLOR}; border: 1px solid {HOLD_COLOR}; }}
.pill-sell {{ background: rgba(239,68,68,0.15); color: {SELL_COLOR}; border: 1px solid {SELL_COLOR}; }}
.section-heading {{
    font-family: 'JetBrains Mono', monospace;
    color: {ACCENT};
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.4rem 0 0.5rem 0;
    border-left: 3px solid {ACCENT};
    padding-left: 0.5rem;
}}
.watch-row-buy {{ border-left: 3px solid {BUY_COLOR}; }}
.watch-row-sell {{ border-left: 3px solid {SELL_COLOR}; }}
.disclaimer {{
    font-size: 0.78rem;
    color: #8b94a8;
    border-top: 1px solid {BORDER};
    padding-top: 0.8rem;
    margin-top: 1.5rem;
}}
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- artifacts
@st.cache_resource
def load_artifacts():
    with open(MODEL_DIR / "signal_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    with open(MODEL_DIR / "metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    return model, scaler, feature_columns, metrics


@st.cache_data(ttl=3600)
def fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def predict_signal(df: pd.DataFrame, model, scaler, feature_columns):
    """Return (label, probabilities dict) for the most recent row."""
    featurized = add_technical_indicators(df).dropna(subset=feature_columns)
    if featurized.empty:
        return None, None
    latest = featurized.iloc[[-1]][feature_columns].values
    latest_scaled = scaler.transform(latest)
    pred = model.predict(latest_scaled)[0]
    proba = model.predict_proba(latest_scaled)[0]
    proba_dict = {LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)}
    return LABEL_NAMES[pred], proba_dict


def pill_class(label: str) -> str:
    return {"Buy": "pill-buy", "Hold": "pill-hold", "Sell": "pill-sell"}.get(label, "pill-hold")


def stat_card(label: str, value: str, delta: str = None, delta_up: bool = True):
    delta_html = ""
    if delta is not None:
        cls = "stat-delta-up" if delta_up else "stat-delta-down"
        arrow = "▲" if delta_up else "▼"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def themed_fig_layout(fig, height):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(family="JetBrains Mono, monospace", color="#c7cedd", size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.03, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ---------------------------------------------------------------- load model
try:
    model, scaler, feature_columns, metrics = load_artifacts()
    model_ready = True
except FileNotFoundError:
    model_ready = False

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="desk-brand">◆ SIGNAL DESK</div>', unsafe_allow_html=True)
    st.markdown('<div class="desk-sub">ML-based Buy / Hold / Sell terminal</div>',
                unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Stock Analysis", "Watchlist Suggestions", "Model Insights"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-heading">Legend</div>', unsafe_allow_html=True)
    st.markdown(
        f'<span class="signal-pill pill-buy">BUY</span> &nbsp;'
        f'<span class="signal-pill pill-hold">HOLD</span> &nbsp;'
        f'<span class="signal-pill pill-sell">SELL</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer">Educational ML project — Random Forest on '
        'technical indicators. Not financial advice.</div>',
        unsafe_allow_html=True,
    )

if not model_ready:
    st.error(
        "No trained model found in `model/`. Run `python train_model.py` first "
        "to download data and train the model, then relaunch the app."
    )
    st.stop()

# ============================================================== PAGE: ANALYSIS
if page == "Stock Analysis":
    top_l, top_r = st.columns([3, 1])
    with top_l:
        ticker = st.text_input("Ticker", value="AAPL",
                                placeholder="e.g. AAPL, TSLA, RELIANCE.NS",
                                label_visibility="collapsed").strip().upper()
    with top_r:
        period = st.selectbox("History", ["6mo", "1y", "2y", "5y"], index=1,
                               label_visibility="collapsed")

    if ticker:
        with st.spinner(f"Pulling {ticker} ..."):
            hist = fetch_history(ticker, period)

        if hist.empty:
            st.warning("No data found for that ticker. Check the symbol and try again.")
        else:
            featurized = add_technical_indicators(hist)
            label, proba = predict_signal(hist, model, scaler, feature_columns)

            last_close = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2])
            change_pct = (last_close / prev_close - 1) * 100

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                stat_card("Last Close", f"{last_close:,.2f}", f"{change_pct:+.2f}%", change_pct >= 0)
            with c2:
                conf = f"{max(proba.values()) * 100:.1f}%" if proba else "—"
                stat_card("Confidence", conf)
            with c3:
                stat_card("RSI (14)", f"{featurized['rsi_14'].iloc[-1]:.1f}")
            with c4:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-label">Model Signal</div>
                    <div style="margin-top:0.3rem;">
                        <span class="signal-pill {pill_class(label) if label else 'pill-hold'}">
                            {label.upper() if label else 'N/A'}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if proba:
                st.markdown('<div class="section-heading">Signal Probabilities</div>',
                             unsafe_allow_html=True)
                proba_df = pd.DataFrame(
                    {"Signal": list(proba.keys()), "Probability": list(proba.values())}
                ).sort_values("Probability", ascending=False)
                st.dataframe(proba_df.style.format({"Probability": "{:.1%}"}),
                             hide_index=True, use_container_width=True)

            st.markdown('<div class="section-heading">Price &amp; Indicators</div>',
                         unsafe_allow_html=True)
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.2, 0.25],
                vertical_spacing=0.04,
                subplot_titles=(f"{ticker}", "MACD", "RSI"),
            )
            fig.add_trace(go.Candlestick(
                x=featurized.index, open=featurized["Open"], high=featurized["High"],
                low=featurized["Low"], close=featurized["Close"], name="Price",
                increasing_line_color=BUY_COLOR, decreasing_line_color=SELL_COLOR,
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=featurized.index, y=featurized["sma_10"],
                                      name="SMA 10", line=dict(width=1, color=ACCENT)), row=1, col=1)
            fig.add_trace(go.Scatter(x=featurized.index, y=featurized["sma_50"],
                                      name="SMA 50", line=dict(width=1, color="#9d7bff")), row=1, col=1)
            fig.add_trace(go.Scatter(x=featurized.index, y=featurized["bb_upper"],
                                      name="BB Upper", line=dict(width=1, dash="dot", color="#4b5568")), row=1, col=1)
            fig.add_trace(go.Scatter(x=featurized.index, y=featurized["bb_lower"],
                                      name="BB Lower", line=dict(width=1, dash="dot", color="#4b5568")), row=1, col=1)

            fig.add_trace(go.Bar(x=featurized.index, y=featurized["macd_hist"], name="MACD Hist",
                                  marker_color=ACCENT), row=2, col=1)
            fig.add_trace(go.Scatter(x=featurized.index, y=featurized["macd"],
                                      name="MACD", line=dict(width=1, color="#e8ebf2")), row=2, col=1)
            fig.add_trace(go.Scatter(x=featurized.index, y=featurized["macd_signal"],
                                      name="Signal", line=dict(width=1, color="#eab308")), row=2, col=1)

            fig.add_trace(go.Scatter(x=featurized.index, y=featurized["rsi_14"],
                                      name="RSI", line=dict(width=1, color="#9d7bff")), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color=SELL_COLOR, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color=BUY_COLOR, row=3, col=1)

            fig.update_xaxes(rangeslider_visible=False, gridcolor=BORDER)
            fig.update_yaxes(gridcolor=BORDER)
            themed_fig_layout(fig, 780)
            st.plotly_chart(fig, use_container_width=True)

            vol_fig = go.Figure(go.Bar(x=featurized.index, y=featurized["Volume"],
                                        name="Volume", marker_color="#4b5568"))
            themed_fig_layout(vol_fig, 220)
            vol_fig.update_xaxes(gridcolor=BORDER)
            vol_fig.update_yaxes(gridcolor=BORDER)
            st.plotly_chart(vol_fig, use_container_width=True)

# ============================================================== PAGE: WATCHLIST
elif page == "Watchlist Suggestions":
    st.markdown('<div class="section-heading">Watchlist Scanner</div>', unsafe_allow_html=True)
    custom = st.text_input(
        "Tickers", value="", placeholder="Comma-separated, e.g. AAPL, TSLA, TCS.NS "
        "(leave blank for default watchlist)", label_visibility="collapsed"
    )
    tickers_to_scan = (
        [t.strip().upper() for t in custom.split(",") if t.strip()]
        if custom.strip() else WATCHLIST
    )

    run = st.button("Run Scan", type="primary")

    if run:
        rows = []
        progress = st.progress(0.0)
        for i, tkr in enumerate(tickers_to_scan):
            hist = fetch_history(tkr, "1y")
            if not hist.empty:
                label, proba = predict_signal(hist, model, scaler, feature_columns)
                if label:
                    last_close = float(hist["Close"].iloc[-1])
                    day_change = (last_close / float(hist["Close"].iloc[-2]) - 1) * 100
                    rows.append({
                        "Ticker": tkr,
                        "Last Close": round(last_close, 2),
                        "Day Change %": round(day_change, 2),
                        "Signal": label,
                        "Confidence": round(max(proba.values()) * 100, 1),
                    })
            progress.progress((i + 1) / len(tickers_to_scan))
        progress.empty()

        if rows:
            result_df = pd.DataFrame(rows).sort_values(
                by=["Signal", "Confidence"],
                key=lambda col: col.map({"Buy": 0, "Hold": 1, "Sell": 2}) if col.name == "Signal" else col,
                ascending=[True, False],
            )

            buy_ct = (result_df["Signal"] == "Buy").sum()
            hold_ct = (result_df["Signal"] == "Hold").sum()
            sell_ct = (result_df["Signal"] == "Sell").sum()
            b1, b2, b3 = st.columns(3)
            with b1:
                stat_card("Buy-rated", str(buy_ct))
            with b2:
                stat_card("Hold-rated", str(hold_ct))
            with b3:
                stat_card("Sell-rated", str(sell_ct))

            def highlight_signal(val):
                colors = {"Buy": f"background-color: rgba(34,197,94,0.15); color:{BUY_COLOR}",
                          "Hold": f"background-color: rgba(234,179,8,0.15); color:{HOLD_COLOR}",
                          "Sell": f"background-color: rgba(239,68,68,0.15); color:{SELL_COLOR}"}
                return colors.get(val, "")

            st.markdown('<div class="section-heading">Ranked Results</div>', unsafe_allow_html=True)
            st.dataframe(
                result_df.style.map(highlight_signal, subset=["Signal"])
                .format({"Confidence": "{:.1f}%", "Day Change %": "{:+.2f}%"}),
                hide_index=True, use_container_width=True,
            )
        else:
            st.warning("Couldn't fetch data for any of the given tickers.")
    else:
        st.markdown(
            f'<div class="card">Default watchlist: '
            f'<span style="font-family:\'JetBrains Mono\',monospace;color:{ACCENT}">'
            f'{", ".join(WATCHLIST)}</span><br><br>'
            f'Click <b>Run Scan</b> to rate every stock on the list.</div>',
            unsafe_allow_html=True,
        )

# ============================================================== PAGE: INSIGHTS
elif page == "Model Insights":
    st.markdown('<div class="section-heading">Held-out Test Performance</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        stat_card("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    with c2:
        stat_card("Test Rows", str(metrics["n_test_rows"]))

    st.markdown('<div class="section-heading">Classification Report</div>', unsafe_allow_html=True)
    st.code(metrics["classification_report"])

    st.markdown('<div class="section-heading">Confusion Matrix</div>', unsafe_allow_html=True)
    cm_df = pd.DataFrame(metrics["confusion_matrix"],
                          index=["Actual Sell", "Actual Hold", "Actual Buy"],
                          columns=["Pred Sell", "Pred Hold", "Pred Buy"])
    st.dataframe(cm_df, use_container_width=True)

    st.markdown('<div class="section-heading">Feature Importance</div>', unsafe_allow_html=True)
    fi = pd.Series(metrics["feature_importances"]).sort_values(ascending=True)
    fi_fig = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color=ACCENT))
    fi_fig.update_xaxes(gridcolor=BORDER)
    fi_fig.update_yaxes(gridcolor=BORDER)
    themed_fig_layout(fi_fig, 450)
    st.plotly_chart(fi_fig, use_container_width=True)

st.markdown(
    '<div class="disclaimer">⚠️ Signal Desk is a machine-learning class project '
    'trained on historical price patterns. It is not investment advice — always '
    'do your own research.</div>',
    unsafe_allow_html=True,
)
