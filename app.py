# app.py — ALPHA TERMINAL v5 — $10M HEDGE FUND AESTHETIC
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime
import requests
from streamlit_local_storage import LocalStorage
import json

# PROFESSIONAL PAGE CONFIG
st.set_page_config(
    page_title="Alpha Terminal v5",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "### Alpha Terminal v5 — Institutional Grade"
    }
)

localS = LocalStorage()

# $10M HEDGE FUND CSS — THIS IS THE MAGIC
st.markdown("""
<style>
    /* Main background */
    .stApp { 
        background: #0e1117; 
        color: #fafafa;
    }
    /* Title */
    h1 { 
        font-family: 'Euclid Circular A', 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -1.5px !important;
        background: linear-gradient(90deg, #00ff88, #00ffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 4.5rem !important;
        margin-bottom: 0 !important;
    }
    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.4rem;
        color: #a0aec0;
        margin-top: -10px;
        margin-bottom: 40px;
    }
    /* Metric cards */
    .stMetric > div { 
        background: #1a1f2e !important;
        border-radius: 16px !important;
        padding: 20px !important;
        border: 1px solid #2d3748 !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .stMetric label { color: #a0aec0 !important; font-weight: 500 !important; }
    .stMetric > div > div:nth-child(2) { color: #00ff88 !important; font-size: 2rem !important; font-weight: 600 !important; }
    /* Sidebar */
    .css-1d391kg { background: #161b26 !important; }
    section[data-testid="stSidebar"] { border-right: 1px solid #2d3748; }
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 20px rgba(102,126,234,0.4); }
    /* AI Report Box */
    .ai-report {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #00ff88;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,255,136,0.2);
    }
    /* Plotly charts */
    .js-plotly-plot { border-radius: 16px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown("<h1>ALPHA TERMINAL v5</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Institutional-Grade AI Trading Intelligence</p>", unsafe_allow_html=True)

# Session state & local storage
for k, v in {"ticker": "NVDA", "watchlist": ["NVDA","AAPL","TSLA","SPY","MSFT","AMD","BTC-USD"], "portfolio": [], "alerts": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

portfolio = localS.getItem("alpha_portfolio_v5")
if portfolio: st.session_state.portfolio = json.loads(portfolio)

ticker = st.session_state.ticker

# Data
@st.cache_data(ttl=180)
def get_data(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="2y", auto_adjust=True)
    info = dict(t.info) if t.info else {}
    return hist, info

hist, info = get_data(ticker)
if hist.empty:
    st.error("No data")
    st.stop()

df = hist.copy()
close = df["Close"]
latest_price = round(close.iloc[-1], 2)
company_name = info.get("longName") or ticker

# ======================== PROFESSIONAL METRICS ROW ========================
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1: st.metric("Price", f"${latest_price:,.2f}", f"{close.pct_change().iloc[-1]:+.2%}")
with col2: st.metric("Volume", f"{df['Volume'].iloc[-1']:,.0f}")
with col3: st.metric("Market Cap", f"${info.get('marketCap',0)/1e9:.1f}B")
with col4: st.metric("P/E", f"{info.get('trailingPE','N/A'):.1f}")
with col5: st.metric("52W High", f"${hist['High'].max():.2f}")
with col6: st.metric("RSI", f"{ta.momentum.RSIIndicator(close).rsi().iloc[-1]:.1f}")

# ======================== PROFESSIONAL CHART ========================
df["EMA20"] = ta.trend.EMAIndicator(close, 20).ema_indicator()
df["EMA50"] = ta.trend.EMAIndicator(close, 50).ema_indicator()
df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
df["MACD"] = ta.trend.MACD(close).macd()

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=[0.55, 0.15, 0.15, 0.15],
    subplot_titles=("Price Action", "Volume", "RSI (14)", "MACD")
)

# Candles + EMAs
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=close, name="Price"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20", line=dict(color="#00ff88", width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA 50", line=dict(color="#ff00ff", width=2)), row=1, col=1)

# Volume
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#4a5568"), row=2, col=1)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#00ffff")), row=3, col=1)
fig.add_hline(y=70, line_dash="dot", line_color="#ff00ff", row=3, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="#ff00ff", row=3, col=1)

# MACD
fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#00ff88")), row=4, col=1)

fig.update_layout(
    height=1000,
    plot_bgcolor='#161b26',
    paper_bgcolor='#0e1117',
    font=dict(color="#fafafa"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ======================== GROK-4 REPORT BUTTON ========================
if st.button("Generate Institutional Grok-4 Report", type="primary"):
    with st.spinner("Grok-4 is analyzing..."):
        try:
            key = st.secrets["GROK_API_KEY"]
            prompt = f"Write a professional hedge-fund style report on {ticker} ({company_name}). Current price ${latest_price:.2f}. Include catalyst, technical setup, bull/bear cases, price target, conviction level. 400 words."
            r = requests.post("https://api.x.ai/v1/chat/completions",
                json={"model": "grok-beta", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
                headers={"Authorization": f"Bearer {key}"}, timeout=60)
            report = r.json()["choices"][0]["message"]["content"]
            st.markdown(f"<div class='ai-report'>{report}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error("Grok-4 credits activating...")

st.markdown("---")
st.caption("Alpha Terminal v5 — Institutional Design • Built with Grok • 2025")
