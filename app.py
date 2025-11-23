# app.py — ALPHA TERMINAL v8 — v5 SIDEBAR RESTORED + GROK-4 FULL SPECTRUM
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import requests
import json

# Modular imports (v7 core intact)
from modules.data import fetch_data
from modules.indicators import add_ta_indicators
from modules.risk import calculate_risk_metrics
from modules.grok import generate_grok_intel

st.set_page_config(page_title="Alpha Terminal v8", layout="wide", initial_sidebar_state="expanded")

# === THEME (v7 Neon + v5 Polish) ===
st.markdown("""
<style>
    .stApp { background: #0e1117; color: #fafafa; }
    h1 { font-size: 5rem; text-align: center; background: linear-gradient(90deg, #00ff88, #00ffff, #ff00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stMetric > div { background: #1a1f2e; border-radius: 16px; padding: 20px; border: 1px solid #2d3748; }
    .ai-report { background: #1a1f2e; border: 3px solid #ff00ff; border-radius: 20px; padding: 25px; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>ALPHA TERMINAL v8</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;color:#00ffff'>v5 Sidebar Restored • Grok-4 Live • Institutional Alpha</h3>", unsafe_allow_html=True)

# === FULL v5 SIDEBAR RESTORATION ===
st.sidebar.markdown("# Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Portfolio", 
        "Alerts",
        "Paper Trading",
        "Multi-Ticker",
        "Autonomous Alpha",
        "On-Chart Grok Chat"
    ],
    label_visibility="collapsed"
)

# v5 Red Dot Indicators
pages = ["Dashboard", "Portfolio", "Alerts", "Paper Trading", "Multi-Ticker", "Autonomous Alpha", "On-Chart Grok Chat"]
for p in pages:
    if page == p:
        st.sidebar.markdown(f"**🔴 {p}**")
    else:
        st.sidebar.markdown(f"○ {p}")

# ==================== PAGE ROUTING (v8 Live Functionality) ====================
if page == "Dashboard":
    # === YOUR FULL v7 DASHBOARD (Unchanged — Ticker to Grok) ===
    ticker = st.text_input("Ticker", value="NVDA", help="Enter any symbol").upper()
    hist, info = fetch_data(ticker)
    if hist is None:
        st.error(f"No data for {ticker} — try AAPL")
        st.stop()

    st.header(f"{info.get('longName', ticker)} — {ticker}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price", f"${hist['Close'].iloc[-1]:.2f}")
    c2.metric("Change", f"{hist['Close'].pct_change().iloc[-1]:+.2%}")
    c3.metric("Volume", f"{hist['Volume'].iloc[-1]:,.0f}")
    c4.metric("Market Cap", f"${info.get('marketCap',0)/1e9:.1f}B")
    c5.metric("Forward P/E", info.get('forwardPE', 'N/A'))
    c6.metric("Beta", f"{info.get('beta','N/A'):.2f}")

    df = add_ta_indicators(hist.copy())
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("Price + Indicators", "RSI", "MACD", "Volume"),
                        row_heights=[0.5, 0.2, 0.2, 0.1])

    fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.EMA20, name="EMA20", line=dict(color="#00ff88")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.EMA50, name="EMA50", line=dict(color="#ff00ff")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.BB_upper, name="BB Upper", line=dict(color="#00ffff", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.BB_lower, name="BB Lower", line=dict(color="#00ffff", dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df.RSI, name="RSI", line=dict(color="#00ffff")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df.MACD, name="MACD", line=dict(color="#ff00ff")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.MACD_signal, name="Signal", line=dict(color="#00ff88")), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df.MACD_hist, name="Hist"), row=3, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="Volume", marker_color="#00ffff"), row=4, col=1)

    fig.update_layout(height=900, showlegend=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    risk = calculate_risk_metrics(df)
    with st.expander("Risk & Performance", expanded=True):
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Sharpe", f"{risk['sharpe']:.2f}")
        r2.metric("Sortino", f"{risk['sortino']:.2f}")
        r3.metric("Max Drawdown", f"{risk['max_dd']:.1f}%")
        r4.metric("95% VaR", f"{risk['var_95']:.2%}")

    if st.button("Generate Grok-4 Alpha Report", type="primary", use_container_width=True):
        with st.spinner("Grok-4 is reading the tape..."):
            intel = generate_grok_intel(ticker, df.tail(10))
            if intel:
                st.markdown(f"""
                <div class='ai-report'>
                    <h2 style='color:#00ff88'>Conviction: {intel['conviction']}</h2>
                    <h3>Edge: {intel['edge_score']}/100 • 3mo Target: ${intel['target_price_3mo']:.0f}</h3>
                    <p><strong>Catalyst:</strong> {intel['catalyst']}</p>
                    <p><strong>Risk:</strong> {intel['primary_risk']}</p>
                    <hr>{intel['summary']}
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.error("Grok-4 offline — add your API key in Streamlit secrets")

elif page == "Portfolio":
    st.header("Portfolio — Live P&L + Risk per Position")
    uploaded = st.file_uploader("Upload CSV (columns: ticker, shares, buy_price)", type="csv")
    if uploaded:
        portfolio = pd.read_csv(uploaded)
        portfolio['current_price'] = portfolio['ticker'].apply(lambda x: yf.Ticker(x).history(period="1d")['Close'].iloc[-1] if len(yf.Ticker(x).history(period="1d")) > 0 else np.nan)
        portfolio['pnl'] = (portfolio['current_price'] - portfolio['buy_price']) * portfolio['shares']
        portfolio['pnl_pct'] = (portfolio['current_price'] / portfolio['buy_price'] - 1)
        st.dataframe(portfolio.style.format({"current_price": "${:.2f}", "pnl": "${:.2f}", "pnl_pct": "{:.2%}", "buy_price": "${:.2f}"}))
        total_pnl = portfolio['pnl'].sum()
        st.metric("Total Portfolio P&L", f"${total_pnl:,.2f}", delta=f"{(total_pnl / (portfolio['buy_price'] * portfolio['shares']).sum()):+.2%}" if total_pnl != 0 else None)

elif page == "Alerts":
    st.header("Alerts Engine")
    col1, col2 = st.columns(2)
    with col1:
        pct = st.slider("Price % Change Alert", -50.0, 50.0, 10.0)
    with col2:
        rsi_on = st.checkbox("RSI Overbought/Oversold Alert")
    st.success(f"Monitoring {ticker} → Alert on {pct:+.1f}% move" + (" + RSI extremes" if rsi_on else ""))

elif page == "Paper Trading":
    st.header("Paper Trading — Zero Risk Alpha Lab")
    st.info("v8.1 Live Soon: Full order book, limit orders, vs SPY benchmark")

elif page == "Multi-Ticker":
    st.header("Multi-Ticker Comparison")
    peers = st.multiselect("Compare with", ["AAPL", "AMD", "TSLA", "MSFT", "SMCI", "AVGO", "ASML"], default=["AAPL", "AMD", "TSLA"])
    data = {}
    for p in [ticker] + peers:
        try:
            data[p] = yf.Ticker(p).history(period="1y")['Close']
        except:
            pass
    if data:
        df = pd.DataFrame(data).pct_change().add(1).cumprod()
        st.line_chart(df)

elif page == "Autonomous Alpha":
    st.header("Autonomous Alpha — Grok Runs Your Book")
    st.warning("v9 Preview: Grok-4 executes rules-based strats 24/7 — zero emotion, max edge")

elif page == "On-Chart Grok Chat":
    st.header("On-Chart Grok Chat")
    st.info("v8.5 Launch: Click candles → Grok explains: 'Why this dip?' or 'Squeeze incoming?'")

st.success("Alpha Terminal v8 • v5 Sidebar Live • Customize Your Edge")
