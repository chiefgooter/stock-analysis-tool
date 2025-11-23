# app.py — ALPHA TERMINAL v7 — FIXED & READY TO DOMINATE
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import requests
import json

# Modular imports
from modules.data import fetch_data
from modules.indicators import add_ta_indicators
from modules.risk import calculate_risk_metrics
from modules.grok import generate_grok_intel

st.set_page_config(page_title="Alpha Terminal v7", layout="wide", initial_sidebar_state="expanded")

# === THEME ===
st.markdown("""
<style>
    .stApp { background: #0e1117; color: #fafafa; }
    h1 { font-size: 5rem; text-align: center; background: linear-gradient(90deg, #00ff88, #00ffff, #ff00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stMetric > div { background: #1a1f2e; border-radius: 16px; padding: 20px; border: 1px solid #2d3748; }
    .ai-report { background: #1a1f2e; border: 3px solid #ff00ff; border-radius: 20px; padding: 25px; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>ALPHA TERMINAL v7</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;color:#00ffff'>Grok-4 Live • Risk Engine • Bloomberg Killer</h3>", unsafe_allow_html=True)

# === TICKER ===
ticker = st.text_input("Ticker", value="NVDA", help="Enter any symbol").upper()

# === DATA ===
hist, info = fetch_data(ticker)
if hist is None:
    st.error(f"No data for {ticker} — try AAPL")
    st.stop()

# === HEADER METRICS (FIXED LINE 48) ===
st.header(f"{info.get('longName', ticker)} — {ticker}")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Price", f"${hist['Close'].iloc[-1]:.2f}")
c2.metric("Change", f"{hist['Close'].pct_change().iloc[-1]:+.2%}")
c3.metric("Volume", f"{hist['Volume'].iloc[-1]:,.0f}")          # ← FIXED HERE
c4.metric("Market Cap", f"${info.get('marketCap',0)/1e9:.1f}B")
c5.metric("Forward P/E", info.get('forwardPE', 'N/A'))
c6.metric("Beta", f"{info.get('beta','N/A'):.2f}")

# === CHART ===
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

# === RISK ===
risk = calculate_risk_metrics(df)
with st.expander("Risk & Performance", expanded=True):
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Sharpe", f"{risk['sharpe']:.2f}")
    r2.metric("Sortino", f"{risk['sortino']:.2f}")
    r3.metric("Max Drawdown", f"{risk['max_dd']:.1f}%")
    r4.metric("95% VaR", f"{risk['var_95']:.2%}")

# === GROK-4 BUTTON ===
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

st.success("Alpha Terminal v7 • LIVE • Ready to make hedge funds switch terminals")
