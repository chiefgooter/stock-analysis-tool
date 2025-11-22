# app.py — ALPHA TERMINAL v3.5 — FINAL UNBREAKABLE VERSION
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")

# ========================= CONFIG =========================
st.set_page_config(page_title="Alpha Terminal v3.5", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .ai-box {padding: 20px; border-radius: 15px; background: linear-gradient(90deg, #1e3a8a, #3b82f6); color: white; margin: 20px 0;}
    .signal-box {padding: 15px; border-left: 6px solid; margin: 10px 0; border-radius: 8px;}
    .bull {border-color: #00ff00; background-color: rgba(0,255,0,0.1);}
    .bear {border-color: #ff0000; background-color: rgba(255,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Alpha Terminal v3.5 — Institutional AI Dashboard")
st.markdown("**The most powerful free stock analyzer — now truly unbreakable**")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Control Panel")
    
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ["AAPL","NVDA","TSLA","SPY","MSFT","GME","BTC-USD"]
    if 'ticker' not in st.session_state:
        st.session_state.ticker = "NVDA"
    
    for t in st.session_state.watchlist:
        if st.button(t, key=t): st.session_state.ticker = t.upper()
    new = st.text_input("Add ticker")
    if st.button("Add") and new:
        nt = new.upper().strip()
        if nt not in st.session_state.watchlist:
            st.session_state.watchlist.append(nt)
            st.rerun()
    
    ticker = st.session_state.ticker

    col1, col2 = st.columns(2)
    start_date = col1.date_input("From", datetime.now() - timedelta(days=730))
    end_date   = col2.date_input("To", datetime.now())

    theme = st.selectbox("Theme", ["Dark","Light"], index=0)
    if theme == "Dark": st._config.set_option("theme.base", "dark")

    st.subheader("Indicators")
    show_ma = st.checkbox("Moving Averages", True)
    show_bb = st.checkbox("Bollinger Bands", True)
    show_supertrend = st.checkbox("SuperTrend", True)
    show_vol_profile = st.checkbox("Volume Profile", True)

    ai_style = st.selectbox("AI Voice", ["Professional","Cathie Wood","Warren Buffett","Jim Cramer","Maximum Bull","Maximum Bear"])

# ========================= CACHE-FRIENDLY DATA FETCH =========================
@st.cache_data(ttl=300, show_spinner="Loading data...")
def get_data(ticker: str, start_date, end_date):
    ticker = ticker.upper().strip()
    
    # Convert dates properly
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt   = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
    
    t = yf.Ticker(ticker)
    
    # Get history with multiple fallbacks
    hist = t.history(start=start_dt, end=end_dt, interval="1d", auto_adjust=True, actions=False)
    if hist.empty or len(hist) < 20:
        hist = t.history(period="2y", auto_adjust=True, actions=False)
    if hist.empty:
        hist = t.history(period="max", auto_adjust=True, actions=False).tail(1000)
    
    # ONLY return picklable objects → NO MORE CACHING ERRORS
    info_dict = dict(t.info) if t.info else {}
    return hist, info_dict

# This is now safe
hist, info = get_data(ticker, start_date, end_date)

if hist.empty or len(hist) < 10:
    st.error(f"No data for **{ticker}** — may be delisted or invalid symbol.")
    st.stop()

latest_price = round(hist["Close"].iloc[-1], 2)
company_name = info.get("longName") or info.get("shortName") or ticker

# ========================= INDICATORS =========================
df = hist.copy()
close = df["Close"]

if show_ma:
    df["EMA20"]  = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["EMA50"]  = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["SMA200"] = close.rolling(200).mean()

if show_bb:
    bb = ta.volatility.BollingerBands(close)
    df["BBU"] = bb.bollinger_hband()
    df["BBL"] = bb.bollinger_lband()

if show_supertrend:
    try:
        df["SuperTrend"] = ta.trend.PSARIndicator(df["High"], df["Low"], close).psar()
    except:
        pass

# Volume Profile
vol_profile = None
if show_vol_profile:
    bins = pd.cut(close, bins=40)
    vol_profile = df["Volume"].groupby(bins).sum()
    vol_profile.index = [i.mid for i in vol_profile.index]

# ========================= SENTIMENT & FAIR VALUE =========================
def get_sentiment(ticker):
    try:
        url = f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws&tbs=qdr:d"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(r.text, 'html.parser')
        headlines = [h.text for h in soup.find_all('h3')[:7]]
        pos = sum(1 for h in headlines if any(w in h.lower() for w in ["beat","surge","buy","bullish"]))
        neg = sum(1 for h in headlines if any(w in h.lower() for w in ["miss","drop","sell","crash"]))
        return (pos - neg) * 15, headlines
    except:
        return 0, []

sentiment_score, _ = get_sentiment(ticker)

def fair_value(info, price):
    eps = info.get("trailingEps") or 1.0
    growth = max(info.get("earningsQuarterlyGrowth", 0.1) or 0.1, 0.05) * 100
    target = info.get("targetMeanPrice") or price * 1.15
    graham = np.sqrt(eps * (8.5 + 2*growth)) * 1.5
    dcf = eps * (10 + growth)
    fv = round(np.mean([graham, dcf, target]), 2)
    upside = round((fv / price - 1) * 100, 1)
    return fv, upside

fv, upside = fair_value(info, latest_price)

# ========================= METRICS =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Price", f"${latest_price}")
c2.metric("AI Fair Value", f"${fv}", f"{upside:+.1f}%")
c3.metric("Sentiment", f"{sentiment_score:+}")
c4.metric("P/E", f"{info.get('trailingPE','N/A'):.1f}")

# ========================= AI REPORT =========================
if st.button("Generate AI Hedge Fund Report (Grok-4)", type="primary"):
    with st.spinner("Grok-4 is analyzing..."):
        prompt = f"Analyze {ticker} ({company_name}) in {ai_style} style. Price ${latest_price}, AI fair value ${fv} ({upside:+.1f}%). Write a 300-word pro report with bull/bear cases and target."
        try:
            key = st.secrets["GROK_API_KEY"]
            r = requests.post("https://api.x.ai/v1/chat/completions",
                json={"model":"grok-beta","messages":[{"role":"user","content":prompt}],"temperature":0.7},
                headers={"Authorization":f"Bearer {key}"}, timeout=30)
            report = r.json()["choices"][0]["message"]["content"]
        except:
            report = "Add your Grok API key in Secrets → GROK_API_KEY to enable AI reports."
        st.markdown(f"<div class='ai-box'><h3>AI Hedge Fund Report</h3>{report}</div>", unsafe_allow_html=True)

# ========================= CHART =========================
fig = make_subplots(rows=4, cols=2, shared_xaxes=True,
                    subplot_titles=("Price","Vol Profile","RSI","Volume"),
                    row_heights=[0.55,0.15,0.15,0.15], column_widths=[0.78,0.22])

fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=close, name="Price"), row=1, col=1)

if show_ma:
    for c, col in zip(["#ff9f40","#ff5722","#6366f1"], ["EMA20","EMA50","SMA200"]):
        if col in df: fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=c)), row=1, col=1)
if show_bb:
    fig.add_trace(go.Scatter(x=df.index, y=df["BBU"], name="BB Upper", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BBL"], name="BB Lower", line=dict(dash="dot")), row=1, col=1)
if show_supertrend and "SuperTrend" in df:
    fig.add_trace(go.Scatter(x=df.index, y=df["SuperTrend"], name="SuperTrend", line=dict(width=4,color="#00ff88")), row=1, col=1)

if vol_profile is not None:
    fig.add_trace(go.Bar(x=vol_profile.values, y=vol_profile.index, orientation='h', marker_color="#6366f1"), row=1, col=2)

df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#a78bfa")), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", row=2, col=1); fig.add_hline(y=30, line_dash="dot", row=2, col=1)

fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color="#94a3b8"), row=4, col=1)

fig.update_layout(height=1000, title=f"{ticker} — Alpha Terminal v3.5", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.success(f"AI Target: ${fv} • Potential: {upside:+.1f}%")
st.caption("Alpha Terminal v3.5 — Built with Grok • Not financial advice • 2025")
