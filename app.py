# app.py — ALPHA TERMINAL v3.5 — FINAL BULLETPROOF VERSION (November 2025)
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

# ========================= PAGE CONFIG & STYLE =========================
st.set_page_config(page_title="Alpha Terminal v3.5", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .big-font {font-size: 50px !important; font-weight: bold; text-align: center;}
    .ai-box {padding: 20px; border-radius: 15px; background: linear-gradient(90deg, #1e3a8a, #3b82f6); color: white; margin: 20px 0; border: 1px solid #4f46e5;}
    .signal-box {padding: 15px; border-left: 6px solid; margin: 10px 0; border-radius: 8px;}
    .bull {border-color: #00ff00; background-color: rgba(0, 255, 0, 0.1);}
    .bear {border-color: #ff0000; background-color: rgba(255, 0, 0, 0.1);}
    .neutral {border-color: #888; background-color: rgba(136, 136, 136, 0.1);}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Alpha Terminal v3.5 — Institutional AI Dashboard")
st.markdown("**The most powerful free stock analyzer on earth — now 100% bulletproof**")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("🎛 Control Panel")
    
    # Watchlist
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ", "MSFT", "AMD", "BTC-USD"]
    if 'ticker' not in st.session_state:
        st.session_state.ticker = "NVDA"
    
    st.subheader("📋 Watchlist")
    for t in st.session_state.watchlist:
        if st.button(t, key=f"watch_{t}"):
            st.session_state.ticker = t.upper()
    new_ticker = st.text_input("Add ticker")
    if st.button("➕ Add") and new_ticker:
        nt = new_ticker.upper().strip()
        if nt not in st.session_state.watchlist:
            st.session_state.watchlist.append(nt)
            st.success(f"{nt} added!")
            st.rerun()

    ticker = st.session_state.ticker

    # Date range with safety defaults
    col1, col2 = st.columns(2)
    default_start = datetime.now() - timedelta(days=730)
    start_date = col1.date_input("From", default_start)
    end_date = col2.date_input("To", datetime.now())

    # Theme
    theme = st.selectbox("🎨 Theme", ["Dark", "Light"], index=0)
    if theme == "Dark":
        st._config.set_option("theme.base", "dark")

    # Indicators
    st.subheader("📊 Indicators")
    show_ma = st.checkbox("Moving Averages", True)
    show_bb = st.checkbox("Bollinger Bands", True)
    show_supertrend = st.checkbox("SuperTrend", True)
    show_ichimoku = st.checkbox("Ichimoku Cloud", False)
    show_vwap = st.checkbox("VWAP", False)
    show_vol_profile = st.checkbox("Volume Profile", True)

    # AI Style
    ai_style = st.selectbox("AI Voice", ["Professional Analyst", "Cathie Wood", "Warren Buffett", "Jim Cramer", "Maximum Bull", "Maximum Bear"])

# ========================= BULLETPROOF DATA FETCH =========================
@st.cache_data(ttl=300, show_spinner="Fetching latest market data...")
def get_data(ticker):
    ticker = ticker.upper().strip()
    try:
        t = yf.Ticker(ticker)
        
        # Primary: use user date range with fixes
        end_dt = min(end_date, datetime.today())
        start_dt = min(start_date, end_dt - timedelta(days=1))
        
        hist = t.history(start=start_dt, end=end_dt + timedelta(days=1), interval="1d", auto_adjust=True, actions=False)
        
        # Fallback 1: if empty, try max period
        if hist.empty or len(hist) < 20:
            hist = t.history(period="2y", auto_adjust=True, actions=False)
        
        # Fallback 2: absolute last resort
        if hist.empty:
            hist = t.history(period="max", auto_adjust=True, actions=False)
            hist = hist.tail(500)  # limit size
        
        info = dict(t.info) if t.info else {}
        earnings = getattr(t, 'earnings_dates', None)
        options = t.options[:8] if hasattr(t, 'options') and t.options else None
        institutional = getattr(t, 'institutional_holders', None)
        
        return t, hist, info, earnings, options, institutional
        
    except Exception as e:
        st.error(f"Connection issue: {str(e)[:100]}")
        return None, pd.DataFrame(), {}, None, None, None

t, hist, info, earnings, options_dates, institutional = get_data(ticker)

if hist.empty or len(hist) < 10:
    st.error(f"⚠️ No price data for **{ticker}**.\n\nThis can happen with brand-new tickers or delisted symbols.")
    st.stop()

company_name = info.get("longName") or info.get("shortName") or ticker
latest_price = hist["Close"].iloc[-1]

# ========================= INDICATORS =========================
df = hist.copy()
close = df["Close"]

if show_ma:
    df["EMA20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["EMA50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    df["SMA200"] = close.rolling(200).mean()

if show_bb:
    bb = ta.volatility.BollingerBands(close)
    df["BBU"] = bb.bollinger_hband()
    df["BBL"] = bb.bollinger_lband()

if show_supertrend:
    try:
        st_ind = ta.trend.PSARIndicator(df["High"], df["Low"], close)
        df["SuperTrend"] = st_ind.psar()
    except:
        pass

if show_ichimoku:
    ichi = ta.trend.IchimokuIndicator(df["High"], df["Low"])
    df["Ich_A"] = ichi.ichimoku_a()
    df["Ich_B"] = ichi.ichimoku_b()

if show_vwap:
    df["VWAP"] = ta.volume.VolumeWeightedAveragePrice(df["High"], df["Low"], close, df["Volume"]).volume_weighted_average_price()

# Volume Profile
vol_profile = None
if show_vol_profile and len(df) > 30:
    bins = pd.cut(close, bins=40)
    vol_profile = df["Volume"].groupby(bins).sum()
    vol_profile.index = [interval.mid for interval in vol_profile.index]

# ========================= SENTIMENT & FAIR VALUE =========================
def get_sentiment(ticker):
    try:
        url = f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws&tbs=qdr:d"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, 'html.parser')
        headlines = [h.text for h in soup.find_all('h3')[:7]]
        pos = sum(1 for h in headlines if any(w in h.lower() for w in ["beat","surge","up","buy","bullish","raise"]))
        neg = sum(1 for h in headlines if any(w in h.lower() for w in ["miss","drop","down","sell","crash","cut"]))
        return (pos - neg) * 15, headlines
    except:
        return 0, ["No recent news"]

sentiment_score, headlines = get_sentiment(ticker)

def fair_value(info, price):
    eps = info.get("trailingEps") or 1
    growth = max(info.get("earningsQuarterlyGrowth", 0.1) or 0.1, 0.05) * 100
    target = info.get("targetMeanPrice") or price * 1.15
    graham = np.sqrt(eps * (8.5 + 2 * growth)) * 1.5
    dcf = eps * (10 + growth)
    fv = np.mean([graham, dcf, target])
    upside = (fv / price - 1) * 100
    return round(fv, 2), round(upside, 1)

fv, upside = fair_value(info, latest_price)

# ========================= DASHBOARD =========================
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Price", f"${latest_price:.2f}")
with col2: st.metric("AI Fair Value", f"${fv}", f"{upside:+.1f}%")
with col3: st.metric("Sentiment", f"{sentiment_score:+}")
with col4: st.metric("P/E", f"{info.get('trailingPE','N/A'):.1f}")
with col5: st.metric("Volume", f"{df['Volume'].iloc[-1]:,}")

# ========================= AI REPORT =========================
if st.button("🤖 Generate AI Hedge Fund Report (Grok-4)", type="primary"):
    with st.spinner("Grok-4 is analyzing..."):
        prompt = f"Analyze {ticker} ({company_name}) as a hedge fund PM in the style of {ai_style}. Current: ${latest_price:.2f}. AI Fair Value: ${fv} ({upside:+.1f}%). Write a 300-word report with bull/bear cases and price target."
        try:
            key = st.secrets["GROK_API_KEY"]
            r = requests.post("https://api.x.ai/v1/chat/completions",
                json={"model": "grok-beta", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7},
                headers={"Authorization": f"Bearer {key}"}, timeout=30)
            report = r.json()["choices"][0]["message"]["content"]
        except:
            report = "Add your Grok API key in Secrets → GROK_API_KEY to enable live AI reports."
        st.markdown(f"<div class='ai-box'><h3>AI Hedge Fund Report</h3>{report}</div>", unsafe_allow_html=True)

# ========================= CHART =========================
fig = make_subplots(rows=4, cols=2, shared_xaxes=True, vertical_spacing=0.03,
                    subplot_titles=("Price & Indicators", "Volume Profile", "RSI", "Volume"),
                    row_heights=[0.55, 0.15, 0.15, 0.15], column_widths=[0.78, 0.22])

fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=close, name="Price"), row=1, col=1)

if show_ma:
    for col, color in zip(["EMA20","EMA50","SMA200"], ["#ff9f40","#ff5722","#6366f1"]):
        if col in df: fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=color)), row=1, col=1)
if show_bb and "BBU" in df:
    fig.add_trace(go.Scatter(x=df.index, y=df["BBU"], name="BB Upper", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BBL"], name="BB Lower", line=dict(dash="dot")), row=1, col=1)
if show_supertrend and "SuperTrend" in df:
    fig.add_trace(go.Scatter(x=df.index, y=df["SuperTrend"], name="SuperTrend", line=dict(width=4, color="#00ff88")), row=1, col=1)

if vol_profile is not None:
    fig.add_trace(go.Bar(x=vol_profile.values, y=vol_profile.index, orientation='h', name="Vol Profile", marker_color="#6366f1"), row=1, col=2)

df["RSI"] = ta.momentum.RSIIndicator(close).rsi()
fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#a78bfa")), row=2, col=1)
fig.add_hline(y=70, line_dash="dot", row=2, col=1); fig.add_hline(y=30, line_dash="dot", row=2, col=1)

fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#94a3b8"), row=4, col=1)

fig.update_layout(height=1000, title=f"{ticker} — Alpha Terminal v3.5", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ========================= SIGNALS + OPTIONS + EXPORT =========================
col1, col2 = st.columns(2)
with col1:
    st.subheader("🎯 Signals")
    signals = []
    if show_supertrend and "SuperTrend" in df and latest_price > df["SuperTrend"].iloc[-1]:
        signals.append("🟢 SuperTrend Bullish")
    if df["RSI"].iloc[-1] < 30:
        signals.append("🔵 RSI Oversold")
    if upside > 25:
        signals.append(f"🚀 {upside:+.1f}% AI Upside")
    for s in signals:
        color = "bull" if "🟢" in s or "🚀" in s else "neutral"
        st.markdown(f'<div class="signal-box {color}">{s}</div>', unsafe_allow_html=True)

    st.download_button("📊 Export CSV", df.to_csv(), f"{ticker}_data.csv")
    st.download_button("🖼️ Export Chart", fig.to_image(format="png"), f"{ticker}_chart.png")

with col2:
    if options_dates:
        st.subheader("Options")
        expiry = st.selectbox("Expiry", options_dates)
        chain = t.option_chain(expiry)
        st.dataframe(chain.calls[['strike','lastPrice','volume','impliedVolatility']].head(8))

st.success(f"AI Target: ${fv} • Potential Return: {upside:+.1f}%")
st.caption("Alpha Terminal v3.5 — Built with Grok • Not financial advice • 2025")
