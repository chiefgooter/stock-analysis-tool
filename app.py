# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta  # pip install ta

# Page config
st.set_page_config(page_title="Pro Stock Analyzer", layout="wide", initial_sidebar_state="expanded")

st.title("🚀 Professional Stock Analysis Dashboard")
st.markdown("Enter any ticker (stocks, ETFs, crypto like BTC-USD, etc.) for instant multi-layer analysis")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
    period = st.selectbox("Time Period", 
                          ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                          index=3)
    ma_fast = st.slider("Fast Moving Average (days)", 5, 50, 20)
    ma_slow = st.slider("Slow Moving Average (days)", 50, 200, 50)
    risk_free_rate = st.number_input("Risk-Free Rate (%)", value=4.5, step=0.1) / 100

# Robust cached data function (this fixes the Streamlit caching error)
@st.cache_data(ttl=3600, show_spinner="Fetching latest market data...")
def get_stock_data(ticker: str, period: str):
    ticker = ticker.upper().strip()
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, auto_adjust=True, actions=False)
        info = stock.info
        # Ensure info is a plain dict (some fields are lazy-loaded)
        info = dict(info) if info else {}
        return hist, info
    except Exception as e:
        st.error(f"Failed to fetch {ticker}: {str(e)}")
        return pd.DataFrame(), {}

# Technical indicators
def add_technical_indicators(df):
    df = df.copy()
    df["MA_Fast"] = df["Close"].rolling(ma_fast).mean()
    df["MA_Slow"] = df["Close"].rolling(ma_slow).mean()
    
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    
    bollinger = ta.volatility.BollingerBands(df["Close"])
    df["BB_Upper"] = bollinger.bollinger_hband()
    df["BB_Middle"] = bollinger.bollinger_mavg()
    df["BB_Lower"] = bollinger.bollinger_lband()
    
    df["Volume_SMA"] = df["Volume"].rolling(20).mean()
    
    return df

# Trend detection
def detect_trend(df):
    latest = df.iloc[-1]
    if latest["Close"] > latest["MA_Fast"] > latest["MA_Slow"]:
        return "Strong Bullish 🟢"
    elif latest["Close"] > latest["MA_Fast"]:
        return "Bullish 🟢"
    elif latest["Close"] < latest["MA_Fast"] < latest["MA_Slow"]:
        return "Strong Bearish 🔴"
    elif latest["Close"] < latest["MA_Fast"]:
        return "Bearish 🔴"
    else:
        return "Sideways ⚪"

# Main app logic
if ticker:
    hist, info = get_stock_data(ticker, period)
    
    if hist.empty or info == {}:
        st.error("No data found. Check the ticker symbol and try again (e.g., AAPL, TSLA, BTC-USD).")
        st.stop()
    
    company_name = info.get("longName") or info.get("shortName") or ticker
    st.header(f"{company_name} ({ticker})")
    
    latest = hist.iloc[-1]
    prev_close = hist["Close"].iloc[-2] if len(hist) > 1 else latest["Close"]
    change = latest["Close"] - prev_close
    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${latest['Close']:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    col2.metric("Volume", f"{latest['Volume']:,.0f}")
    col3.metric("52W High", f"${hist['High'].tail(252).max():.2f}" if len(hist) >= 252 else "N/A")
    col4.metric("52W Low", f"${hist['Low'].tail(252).min():.2f}" if len(hist) >= 252 else "N/A")
    
    df = add_technical_indicators(hist)
    trend = detect_trend(df)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.success(f"**Trend**: {trend}")
    rsi_val = df["RSI"].iloc[-1]
    rsi_status = "Oversold 🔵" if rsi_val < 30 else "Overbought 🔴" if rsi_val > 70 else "Neutral"
    col2.info(f"**RSI (14)**: {rsi_val:.1f} → {rsi_status}")
    volatility = df["Close"].pct_change().std() * np.sqrt(252) * 100
    col3.info(f"**Annual Volatility**: {volatility:.1f}%")
    col4.info(f"**Beta**: {info.get('beta', 'N/A')}")
    
    # Interactive Plotly chart
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=("Price, MAs & Bollinger Bands", "Volume", "RSI", "MACD"),
                        row_heights=[0.5, 0.2, 0.15, 0.15])
    
    # Candlestick + MAs + BB
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_Fast"], name=f"MA{ma_fast}", line=dict(color="orange")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA_Slow"], name=f"MA{ma_slow}", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(dash="dot", color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(dash="dot", color="gray")), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="lightblue"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Volume_SMA"], name="Vol SMA20", line=dict(color="red")), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="purple")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", annotation_text="Overbought", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", annotation_text="Oversold", row=3, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="blue")), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="orange")), row=4, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram", marker_color="gray"), row=4, col=1)
    
    fig.update_layout(height=1000, title_text=f"{ticker} – Technical Dashboard", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Signals summary
    st.subheader("🎯 Current Signals")
    signals = []
    if latest["Close"] > df["MA_Fast"].iloc[-1] > df["MA_Slow"].iloc[-1]:
        signals.append("🟢 Bullish moving average alignment")
    if df["MACD"].iloc[-1] > df["MACD_Signal"].iloc[-1]:
        signals.append("🟢 MACD bullish crossover")
    if rsi_val < 30:
        signals.append("🔵 RSI oversold – potential bounce")
    if rsi_val > 70:
        signals.append("🔴 RSI overbought – caution")
    if latest["Close"] < df["BB_Lower"].iloc[-1]:
        signals.append("🔵 Price touching lower Bollinger – extreme")
    
    if signals:
        for s in signals:
            st.write(s)
    else:
        st.info("No strong directional signals – consolidation phase")
    
    # Fundamentals table
    st.subheader("💼 Key Fundamentals")
    fund_items = {
        "Market Cap": info.get("marketCap"),
        "P/E Ratio": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "EPS (TTM)": info.get("trailingEps"),
        "Dividend Yield": info.get("dividendYield"),
        "52-Week Change": info.get("52WeekChange"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
    }
    fund_df = pd.DataFrame(fund_items.items(), columns=["Metric", "Value"])
    def format_val(x):
        if isinstance(x, (int, float)):
            if x > 1e9: return f"${x/1e9:.2f}B"
            if x > 1e6: return f"${x/1e6:.2f}M"
            if isinstance(x, float) and x < 10: return f"{x:.2f}%"
            return f"${x:,.2f}" if x > 1000 else f"{x:.2f}"
        return str(x) if x is not None else "N/A"
    fund_df["Value"] = fund_df["Value"].apply(format_val)
    st.table(fund_df)

st.markdown("---")
st.caption("Data: Yahoo Finance • Not financial advice • For educational purposes only")
