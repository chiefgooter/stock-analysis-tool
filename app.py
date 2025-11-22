# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta  # technical analysis library

# Page config
st.set_page_config(page_title="Professional Stock Analyzer", layout="wide")
st.title("🚀 Professional Stock Market Analysis Tool")
st.markdown("Enter any stock ticker (e.g., AAPL, TSLA, BTC-USD) for instant professional-grade analysis")

# Sidebar
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    period = st.selectbox("Analysis Period", 
                         ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                         index=3)
    
    ma_fast = st.slider("Fast Moving Average", 5, 50, 20)
    ma_slow = st.slider("Slow Moving Average", 50, 200, 50)
    risk_free_rate = st.number_input("Risk-Free Rate (%)", value=4.5) / 100

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info, stock
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None, None, None

def add_technical_indicators(df):
    df = df.copy()
    # Moving Averages
    df['MA_Fast'] = df['Close'].rolling(ma_fast).mean()
    df['MA_Slow'] = df['Close'].rolling(ma_slow).mean()
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    df['BB_Lower'] = bollinger.bollinger_lband()
    
    # Volume SMA
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    
    # Support & Resistance (simple pivot points)
    high = df['High'].rolling(20).max()
    low = df['Low'].rolling(20).min()
    df['Resistance'] = high.shift(1)
    df['Support'] = low.shift(1)
    
    return df

def detect_trend(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    if latest['Close'] > latest['MA_Fast'] > latest['MA_Slow']:
        trend = "Strong Bullish 🟢"
    elif latest['Close'] > latest['MA_Fast'] and latest['MA_Slow'] > latest['MA_Fast']:
        trend = "Weak Bullish 🟡"
    elif latest['Close'] < latest['MA_Fast'] < latest['MA_Slow']:
        trend = "Strong Bearish 🔴"
    elif latest['Close'] < latest['MA_Fast'] and latest['MA_Slow'] < latest['MA_Fast']:
        trend = "Weak Bearish 🟠"
    else:
        trend = "Sideways ⚪"
    
    return trend

# Main app
if ticker:
    hist, info, stock = get_stock_data(ticker, period)
    
    if hist is not None and not hist.empty:
        name = info.get('longName', ticker)
        st.header(f"{name} ({ticker})")
        
        # Current price & change
        latest = hist.iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else latest['Close']
        change = latest['Close'] - prev_close
        change_pct = (change / prev_close) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${latest['Close']:.2f}", 
                     f"{change:+.2f} ({change_pct:+.2f}%)")
        with col2:
            st.metric("Volume", f"{latest['Volume']:,.0f}")
        with col3:
            st.metric("52W High", f"${hist['High'].tail(252).max():.2f}")
        with col4:
            st.metric("52W Low", f"${hist['Low'].tail(252).min():.2f}")
        
        # Add indicators
        df = add_technical_indicators(hist)
        trend = detect_trend(df)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.success(f"**Trend**: {trend}")
        with col2:
            rsi = df['RSI'].iloc[-1]
            rsi_status = "Oversold 🔵" if rsi < 30 else "Overbought 🔴" if rsi > 70 else "Neutral"
            st.info(f"**RSI**: {rsi:.1f} → {rsi_status}")
        with col3:
            st.info(f"**Volatility**: {(df['Close'].pct_change().std() * np.sqrt(252)*100):.1f}%")
        with col4:
            beta = info.get('beta', 'N/A')
            st.info(f"**Beta**: {beta}")
        
        # Charts
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=('Price & Moving Averages', 'Volume', 'RSI', 'MACD'),
            row_heights=[0.5, 0.2, 0.15, 0.15]
        )
        
        # Price + MAs + Bollinger
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'],
                                     name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA_Fast'], name=f"MA{ma_fast}", line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MA_Slow'], name=f"MA{ma_slow}", line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="BB Upper", line=dict(dash='dot', color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="BB Lower", line=dict(dash='dot', color='gray')), row=1, col=1)
        
        # Volume
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Volume_SMA'], name="Vol SMA20", line=dict(color='red')), row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", row=3, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal"), row=4, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name="Histogram"), row=4, col=1)
        
        fig.update_layout(height=1000, title_text=f"{ticker} - Technical Analysis Dashboard")
        st.plotly_chart(fig, use_container_width=True)
        
        # Signals summary
        st.subheader("📊 Current Signals Summary")
        signals = []
        
        latest = df.iloc[-1]
        if latest['Close'] > latest['MA_Fast'] > latest['MA_Slow']:
            signals.append("🟢 Golden Cross confirmed (or in progress)")
        if latest['MACD'] > latest['MACD_Signal']:
            signals.append("🟢 MACD Bullish")
        if latest['RSI'] < 30:
            signals.append("🔵 Potentially Oversold - Reversal candidate")
        elif latest['RSI'] > 70:
            signals.append("🔴 Potentially Overbought - Caution")
        if latest['Close'] < latest['BB_Lower']:
            signals.append("🔵 Price below Lower Bollinger - Extreme oversold")
        
        for signal in signals:
            st.write(signal)
        
        if not signals:
            st.info("No strong signals at the moment - market in consolidation")
            
        # Fundamentals (if available)
        st.subheader("💼 Fundamentals")
        fundamentals = {
            "Market Cap": info.get('marketCap'),
            "P/E Ratio": info.get('trailingPE'),
            "EPS (TTM)": info.get('trailingEps'),
            "Dividend Yield": info.get('dividendYield'),
            "52W Change": info.get('52WeekChange'),
            "Sector": info.get('sector'),
            "Industry": info.get('industry')
        }
        
        fund_df = pd.DataFrame(list(fundamentals.items()), columns=['Metric', 'Value'])
        fund_df['Value'] = fund_df['Value'].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int,float)) and x > 1000 else 
                                                 f"{x:.2f}%" if isinstance(x, float) and x < 10 else str(x))
        st.table(fund_df)
        
    else:
        st.error("No data found for this ticker. Try AAPL, MSFT, or BTC-USD")

st.markdown("---")
st.caption("Data provided by Yahoo Finance • Not financial advice • Use at your own risk")
