import streamlit as st
import yfinance as yf

@st.cache_data(ttl=300)
def fetch_data(ticker):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2y")
        info = t.info
        if hist.empty:
            return None, None
        return hist, info
    except:
        return None, None
