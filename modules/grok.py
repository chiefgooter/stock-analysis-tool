import streamlit as st
import requests
import json

def generate_grok_intel(ticker, recent_df):
    api_key = st.secrets.get("GROK_API_KEY")
    if not api_key:
        return {
            "conviction": "DEMO MODE",
            "edge_score": 99,
            "target_price_3mo": 999,
            "catalyst": "Get your key at x.ai/api",
            "primary_risk": "None — you're in demo heaven",
            "summary": "**Alpha Terminal v7 is live** — Grok-4 full power unlocked with your key."
        }

    prompt = f"""
You are Grok-4, senior PM at a $20B hedge fund.
Analyze {ticker} and return ONLY valid JSON:

{{
  "conviction": "STRONG BUY|BUY|NEUTRAL|SELL|STRONG SELL",
  "target_price_3mo": number,
  "catalyst": "single sentence",
  "primary_risk": "single sentence",
  "edge_score": number_between_0_and_100,
  "summary": "markdown under 200 chars"
}}

Latest prices: {recent_df['Close'].to_json()}
"""

    try:
        r = requests.post(
            "https://api.x.ai/v1/chat/completions",
            json={
                "model": "grok-beta",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"}
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=20
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except:
        pass
    return None
