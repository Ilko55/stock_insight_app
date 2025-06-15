
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objs as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

# Helper functions
def calculate_indicators(df):
    df["EMA_50"] = df["Close"].ewm(span=50).mean()
    df["EMA_200"] = df["Close"].ewm(span=200).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["ATR"] = compute_atr(df, 14)
    df["MACD"], df["MACD_signal"] = compute_macd(df["Close"])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()

def compute_macd(series):
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def sentiment_score(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)["compound"]
    return score

def get_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey=YOUR_NEWSAPI_KEY"
    response = requests.get(url)
    articles = []
    if response.status_code == 200:
        data = response.json()
        for article in data["articles"][:5]:
            score = sentiment_score(article["title"])
            sentiment = "🟢" if score > 0.2 else "🔴" if score < -0.2 else "🟡"
            articles.append(f"{sentiment} [{article['title']}]({article['url']})")
    return articles

# Streamlit App
st.title("📊 Stock Insight App")

ticker = st.text_input("Enter stock ticker (e.g. AAPL)", "AAPL").upper()
df = yf.download(ticker, period="6mo", interval="1d")
if df.empty:
    st.error("Ticker not found or no data.")
else:
    df = calculate_indicators(df)

    st.subheader("📈 Price Chart with Signals")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"], mode="lines", name="EMA 50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_200"], mode="lines", name="EMA 200"))
    st.plotly_chart(fig)

    st.subheader("🧠 Signal Summary")
    latest = df.iloc[-1]
    signal = "BUY" if latest["MACD"] > latest["MACD_signal"] and latest["RSI"] < 70 else "SELL" if latest["RSI"] > 70 else "HOLD"
    confidence = abs(latest["MACD"] - latest["MACD_signal"]) / latest["Close"] * 100
    st.write(f"Signal: **{signal}** with confidence: **{confidence:.2f}%**")
    st.write(f"Volatility (ATR): **{latest['ATR']:.2f}**")

    st.subheader("📰 Recent News Headlines")
    for item in get_news(ticker):
        st.markdown(f"- {item}")
