import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Elite Stock Dashboard", layout="wide")
st_autorefresh(interval=60000, key="refresh")

st.title("📈 TCS Elite Quant Trading Dashboard")

# =====================================================
# 1️⃣ LIVE INTRADAY DATA (5 DAYS)
# =====================================================
ticker = yf.Ticker("TCS.NS")
live_data = ticker.history(period="5d", interval="1m")

if isinstance(live_data.columns, pd.MultiIndex):
    live_data.columns = live_data.columns.get_level_values(0)

current_price = float(live_data["Close"].iloc[-1])
previous_price = float(live_data["Close"].iloc[-2])
current_volume = int(live_data["Volume"].iloc[-1])

# =====================================================
# 2️⃣ MARKET STATUS + METRICS
# =====================================================
now = datetime.now()
market_status = "🟢 Market Open" if 9 <= now.hour < 15 else "🔴 Market Closed"

col1, col2, col3 = st.columns(3)
col1.metric("Live Price", f"₹ {current_price:.2f}", delta=f"{current_price - previous_price:.2f}")
col2.metric("Volume (Last Minute)", current_volume)
col3.markdown(f"### {market_status}")

# =====================================================
# 3️⃣ VWAP + CANDLESTICK CHART
# =====================================================
live_data["VWAP"] = (live_data["Close"] * live_data["Volume"]).cumsum() / live_data["Volume"].cumsum()

st.subheader("📊 5-Day Intraday Candlestick Chart")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=live_data.index,
    open=live_data['Open'],
    high=live_data['High'],
    low=live_data['Low'],
    close=live_data['Close'],
    name="Candlestick"
))
fig.add_trace(go.Scatter(
    x=live_data.index,
    y=live_data["VWAP"],
    mode="lines",
    name="VWAP"
))
fig.update_layout(xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# 4️⃣ 5-DAY CANDLE BIAS
# =====================================================
st.subheader("📊 5-Day Candle Bias")

daily_data = ticker.history(period="5d", interval="1d")

bullish_days = (daily_data["Close"] > daily_data["Open"]).sum()
bearish_days = (daily_data["Close"] < daily_data["Open"]).sum()
neutral_days = (daily_data["Close"] == daily_data["Open"]).sum()

pie_fig = go.Figure(data=[go.Pie(
    labels=["Bullish", "Bearish", "Neutral"],
    values=[bullish_days, bearish_days, neutral_days],
    hole=0.6,
    marker=dict(colors=["#00cc96", "#ef553b", "#a6a6a6"])
)])
pie_fig.update_layout(annotations=[dict(text="5-Day Bias", x=0.5, y=0.5, showarrow=False)])
st.plotly_chart(pie_fig, use_container_width=True)

# =====================================================
# 5️⃣ 5-YEAR HISTORICAL + INDICATORS
# =====================================================
st.subheader("📈 5-Year Historical & Technical Indicators")

hist_data = yf.download("TCS.NS", period="5y")

if isinstance(hist_data.columns, pd.MultiIndex):
    hist_data.columns = hist_data.columns.get_level_values(0)

hist_data["MA_20"] = hist_data["Close"].rolling(20).mean()
hist_data["MA_50"] = hist_data["Close"].rolling(50).mean()
hist_data["Return"] = hist_data["Close"].pct_change()

# RSI
delta = hist_data["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
hist_data["RSI"] = 100 - (100 / (1 + rs))

# MACD
ema12 = hist_data["Close"].ewm(span=12, adjust=False).mean()
ema26 = hist_data["Close"].ewm(span=26, adjust=False).mean()
hist_data["MACD"] = ema12 - ema26
hist_data["Signal_Line"] = hist_data["MACD"].ewm(span=9, adjust=False).mean()

hist_data.dropna(inplace=True)

st.line_chart(hist_data["Close"])

latest = hist_data.iloc[-1]
latest_rsi = float(latest["RSI"])
latest_macd = float(latest["MACD"])
latest_signal = float(latest["Signal_Line"])

colA, colB = st.columns(2)
colA.metric("RSI (14)", f"{latest_rsi:.2f}")
colB.metric("MACD", f"{latest_macd:.2f}")

# =====================================================
# 6️⃣ ML PREDICTION
# =====================================================
features = [
    float(latest["Close"]),
    float(latest["MA_20"]),
    float(latest["MA_50"]),
    latest_rsi,
    latest_macd,
    latest_signal,
    float(latest["Return"])
]

try:
    response = requests.post(
        "http://127.0.0.1:5000/predict",
        json={"features": features}
    )
    prediction = response.json()["predicted_next_day_price"]

    st.subheader("🔮 Predicted Next-Day Price")
    st.success(f"₹ {prediction:.2f}")

    # =====================================================
    # 7️⃣ MULTI-FACTOR SIGNAL ENGINE
    # =====================================================
    score = 0

    if latest_rsi < 30:
        score += 2
    elif latest_rsi > 70:
        score -= 2

    if latest_macd > latest_signal:
        score += 1
    else:
        score -= 1

    if bullish_days > bearish_days:
        score += 1
    elif bearish_days > bullish_days:
        score -= 1

    if prediction > current_price:
        score += 1
    else:
        score -= 1

    if score >= 4:
        final_signal = "🟢 STRONG BUY"
    elif score >= 2:
        final_signal = "🟢 BUY"
    elif score >= 0:
        final_signal = "🟡 HOLD"
    elif score <= -2:
        final_signal = "🔴 SELL"
    else:
        final_signal = "🔴 STRONG SELL"

    st.subheader("🎯 Final Trading Signal")
    st.success(final_signal)
    st.write(f"Signal Score: {score}")

    # =====================================================
    # 8️⃣ RISK / REWARD
    # =====================================================
    st.subheader("📊 Risk Management")

    stop_loss = current_price * 0.98
    risk = current_price - stop_loss
    reward = prediction - current_price
    rr_ratio = reward / risk if risk > 0 else 0

    colR1, colR2, colR3 = st.columns(3)
    colR1.metric("Stop Loss", f"₹ {stop_loss:.2f}")
    colR2.metric("Target", f"₹ {prediction:.2f}")
    colR3.metric("Risk/Reward", f"{rr_ratio:.2f}")

    # =====================================================
    # 9️⃣ SIMPLE BACKTEST (30 DAYS)
    # =====================================================
    st.subheader("📈 Recent Signal Accuracy")

    backtest_data = hist_data.tail(30)
    correct = 0

    for i in range(1, len(backtest_data)):
        prev = backtest_data["Close"].iloc[i-1]
        curr = backtest_data["Close"].iloc[i]

        actual = 1 if curr > prev else -1
        predicted_dir = 1 if latest_macd > latest_signal else -1

        if actual == predicted_dir:
            correct += 1

    accuracy = (correct / 29) * 100
    st.metric("30-Day Accuracy", f"{accuracy:.1f}%")

    # =====================================================
    # 🔟 PORTFOLIO SIMULATION
    # =====================================================
    st.subheader("💼 Portfolio Simulation")

    capital = 100000
    shares = capital / hist_data["Close"].iloc[-30]
    portfolio_value = shares * current_price
    profit = portfolio_value - capital

    colP1, colP2, colP3 = st.columns(3)
    colP1.metric("Initial Capital", f"₹ {capital}")
    colP2.metric("Current Value", f"₹ {portfolio_value:.2f}")
    colP3.metric("Net Profit / Loss", f"₹ {profit:.2f}")

except:
    st.warning("Prediction API not running.")
