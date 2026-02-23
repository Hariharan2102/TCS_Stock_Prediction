import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import joblib

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Elite AI Quant Dashboard", layout="wide")
st_autorefresh(interval=60000, key="refresh")

st.title("📈 Elite AI Quant Trading Dashboard")

# =====================================
# SIDEBAR STOCK SELECTION
# =====================================
st.sidebar.title("Stock Selection")

stock_dict = {
    "TCS": "TCS.NS",
    "Reliance": "RELIANCE.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "NIFTY 50": "^NSEI"
}

selected_stock = st.sidebar.selectbox(
    "Choose Stock",
    list(stock_dict.keys())
)

ticker_symbol = stock_dict[selected_stock]
ticker = yf.Ticker(ticker_symbol)

# =====================================
# LOAD MODEL
# =====================================
try:
    model = joblib.load("tcs_lr_model.pkl")
    scaler = joblib.load("tcs_scaler.pkl")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# =====================================
# LIVE DATA
# =====================================
live_data = ticker.history(period="5d", interval="1m")

if isinstance(live_data.columns, pd.MultiIndex):
    live_data.columns = live_data.columns.get_level_values(0)

current_price = float(live_data["Close"].iloc[-1])
previous_price = float(live_data["Close"].iloc[-2])
current_volume = int(live_data["Volume"].iloc[-1])

# =====================================
# MARKET STATUS
# =====================================
now = datetime.now()

if now.weekday() >= 5:
    market_status = "🔴 Closed (Weekend)"
elif 9 <= now.hour < 15:
    market_status = "🟢 Open"
else:
    market_status = "🔴 Closed"

# =====================================
# METRICS
# =====================================
col1, col2, col3 = st.columns(3)

col1.metric(
    "Live Price",
    f"₹ {current_price:.2f}",
    delta=f"{current_price - previous_price:.2f}"
)

col2.metric("Volume", current_volume)

col3.metric("Market Status", market_status)

# =====================================
# INTRADAY CANDLESTICK
# =====================================
st.subheader("📊 Intraday Candlestick")

live_data["VWAP"] = (
    live_data["Close"] * live_data["Volume"]
).cumsum() / live_data["Volume"].cumsum()

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=live_data.index,
    open=live_data["Open"],
    high=live_data["High"],
    low=live_data["Low"],
    close=live_data["Close"],
    name="Price"
))

fig.add_trace(go.Scatter(
    x=live_data.index,
    y=live_data["VWAP"],
    mode="lines",
    name="VWAP"
))

fig.update_layout(template="plotly_dark")

st.plotly_chart(fig, config={"responsive": True})

# =====================================
# HISTORICAL DATA
# =====================================
st.subheader("📈 Professional Trading Chart")

hist_data = yf.download(ticker_symbol, period="5y")

if isinstance(hist_data.columns, pd.MultiIndex):
    hist_data.columns = hist_data.columns.get_level_values(0)

# Indicators
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
ema12 = hist_data["Close"].ewm(span=12).mean()
ema26 = hist_data["Close"].ewm(span=26).mean()

hist_data["MACD"] = ema12 - ema26
hist_data["Signal"] = hist_data["MACD"].ewm(span=9).mean()

hist_data.dropna(inplace=True)

latest = hist_data.iloc[-1]

# =====================================
# ML PREDICTION
# =====================================
features = [[
    float(latest["Close"]),
    float(latest["MA_20"]),
    float(latest["MA_50"]),
    float(latest["RSI"]),
    float(latest["MACD"]),
    float(latest["Signal"]),
    float(latest["Return"])
]]

scaled = scaler.transform(features)
prediction = model.predict(scaled)[0]

# =====================================
# PROFESSIONAL CHART WITH SIGNALS
# =====================================
chart = go.Figure()

chart.add_trace(go.Scatter(
    x=hist_data.index,
    y=hist_data["Close"],
    name="Price"
))

chart.add_trace(go.Scatter(
    x=hist_data.index,
    y=hist_data["MA_20"],
    name="MA20"
))

chart.add_trace(go.Scatter(
    x=hist_data.index,
    y=hist_data["MA_50"],
    name="MA50"
))

# BUY marker
if prediction > current_price:
    chart.add_trace(go.Scatter(
        x=[hist_data.index[-1]],
        y=[current_price],
        mode="markers",
        marker=dict(color="green", size=15),
        name="BUY"
    ))

# SELL marker
if prediction < current_price:
    chart.add_trace(go.Scatter(
        x=[hist_data.index[-1]],
        y=[current_price],
        mode="markers",
        marker=dict(color="red", size=15),
        name="SELL"
    ))

# Prediction marker
chart.add_trace(go.Scatter(
    x=[hist_data.index[-1] + timedelta(days=1)],
    y=[prediction],
    mode="markers",
    marker=dict(color="yellow", size=15),
    name="Prediction"
))

chart.update_layout(template="plotly_dark")

st.plotly_chart(chart, config={"responsive": True})

# =====================================
# PREDICTION METRICS
# =====================================
st.subheader("🔮 AI Forecast")

colP1, colP2 = st.columns(2)

colP1.metric(
    "Predicted Price",
    f"₹ {prediction:.2f}"
)

change = prediction - current_price

colP2.metric(
    "Expected Change",
    f"₹ {change:.2f}",
    delta=f"{change:.2f}"
)

# =====================================
# CONFIDENCE
# =====================================
confidence = 100 - abs(change/current_price*100)

st.metric("AI Confidence", f"{confidence:.2f}%")

# =====================================
# BUY SELL ZONES
# =====================================
buy_zone = prediction * 0.98
sell_zone = prediction * 1.02

colZ1, colZ2 = st.columns(2)

colZ1.metric("Buy Below", f"₹ {buy_zone:.2f}")
colZ2.metric("Sell Above", f"₹ {sell_zone:.2f}")

# =====================================
# SIGNAL
# =====================================
score = 0

if prediction > current_price:
    score += 2
else:
    score -= 2

if latest["RSI"] < 30:
    score += 1

if latest["RSI"] > 70:
    score -= 1

signal = "BUY" if score > 0 else "SELL"

st.subheader("Trading Signal")
st.success(signal)

# =====================================
# PORTFOLIO SIMULATOR
# =====================================
st.subheader("Portfolio Simulator")

investment = st.sidebar.number_input(
    "Investment Amount",
    value=100000
)

shares = investment / current_price

future_value = shares * prediction

profit = future_value - investment

col1, col2, col3 = st.columns(3)

col1.metric("Investment", f"₹ {investment}")
col2.metric("Future Value", f"₹ {future_value:.2f}")
col3.metric("Profit", f"₹ {profit:.2f}")

# =====================================
# FORECAST
# =====================================
st.subheader("30 Day Forecast")

future_prices = []
temp = prediction

for i in range(30):
    temp = temp * (1 + np.random.normal(0.001, 0.01))
    future_prices.append(temp)

future_dates = pd.date_range(
    start=datetime.now(),
    periods=30
)

forecast_df = pd.DataFrame(
    future_prices,
    index=future_dates
)

st.line_chart(forecast_df)
