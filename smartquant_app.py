import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import cvxpy as cp
from datetime import datetime, timedelta

# === Streamlit Setup ===
st.set_page_config(layout="wide")
st.title("SmartQuant Pro: ML Portfolio Optimizer & Risk Dashboard")
st.markdown("Empowering investment strategy with machine learning and quant analytics.")

# === Sidebar Inputs ===
st.sidebar.header("Configuration")
tickers = st.sidebar.multiselect("Select Assets", ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'NVDA', 'META', 'XOM'], default=['AAPL', 'MSFT', 'GOOGL'])

# Date inputs
default_start = pd.to_datetime("2023-01-01")
default_end = pd.to_datetime("today")

# Real-time mode toggle
st.sidebar.markdown("---")
real_time = st.sidebar.checkbox("🔄 Enable Real-Time Updates (1-min interval)", value=False)

# If real-time, override dates and use 1m interval
if real_time:
    start = datetime.now() - timedelta(days=1)
    end = datetime.now()
    interval = "1m"
else:
    start = st.sidebar.date_input("Start Date", default_start)
    end = st.sidebar.date_input("End Date", default_end)
    interval = "1d"

# Model selection
model_choice = st.sidebar.radio("ML Model", ['Lasso', 'Random Forest'])

# Manual refresh (for real-time mode)
if real_time:
    if st.sidebar.button("🔁 Refresh Now"):
        st.rerun()


# === Load Data ===
@st.cache_data(ttl=60)  # Refresh every 60s if real-time
def load_data(tickers, start, end, interval):
    df = yf.download(tickers, start=start, end=end, interval=interval, group_by='ticker', auto_adjust=False, progress=False)

    # Handle single ticker separately
    if len(tickers) == 1:
        return df['Adj Close'].to_frame(name=tickers[0])

    adj_close = pd.DataFrame()
    for ticker in tickers:
        try:
            adj_close[ticker] = df[ticker]['Adj Close']
        except Exception as e:
            st.warning(f"⚠️ Data error for {ticker}: {e}")
    return adj_close.dropna()

data = load_data(tickers, start, end, interval)

if data.empty:
    st.error("❌ No data fetched. Try selecting different tickers or disabling real-time mode.")
    st.stop()

# === Feature Engineering ===
def compute_features(prices):
    df = pd.DataFrame(index=prices.index)
    df['return_1d'] = prices.pct_change()
    df['volatility_5d'] = prices.pct_change().rolling(5).std()
    df['momentum_5d'] = prices / prices.shift(5) - 1
    df['sma_diff'] = prices.rolling(5).mean() - prices.rolling(10).mean()
    df['target'] = prices.pct_change().shift(-1)
    return df.dropna()

# === Train ML Models ===
X_all, y_all, pred_all = [], [], []

for ticker in tickers:
    df = compute_features(data[ticker])
    features = ['return_1d', 'volatility_5d', 'momentum_5d', 'sma_diff']
    X, y = df[features], df['target']
    
    if model_choice == 'Lasso':
        model = Lasso(alpha=0.001)
    else:
        model = RandomForestRegressor(n_estimators=100)

    try:
        model.fit(X, y)
        pred = model.predict(X[-100:])
        pred_all.append(np.mean(pred))
    except Exception as e:
        st.warning(f"⚠️ Model training failed for {ticker}: {e}")
        pred_all.append(0.0)

# === Optimization ===
returns = np.array(pred_all)
cov = data[tickers].pct_change().dropna().cov().values
w = cp.Variable(len(tickers))
objective = cp.Minimize(cp.quad_form(w, cov))
constraints = [cp.sum(w) == 1, w >= 0, returns @ w >= np.mean(returns)]
prob = cp.Problem(objective, constraints)
prob.solve()

# Check if solution is optimal
if prob.status != 'optimal':
    st.error(f"⚠️ Optimization failed using {model_choice}. Try fewer assets or a different model.")
    st.stop()

# Clean weights
weights = w.value
weights = np.clip(weights, 0, None)
weights = weights / weights.sum()

# === Portfolio Returns ===
daily_ret = data[tickers].pct_change().dropna()
port_ret = daily_ret @ weights
cum_ret = (1 + port_ret).cumprod()

# === Charts ===
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Portfolio Allocation")
    fig1, ax1 = plt.subplots()
    ax1.pie(weights, labels=tickers, autopct='%1.1f%%')
    st.pyplot(fig1)

with col2:
    st.subheader("📈 Cumulative Portfolio Return")
    st.line_chart(cum_ret)

# === Risk Metrics ===
st.subheader("⚠️ Risk Analysis")
VaR_95 = -np.percentile(port_ret, 5)
CVaR_95 = -port_ret[port_ret <= -VaR_95].mean()
drawdown = cum_ret / cum_ret.cummax() - 1

col1, col2, col3 = st.columns(3)
col1.metric("VaR (95%)", f"{VaR_95:.2%}")
col2.metric("CVaR (95%)", f"{CVaR_95:.2%}")
col3.metric("Max Drawdown", f"{drawdown.min():.2%}")

st.line_chart(port_ret.rolling(20).std())
