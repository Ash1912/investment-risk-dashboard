import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Investment Risk Management Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= THEME =================
st.markdown("""
<style>
.block-container { padding-top: 0.8rem; }
section[data-testid="stSidebar"] > div { padding-top: 0.6rem; }
header {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stSidebar"] { background-color: #111827; }
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("📊 Investment Risk Management & Diversification Dashboard")
st.caption("Risk Analytics | Portfolio Comparison | Synthetic Indian Equity Data")

# ================= SIDEBAR =================
st.sidebar.subheader("📅 Date Range")
start_date, end_date = st.sidebar.date_input(
    "Select Analysis Period",
    value=(pd.to_datetime("2026-01-01"), pd.to_datetime("2026-01-31"))
)

st.sidebar.subheader("📁 Portfolios to Compare")
portfolio_types = st.sidebar.multiselect(
    "Select Portfolios",
    ["Young Investor", "Middle-aged Investor", "Senior Investor", "Passive Index", "Custom Portfolio"],
    default=["Young Investor", "Middle-aged Investor"]
)

st.sidebar.subheader("⚙️ Risk Parameters")
confidence = st.sidebar.selectbox("VaR Confidence Level", ["95%", "99%"])
confidence_level = 5 if confidence == "95%" else 1
rf = 0.06 / 252

st.sidebar.markdown("---")
st.sidebar.subheader("🎚 Custom Portfolio Allocation")

small = st.sidebar.slider("Small-Cap", 0.0, 1.0, 0.4, 0.05)
mid = st.sidebar.slider("Mid-Cap", 0.0, 1.0, 0.3, 0.05)
large = st.sidebar.slider("Large-Cap", 0.0, 1.0, 0.2, 0.05)
mf = st.sidebar.slider("Mutual Fund", 0.0, 1.0, 0.1, 0.05)

weights_norm = np.array([small, mid, large, mf])
weights_norm /= weights_norm.sum()

# ================= PORTFOLIO PROFILES =================
PORTFOLIO_PROFILES = {
    "Young Investor": np.array([0.4, 0.3, 0.2, 0.1]),
    "Middle-aged Investor": np.array([0.20, 0.3, 0.3, 0.2]),
    "Senior Investor": np.array([0.05, 0.15, 0.5, 0.3]),
    "Passive Index": np.array([0, 0, 0, 1])
}

# ================= DATA GENERATION =================
np.random.seed()
dates = pd.date_range(start=start_date, end=end_date, freq="B")
stocks = [f"Stock_{i}" for i in range(1, 11)]
assets = stocks + ["Index_Fund"]

prices = pd.DataFrame(index=dates, columns=assets)
for asset in assets:
    prices[asset] = 100 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod()

returns = prices.pct_change().dropna()
market_return = returns[stocks].mean(axis=1)

# ================= STOCK METRICS =================
stock_metrics = []
for s in stocks:
    beta = np.cov(returns[s], market_return)[0, 1] / np.var(market_return)
    stock_metrics.append([s, returns[s].mean(), returns[s].std(), beta])

stock_metrics_df = pd.DataFrame(
    stock_metrics,
    columns=["Stock", "Avg Return", "Volatility", "Beta"]
)

# ================= WEIGHT BUILDER =================
def build_weights(profile):
    w_raw = weights_norm if profile == "Custom Portfolio" else PORTFOLIO_PROFILES[profile]
    w = {}
    for s in stocks[:3]: w[s] = w_raw[0] / 3
    for s in stocks[3:6]: w[s] = w_raw[1] / 3
    for s in stocks[6:10]: w[s] = w_raw[2] / 4
    w["Index_Fund"] = w_raw[3]
    return pd.Series(w)

weights_df = pd.DataFrame({p: build_weights(p) for p in portfolio_types})

# ================= PORTFOLIO RETURNS & METRICS =================
portfolio_returns = {}
metrics = []
var_es = []

for p in portfolio_types:
    pr = (returns * build_weights(p)).sum(axis=1)
    portfolio_returns[p] = pr

    beta = np.cov(pr, market_return)[0, 1] / np.var(market_return)
    sharpe = (pr.mean() - rf) / pr.std()
    VaR = np.percentile(pr, confidence_level)
    ES = pr[pr <= VaR].mean()

    metrics.append([p, pr.mean(), pr.std(), beta, sharpe])
    var_es.append([p, VaR, ES])

risk_metrics_df = pd.DataFrame(
    metrics,
    columns=["Portfolio", "Return", "Volatility", "Beta", "Sharpe"]
)

var_es_df = pd.DataFrame(var_es, columns=["Portfolio", "VaR", "Expected Shortfall"])

comparison_df = risk_metrics_df.merge(var_es_df, on="Portfolio")

comparison_df["Bubble_Size"] = comparison_df["Sharpe"].clip(lower=0.01) * 80

# ================= TABS =================
tabs = st.tabs([
    "Raw Prices",
    "Daily Returns",
    "Market Return",
    "Stock Risk Metrics",
    "Portfolio Weights",
    "Portfolio Returns",
    "Risk Adjusted Metrics",
    "VaR & ES",
    "Passive vs Active",
    "Summary Insights"
])

tabs[0].dataframe(prices.round(2))
tabs[1].dataframe(returns.round(4))
tabs[2].dataframe(market_return.round(4))
tabs[3].dataframe(stock_metrics_df.round(4))
tabs[4].dataframe(weights_df.round(4))

tabs[5].plotly_chart(
    px.line(pd.DataFrame(portfolio_returns), title="Portfolio Returns Comparison")
)

tabs[6].dataframe(risk_metrics_df.round(4))

tabs[7].plotly_chart(
    px.bar(var_es_df, x="Portfolio", y=["VaR", "Expected Shortfall"],
           barmode="group", title="Downside Risk Comparison")
)

tabs[8].dataframe(comparison_df.round(4))

tabs[9].plotly_chart(
    px.scatter(
        comparison_df,
        x="Volatility",
        y="Return",
        size="Bubble_Size",
        color="Portfolio",
        hover_data=["Sharpe", "Beta"],
        title="Risk vs Return (Sharpe-Adjusted View)"
    )
)
