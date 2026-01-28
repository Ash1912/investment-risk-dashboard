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
st.caption("Risk Analytics | Portfolio Comparison | Indian Equity Market Data")

# ================= LOAD RAW DATA =================
@st.cache
def load_data():
    try:
        df = pd.read_excel("RawPrices.xlsx")
    except Exception:
        df = pd.read_csv("RawPrices.csv")

    df.columns = df.columns.str.strip()

    # ✅ FIX DATE PARSING
    if np.issubdtype(df["Date"].dtype, np.number):
        df["Date"] = pd.to_datetime(df["Date"], origin="1899-12-30", unit="D")
    else:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Drop invalid dates
    df = df.dropna(subset=["Date"])

    df.set_index("Date", inplace=True)

    return df

prices_full = load_data()

# FORCE SORT + CLEAN INDEX
prices_full = prices_full.sort_index()
prices_full = prices_full[~prices_full.index.duplicated()]

market_index = "NIFTY 50"

stocks = prices_full.columns.drop("NIFTY 50")

if market_index not in prices_full.columns:
    st.error("❌ Market index 'NIFTY 50' not found in dataset")
    st.stop()

# ================= SIDEBAR =================
st.sidebar.subheader("📅 Date Range")

# Dataset boundaries
data_start = prices_full.index.min().date()
data_end = prices_full.index.max().date()

# Default dates (from file)
default_start = pd.to_datetime("2025-12-31").date()
default_end = pd.to_datetime("2026-01-10").date()

# Ensure defaults are within data range
default_start = max(default_start, data_start)
default_end = min(default_end, data_end)

start_date, end_date = st.sidebar.date_input(
    "Select Analysis Period",
    value=(default_start, default_end),
    min_value=data_start,
    max_value=data_end
)

# Convert back to pandas datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter data
prices = prices_full.loc[start_date:end_date]

# Graceful handling of missing data
if prices.empty or prices.shape[0] < 2:
    st.warning(
        "⚠️ We don't have sufficient data for the selected date range.\n\n"
        f"Available data is from **{data_start}** to **{data_end}**.\n\n"
        "Please adjust the date range."
    )
    st.stop()

st.sidebar.subheader("📁 Portfolios to Compare")
portfolio_types = st.sidebar.multiselect(
    "Select Portfolios",
    ["Young Investor", "Middle-aged Investor", "Senior Investor", "Passive Index", "Custom Portfolio"],
    default=["Young Investor", "Middle-aged Investor"]
)

st.sidebar.subheader("⚙️ Risk Parameters")
confidence = st.sidebar.selectbox("VaR Confidence Level", ["95%", "99%"])
confidence_level = 5 if confidence == "95%" else 1

# Average 10Y Indian T-Bill Yield (2015–2025 approx)
TBILL_10Y_AVG = 0.062   # 6.2%

rf = TBILL_10Y_AVG / 252

st.sidebar.markdown("---")
st.sidebar.subheader("🎚 Custom Portfolio Allocation")

small = st.sidebar.slider("Small-Cap", 0.0, 1.0, 0.4, 0.05)
mid = st.sidebar.slider("Mid-Cap", 0.0, 1.0, 0.3, 0.05)
large = st.sidebar.slider("Large-Cap", 0.0, 1.0, 0.2, 0.05)
mf = st.sidebar.slider("Mutual Fund", 0.0, 1.0, 0.1, 0.05)

weights_norm = np.array([small, mid, large, mf])
if weights_norm.sum() == 0:
    weights_norm = np.array([0.4, 0.3, 0.2, 0.1])
else:
    weights_norm /= weights_norm.sum()


# ================= PORTFOLIO PROFILES =================
PORTFOLIO_PROFILES = {
    "Young Investor": np.array([0.4, 0.3, 0.2, 0.1]),
    "Middle-aged Investor": np.array([0.20, 0.3, 0.3, 0.2]),
    "Senior Investor": np.array([0.05, 0.15, 0.5, 0.3]),
    "Passive Index": np.array([0, 0, 0, 1])
}

# ================= RETURNS =================
if prices.shape[0] < 2:
    st.error("⚠️ Not enough data points for return calculation. Please select a wider date range.")
    st.stop()

returns = prices.pct_change().dropna(how="all")

market_return = returns[market_index]

# ================= CORRELATION HEATMAP =================

corr_matrix = returns.corr().copy()

corr_fig = px.imshow(
    corr_matrix,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu",
    title="Asset Correlation Heatmap"
)


# ================= STOCK METRICS =================
stock_metrics = []

for s in stocks:
    market_var = market_return.var()
    beta = returns[s].cov(market_return) / market_var if market_var != 0 else np.nan
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
    w[market_index] = w_raw[3]
    return pd.Series(w)

weights_df = pd.DataFrame({p: build_weights(p) for p in portfolio_types})

# ================= PORTFOLIO METRICS =================
portfolio_returns = {}
metrics = []
var_es = []

for p in portfolio_types:
    pr = (returns * build_weights(p)).sum(axis=1)
    portfolio_returns[p] = pr

    beta = pr.cov(market_return) / market_return.var()
    sharpe = (pr.mean() * 252 - TBILL_10Y_AVG) / (pr.std() * np.sqrt(252))
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

index_daily_return = market_return.mean()
index_annual_return = index_daily_return * 252

portfolio_summary = comparison_df.copy()
portfolio_summary["Annual Return %"] = portfolio_summary["Return"] * 252 * 100
portfolio_summary["Excess Return over T-Bill %"] = (
    portfolio_summary["Annual Return %"] - TBILL_10Y_AVG * 100
)

# ================= MONTE CARLO SIMULATION =================

def monte_carlo_simulation(returns, simulations=500, days=252):
    mean = returns.mean()
    std = returns.std()

    simulations = np.random.normal(mean, std, (days, simulations))
    price_paths = np.cumprod(1 + simulations, axis=0)

    simulations_df = pd.DataFrame(price_paths)
    return simulations_df

mc_results = {}
for p in portfolio_types:
    mc_results[p] = monte_carlo_simulation(portfolio_returns[p])


# ================= EXTRA RISK METRICS =================

def max_drawdown(series):
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

portfolio_summary["Annual Volatility %"] = portfolio_summary["Volatility"] * np.sqrt(252) * 100

portfolio_summary["Max Drawdown %"] = [
    max_drawdown(portfolio_returns[p]) * 100 for p in portfolio_summary["Portfolio"]
]

portfolio_summary["Beats T-Bill"] = portfolio_summary["Annual Return %"] > TBILL_10Y_AVG * 100
portfolio_summary["Beats NIFTY 50"] = portfolio_summary["Annual Return %"] > index_annual_return * 100

index_row = {
    "Portfolio": "NIFTY 50 Index",
    "Annual Return %": index_annual_return * 100,
    "Annual Volatility %": market_return.std() * np.sqrt(252) * 100,
    "Sharpe": (index_annual_return - TBILL_10Y_AVG) / (market_return.std() * np.sqrt(252)),
    "Excess Return over T-Bill %": (index_annual_return - TBILL_10Y_AVG) * 100,
    "Max Drawdown %": max_drawdown(market_return) * 100,
    "Beats T-Bill": True,
    "Beats NIFTY 50": False
}

t_bill_row = {
    "Portfolio": "10Y T-Bill (Risk Free)",
    "Annual Return %": TBILL_10Y_AVG * 100,
    "Annual Volatility %": 0,
    "Sharpe": np.nan,
    "Excess Return over T-Bill %": 0,
    "Max Drawdown %": 0,
    "Beats T-Bill": False,
    "Beats NIFTY 50": False
}

portfolio_summary = pd.concat(
    [portfolio_summary, pd.DataFrame([index_row, t_bill_row])],
    ignore_index=True
)

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
    "T-Bill vs Index vs Portfolio",
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
    px.bar(
        var_es_df,
        x="Portfolio",
        y=["VaR", "Expected Shortfall"],
        barmode="group",
        title=f"Downside Risk Comparison ({confidence})"
    )
)

tabs[8].subheader("📊 Portfolio Risk Comparison")
tabs[8].dataframe(comparison_df.round(4))

tabs[8].subheader("🔗 Asset Correlation Heatmap")
tabs[8].plotly_chart(corr_fig, use_container_width=True)

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

for p in portfolio_types:
    mc_df = mc_results[p].copy()
    tabs[9].plotly_chart(
        px.line(
            mc_df,
            title=f"Monte Carlo Simulation – {p}",
            labels={"value": "Portfolio Value", "index": "Days"}
            ),
            use_container_width=True
        )

tabs[10].subheader("📊 Detailed Portfolio vs Index vs T-Bill Comparison")

tabs[10].dataframe(
    portfolio_summary[[
        "Portfolio",
        "Annual Return %",
        "Annual Volatility %",
        "Sharpe",
        "Max Drawdown %",
        "Excess Return over T-Bill %",
        "Beats T-Bill",
        "Beats NIFTY 50"
    ]].round(2)
)
