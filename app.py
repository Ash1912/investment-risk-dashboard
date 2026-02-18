import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import norm
import warnings
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", category=PerformanceWarning)


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Investment Risk Management Dashboard",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================
st.title("📊 5-Year Portfolio Risk Analysis (2021–2025)")
st.caption("Comprehensive Risk & Return Evaluation using Annual Open–Close Prices")

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.header("⚙ Portfolio Controls")

rf_rate = st.sidebar.slider(
    "Risk-Free Rate (Annual %)", 3.0, 8.0, 6.2, 0.1
) / 100

var_conf = st.sidebar.selectbox("VaR Confidence Level", ["90%", "95%", "99%"])
var_level = {"90%": 10, "95%": 5, "99%": 1}[var_conf]

# =====================================================
# LOAD DATA (Streamlit-safe caching)
# =====================================================
@st.cache
def load_data():
    raw = pd.read_excel(
        "Opening_Closing_Stock_Data_2021_2025.xlsx",
        header=None
    )

    # Find rows where first column == 'Asset'
    header_rows = raw[raw[0] == "Asset"].index

    all_tables = []

    for i in range(len(header_rows)):
        start = header_rows[i]
        end = header_rows[i + 1] if i + 1 < len(header_rows) else len(raw)

        temp = raw.iloc[start:end].copy()
        temp.columns = temp.iloc[0]      # set header
        temp = temp.iloc[1:]             # drop header row
        all_tables.append(temp)

    df = pd.concat(all_tables, ignore_index=True)

    # Clean
    df.columns = df.columns.str.strip()
    df["Asset"] = df["Asset"].astype(str).str.strip()

    for col in df.columns:
        if col != "Asset":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

df = load_data()
years = ["2021", "2022", "2023", "2024", "2025"]

# =====================================================
# ASSET CLASS MAPPING
# =====================================================
ASSET_CLASS_MAP = {
    "HBL Power": "Equity", "Mazagon Dock": "Equity", "KPIT Tech": "Equity",
    "Trent Ltd": "Equity", "Persistent": "Equity", "Cummins India": "Equity",
    "Reliance": "Equity", "TCS": "Equity", "HDFC Bank": "Equity",

    "Gold BeES (ETF)": "ETF",

    "Embassy Office Parks REIT": "REIT",
    "Mindspace Business Parks REIT": "REIT",

    "IRB InvIT Fund": "InvIT",

    "India Grid Trust": "Bonds",
    "REC Bond": "Bonds",
    "NABARD Bond": "Bonds",

    "Nifty 50 Index": "Index"
}

# =====================================================
# ANNUAL RETURNS
# =====================================================
annual_returns = {}

for _, row in df.iterrows():
    asset = row["Asset"]
    returns = []

    for y in years:
        o = row.get(f"{y} Open (Rs)")
        c = row.get(f"{y} Close (Rs)")
        if pd.notna(o) and pd.notna(c) and o != 0:
            returns.append((c - o) / o)
        else:
            returns.append(0.0)  # avoid NaN in UI

    annual_returns[asset] = returns

annual_returns_df = pd.DataFrame(annual_returns, index=years).T
# 🔒 Force numeric dtype for year columns (CRITICAL)
annual_returns_df[years] = (
    annual_returns_df[years]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0.0)
)
annual_returns_df["Asset Class"] = annual_returns_df.index.map(ASSET_CLASS_MAP)

st.subheader("📅 Annual Asset Returns (%)")
display_df = annual_returns_df.copy()
display_df[years] = (display_df[years] * 100).round(2)

# ✅ FIX: Make index a proper column
display_df = display_df.reset_index()
display_df = display_df.rename(columns={"index": "Assets"})
st.dataframe(display_df)

# =====================================================
# ASSET CLASS ANALYSIS (⭐ NEW ⭐)
# =====================================================
st.subheader("🏷 Asset Class Performance Comparison")

asset_class_returns = annual_returns_df.groupby("Asset Class")[years].mean()

asset_metrics = []

for cls in asset_class_returns.index:
    r = asset_class_returns.loc[cls]
    mean_r = r.mean()
    vol = r.std()
    sharpe = (mean_r - rf_rate) / vol if vol > 0 else 0
    cumulative = (1 + r).cumprod()
    # max_dd = ((cumulative / cumulative.cummax()) - 1).min()
    CAGR = cumulative.iloc[-1] ** (1 / len(r)) - 1

    asset_metrics.append([
        cls, mean_r*100, CAGR*100, vol*100, sharpe
    ])

asset_metrics_df = pd.DataFrame(asset_metrics, columns=[
    "Asset Class", "Avg Return %", "CAGR %", "Volatility %", "Sharpe Ratio"
])

st.dataframe(asset_metrics_df.round(2))

asset_metrics_df["Size"] = asset_metrics_df["Sharpe Ratio"].abs().clip(lower=0.1)

fig = px.scatter(
    asset_metrics_df,
    x="Volatility %",
    y="Avg Return %",
    size="Size",
    color="Asset Class",
    title="Asset Class Risk–Return Comparison"
)

st.plotly_chart(fig, use_container_width=True)

best_asset = asset_metrics_df.loc[
    asset_metrics_df["Sharpe Ratio"].idxmax()
]

st.success(f"""
🧠 **Best Asset Class (Risk-Adjusted)**  
**{best_asset['Asset Class']}**

• Avg Return: {best_asset['Avg Return %']:.2f}%  
• Volatility: {best_asset['Volatility %']:.2f}%  
• Sharpe Ratio: {best_asset['Sharpe Ratio']:.2f}
""")

# =====================================================
# PORTFOLIOS
# =====================================================
PORTFOLIOS = {
    "Young Investor": {
        "HBL Power": 0.12, "Mazagon Dock": 0.12, "KPIT Tech": 0.12,
        "Trent Ltd": 0.12, "Persistent": 0.12,
        "Reliance": 0.10, "Gold BeES (ETF)": 0.10
    },
    "Middle-aged Investor": {
        "Reliance": 0.18, "TCS": 0.18, "HDFC Bank": 0.18,
        "Persistent": 0.12, "KPIT Tech": 0.12,
        "Gold BeES (ETF)": 0.10, "Embassy Office Parks REIT": 0.06,
    "REC Bond": 0.06
    },
    "Senior Investor": {
    "HDFC Bank": 0.20,
    "TCS": 0.15,
    "Reliance": 0.15,
    "Gold BeES (ETF)": 0.15,
    "REC Bond": 0.15,
    "NABARD Bond": 0.10,
    "India Grid Trust": 0.10
    }
}

selected = st.sidebar.multiselect(
    "Select Portfolios",
    list(PORTFOLIOS.keys()),
    default=list(PORTFOLIOS.keys())
)

if not selected:
    st.warning("⚠ Please select at least one portfolio.")
    st.stop()

# =====================================================
# PORTFOLIO WEIGHTS TABLE
# =====================================================

st.subheader("⚖️ Portfolio Asset Weights")

# Define MASTER ASSET LIST (all possible assets in system)
MASTER_ASSETS = sorted(
    set(asset for p in PORTFOLIOS.values() for asset in p.keys())
)

# Build weight table
weights_df = pd.DataFrame(index=MASTER_ASSETS)

for portfolio, assets in PORTFOLIOS.items():
    weights_df[portfolio] = [
        assets.get(asset, 0) * 100 for asset in MASTER_ASSETS
    ]

# Add total allocation column (for sorting)
weights_df["Total Allocation (%)"] = weights_df.sum(axis=1)

# Sort assets by importance
weights_df = weights_df.sort_values(
    by="Total Allocation (%)",
    ascending=False
)

# Reset index for display
weights_df = weights_df.reset_index()
weights_df = weights_df.rename(columns={"index": "Asset"})

# Display
st.dataframe(weights_df.round(2))

# =====================================================
# STACKED BAR CHART – PORTFOLIO COMPOSITION
# =====================================================
st.subheader("📊 Portfolio Asset Allocation (Stacked)")

stack_df = weights_df.drop(columns=["Total Allocation (%)"])
stack_df = stack_df.melt(
    id_vars="Asset",
    var_name="Portfolio",
    value_name="Weight %"
)

stack_df = stack_df[stack_df["Weight %"] > 0]

fig = px.bar(
    stack_df,
    x="Portfolio",
    y="Weight %",
    color="Asset",
    title="Portfolio Asset Composition",
    text_auto=True
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PIE CHART – INDIVIDUAL PORTFOLIOS
# =====================================================
st.subheader("🥧 Individual Portfolio Allocation")

pie_portfolio = st.selectbox(
    "Select Portfolio for Pie Chart",
    list(PORTFOLIOS.keys()),
    key="pie_select"
)

pie_data = pd.DataFrame({
    "Asset": list(PORTFOLIOS[pie_portfolio].keys()),
    "Weight %": [v * 100 for v in PORTFOLIOS[pie_portfolio].values()]
})

fig = px.pie(
    pie_data,
    names="Asset",
    values="Weight %",
    title=f"{pie_portfolio} – Asset Allocation",
    hole=0.4
)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PORTFOLIO RETURNS
# =====================================================
portfolio_data = {}

for p in selected:
    total = np.zeros(len(years))
    for asset, w in PORTFOLIOS[p].items():
        if asset in annual_returns_df.index:
            asset_returns = annual_returns_df.loc[asset, years].to_numpy(dtype=float)
            total += asset_returns * w
    portfolio_data[p] = total

portfolio_df = pd.DataFrame(portfolio_data, index=years).copy()

st.subheader("📊 Portfolio Annual Returns (%)")
st.dataframe((portfolio_df * 100).round(2))

# =====================================================
# RISK METRICS
# =====================================================
metrics = []

for p in portfolio_df.columns:
    r = portfolio_df[p]

    mean_r = r.mean()
    vol = r.std()
    downside_vol = r[r < 0].std()

    sharpe = (mean_r - rf_rate) / vol if vol > 0 else 0
    sortino = (mean_r - rf_rate) / downside_vol if downside_vol > 0 else 0

    VaR = np.percentile(r, var_level)
    ES = r[r <= VaR].mean()

    cumulative = (1 + r).cumprod()
    # max_dd = ((cumulative / cumulative.cummax()) - 1).min()
    CAGR = cumulative.iloc[-1] ** (1 / len(r)) - 1

    metrics.append([
        p,
        mean_r * 100,
        CAGR * 100,
        vol * 100,
        sharpe,
        sortino,
        VaR * 100,
        ES * 100
    ])

metrics_df = pd.DataFrame(metrics, columns=[
    "Portfolio", "Avg Return %", "CAGR %",
    "Volatility %", "Sharpe Ratio", "Sortino Ratio",
    f"VaR ({var_conf}) %", "Expected Shortfall %"
])

# =====================================================
# KPI CARDS
# =====================================================
st.subheader("📌 KPI Comparison")

for _, row in metrics_df.iterrows():
    st.markdown(f"### {row['Portfolio']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Return (%)", f"{row['Avg Return %']:.2f}")
    c2.metric("CAGR (%)", f"{row['CAGR %']:.2f}")
    c3.metric("Volatility (%)", f"{row['Volatility %']:.2f}")
    # c4.metric("Max Drawdown (%)", f"{row['Max Drawdown %']:.2f}")

# =====================================================
# AUTO RECOMMENDATION
# =====================================================
best_sharpe = metrics_df.loc[metrics_df["Sharpe Ratio"].idxmax()]
low_risk = metrics_df.loc[metrics_df["Volatility %"].idxmin()]
high_return = metrics_df.loc[metrics_df["CAGR %"].idxmax()]

st.subheader("🧠 Auto Portfolio Recommendation")

st.success(f"✅ Best Risk-Adjusted: **{best_sharpe['Portfolio']}**")
st.info(f"🛡 Lowest Risk: **{low_risk['Portfolio']}**")
st.warning(f"🚀 Highest Return: **{high_return['Portfolio']}**")

# =====================================================
# RISK–RETURN SCATTER (NaN-safe)
# =====================================================
metrics_df["Sharpe Size"] = metrics_df["Sharpe Ratio"].abs().clip(lower=0.1)

fig = px.scatter(
    metrics_df,
    x="Volatility %",
    y="Avg Return %",
    size="Sharpe Size",
    color="Portfolio",
    title="Risk–Return Trade-off"
)
st.plotly_chart(fig)

# =====================================================
# CORRELATION HEATMAP
# =====================================================
st.subheader("🔗 Portfolio Correlation Matrix")

fig = px.imshow(
    portfolio_df.corr(),
    text_auto=".2f",
    color_continuous_scale="RdBu"
)
st.plotly_chart(fig)

# =====================================================
# FOCUS PORTFOLIO (FIXED ERROR)
# =====================================================
st.subheader("📈 Return Distribution & VaR")

focus_portfolio = st.selectbox(
    "Select Portfolio for Distribution Analysis",
    portfolio_df.columns
)

focus_row = metrics_df[metrics_df["Portfolio"] == focus_portfolio].iloc[0]
r = portfolio_df[focus_portfolio]

mu, sigma = r.mean(), r.std()

if sigma > 0:
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    y = norm.pdf(x, mu, sigma)
    VaR_value = np.percentile(r, var_level)

    fig = px.line(
        x=x * 100,
        y=y,
        labels={"x": "Return (%)", "y": "Probability Density"},
        title=f"{focus_portfolio} – Return Distribution"
    )
    fig.add_vline(x=mu * 100, line_dash="dash", annotation_text="Mean")
    fig.add_vline(x=VaR_value * 100, line_color="red",
                  annotation_text=f"VaR {var_conf}")
    st.plotly_chart(fig)

# =====================================================
# MONTE CARLO SIMULATION
# =====================================================
# st.subheader("📈 Monte Carlo Simulation – Future Portfolio Forecast")

# n_simulations = st.slider(
#     "Number of Simulations", 500, 5000, 2000, step=500
# )
# n_years = st.slider(
#     "Forecast Years", 1, 10, 5
# )

# mean_return = r.mean()
# volatility = r.std()

# simulated_returns = np.random.normal(
#     mean_return,
#     volatility,
#     size=(n_simulations, n_years)
# )

# simulated_cumulative = (1 + simulated_returns).cumprod(axis=1)

# # Convert to DataFrame for plotting
# sim_df = pd.DataFrame(
#     simulated_cumulative.T,
#     columns=[f"Sim {i}" for i in range(simulated_cumulative.shape[0])]
# ).copy()

# fig = px.line(
#     sim_df,
#     title=f"Monte Carlo Simulation – {focus_portfolio}",
#     labels={"value": "Portfolio Value", "index": "Year"}
# )
# fig.update_traces(line=dict(width=1), opacity=0.1)
# st.plotly_chart(fig)

# # Monte Carlo Insights
# final_values = simulated_cumulative[:, -1]

# st.info(f"""
# 📊 **Monte Carlo Insights ({n_years} Years)**  
# • Expected Portfolio Value: **{final_values.mean():.2f}x**  
# • Worst 5% Outcome: **{np.percentile(final_values, 5):.2f}x**  
# • Best 95% Outcome: **{np.percentile(final_values, 95):.2f}x**
# """)

# # =====================================================
# # CONFIDENCE CONE (MONTE CARLO)
# # =====================================================
# st.subheader("📉 Monte Carlo Confidence Cone")

# percentiles = [5, 25, 50, 75, 95]

# percentile_values = np.percentile(simulated_cumulative, percentiles, axis=0)

# cone_df = pd.DataFrame(
#     percentile_values.T,
#     columns=[f"P{p}" for p in percentiles]
# ).copy()

# cone_df["Year"] = range(1, n_years + 1)

# fig = px.line(
#     cone_df,
#     x="Year",
#     y=[f"P{p}" for p in percentiles],
#     title=f"Confidence Cone – {focus_portfolio}",
#     labels={"value": "Portfolio Value", "variable": "Percentile"}
# )

# st.plotly_chart(fig, use_container_width=True)

# =====================================================
# CAPM & BETA ANALYSIS (CORRECT MARKET-BASED)
# =====================================================
st.subheader("🧮 CAPM & Beta Analysis")

# ✅ Use Nifty 50 as Market Proxy
if "Nifty 50 Index" not in annual_returns_df.index:
    st.error("❌ Nifty 50 Index data not found for Beta calculation.")
    st.stop()

market_returns = annual_returns_df.loc["Nifty 50 Index", years].astype(float)
portfolio_returns = portfolio_df[focus_portfolio].astype(float)

# Align lengths (safety)
market_returns = market_returns.values
portfolio_returns = portfolio_returns.values

# ✅ Proper Beta formula
covariance = np.cov(portfolio_returns, market_returns)[0, 1]
market_variance = np.var(market_returns)

beta = covariance / market_variance if market_variance > 0 else 0

# CAPM Expected Return
expected_capm_return = rf_rate + beta * (market_returns.mean() - rf_rate)
actual_return = portfolio_returns.mean()

capm_df = pd.DataFrame({
    "Metric": [
        "Beta",
        "Expected Return (CAPM %)",
        "Actual Return (%)"
    ],
    "Value": [
        round(beta, 3),
        round(expected_capm_return * 100, 2),
        round(actual_return * 100, 2)
    ]
})

st.table(capm_df)

# =====================================================
# ALPHA CALCULATION
# =====================================================
alpha = actual_return - expected_capm_return

st.subheader("📊 Alpha Analysis")

st.metric(
    label="Alpha (%)",
    value=f"{alpha * 100:.2f}",
    delta="Outperformance" if alpha > 0 else "Underperformance"
)

st.info(f"""
### 📌 CAPM & Alpha Interpretation – {focus_portfolio}

• **Beta:** {beta:.2f}  
• **Expected Return (CAPM):** {expected_capm_return*100:.2f}%  
• **Actual Return:** {actual_return*100:.2f}%  
• **Alpha:** {alpha*100:.2f}%

{"✅ Positive Alpha → Portfolio manager added value"
 if alpha > 0 else
 "⚠ Negative Alpha → Portfolio underperformed market expectations"}
""")

# =====================================================
# BETA COMPARISON (Portfolio vs Market)
# =====================================================
st.subheader("📐 Portfolio Beta vs Market Beta")

beta_df = pd.DataFrame({
    "Type": ["Market", focus_portfolio],
    "Beta": [1.0, beta]
})

fig = px.bar(
    beta_df,
    x="Type",
    y="Beta",
    color="Type",
    text="Beta",
    title="Systematic Risk Comparison (Beta)"
)

fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.add_hline(y=1, line_dash="dash", annotation_text="Market Beta = 1")

st.plotly_chart(fig, use_container_width=True)

# Interpretation
if beta > 1:
    st.warning("📈 Portfolio is MORE volatile than the market (Aggressive)")
elif beta < 1:
    st.success("🛡 Portfolio is LESS volatile than the market (Defensive)")
else:
    st.info("⚖ Portfolio risk matches the market")

# =====================================================
# PORTFOLIO vs MARKET REGRESSION (CAPM LINE)
# =====================================================
st.subheader("📉 Portfolio vs Market Regression (CAPM)")

reg_df = pd.DataFrame({
    "Market Return (%)": market_returns * 100,
    "Portfolio Return (%)": portfolio_returns * 100
})

fig = px.scatter(
    reg_df,
    x="Market Return (%)",
    y="Portfolio Return (%)",
    trendline="ols",
    title=f"{focus_portfolio}: Portfolio vs Market Returns"
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
📌 **Interpretation**  
• Slope of regression line = **Beta**  
• Steeper line → higher market sensitivity  
• Flatter line → defensive portfolio
""")

# =====================================================
# EFFICIENT FRONTIER (FIXED & CORRECT)
# =====================================================
# st.subheader("📈 Efficient Frontier")

# # Assets used in selected portfolios
# selected_assets = list(
#     set().union(*[PORTFOLIOS[p].keys() for p in selected])
# )

# assets_df = annual_returns_df.loc[
#     annual_returns_df.index.intersection(selected_assets),
#     years
# ].astype(float)

# # Convert to NumPy
# returns_matrix = assets_df.values
# mean_returns = returns_matrix.mean(axis=1)
# cov_matrix = np.cov(returns_matrix)

# n_assets = len(mean_returns)
# n_ports = 3000

# results = np.zeros((n_ports, 3))

# for i in range(n_ports):
#     weights = np.random.random(n_assets)
#     weights /= np.sum(weights)

#     port_return = np.dot(weights, mean_returns)
#     port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
#     sharpe = (port_return - rf_rate) / port_vol if port_vol > 0 else 0

#     results[i] = [port_vol * 100, port_return * 100, sharpe]

# ef_df = pd.DataFrame(
#     results,
#     columns=["Volatility %", "Return %", "Sharpe Ratio"]
# )

# fig = px.scatter(
#     ef_df,
#     x="Volatility %",
#     y="Return %",
#     color="Sharpe Ratio",
#     title="Efficient Frontier (Random Portfolios)",
#     color_continuous_scale="Viridis"
# )

# st.plotly_chart(fig, use_container_width=True)

# =====================================================
# FINAL INSIGHTS
# =====================================================
st.success(f"""
### 🔍 Final Investment Summary – {focus_portfolio}

✔ **Risk–Return Profile:** {"High Growth" if focus_row['Volatility %'] > 20 else "Balanced / Stable"}  
✔ **Volatility Level:** {focus_row['Volatility %']:.2f}%  
✔ **Risk-Adjusted Performance (Sharpe):** {focus_row['Sharpe Ratio']:.2f}

📌 *This portfolio aligns well with its intended risk profile and investment horizon.*
""")
