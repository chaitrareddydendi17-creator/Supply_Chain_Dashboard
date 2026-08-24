"""
Supply Chain Demand & Inventory Dashboard - Streamlit App
------------------------------------------------------------
Run locally:
    pip install streamlit pandas numpy plotly
    streamlit run streamlit_dashboard.py

Deploy for free (shareable link):
    1. Push this file + requirements.txt to a public GitHub repo
    2. Go to https://share.streamlit.io -> "New app" -> pick your repo
    3. You'll get a live URL like https://yourname-supplychain.streamlit.app

Expected CSV columns (matches the Kaggle "Retail Store Inventory
Forecasting Dataset"): Date, Store ID, Product ID, Units Sold,
Inventory Level, Units Ordered  (Category/Region optional)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data
def generate_demo_data():
    rng = np.random.default_rng(42)
    stores = ["Store A", "Store B"]
    products = [
        ("Widget X", "Hardware", 28, 420, 260),
        ("Widget Y", "Hardware", 16, 260, 180),
        ("Gadget Z", "Electronics", 40, 520, 320),
        ("Gadget W", "Electronics", 12, 200, 150),
    ]
    dates = pd.date_range("2026-02-06", periods=180, freq="D")
    rows = []
    for store in stores:
        for name, cat, base, cap, restock in products:
            inventory = cap
            for d in dates:
                weekend_bump = 1.25 if d.dayofweek >= 5 else 1.0
                trend = 1 + (d - dates[0]).days / 180 * 0.15
                noise = rng.uniform(0.75, 1.25)
                units_sold = max(0, round(base * weekend_bump * trend * noise))
                inventory = max(0, inventory - units_sold)
                units_ordered = 0
                if inventory < restock * 0.4:
                    units_ordered = restock + rng.integers(0, 60)
                    inventory += units_ordered
                rows.append([d, store, name, cat, units_sold, inventory, units_ordered])
    return pd.DataFrame(rows, columns=[
        "Date", "Store ID", "Product ID", "Category",
        "Units Sold", "Inventory Level", "Units Ordered"
    ])


def robust_stats(series):
    """Mean/std that ignore extreme spikes (e.g. promo days, holiday surges)
    so a handful of outlier days don't blow up the safety-stock formula."""
    s = series.dropna()
    if len(s) < 5:
        return s.mean(), s.std()
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    clipped = s.clip(lo, hi)
    return clipped.mean(), clipped.std()


st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    df["Date"] = pd.to_datetime(df["Date"])
    is_demo = False
else:
    df = generate_demo_data()
    is_demo = True
    st.sidebar.info("Showing simulated demo data — upload a CSV to use your own.")

# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
st.sidebar.header("Filters")
store = st.sidebar.selectbox("Store", sorted(df["Store ID"].unique()))
products_in_store = sorted(df[df["Store ID"] == store]["Product ID"].unique())
product = st.sidebar.selectbox("Product", products_in_store)

st.sidebar.header("Assumptions")
lead_time = st.sidebar.number_input("Lead time (days)", min_value=1, max_value=30, value=5)
service_z = st.sidebar.selectbox("Service level", [1.28, 1.65, 2.05],
                                  format_func=lambda z: f"{z} (~{'90%' if z==1.28 else '95%' if z==1.65 else '98%'})",
                                  index=1)

# ----------------------------------------------------------------------------
# Compute KPIs for selected store/product
# ----------------------------------------------------------------------------
sub = df[(df["Store ID"] == store) & (df["Product ID"] == product)].sort_values("Date")
sub["MA_7"] = sub["Units Sold"].rolling(7).mean()
sub["MA_30"] = sub["Units Sold"].rolling(30).mean()

avg_daily, std_daily = robust_stats(sub["Units Sold"])
safety_stock = service_z * std_daily * np.sqrt(lead_time)
reorder_point = avg_daily * lead_time + safety_stock
current_inventory = sub["Inventory Level"].iloc[-1] if "Inventory Level" in sub else np.nan
days_of_cover = current_inventory / avg_daily if avg_daily > 0 else np.nan

# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
st.title("Demand & Inventory Dashboard")
st.caption("Supply chain analytics — demand forecasting, safety stock & reorder points")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg Daily Demand", f"{avg_daily:.1f}")
c2.metric("Demand Volatility (σ)", f"{std_daily:.1f}")
c3.metric("Safety Stock", f"{safety_stock:.0f}")
c4.metric("Reorder Point", f"{reorder_point:.0f}")
c5.metric("Days of Cover", f"{days_of_cover:.1f}",
          delta=None if pd.isna(days_of_cover) else f"{'below' if days_of_cover < lead_time else 'ok'} lead time")

st.subheader(f"{product} — {store} — daily demand")
fig = go.Figure()
fig.add_trace(go.Scatter(x=sub["Date"], y=sub["Units Sold"], name="Actual",
                          line=dict(color="#8891A0", width=1), opacity=0.5))
fig.add_trace(go.Scatter(x=sub["Date"], y=sub["MA_7"], name="7-day avg",
                          line=dict(color="#2C6E63", width=2)))
fig.add_trace(go.Scatter(x=sub["Date"], y=sub["MA_30"], name="30-day avg",
                          line=dict(color="#28344A", width=2, dash="dash")))
fig.add_hline(y=reorder_point, line_dash="dot", line_color="#B23A20",
              annotation_text="Reorder point", annotation_position="top left")
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                   plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Inventory runway across all products in the selected store
# ----------------------------------------------------------------------------
st.subheader(f"Inventory runway — {store}")

runway_rows = []
for p in products_in_store:
    p_df = df[(df["Store ID"] == store) & (df["Product ID"] == p)].sort_values("Date")
    p_avg, p_std = robust_stats(p_df["Units Sold"])
    p_ss = service_z * p_std * np.sqrt(lead_time)
    p_rop = p_avg * lead_time + p_ss
    p_current = p_df["Inventory Level"].iloc[-1] if "Inventory Level" in p_df else np.nan
    status = "Reorder now" if p_current <= p_rop else ("Watch" if p_current <= p_rop * 1.25 else "OK")
    runway_rows.append({
        "Product": p, "Current Inventory": int(round(p_current, 0)),
        "Reorder Point": int(round(p_rop, 0)), "Status": status
    })

runway_df = pd.DataFrame(runway_rows)


def highlight_status(row):
    if row["Status"] == "Reorder now":
        bg, text = "#F5DCD4", "#7A2A16"
    elif row["Status"] == "Watch":
        bg, text = "#F3E4CB", "#7A5310"
    else:
        bg, text = "#DCEAE6", "#1E4A42"
    return [f"background-color: {bg}; color: {text}; font-weight: 500"] * len(row)


st.dataframe(runway_df.style.apply(highlight_status, axis=1), use_container_width=True, hide_index=True)

st.caption("Red = place a purchase order now. Amber = approaching reorder point. Green = healthy stock.")
