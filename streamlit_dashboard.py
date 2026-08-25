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

Works with (almost) any retail/inventory CSV. On upload, the app asks
you to confirm which of your columns map to Date / Store / Product /
Units Sold / etc. Only Date, Store, Product, and Units Sold are
required - everything else is optional and the app degrades gracefully
if it's missing.

Design notes (why this file is built the way it is - kept as comments
so this reads like reasoning, not just code):

  * "Inventory Level" and "Units Ordered" columns in most public retail
    datasets (incl. the Kaggle "Retail Store Inventory Forecasting
    Dataset") turn out not to behave like a real depleting/replenishing
    stock balance - they're generated independently of each other and of
    Units Sold. Verified this two ways: (1) reconstructing a cumulative
    balance from them collapses to zero for most SKUs, meaning Units
    Ordered doesn't track real restocking; (2) even trusting the raw
    Inventory Level column directly, its typical scale (often only ~1-2
    days of average demand) is structurally too low to ever clear a
    realistic lead-time + safety-stock reorder point, regardless of what
    lead time is assumed - so literally every SKU gets flagged, which is
    a data-compatibility problem, not a real network-wide crisis.
    Rather than force a broken assumption onto data that doesn't support
    it, this app SIMULATES inventory forward from each SKU's real demand
    history under an explicit, stated (s, S) reorder-point policy. See
    simulate_policy_inventory(). This is a deliberate, disclosed choice:
    "Current Inventory" here means "what inventory would look like under
    this stated policy, given your real demand" - not a reported fact
    from the source file.

  * "Demand forecasting" needs an actual forward-looking prediction, not
    just a trailing moving average. See forecast_demand() - a simple,
    explainable weekly-seasonality + exponential smoothing forecast.
    Deliberately NOT using a black-box model: a supply chain audience
    trusts simple/explainable methods more than opaque ones, and the
    story here is business judgment, not ML sophistication.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

# ----------------------------------------------------------------------------
# 1. Column mapping - makes the app work with (almost) any dataset
# ----------------------------------------------------------------------------
CANDIDATE_COLUMNS = {
    "Date": ["date", "order date", "week", "day", "period"],
    "Store ID": ["store id", "store_id", "storeid", "store", "location", "shop", "outlet",
                 "warehouse", "branch", "region"],
    "Product ID": ["product id", "product_id", "productid", "product", "sku", "item id", "item"],
    "Units Sold": ["units sold", "sales", "demand", "qty sold", "quantity sold", "sold", "sales qty"],
    "Inventory Level": ["inventory level", "inventory", "stock", "on hand", "stock level", "on-hand"],
    "Units Ordered": ["units ordered", "ordered", "replenishment", "reorder qty", "purchase qty", "po qty"],
    "Category": ["category", "product category", "type", "segment"],
    "Price": ["price", "unit price", "selling price", "unit cost", "cost"],
}
REQUIRED_FIELDS = ["Date", "Store ID", "Product ID", "Units Sold"]
OPTIONAL_FIELDS = ["Inventory Level", "Units Ordered", "Category", "Price"]


def guess_column(canonical_name, actual_columns):
    """Best-effort auto-match of a canonical field to an uploaded column name."""
    candidates = CANDIDATE_COLUMNS[canonical_name]
    lowered = {c: c.lower().strip() for c in actual_columns}
    for cand in candidates:
        for col, low in lowered.items():
            if low == cand:
                return col
    for cand in candidates:
        for col, low in lowered.items():
            if cand in low:
                return col
    return None


@st.cache_data
def generate_demo_data():
    """Synthetic fallback dataset so the app is never blank on first load."""
    rng = np.random.default_rng(42)
    stores = ["Store A", "Store B"]
    products = [
        ("Widget X", "Hardware", 28, 420, 260, 18.0),
        ("Widget Y", "Hardware", 16, 260, 180, 24.5),
        ("Gadget Z", "Electronics", 40, 520, 320, 42.0),
        ("Gadget W", "Electronics", 12, 200, 150, 89.0),
    ]
    dates = pd.date_range("2026-02-06", periods=180, freq="D")
    rows = []
    for store in stores:
        for name, cat, base, cap, restock, price in products:
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
                rows.append([d, store, name, cat, units_sold, inventory, units_ordered, price])
    return pd.DataFrame(rows, columns=[
        "Date", "Store ID", "Product ID", "Category",
        "Units Sold", "Inventory Level", "Units Ordered", "Price"
    ])


st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

SELECT_PLACEHOLDER = "— select a column —"

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    is_demo = False

    st.sidebar.subheader("Map your columns")
    st.sidebar.caption("Auto-detected where confident - please confirm these are right before continuing.")
    mapping = {}
    actual_cols = list(raw.columns)
    for field in REQUIRED_FIELDS:
        guess = guess_column(field, actual_cols)
        options = [SELECT_PLACEHOLDER] + actual_cols
        # Only pre-select a real column if we found a genuine match - never
        # silently default to "whatever column happens to be first in the
        # file" for a required field. A wrong required-field guess corrupts
        # every downstream number, so an unconfident guess must force the
        # user to actively choose rather than picking for them.
        idx = options.index(guess) if guess in actual_cols else 0
        choice = st.sidebar.selectbox(f"{field} *", options, index=idx, key=f"map_{field}")
        mapping[field] = None if choice == SELECT_PLACEHOLDER else choice
    for field in OPTIONAL_FIELDS:
        guess = guess_column(field, actual_cols)
        options = ["(none)"] + actual_cols
        idx = options.index(guess) if guess in actual_cols else 0
        choice = st.sidebar.selectbox(field, options, index=idx, key=f"map_{field}")
        mapping[field] = None if choice == "(none)" else choice

    # Validate before doing anything else with the data. Two failure modes
    # to catch explicitly rather than let silently corrupt every downstream
    # number: a required field left unmapped, or two different fields
    # accidentally pointing at the same source column (e.g. Store ID and
    # Category both mapped to "Category").
    missing_required = [f for f in REQUIRED_FIELDS if mapping.get(f) is None]
    chosen = [c for c in mapping.values() if c is not None]
    duplicates = sorted({c for c in chosen if chosen.count(c) > 1})

    if missing_required:
        st.error(f"Please map the required column(s) in the sidebar: {', '.join(missing_required)}.")
        st.stop()
    if duplicates:
        st.error(
            f"These source column(s) are mapped to more than one field, which will corrupt the numbers: "
            f"**{', '.join(duplicates)}**. Please make each field in the sidebar point to a different column."
        )
        st.stop()

    df = pd.DataFrame()
    for canonical, source_col in mapping.items():
        if source_col is not None:
            df[canonical] = raw[source_col]
    df["Date"] = pd.to_datetime(df["Date"])
    for opt_col in ["Inventory Level", "Units Ordered", "Category", "Price"]:
        if opt_col not in df.columns:
            df[opt_col] = np.nan
else:
    df = generate_demo_data()
    is_demo = True
    st.sidebar.info("Showing simulated demo data - upload a CSV to use your own.")

has_price = df["Price"].notna().any()

# ----------------------------------------------------------------------------
# 2. Sidebar controls / assumptions
# ----------------------------------------------------------------------------
st.sidebar.header("Filters")
store = st.sidebar.selectbox("Store", sorted(df["Store ID"].unique()))
products_in_store = sorted(df[df["Store ID"] == store]["Product ID"].unique())
product = st.sidebar.selectbox("Product", products_in_store)

st.sidebar.header("Assumptions")

# Data-driven default lead time: hardcoding "5 days" (fine for the built-in
# demo data) breaks on real datasets where on-hand inventory only ever
# covers a day or two of demand - every SKU ends up flagged "reorder now"
# not because anything is wrong, but because the lead-time assumption
# doesn't match how that dataset represents stock. So: estimate typical
# days of on-hand cover from the data itself and use that as the default,
# while still letting the user override it if they know the real supplier
# lead time.
if df["Inventory Level"].notna().any():
    per_sku_cover = df.groupby(["Store ID", "Product ID"]).agg(
        inv_mean=("Inventory Level", "mean"), sold_mean=("Units Sold", "mean")
    )
    per_sku_cover["cover_days"] = per_sku_cover["inv_mean"] / per_sku_cover["sold_mean"].replace(0, np.nan)
    median_cover = per_sku_cover["cover_days"].median(skipna=True)
    suggested_lead_time = int(np.clip(round(median_cover), 1, 30)) if pd.notna(median_cover) else 5
else:
    suggested_lead_time = 5

lead_time = st.sidebar.number_input("Lead time (days)", min_value=1, max_value=60, value=suggested_lead_time)
st.sidebar.caption(
    f"Defaulted to {suggested_lead_time} day(s), estimated from typical on-hand cover in your data "
    "(median Inventory Level ÷ average daily demand, across all SKUs). Override if you know the real "
    "supplier lead time - just be aware a lead time much longer than typical on-hand cover will flag "
    "most SKUs as needing reorder, since the data itself never carries that much buffer stock."
)
service_z = st.sidebar.selectbox(
    "Service level", [1.28, 1.65, 2.05],
    format_func=lambda z: f"{z} (~{'90%' if z == 1.28 else '95%' if z == 1.65 else '98%'})",
    index=1,
)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", 7, 60, 30)
order_multiplier = st.sidebar.slider(
    "Order-up-to buffer (x lead-time demand)", 1.0, 4.0, 1.5, step=0.5
)
st.sidebar.caption(
    "Controls how much extra stock is ordered above the reorder point each cycle. Lower = leaner but more "
    "frequent reorder flags, especially for volatile SKUs. Higher = fewer urgent flags but more holding cost."
)

st.sidebar.subheader("Cost assumptions")
st.sidebar.caption("Used for $ risk estimates - override if you know real figures.")
default_unit_cost = float(df["Price"].dropna().mean()) if has_price else 25.0
unit_cost = st.sidebar.number_input("Assumed unit cost ($)", min_value=0.1, value=round(default_unit_cost, 2))
holding_rate = st.sidebar.slider("Annual holding cost rate (% of unit cost)", 5, 40, 20) / 100
stockout_margin_multiplier = st.sidebar.slider(
    "Stockout cost multiplier (x unit cost, lost-sale + goodwill)", 1.0, 5.0, 2.0
)

# ----------------------------------------------------------------------------
# 3. Core calculations
# ----------------------------------------------------------------------------
def robust_stats(series):
    """Mean/std that ignore extreme spikes (e.g. promo days, holiday surges)
    so a handful of outlier days don't blow up the safety-stock formula."""
    s = series.dropna()
    if len(s) < 5:
        return s.mean(), s.std()
    lo, hi = s.quantile(0.05), s.quantile(0.95)
    clipped = s.clip(lo, hi)
    return clipped.mean(), clipped.std()


def simulate_policy_inventory(dates, demand, lead_time, service_z, avg_daily, std_daily, order_multiplier=1.5):
    """Simulate on-hand inventory forward from REAL demand under an explicit
    (s, S) reorder-point policy, rather than trusting the source file's
    Inventory Level / Units Ordered columns (see design notes at top of
    file for why those can't be trusted across most public datasets).

    s (reorder point)  = lead-time demand + safety stock, same formula used
                          everywhere else in this app.
    S (order-up-to)     = s + order_multiplier x one lead-time's worth of
                          average demand. order_multiplier is a user-facing
                          lever: too small and volatile SKUs spend most of
                          their time flagged "reorder now" simply because
                          each order barely covers one cycle; larger values
                          trade more holding cost for fewer urgent flags.

    Mechanics: inventory depletes by that day's actual demand. Whenever it
    drops to/below s, a replenishment order for (S - inventory) is placed
    and arrives `lead_time` days later. Simplification: only one order can
    be in transit at a time (reasonable for a single-supplier, single-lead-
    time model; noted as a limitation in the UI).

    Returns the day-by-day simulated inventory array plus s and S, so the
    detail view can plot the resulting sawtooth alongside the reorder
    point line.
    """
    n = len(demand)
    demand = np.nan_to_num(np.asarray(demand, dtype=float), nan=0.0)
    s = avg_daily * lead_time + service_z * (std_daily or 0) * np.sqrt(lead_time)
    S = s + avg_daily * lead_time * order_multiplier
    inv = S  # assume the SKU starts fully stocked
    pending = None  # (arrival_index, qty) - at most one order in transit
    levels = np.empty(n)
    for i in range(n):
        if pending is not None and pending[0] == i:
            inv += pending[1]
            pending = None
        inv = max(inv - demand[i], 0.0)
        if inv <= s and pending is None:
            qty = max(S - inv, 0.0)
            arrival = i + lead_time
            if arrival < n:
                pending = (arrival, qty)
        levels[i] = inv
    return levels, s, S


def forecast_demand(sub, horizon):
    """Simple, explainable forecast: weekly-seasonality factors applied on
    top of exponentially-smoothed demand level. Not a black-box model -
    deliberately chosen so the method is easy to explain in an interview."""
    d = sub[["Date", "Units Sold"]].dropna()
    if len(d) < 14:
        last_val = d["Units Sold"].mean() if len(d) else 0
        future_dates = pd.date_range(sub["Date"].max() + pd.Timedelta(days=1), periods=horizon)
        return future_dates, np.full(horizon, last_val), d["Units Sold"].std() or 0

    d = d.copy()
    d["dow"] = d["Date"].dt.dayofweek
    overall_mean = d["Units Sold"].mean() or 1
    dow_factor = (d.groupby("dow")["Units Sold"].mean() / overall_mean).to_dict()

    alpha = 0.3
    level = d["Units Sold"].iloc[0] / dow_factor.get(d["dow"].iloc[0], 1.0)
    fitted = []
    for _, row in d.iterrows():
        deseasoned = row["Units Sold"] / dow_factor.get(row["dow"], 1.0)
        level = alpha * deseasoned + (1 - alpha) * level
        fitted.append(level * dow_factor.get(row["dow"], 1.0))
    residual_std = (d["Units Sold"] - pd.Series(fitted, index=d.index)).std()

    future_dates = pd.date_range(sub["Date"].max() + pd.Timedelta(days=1), periods=horizon)
    forecast_vals = [level * dow_factor.get(dt.dayofweek, 1.0) for dt in future_dates]
    return future_dates, np.array(forecast_vals), (residual_std if not np.isnan(residual_std) else 0)


@st.cache_data(show_spinner=False)
def compute_all_sku_metrics(df, lead_time, service_z, unit_cost_default, holding_rate, stockout_mult, has_price, order_multiplier):
    """One pass over every Store/Product combo - powers the portfolio view."""
    rows = []
    for (s, p), grp in df.groupby(["Store ID", "Product ID"]):
        grp = grp.sort_values("Date")
        avg_daily, std_daily = robust_stats(grp["Units Sold"])
        levels, reorder_point, order_up_to = simulate_policy_inventory(
            grp["Date"].to_numpy(), grp["Units Sold"].to_numpy(), lead_time, service_z, avg_daily, std_daily, order_multiplier
        )
        safety_stock = service_z * (std_daily or 0) * np.sqrt(lead_time)
        current_inv = levels[-1]
        days_of_cover = current_inv / avg_daily if avg_daily > 0 else np.nan

        if avg_daily > 0:
            days_to_breach = max((current_inv - reorder_point) / avg_daily, 0)
        else:
            days_to_breach = np.nan

        if current_inv <= reorder_point:
            status = "Reorder now"
        elif current_inv <= reorder_point * 1.25:
            status = "Watch"
        else:
            status = "OK"

        cv = (std_daily / avg_daily) if avg_daily > 0 else 0
        variability_tag = "X (steady)" if cv < 0.5 else ("Y (moderate)" if cv < 1.0 else "Z (volatile)")

        sku_price = grp["Price"].dropna().mean() if grp["Price"].notna().any() else unit_cost_default
        annual_revenue_proxy = avg_daily * 365 * sku_price

        holding_cost = safety_stock * sku_price * holding_rate
        shortfall = max(reorder_point - current_inv, 0)
        stockout_cost_estimate = shortfall * sku_price * stockout_mult if status != "OK" else 0.0

        rows.append({
            "Store ID": s, "Product ID": p, "Category": grp["Category"].dropna().iloc[0] if grp["Category"].notna().any() else "-",
            "Avg Daily Demand": avg_daily, "Demand Std": std_daily, "CV": cv, "Variability (XYZ)": variability_tag,
            "Safety Stock": safety_stock, "Reorder Point": reorder_point,
            "Current Inventory": current_inv, "Days of Cover": days_of_cover, "Days to Reorder Breach": days_to_breach,
            "Status": status, "Unit Price ($)": sku_price, "Annual Revenue Proxy ($)": annual_revenue_proxy,
            "Holding Cost/yr ($)": holding_cost, "Est. Stockout Cost ($)": stockout_cost_estimate,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("Annual Revenue Proxy ($)", ascending=False).reset_index(drop=True)
    out["Cum Revenue %"] = out["Annual Revenue Proxy ($)"].cumsum() / out["Annual Revenue Proxy ($)"].sum() * 100
    out["Value Tier (ABC)"] = np.where(out["Cum Revenue %"] <= 80, "A",
                                 np.where(out["Cum Revenue %"] <= 95, "B", "C"))
    out["Segment"] = out["Value Tier (ABC)"] + out["Variability (XYZ)"].str[0]
    return out


portfolio = compute_all_sku_metrics(df, lead_time, service_z, unit_cost, holding_rate, stockout_margin_multiplier, has_price, order_multiplier)

# Self-check: even with a simulated policy, if almost everything still
# flags, say so explicitly rather than silently presenting numbers that
# all look alarming (e.g. can happen with very bursty/volatile demand
# pushing safety stock requirements very high across the board).
frac_reorder = (portfolio["Status"] == "Reorder now").mean()
median_cover_days = portfolio["Days of Cover"].median(skipna=True)
if frac_reorder > 0.6 and pd.notna(median_cover_days):
    st.warning(
        f"⚠️ {frac_reorder * 100:.0f}% of SKUs are flagged **Reorder now** under a {lead_time}-day lead time, "
        f"current service-level, and {order_multiplier}x order-up-to buffer. Median simulated days of cover across "
        f"the network is **{median_cover_days:.1f} days**. If this feels too aggressive, try a shorter lead time, "
        f"a lower service level, or a larger order-up-to buffer in the sidebar - high-CV demand combined with a "
        f"tight order-up-to buffer means a SKU spends most of its cycle below the reorder point by construction."
    )

# ----------------------------------------------------------------------------
# 4. Layout - plain-English summary first, for anyone skimming
# ----------------------------------------------------------------------------
st.title("Demand & Inventory Dashboard")
st.caption("Supply chain analytics - demand forecasting, safety stock, reorder points & $ risk")
if not has_price:
    st.caption(f"No price column found - using an assumed unit cost of ${unit_cost:,.2f} for all $ estimates (editable in the sidebar).")

n_reorder = int((portfolio["Status"] == "Reorder now").sum())
n_watch = int((portfolio["Status"] == "Watch").sum())
n_ok = int((portfolio["Status"] == "OK").sum())
n_total = len(portfolio)
total_stockout_risk = portfolio["Est. Stockout Cost ($)"].sum()
total_holding_cost = portfolio["Holding Cost/yr ($)"].sum()

# --- Executive summary: written for someone with 30 seconds and no supply
# chain background - the KPIs and jargon-heavy tables below are for anyone
# who wants to dig in, but this paragraph should stand alone.
top_risk = portfolio.sort_values("Est. Stockout Cost ($)", ascending=False).iloc[0] if n_total else None
worst_segment = (
    portfolio[portfolio["Status"] == "Reorder now"]["Segment"].value_counts().idxmax()
    if n_reorder > 0 else None
)
summary_bits = [
    f"This dashboard tracks **{n_total} product/store combinations**. Right now, **{n_reorder} need a "
    f"purchase order placed immediately**, **{n_watch} should be watched closely**, and **{n_ok} are healthy**."
]
if total_stockout_risk > 0:
    summary_bits.append(
        f"If the urgent items aren't restocked, we estimate roughly **${total_stockout_risk:,.0f}** in lost-sale "
        f"risk across the network."
    )
if top_risk is not None and top_risk["Status"] != "OK":
    summary_bits.append(
        f"The single biggest risk right now is **{top_risk['Product ID']}** at **{top_risk['Store ID']}** "
        f"(~${top_risk['Est. Stockout Cost ($)']:,.0f} at risk)."
    )
if worst_segment:
    summary_bits.append(
        f"Most urgent items fall in segment **{worst_segment}** (A/B/C = how much revenue a product drives, "
        f"X/Y/Z = how unpredictable its demand is - a 'Z' means demand swings a lot day to day, which is why "
        f"it's harder to keep in stock)."
    )
st.info("**In plain English:** " + " ".join(summary_bits))

k1, k2, k3, k4 = st.columns(4)
k1.metric("SKUs needing reorder now", n_reorder)
k2.metric("SKUs to watch", n_watch)
k3.metric("Est. stockout $ at risk", f"${total_stockout_risk:,.0f}")
k4.metric("Annual safety-stock holding cost", f"${total_holding_cost:,.0f}")

# --- Forward-looking recommendations: broader than any single SKU, meant
# to read like advice a consultant would leave behind, not just a status
# report of what's currently on fire.
st.markdown("#### Recommended next steps")
recs = []
frac_reorder_all = n_reorder / n_total if n_total else 0
z_share = (portfolio["Variability (XYZ)"].str.startswith("Z")).mean() if n_total else 0
az_bz_count = portfolio["Segment"].isin(["AZ", "BZ"]).sum()

if frac_reorder_all > 0.3:
    recs.append(
        f"**Reduce supplier lead time or review order frequency.** {frac_reorder_all*100:.0f}% of SKUs are "
        f"flagged urgent at once - that's usually a sign the replenishment cycle (lead time + order size) is "
        f"too tight for how fast stock moves, not that every product independently went wrong."
    )
if az_bz_count > 0:
    recs.append(
        f"**Give the {az_bz_count} high-value, high-volatility (AZ/BZ) SKUs closer, more frequent review** "
        f"instead of the same flat reorder rule as everything else - they drive real revenue and are the "
        f"hardest to forecast well, so they carry the most downside if mismanaged."
    )
if total_holding_cost > total_stockout_risk * 2 and total_stockout_risk > 0:
    recs.append(
        "**Consider trimming the order-up-to buffer.** Estimated holding cost is well above estimated stockout "
        "risk right now, which can mean more cash is tied up in safety stock than the actual risk justifies."
    )
elif total_stockout_risk > total_holding_cost * 2 and total_holding_cost > 0:
    recs.append(
        "**Consider increasing the order-up-to buffer for urgent SKUs.** Estimated stockout risk is well above "
        "estimated holding cost, suggesting it's currently cheaper to carry a bit more stock than to risk running out."
    )
if not recs:
    recs.append("No major red flags at the network level under current assumptions - spot-check the priority table below for individual SKUs.")
for r in recs:
    st.markdown(f"- {r}")
st.caption("These are generated from the current sidebar assumptions (lead time, service level, order-up-to buffer) - adjust them to stress-test the picture.")

# ----------------------------------------------------------------------------
# 5. Portfolio view - what to look at first, across the whole network
# ----------------------------------------------------------------------------
st.subheader("Portfolio priority view - all SKUs, ranked by urgency")
st.caption("Sorted by estimated $ at risk. A/B/C = share of revenue. X/Y/Z = demand steadiness (X=steady, Z=volatile).")

priority = portfolio.sort_values(["Est. Stockout Cost ($)", "Status"], ascending=[False, True])
display_cols = ["Store ID", "Product ID", "Category", "Segment", "Status",
                 "Current Inventory", "Reorder Point", "Days to Reorder Breach", "Est. Stockout Cost ($)"]
show = priority[display_cols].copy()
show["Current Inventory"] = show["Current Inventory"].round(0).astype(int)
show["Reorder Point"] = show["Reorder Point"].round(0).astype(int)
show["Days to Reorder Breach"] = show["Days to Reorder Breach"].round(1)
show["Est. Stockout Cost ($)"] = show["Est. Stockout Cost ($)"].round(0)


def highlight_status(row):
    if row["Status"] == "Reorder now":
        bg, text = "#F5DCD4", "#7A2A16"
    elif row["Status"] == "Watch":
        bg, text = "#F3E4CB", "#7A5310"
    else:
        bg, text = "#DCEAE6", "#1E4A42"
    return [f"background-color: {bg}; color: {text}; font-weight: 500"] * len(row)


st.dataframe(show.head(15).style.apply(highlight_status, axis=1), use_container_width=True, hide_index=True)
st.caption("Red = place a purchase order now. Amber = approaching reorder point. Green = healthy stock. Showing top 15 by $ at risk.")

# ----------------------------------------------------------------------------
# 6. ABC/XYZ segmentation summary
# ----------------------------------------------------------------------------
st.subheader("ABC/XYZ segmentation")
st.caption("A = top ~80% of revenue, B = next ~15%, C = remaining ~5%. X = steady demand, Y = moderate, Z = volatile. "
           "AZ/BZ SKUs are the highest-risk to manage on a flat reorder rule and deserve tighter, more frequent review.")
seg_summary = portfolio.groupby("Segment").agg(
    SKUs=("Product ID", "count"),
    Avg_CV=("CV", "mean"),
    Total_Revenue_Proxy=("Annual Revenue Proxy ($)", "sum"),
).reset_index().sort_values("Total_Revenue_Proxy", ascending=False)
seg_summary["Total_Revenue_Proxy"] = seg_summary["Total_Revenue_Proxy"].round(0)
seg_summary["Avg_CV"] = seg_summary["Avg_CV"].round(2)
st.dataframe(seg_summary, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# 7. Single-SKU detail view (with forecast)
# ----------------------------------------------------------------------------
st.divider()
st.subheader(f"{product} - {store} - detail view")

sub = df[(df["Store ID"] == store) & (df["Product ID"] == product)].sort_values("Date")
sub["MA_7"] = sub["Units Sold"].rolling(7).mean()

avg_daily, std_daily = robust_stats(sub["Units Sold"])
sim_levels, reorder_point, order_up_to = simulate_policy_inventory(
    sub["Date"].to_numpy(), sub["Units Sold"].to_numpy(), lead_time, service_z, avg_daily, std_daily, order_multiplier
)
sub["Simulated Inventory"] = sim_levels
safety_stock = service_z * (std_daily or 0) * np.sqrt(lead_time)
current_inventory = sim_levels[-1]
days_of_cover = current_inventory / avg_daily if avg_daily > 0 else np.nan

future_dates, forecast_vals, resid_std = forecast_demand(sub, forecast_horizon)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg Daily Demand", f"{avg_daily:.1f}")
c2.metric("Demand Volatility (σ)", f"{std_daily:.1f}")
c3.metric("Safety Stock", f"{safety_stock:.0f}")
c4.metric("Reorder Point", f"{reorder_point:.0f}")
c5.metric("Days of Cover", f"{days_of_cover:.1f}",
          delta=None if pd.isna(days_of_cover) else f"{'below' if days_of_cover < lead_time else 'ok'} lead time")

fig = go.Figure()
fig.add_trace(go.Scatter(x=sub["Date"], y=sub["Units Sold"], name="Actual daily sales",
                          line=dict(color="#8891A0", width=1), opacity=0.5))
fig.add_trace(go.Scatter(x=sub["Date"], y=sub["MA_7"], name="7-day average",
                          line=dict(color="#2C6E63", width=2)))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_vals, name=f"{forecast_horizon}-day forecast",
                          line=dict(color="#28344A", width=2, dash="dash")))
fig.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates[::-1]),
    y=list(forecast_vals + resid_std) + list((forecast_vals - resid_std)[::-1]),
    fill="toself", fillcolor="rgba(40,52,74,0.12)", line=dict(color="rgba(0,0,0,0)"),
    name="Forecast uncertainty range", hoverinfo="skip",
))
fig.add_hline(y=reorder_point, line_dash="dot", line_color="#B23A20",
              annotation_text="Reorder point", annotation_position="bottom right",
              annotation_font=dict(size=13, color="#B23A20"),
              annotation_bgcolor="rgba(255,255,255,0.85)")
fig.update_layout(
    height=430, margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="white", paper_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.35, x=0, font=dict(size=12)),
    font=dict(size=13),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.caption(
    "**How to read this:** gray = actual daily sales. Dark green = 7-day rolling average (smooths out "
    "day-to-day noise so the trend is visible). Dashed navy = the forecast for the next "
    f"{forecast_horizon} days. Shaded band = how uncertain that forecast is - wider means less confident. "
    "Red dotted line = reorder point: the stock level at which a new purchase order should go out."
)

st.markdown(f"**Simulated inventory position under an assumed reorder policy** (s={reorder_point:.0f}, S={order_up_to:.0f})")
st.caption(
    "This is NOT the source file's inventory column - it's what stock on hand would look like if this SKU's "
    "real demand history were managed under a standard reorder-point policy, since the source data's own "
    "inventory fields don't reliably reflect a real depleting/replenishing balance (see limitations below)."
)
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=sub["Date"], y=sub["Simulated Inventory"], name="Simulated stock on hand",
                           line=dict(color="#2C6E63", width=1.5), fill="tozeroy", fillcolor="rgba(44,110,99,0.10)"))
fig2.add_hline(y=reorder_point, line_dash="dot", line_color="#B23A20",
               annotation_text="Reorder point (s)", annotation_position="bottom right",
               annotation_font=dict(size=13, color="#B23A20"),
               annotation_bgcolor="rgba(255,255,255,0.85)")
fig2.add_hline(y=order_up_to, line_dash="dash", line_color="#5B6472",
               annotation_text="Order-up-to (S)", annotation_position="top right",
               annotation_font=dict(size=13, color="#5B6472"),
               annotation_bgcolor="rgba(255,255,255,0.85)")
fig2.update_layout(
    height=300, margin=dict(l=10, r=10, t=30, b=10),
    plot_bgcolor="white", paper_bgcolor="white", font=dict(size=13), showlegend=False,
)
st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
st.caption(
    "**How to read this:** the green sawtooth is simulated stock on hand - it drops as items sell and jumps "
    "back up when a restock arrives. Red dotted line = reorder point (order more once stock falls to here). "
    "Gray dashed line = order-up-to level (how much a restock brings inventory back to). Time spent below the "
    "red line means this SKU is currently waiting on a purchase order."
)

# ----------------------------------------------------------------------------
# 8. Plain-English recommendation (this is the part a manager actually reads)
# ----------------------------------------------------------------------------
sku_row = portfolio[(portfolio["Store ID"] == store) & (portfolio["Product ID"] == product)].iloc[0]

lines = []
lines.append(
    f"**{product} at {store}** sells an average of **{avg_daily:.0f} units/day** "
    f"(volatility σ={std_daily:.0f}, CV={sku_row['CV']:.2f} -> classified **{sku_row['Variability (XYZ)']}**, "
    f"value tier **{sku_row['Value Tier (ABC)']}**)."
)
if sku_row["Status"] == "Reorder now":
    lines.append(
        f"Current simulated inventory is **{current_inventory:.0f} units**, already below the reorder point of "
        f"**{reorder_point:.0f} units**. Recommend placing a purchase order now for at least "
        f"**{max(reorder_point - current_inventory, 0):.0f} units** to restore cover, given a {lead_time}-day lead time. "
        f"Estimated stockout exposure if unaddressed: **${sku_row['Est. Stockout Cost ($)']:,.0f}**."
    )
elif sku_row["Status"] == "Watch":
    lines.append(
        f"Current simulated inventory is **{current_inventory:.0f} units**, within 25% of the reorder point "
        f"(**{reorder_point:.0f} units**). At current demand, this SKU crosses its reorder point in "
        f"roughly **{sku_row['Days to Reorder Breach']:.0f} days** - recommend queuing a purchase order now "
        f"so it lands before the breach, given a {lead_time}-day lead time."
    )
else:
    lines.append(
        f"Current simulated inventory is **{current_inventory:.0f} units**, comfortably above the reorder point "
        f"(**{reorder_point:.0f} units**), giving roughly **{days_of_cover:.0f} days of cover**. No action needed."
    )
if sku_row["Variability (XYZ)"].startswith("Z"):
    lines.append(
        "This SKU's demand is highly volatile - a flat safety-stock formula understates real risk here. "
        "Consider reviewing it more frequently than steadier SKUs, or holding extra buffer beyond the formula."
    )

st.markdown("#### Recommendation")
st.info("\n\n".join(lines))

st.caption(
    "Limitations / assumptions: reorder point assumes stationary (non-seasonal) daily demand and fixed lead time; "
    "$ estimates use an assumed or averaged unit cost and a flat stockout-cost multiplier rather than true margin data; "
    "the inventory chart above is simulated from an assumed reorder policy applied to real demand, not read from the "
    "source file, because the source file's inventory/replenishment columns did not reliably behave like a real "
    "depleting stock balance (checked directly - see design notes in the code); only one purchase order is modeled "
    "in transit at a time."
)
