import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AirFly Insights", page_icon="✈️", layout="wide")

# ====================== STYLING ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; color: #f8fafc; }
.main-hero { background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f2744 100%); border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem; }
.obs-box { background: linear-gradient(90deg, #eff6ff, #f8fbff); border-left: 4px solid #3b82f6; border-radius: 0 8px 8px 0; padding: 0.8rem 1.2rem; font-size: 0.88rem; color: #1e40af; margin: 0.6rem 0 1.5rem 0; }
.section-header { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: #1e3a5f; padding: 0.6rem 0 0.6rem 1rem; border-left: 4px solid #3b82f6; background: linear-gradient(90deg, rgba(59,130,246,0.05), transparent); border-radius: 0 8px 8px 0; }
</style>
""", unsafe_allow_html=True)

# ====================== TITLE ======================
st.markdown('<div class="main-hero"><h1 class="hero-title">✈ AirFly Insights</h1><p style="color:#94a3b8;">2015 US Domestic Flight Dashboard</p></div>', unsafe_allow_html=True)

# ====================== FILE UPLOADER ======================
st.sidebar.header("Upload Data File")
uploaded_file = st.sidebar.file_uploader("Upload airline_preprocessed.parquet (157 MB)", type=["parquet"])

if uploaded_file is None:
    st.warning("⚠️ Please upload the `airline_preprocessed.parquet` file to continue.")
    st.info("Download it from: https://drive.google.com/uc?id=1Xz4srzZ6mRK5GJJqyB3UY8WXQQCLdfrB")
    st.stop()

@st.cache_data(show_spinner="Loading flight data...")
def load_data(uploaded_file):
    df = pd.read_parquet(uploaded_file)
    df.columns = df.columns.str.strip().str.upper()
    if 'ROUTE' not in df.columns:
        df['ROUTE'] = df['ORIGIN_AIRPORT'].astype(str) + "_" + df['DESTINATION_AIRPORT'].astype(str)
    df['SEASON'] = df['MONTH'].apply(lambda m: 'Winter' if m in [12,1,2] else 'Spring' if m in [3,4,5] else 'Summer' if m in [6,7,8] else 'Fall')
    return df

data = load_data(uploaded_file)

# Global Filters
st.sidebar.header("Filters")
airlines_list = sorted(data['AIRLINE'].dropna().unique())
selected_airlines = st.sidebar.multiselect("Airlines", options=airlines_list, default=airlines_list)

month_list = sorted(data['MONTH'].unique())
selected_months = st.sidebar.multiselect("Months", options=month_list, default=month_list, format_func=lambda x: f"{x:02d}")

filtered_data = data[
    (data['AIRLINE'].isin(selected_airlines)) &
    (data['MONTH'].isin(selected_months))
].copy()

# ====================== HELPERS ======================
def obs(txt):
    st.markdown(f'<div class="obs-box">💡 <strong>Insight:</strong> {txt}</div>', unsafe_allow_html=True)

def section(txt):
    st.markdown(f'<div class="section-header">{txt}</div>', unsafe_allow_html=True)

def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

sns.set_theme(style="whitegrid", font_scale=0.95)

# ====================== PAGES ======================
page = st.sidebar.radio("Navigation", ["🏠 Overview", "📊 Milestone 2 – Delay Analysis", "🚫 Milestone 3 – Cancellations & Routes"])

if "Overview" in page:
    st.success("✅ Data loaded successfully!")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Flights", f"{len(filtered_data):,}")
    with c2: st.metric("Airlines", filtered_data['AIRLINE'].nunique())
    with c3: st.metric("Unique Routes", filtered_data['ROUTE'].nunique())
    with c4: st.metric("Cancellation Rate", f"{filtered_data['CANCELLED'].mean()*100:.1f}%")

# ====================== MILESTONE 2 ======================
elif "Milestone 2" in page:
    st.subheader("📊 Milestone 2 – Delay Analysis")

    # 1. Top 15 Airports
    section("1 · Top 15 Airports with Highest Average Departure Delay")
    ap = filtered_data[filtered_data['CANCELLED']==0].groupby('ORIGIN_AIRPORT').agg(
        avg_delay=('DEPARTURE_DELAY','mean'), count=('DEPARTURE_DELAY','count')).reset_index()
    ap = ap[ap['count'] > 1000]
    top15 = ap.nlargest(15, 'avg_delay')
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(top15['ORIGIN_AIRPORT'], top15['avg_delay'], color='tomato')
    ax.set_title("Top 15 Airports – Highest Avg Departure Delay")
    ax.set_xlabel("Avg Delay (minutes)")
    ax.invert_yaxis()
    plt.tight_layout()
    show(fig)
    obs("Mid-size congested airports show worse delays than major hubs.")

    # 2. Monthly Trend
    section("2 · Monthly Flight Volume Trend")
    m = filtered_data.groupby('MONTH').size()
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(m.index, m.values, marker='o', color='#3b82f6')
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_title("Monthly Flight Volume Trend")
    plt.tight_layout()
    show(fig)

    # 3. Route Congestion
    section("3 · Route Congestion vs Average Arrival Delay")
    rs = filtered_data.groupby('ROUTE').agg(ARRIVAL_DELAY=('ARRIVAL_DELAY','mean'), flight_count=('ROUTE','count')).reset_index()
    rs = rs[rs['flight_count'] > 500]
    fig, ax = plt.subplots(figsize=(10,5))
    sc = ax.scatter(rs['flight_count'], rs['ARRIVAL_DELAY'], c=rs['ARRIVAL_DELAY'], cmap='RdYlGn_r', s=rs['flight_count']/100, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Avg Delay (min)')
    ax.set_title("Route Congestion vs Average Delay")
    ax.set_xlabel("Number of Flights")
    ax.set_ylabel("Avg Arrival Delay (min)")
    plt.tight_layout()
    show(fig)

    # 4. Delay Distribution by Top 5 Airlines
    section("4 · Delay Distribution by Top 5 Airlines")
    top5 = filtered_data['AIRLINE'].value_counts().head(5).index
    flt = filtered_data[filtered_data['AIRLINE'].isin(top5)]
    flt = flt[flt['ARRIVAL_DELAY'].between(-30,120)]
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=flt, x='AIRLINE', y='ARRIVAL_DELAY', palette='Set2', ax=ax)
    ax.axhline(0, color='red', linestyle='--', label='On Time')
    ax.set_title("Arrival Delay Distribution — Top 5 Airlines")
    ax.legend()
    plt.tight_layout()
    show(fig)

    # 5. Top 10 Busiest Routes
    section("5 · Top 10 Busiest Routes")
    tr = filtered_data['ROUTE'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(tr.index, tr.values, color='orange')
    ax.set_title("Top 10 Busiest Routes")
    ax.invert_yaxis()
    plt.tight_layout()
    show(fig)

    # 6. Stacked Delay Causes
    section("6 · Average Delay Causes by Airline")
    cols = [c for c in ['AIR_SYSTEM_DELAY','SECURITY_DELAY','WEATHER_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY'] if c in filtered_data.columns]
    dc = filtered_data.groupby('AIRLINE')[cols].mean().head(10)
    fig, ax = plt.subplots(figsize=(12,6))
    dc.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title("Average Delay Causes by Airline (minutes)")
    plt.tight_layout()
    show(fig)

    # 7. Histograms
    section("7 · Arrival & Departure Delay Distributions")
    col1, col2 = st.columns(2)
    with col1:
        da = filtered_data['ARRIVAL_DELAY'].dropna()
        da = da[(da>=-60)&(da<=180)]
        fig, ax = plt.subplots()
        ax.hist(da, bins=80, color='#3b82f6', alpha=0.85)
        ax.set_title("Arrival Delay Distribution")
        show(fig)
    with col2:
        dd = filtered_data['DEPARTURE_DELAY'].dropna()
        dd = dd[(dd>=-60)&(dd<=180)]
        fig, ax = plt.subplots()
        ax.hist(dd, bins=80, color='#f97316', alpha=0.85)
        ax.set_title("Departure Delay Distribution")
        show(fig)

# ====================== MILESTONE 3 ======================
elif "Milestone 3" in page:
    st.subheader("🚫 Milestone 3 – Cancellations & Route Analysis")

    # 1. Cancellation by Season
    section("1 · Cancellation Reasons by Season")
    season_cancel = filtered_data[filtered_data['CANCELLED']==1].groupby(['SEASON','CANCELLATION_REASON']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10,5))
    season_cancel.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title("Cancellation Reasons by Season")
    plt.tight_layout()
    show(fig)
    obs("Winter has 3x more weather cancellations than any other season.")

    # 2. Top Cancelled Routes
    section("2 · Top 10 Routes with Highest Cancellation Rate")
    rc = filtered_data.groupby('ROUTE').agg(cancel_rate=('CANCELLED','mean'), cnt=('CANCELLED','count')).reset_index()
    rc = rc[rc['cnt']>1000]
    rc['cancel_pct'] = rc['cancel_rate']*100
    top10c = rc.nlargest(10, 'cancel_pct')
    fig, ax = plt.subplots(figsize=(11,5))
    ax.barh(top10c['ROUTE'], top10c['cancel_pct'], color='crimson')
    ax.set_title("Top 10 Routes – Highest Cancellation Rate")
    ax.invert_yaxis()
    plt.tight_layout()
    show(fig)

    # 3. Monthly Cancellation Trend
    section("3 · Monthly Cancellation Trend by Reason")
    mc = filtered_data[filtered_data['CANCELLED']==1].groupby(['MONTH','CANCELLATION_REASON']).size().unstack(fill_value=0)
    mc.index = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][:len(mc)]
    fig, ax = plt.subplots(figsize=(12,6))
    mc.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title("Monthly Cancellation Trend by Reason")
    plt.tight_layout()
    show(fig)

# Footer
st.markdown("---")
st.caption("AirFly Insights • All charts from Milestone 2 & 3 • Upload parquet file to run")
