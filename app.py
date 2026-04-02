import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AirFly Insights", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

# ====================== STYLING ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(160deg, #0a0f1e 0%, #0d1b35 60%, #0a1628 100%); }
.main-hero { background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f2744 100%); border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; color: #f8fafc; }
.metric-card { background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid rgba(59,130,246,0.2); border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
.metric-val { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: #60a5fa; }
.obs-box { background: linear-gradient(90deg, #eff6ff, #f8fbff); border-left: 4px solid #3b82f6; border-radius: 0 8px 8px 0; padding: 0.8rem 1.2rem; font-size: 0.88rem; color: #1e40af; margin: 0.6rem 0 1.5rem 0; }
.section-header { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: #1e3a5f; padding: 0.6rem 0 0.6rem 1rem; border-left: 4px solid #3b82f6; background: linear-gradient(90deg, rgba(59,130,246,0.05), transparent); border-radius: 0 8px 8px 0; }
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown('<div style="font-family:Syne;font-size:1.4rem;font-weight:800;color:#60a5fa;">✈ AirFly Insights</div>', unsafe_allow_html=True)
    st.caption("2015 US Domestic Flights • 5.8M records")
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Overview", "📊 Milestone 2 – Delay Analysis", "🚫 Milestone 3 – Cancellations & Routes"])
    st.markdown("---")
    st.subheader("🔍 Global Filters")
    st.caption("Filters apply to all charts below")

# ====================== LOAD DATA ======================
@st.cache_data(show_spinner="Loading 5.8M flights...")
def load_data():
    import gdown
    FILE_ID = "1Xz4srzZ6mRK5GJJqyB3UY8WXQQCLdfrB"
    dest = "airline_preprocessed.parquet"
    if not os.path.exists(dest):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", dest, quiet=False)
    df = pd.read_parquet(dest)
    df.columns = df.columns.str.strip().str.upper()
    if 'ROUTE' not in df.columns:
        df['ROUTE'] = df['ORIGIN_AIRPORT'].astype(str) + "_" + df['DESTINATION_AIRPORT'].astype(str)
    df['SEASON'] = df['MONTH'].apply(lambda m: 'Winter' if m in [12,1,2] else 'Spring' if m in [3,4,5] else 'Summer' if m in [6,7,8] else 'Fall')
    return df

data = load_data()

# Global Filters
airlines_list = sorted(data['AIRLINE'].dropna().unique())
selected_airlines = st.sidebar.multiselect("Select Airlines", options=airlines_list, default=airlines_list)

month_list = sorted(data['MONTH'].unique())
selected_months = st.sidebar.multiselect("Select Months", options=month_list, default=month_list, format_func=lambda x: f"{x:02d}")

# Filtered Data
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

BG = "#fafbff"
sns.set_theme(style="whitegrid", font_scale=0.95)

# ====================== PAGES ======================
if "Overview" in page:
    st.markdown("""<div class="main-hero"><h1 class="hero-title">AirFly Insights</h1><p style="color:#94a3b8;">Interactive 2015 US Domestic Flight Dashboard</p></div>""", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Total Flights", f"{len(filtered_data):,}")
    with c2: st.metric("Airlines", filtered_data['AIRLINE'].nunique())
    with c3: st.metric("Unique Routes", filtered_data['ROUTE'].nunique())
    with c4: st.metric("Cancellation Rate", f"{filtered_data['CANCELLED'].mean()*100:.1f}%")

# ====================== MILESTONE 2 ======================
elif "Milestone 2" in page:
    st.markdown("""<div class="main-hero"><span style="background:#3b82f6;color:white;padding:4px 12px;border-radius:20px;font-size:0.85rem;">Milestone 2</span><h1 class="hero-title" style="font-size:1.9rem;">Delay Analysis</h1></div>""", unsafe_allow_html=True)

    # 1. Top 15 Airports Highest Avg Departure Delay
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
    obs("Mid-size congested airports show worse delays than major hubs which have better scheduling buffers.")

    # 2. Monthly Flight Volume Trend
    section("2 · Monthly Flight Volume Trend")
    m = filtered_data.groupby('MONTH').size()
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(m.index, m.values, marker='o', color='#3b82f6', linewidth=2.5)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_title("Monthly Flight Volume Trend")
    ax.set_ylabel("Total Flights")
    plt.tight_layout()
    show(fig)
    obs("Two peaks (March, July) and two troughs (February, September) reveal clear bi-annual seasonality.")

    # 3. Route Congestion vs Average Arrival Delay
    section("3 · Route Congestion vs Average Arrival Delay")
    rs = filtered_data.groupby('ROUTE').agg(ARRIVAL_DELAY=('ARRIVAL_DELAY','mean'), flight_count=('ROUTE','count')).reset_index()
    rs = rs[rs['flight_count'] > 500]
    fig, ax = plt.subplots(figsize=(10,5))
    sc = ax.scatter(rs['flight_count'], rs['ARRIVAL_DELAY'], c=rs['ARRIVAL_DELAY'], cmap='RdYlGn_r', s=rs['flight_count']/80, alpha=0.55)
    plt.colorbar(sc, ax=ax, label='Avg Delay (min)')
    ax.set_title("Route Congestion vs Average Delay")
    ax.set_xlabel("Number of Flights")
    ax.set_ylabel("Avg Arrival Delay (min)")
    plt.tight_layout()
    show(fig)
    obs("High-frequency routes cluster near zero — operational maturity brings punctuality.")

    # 4. Delay Distribution by Top 5 Airlines (Boxplot)
    section("4 · Delay Distribution by Top 5 Airlines")
    top5 = filtered_data['AIRLINE'].value_counts().head(5).index
    flt = filtered_data[filtered_data['AIRLINE'].isin(top5)].copy()
    flt = flt[flt['ARRIVAL_DELAY'].between(-30,120)]
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=flt, x='AIRLINE', y='ARRIVAL_DELAY', palette='Set2', ax=ax)
    ax.axhline(0, color='#ef4444', linestyle='--', label='On Time')
    ax.set_title("Arrival Delay Distribution — Top 5 Airlines")
    ax.set_ylabel("Arrival Delay (minutes)")
    ax.legend()
    plt.tight_layout()
    show(fig)
    obs("All medians below zero — most flights arrive early.")

    # 5. Top 10 Busiest Routes
    section("5 · Top 10 Busiest Routes")
    tr = filtered_data['ROUTE'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.barh(tr.index, tr.values, color='orange')
    ax.set_title("Top 10 Busiest Routes")
    ax.set_xlabel("Number of Flights")
    ax.invert_yaxis()
    plt.tight_layout()
    show(fig)
    obs("SFO↔LAX is the busiest corridor. LAX appears in 7 of 10 routes.")

    # 6. Average Delay Causes by Airline (Stacked)
    section("6 · Average Delay Causes by Airline (Minutes)")
    avail = [c for c in ['AIR_SYSTEM_DELAY','SECURITY_DELAY','WEATHER_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY'] if c in filtered_data.columns]
    dc = filtered_data.groupby('AIRLINE')[avail].mean().head(10)
    fig, ax = plt.subplots(figsize=(12,6))
    dc.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title("Average Delay Causes by Airline (minutes)")
    ax.set_ylabel("Avg Delay (minutes)")
    plt.tight_layout()
    show(fig)
    obs("Late Aircraft Delay dominates every airline.")

    # 7. Delay Cause % Breakdown
    section("7 · Delay Cause % Breakdown by Airline")
    cmap2 = {'AIRLINE_DELAY':'Carrier','WEATHER_DELAY':'Weather','AIR_SYSTEM_DELAY':'NAS','SECURITY_DELAY':'Security','LATE_AIRCRAFT_DELAY':'Late Aircraft'}
    cols2 = [c for c in cmap2 if c in filtered_data.columns]
    dc2 = filtered_data[filtered_data['ARRIVAL_DELAY']>0].groupby('AIRLINE')[cols2].mean()
    dc2.columns = [cmap2[c] for c in cols2]
    dpct = dc2.div(dc2.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(13,6))
    dpct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title("Delay Cause Breakdown by Airline (%)")
    ax.set_ylabel("% of Total Delay")
    plt.tight_layout()
    show(fig)
    obs("Carrier + Late Aircraft account for 70–80% of total delay.")

    # 8. Weather Delay by Month
    section("8 · Average Weather Delay by Month")
    wm = filtered_data.groupby('MONTH')['WEATHER_DELAY'].mean()
    fig, ax = plt.subplots(figsize=(11,4))
    ax.plot(wm.index, wm.values, marker='s', color='#8b5cf6', linewidth=2.5)
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_title("Average Weather Delay by Month")
    ax.set_ylabel("Avg Weather Delay (min)")
    plt.tight_layout()
    show(fig)
    obs("February and June–July show highest weather delays.")

    # 9. Histograms Side by Side
    section("9 · Arrival & Departure Delay Distributions")
    c1, c2 = st.columns(2)
    with c1:
        da = filtered_data['ARRIVAL_DELAY'].dropna()
        da = da[(da >= -60) & (da <= 180)]
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(da, bins=80, color='#3b82f6', alpha=0.85)
        ax.axvline(da.mean(), color='red', linestyle='--', label=f'Mean: {da.mean():.1f}')
        ax.set_title("Arrival Delay Distribution")
        ax.legend()
        plt.tight_layout()
        show(fig)
    with c2:
        dd = filtered_data['DEPARTURE_DELAY'].dropna()
        dd = dd[(dd >= -60) & (dd <= 180)]
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(dd, bins=80, color='#f97316', alpha=0.85)
        ax.axvline(dd.mean(), color='red', linestyle='--', label=f'Mean: {dd.mean():.1f}')
        ax.set_title("Departure Delay Distribution")
        ax.legend()
        plt.tight_layout()
        show(fig)
    obs("Mean >> Median — extreme delays pull the average up.")

# ====================== MILESTONE 3 ======================
elif "Milestone 3" in page:
    st.markdown("""<div class="main-hero"><span style="background:#3b82f6;color:white;padding:4px 12px;border-radius:20px;font-size:0.85rem;">Milestone 3</span><h1 class="hero-title" style="font-size:1.9rem;">Cancellations & Route Analysis</h1></div>""", unsafe_allow_html=True)

    # 1. Cancellation Reasons by Season
    section("1 · Cancellation Reasons by Season")
    season_cancel = filtered_data[filtered_data['CANCELLED']==1].groupby(['SEASON','CANCELLATION_REASON']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10,5))
    season_cancel.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title("Cancellation Reasons by Season")
    ax.set_ylabel("Number of Cancellations")
    plt.tight_layout()
    show(fig)
    obs("Winter has 3x more weather cancellations than any other season.")

    # 2. Top 10 Routes with Highest Cancellation Rate
    section("2 · Top 10 Routes with Highest Cancellation Rate")
    rc = filtered_data.groupby('ROUTE').agg(cancel_rate=('CANCELLED','mean'), cnt=('CANCELLED','count')).reset_index()
    rc = rc[rc['cnt']>1000]
    rc['cancel_pct'] = rc['cancel_rate']*100
    top10c = rc.nlargest(10,'cancel_pct')
    fig, ax = plt.subplots(figsize=(11,5))
    ax.barh(top10c['ROUTE'], top10c['cancel_pct'], color='red')
    ax.set_title("Top 10 Routes — Highest Cancellation Rate")
    ax.set_xlabel("Cancellation Rate (%)")
    ax.invert_yaxis()
    plt.tight_layout()
    show(fig)
    obs("JFK→DCA leads at ~10.6%. Northeast short-haul routes are most vulnerable.")

    # 3. Monthly Cancellation Trend by Reason
    section("3 · Monthly Cancellation Trend by Reason")
    mc = filtered_data[filtered_data['CANCELLED']==1].groupby(['MONTH','CANCELLATION_REASON']).size().unstack(fill_value=0)
    mc.index = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][:len(mc)]
    fig, ax = plt.subplots(figsize=(12,6))
    mc.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title("Monthly Cancellation Trend by Reason")
    ax.set_ylabel("Number of Cancellations")
    plt.tight_layout()
    show(fig)

    # 4. Average Arrival Delay — Top 10 Busiest Routes
    section("4 · Average Arrival Delay — Top 10 Busiest Routes")
    rd = filtered_data[filtered_data['CANCELLED']==0].groupby('ROUTE').agg(avg_delay=('ARRIVAL_DELAY','mean'), cnt=('ARRIVAL_DELAY','count')).reset_index()
    rdt = rd[rd['ROUTE'].isin(filtered_data['ROUTE'].value_counts().head(10).index)].sort_values('avg_delay', ascending=False)
    fig, ax = plt.subplots(figsize=(11,5))
    colors = ['#ef4444' if v>5 else '#10b981' for v in rdt['avg_delay']]
    ax.barh(rdt['ROUTE'], rdt['avg_delay'], color=colors)
    ax.set_title("Avg Arrival Delay — Top 10 Busiest Routes")
    ax.set_xlabel("Avg Arrival Delay (minutes)")
    ax.invert_yaxis()
    plt.tight_layout()
    show(fig)

    # 5. Weather Delay Distribution by Season (Violin)
    section("5 · Weather Delay Distribution by Season")
    wf = filtered_data[(filtered_data['CANCELLED']==0) & (filtered_data['WEATHER_DELAY']>0) & (filtered_data['WEATHER_DELAY']<120)]
    fig, ax = plt.subplots(figsize=(11,5))
    sns.violinplot(data=wf, x='SEASON', y='WEATHER_DELAY', palette='coolwarm', ax=ax)
    ax.set_title("Weather Delay Distribution by Season")
    ax.set_ylabel("Weather Delay (minutes)")
    plt.tight_layout()
    show(fig)
    obs("Winter shows widest and tallest distribution — most severe weather delays.")

# Footer
st.markdown("---")
st.markdown("""<div style='text-align:center;color:#64748b;font-size:0.8rem;'>
✈️ <strong>AirFly Insights</strong>
</div>""", unsafe_allow_html=True)
