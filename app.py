import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AirFly Insights", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(160deg, #0a0f1e 0%, #0d1b35 60%, #0a1628 100%); border-right: 1px solid rgba(59,130,246,0.15); }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
.main-hero { background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f2744 100%); border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem; border: 1px solid rgba(59,130,246,0.2); position: relative; overflow: hidden; }
.main-hero::before { content: '✈'; position: absolute; right: 3rem; top: 50%; transform: translateY(-50%); font-size: 8rem; opacity: 0.04; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800; color: #f8fafc; margin: 0 0 0.4rem 0; letter-spacing: -0.5px; }
.hero-sub { color: #94a3b8; font-size: 1.05rem; margin: 0; font-weight: 300; }
.metric-card { background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid rgba(59,130,246,0.2); border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
.metric-val { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: #60a5fa; display: block; }
.metric-lbl { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
.section-header { font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 700; color: #1e3a5f; padding: 0.6rem 0 0.6rem 1rem; border-left: 4px solid #3b82f6; margin: 2rem 0 0.8rem 0; background: linear-gradient(90deg, rgba(59,130,246,0.05), transparent); border-radius: 0 8px 8px 0; }
.obs-box { background: linear-gradient(90deg, #eff6ff, #f8fbff); border-left: 4px solid #3b82f6; border-radius: 0 8px 8px 0; padding: 0.8rem 1.2rem; font-size: 0.88rem; color: #1e40af; margin: 0.6rem 0 1.5rem 0; line-height: 1.5; }
.page-tag { display: inline-block; background: rgba(59,130,246,0.1); color: #3b82f6; font-size: 0.75rem; font-weight: 600; padding: 3px 10px; border-radius: 20px; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.5rem; }
.sidebar-logo { font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 800; color: #60a5fa !important; letter-spacing: -0.5px; }
.sidebar-sub { font-size: 0.78rem; color: #475569 !important; margin-top: -4px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-logo">✈ AirFly Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">2015 US Domestic Aviation</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Overview", "📊 Milestone 2 · Delay Analysis", "🚫 Milestone 3 · Cancellations & Routes"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption("Source: Kaggle Airlines Dataset")
    st.caption("5.8M flights · 31 columns · 2015")
    st.markdown("---")
    st.markdown("**Tech Stack**")
    st.caption("Python · Pandas · Matplotlib · Seaborn · Streamlit")

def obs(txt):
    st.markdown(f'<div class="obs-box">💡 <strong>Insight:</strong> {txt}</div>', unsafe_allow_html=True)

def section(txt):
    st.markdown(f'<div class="section-header">{txt}</div>', unsafe_allow_html=True)

def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

BG = "#fafbff"
sns.set_theme(style="whitegrid", font_scale=0.95)

@st.cache_data(show_spinner="⏳ Loading 5.8M flight records…")
def load_data():
    import gdown
    FILE_ID = "1Xz4srzZ6mRK5GJJqyB3UY8WXQQCLdfrB"
    dest = "airline_preprocessed.parquet"
    if not os.path.exists(dest):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", dest, quiet=False)
    df = pd.read_parquet(dest)
    df.columns = df.columns.str.strip().str.upper()
    if 'ROUTE' not in df.columns:
        df['ROUTE'] = df['ORIGIN_AIRPORT'].astype(str) + '_' + df['DESTINATION_AIRPORT'].astype(str)
    for col in ['AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce').fillna(1).astype(int)
    return df

try:
    data = load_data()
    data.columns = data.columns.str.strip().str.upper()
    data_ready = True
except Exception as e:
    data_ready = False
    st.error(f"❌ Failed to load data: {e}")

def need_data():
    st.error("❌ Data unavailable.")
    st.stop()

# ══ OVERVIEW ══════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("""<div class="main-hero"><p style="color:#60a5fa;font-size:0.8rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.5rem;">Data Visualization Project · 2015</p><h1 class="hero-title">AirFly Insights</h1><p class="hero-sub">Uncovering delay patterns, cancellation trends, and route performance across 5.8 million US domestic flights.</p></div>""", unsafe_allow_html=True)

    if data_ready:
        c1,c2,c3,c4 = st.columns(4)
        for col, (val, lbl) in zip([c1,c2,c3,c4], [
            (f"{len(data):,}", "Total Flights"),
            (str(data['AIRLINE'].nunique()), "Airlines"),
            (f"{data['ROUTE'].nunique():,}", "Unique Routes"),
            (f"{data['CANCELLED'].mean()*100:.1f}%", "Cancellation Rate"),
        ]):
            with col:
                st.markdown(f'<div class="metric-card"><span class="metric-val">{val}</span><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.1,1])
    with col1:
        st.markdown("### 📖 Project Storyline")
        st.markdown("""| # | Milestone | Focus |
|---|-----------|-------|
| 1 | **Data Foundation** | Cleaning · Feature Engineering · Preprocessing |
| 2 | **Delay Analysis** | Airlines · Routes · Causes · Time Patterns |
| 3 | **Cancellations & Routes** | Seasonal Trends · Route Risk · Why Flights Cancel |

Use the **sidebar** to explore each milestone.""")
    with col2:
        st.markdown("### 🔑 Key Findings")
        for f in ["61.8% of flights arrive **early** (median = -5 min)","**Late Aircraft** is #1 delay cause across all airlines","**February** has 4.7% cancellation rate — peak winter storms","**JFK→DCA** is the most cancelled route (10.6%)","**Busiest routes** are NOT the most delayed"]:
            st.markdown(f"- {f}")

# ══ MILESTONE 2 ════════════════════════════════════════════════════════════════
elif "Milestone 2" in page:
    st.markdown("""<div class="main-hero"><span class="page-tag">Milestone 2</span><h1 class="hero-title" style="font-size:1.9rem;">📊 Delay Analysis</h1><p class="hero-sub">Airlines · Routes · Causes · Distributions · Airport Performance</p></div>""", unsafe_allow_html=True)
    if not data_ready: need_data()

    with st.expander("🔧 Chart Filters", expanded=False):
        col1, col2 = st.columns(2)
        with col1: top_n = st.slider("Top N Airlines to show", 5, 14, 10)
        with col2: clip_max = st.slider("Max delay (min) for histograms", 60, 300, 180)

    # 1. Top airlines
    section("1 · Top Airlines by Flight Volume")
    top_a = data['AIRLINE'].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bars = ax.barh(top_a.index, top_a.values, color=sns.color_palette("Blues_r",len(top_a)), edgecolor='white', height=0.65)
    for bar,v in zip(bars,top_a.values): ax.text(v+max(top_a.values)*0.01, bar.get_y()+bar.get_height()/2, f"{v:,}", va='center', fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    ax.set_title(f"Top {top_n} Airlines by Flight Volume (2015)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Flights"); ax.invert_yaxis(); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("Southwest (WN) dominates with ~1.26M flights — nearly double Delta. Top 3 carriers (WN, DL, AA) handle over 50% of all domestic flights.")
    st.markdown("---")

    # 2. Monthly trend
    section("2 · Monthly Flight Volume Trend")
    m = data.groupby('MONTH').size()
    fig, ax = plt.subplots(figsize=(11,4)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.plot(m.index, m.values, marker='o', color='#3b82f6', linewidth=2.5, markersize=7, markerfacecolor='white', markeredgewidth=2)
    ax.fill_between(m.index, m.values, alpha=0.1, color='#3b82f6')
    ax.set_xticks(range(1,13)); ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    ax.set_title("Monthly Flight Volume Trend (2015)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Month"); ax.set_ylabel("Total Flights"); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("Two peaks (March, July) and two troughs (February, September) reveal clear bi-annual seasonality in US domestic aviation.")
    st.markdown("---")

    # 3. Route congestion scatter
    section("3 · Route Congestion vs Average Arrival Delay")
    rs = data.groupby('ROUTE').agg(ARRIVAL_DELAY=('ARRIVAL_DELAY','mean'), flight_count=('ROUTE','count')).reset_index()
    rs = rs[rs['flight_count'] > 500]
    fig, ax = plt.subplots(figsize=(10,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    sc = ax.scatter(rs['flight_count'], rs['ARRIVAL_DELAY'], c=rs['ARRIVAL_DELAY'], cmap='RdYlGn_r', s=rs['flight_count']/80, alpha=0.55, linewidths=0)
    plt.colorbar(sc, ax=ax, label='Avg Delay (min)')
    ax.axhline(0, color='#374151', linestyle='--', lw=1, alpha=0.5)
    ax.set_title("Route Congestion vs Average Delay\n(bubble size = flight volume)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Flights"); ax.set_ylabel("Avg Arrival Delay (min)"); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("High-frequency routes cluster near zero — funnel shape narrowing left-to-right proves operational maturity brings punctuality. Worst delays (red) exclusively on low-volume routes under 2,000 flights.")
    st.markdown("---")

    # 4. Boxplot
    section("4 · Delay Distribution by Top 5 Airlines (Boxplot)")
    top5 = data['AIRLINE'].value_counts().head(5).index
    flt = data[data['AIRLINE'].isin(top5)].copy()
    flt = flt[flt['ARRIVAL_DELAY'].between(-30,120)]
    fig, ax = plt.subplots(figsize=(10,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    sns.boxplot(data=flt, x='AIRLINE', y='ARRIVAL_DELAY', hue='AIRLINE', palette='Set2', legend=False, flierprops=dict(marker='.', markersize=2, alpha=0.3), ax=ax)
    ax.axhline(0, color='#ef4444', linestyle='--', lw=1.5, label='On Time')
    ax.set_title("Arrival Delay Distribution — Top 5 Airlines", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Airline"); ax.set_ylabel("Arrival Delay (minutes)"); ax.legend(fontsize=9); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("All medians below zero — most flights arrive early. Long upper whiskers show extreme delays exist but are rare.")
    st.markdown("---")

    # 5. Busiest routes
    section("5 · Top 10 Busiest Routes")
    tr = data['ROUTE'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bars = ax.barh(tr.index, tr.values, color=sns.color_palette("Oranges_r",len(tr)), edgecolor='white', height=0.65)
    for bar,v in zip(bars,tr.values): ax.text(v+100, bar.get_y()+bar.get_height()/2, f"{v:,}", va='center', fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    ax.set_title("Top 10 Busiest Routes (2015)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Flights"); ax.invert_yaxis(); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("SFO↔LAX is the busiest corridor (~27K combined flights). LAX appears in 7 of 10 routes — the most connected hub in US domestic aviation.")
    st.markdown("---")

    # 6. Stacked delay causes minutes
    section("6 · Average Delay Causes by Airline (Minutes)")
    avail = [c for c in ['AIR_SYSTEM_DELAY','SECURITY_DELAY','WEATHER_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY'] if c in data.columns]
    dc = data.groupby('AIRLINE')[avail].mean().head(10)
    fig, ax = plt.subplots(figsize=(12,6)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    dc.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='white', width=0.7)
    ax.set_title("Average Delay Causes by Airline (minutes)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Avg Delay (minutes)"); ax.set_xlabel("Airline"); ax.tick_params(axis='x', rotation=30)
    ax.legend(title="Cause", bbox_to_anchor=(1.01,1), loc='upper left', fontsize=9); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("Late Aircraft Delay dominates every airline — a cascading effect. NK and F9 show highest total delay minutes with weaker operational buffers.")
    st.markdown("---")

    # 7. % breakdown
    section("7 · Delay Cause % Breakdown by Airline")
    cmap2 = {'AIRLINE_DELAY':'Carrier','WEATHER_DELAY':'Weather','AIR_SYSTEM_DELAY':'NAS','SECURITY_DELAY':'Security','LATE_AIRCRAFT_DELAY':'Late Aircraft'}
    cols2 = [c for c in cmap2 if c in data.columns]
    dc2 = data[data['ARRIVAL_DELAY']>0].groupby('AIRLINE')[cols2].mean()
    dc2.columns = [cmap2[c] for c in cols2]
    dpct = dc2.div(dc2.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(13,6)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    dpct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='white', width=0.7)
    for stk in ax.containers:
        for bar in stk:
            h = bar.get_height()
            if h > 6: ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+h/2, f'{h:.0f}%', ha='center', va='center', fontsize=7.5, color='#111', fontweight='bold')
    ax.set_title("Delay Cause Breakdown by Airline (% of Total)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("% of Total Delay"); ax.set_xlabel("Airline"); ax.set_ylim(0,108)
    ax.legend(title="Cause", bbox_to_anchor=(1.01,1), loc='upper left', fontsize=9); ax.tick_params(axis='x', rotation=30); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("Carrier + Late Aircraft account for 70–80% of total delay regardless of airline. Security delay is <1% everywhere.")
    st.markdown("---")

    # 8. Weather delay by month
    section("8 · Average Weather Delay by Month")
    wm = data.groupby('MONTH')['WEATHER_DELAY'].mean()
    fig, ax = plt.subplots(figsize=(11,4)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.plot(wm.index, wm.values, marker='s', color='#8b5cf6', linewidth=2.5, markersize=7, markerfacecolor='white', markeredgewidth=2)
    ax.fill_between(wm.index, wm.values, alpha=0.12, color='#8b5cf6')
    ax.set_xticks(range(1,13)); ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_title("Average Weather Delay by Month", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Month"); ax.set_ylabel("Avg Weather Delay (min)"); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("Two-peak pattern: February spike from winter blizzards, June–July spike from thunderstorms. September–October is the weather-safest window.")
    st.markdown("---")

    # 9. Histograms side by side
    section("9 · Arrival & Departure Delay Distributions")
    c1,c2 = st.columns(2)
    with c1:
        da = data['ARRIVAL_DELAY'].dropna(); da = da[(da>=-60)&(da<=clip_max)]
        fig, ax = plt.subplots(figsize=(7,4)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.hist(da, bins=80, color='#3b82f6', edgecolor='white', alpha=0.85)
        ax.axvline(da.mean(), color='#ef4444', linestyle='--', lw=1.8, label=f'Mean: {da.mean():.1f}m')
        ax.axvline(da.median(), color='#10b981', linestyle='--', lw=1.8, label=f'Median: {da.median():.1f}m')
        ax.axvline(da.quantile(0.95), color='#111', linestyle=':', lw=1.5, label=f'95th: {da.quantile(0.95):.0f}m')
        ax.set_title("Arrival Delay Distribution", fontsize=11, fontweight='bold')
        ax.set_xlabel("Delay (min)"); ax.set_ylabel("Flights"); ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); show(fig)
        st.caption(f"Early: {(da<0).mean()*100:.1f}% · On-time(0-15m): {((da>=0)&(da<=15)).mean()*100:.1f}% · Severe(>60m): {(da>60).mean()*100:.1f}%")
    with c2:
        dd = data['DEPARTURE_DELAY'].dropna(); dd = dd[(dd>=-60)&(dd<=clip_max)]
        fig, ax = plt.subplots(figsize=(7,4)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        ax.hist(dd, bins=80, color='#f97316', edgecolor='white', alpha=0.85)
        ax.axvline(dd.mean(), color='#ef4444', linestyle='--', lw=1.8, label=f'Mean: {dd.mean():.1f}m')
        ax.axvline(dd.median(), color='#10b981', linestyle='--', lw=1.8, label=f'Median: {dd.median():.1f}m')
        ax.set_title("Departure Delay Distribution", fontsize=11, fontweight='bold')
        ax.set_xlabel("Delay (min)"); ax.set_ylabel("Flights"); ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); show(fig)
        st.caption(f"Mean: {dd.mean():.1f}m · Median: {dd.median():.1f}m")
    obs("Mean >> Median for both distributions — most flights on-time/early while extreme delays pull the average up significantly.")
    st.markdown("---")

    # 10. Delay severity
    section("10 · Delay Severity Composition by Airline")
    dt2 = data.copy()
    dt2['delay_severity'] = pd.cut(dt2['ARRIVAL_DELAY'], bins=[-1000,0,15,60,10000], labels=['On Time','Minor','Moderate','Severe'])
    sev = dt2.groupby('AIRLINE')['delay_severity'].value_counts(normalize=True).unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(12,6)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    sev.plot(kind='bar', stacked=True, ax=ax, color=['#10b981','#3b82f6','#f59e0b','#ef4444'], edgecolor='white', width=0.75)
    ax.set_title("Delay Severity Composition by Airline", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Proportion"); ax.set_xlabel("Airline"); ax.tick_params(axis='x', rotation=30)
    ax.legend(title="Severity", bbox_to_anchor=(1.01,1), loc='upper left', fontsize=9); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("Hawaiian (HA) and Delta (DL) have the highest on-time proportion. Spirit (NK) and Frontier (F9) show the largest share of moderate-to-severe delays.")
    st.markdown("---")

    # 11. Top 15 airports
    section("11 · Top 15 Airports — Highest Average Departure Delay")
    ap = data[data['CANCELLED']==0].groupby('ORIGIN_AIRPORT').agg(avg_delay=('DEPARTURE_DELAY','mean'), cnt=('DEPARTURE_DELAY','count')).reset_index()
    ap = ap[ap['cnt']>1000]; top15 = ap.nlargest(15,'avg_delay')
    fig, ax = plt.subplots(figsize=(10,6)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bars = ax.barh(top15['ORIGIN_AIRPORT'], top15['avg_delay'], color=sns.color_palette("Reds_r",len(top15)), edgecolor='white', height=0.65)
    for bar,v in zip(bars,top15['avg_delay']): ax.text(v+0.1, bar.get_y()+bar.get_height()/2, f'{v:.1f}m', va='center', fontsize=9)
    ax.set_title("Top 15 Airports — Highest Avg Departure Delay", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Avg Departure Delay (minutes)"); ax.invert_yaxis(); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("Mid-size congested airports outrank major hubs — large airports have better ground crew resources and scheduling buffers.")
    st.markdown("---")

    # 12. Distance vs delay
    section("12 · Flight Distance vs Arrival Delay")
    if 'DISTANCE' in data.columns:
        samp = data[(data['ARRIVAL_DELAY'].between(-60,180))&(data['CANCELLED']==0)].sample(10000, random_state=42).copy()
        samp['DIST_BUCKET'] = pd.cut(samp['DISTANCE'], bins=[0,500,1000,2000,5000], labels=['Short (<500mi)','Medium (500-1000mi)','Long (1000-2000mi)','Ultra (>2000mi)'])
        fig, ax = plt.subplots(figsize=(11,6)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        sns.scatterplot(data=samp, x='DISTANCE', y='ARRIVAL_DELAY', hue='DIST_BUCKET', palette='coolwarm', alpha=0.35, s=12, ax=ax)
        z = np.polyfit(samp['DISTANCE'], samp['ARRIVAL_DELAY'], 1); p = np.poly1d(z)
        xl = np.linspace(samp['DISTANCE'].min(), samp['DISTANCE'].max(), 100)
        ax.plot(xl, p(xl), color='#ef4444', lw=2.5, linestyle='--', label='Trend Line')
        ax.axhline(0, color='#374151', linestyle=':', lw=1, label='On Time')
        ax.set_title("Flight Distance vs Arrival Delay", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Distance (miles)"); ax.set_ylabel("Arrival Delay (min)")
        ax.legend(title="Distance", bbox_to_anchor=(1.01,1), loc='upper left', fontsize=9); ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); show(fig)
        obs(f"Trend line slopes downward (correlation: {samp['DISTANCE'].corr(samp['ARRIVAL_DELAY']):.3f}) — longer flights recover delays in air. Short-haul shows highest variance.")
    else:
        st.warning("DISTANCE column not found.")

# ══ MILESTONE 3 ════════════════════════════════════════════════════════════════
elif "Milestone 3" in page:
    st.markdown("""<div class="main-hero"><span class="page-tag">Milestone 3</span><h1 class="hero-title" style="font-size:1.9rem;">🚫 Cancellations & Route Analysis</h1><p class="hero-sub">Seasonal patterns · Route risk · Why flights cancel · When to avoid flying</p></div>""", unsafe_allow_html=True)
    if not data_ready: need_data()

    data['SEASON'] = data['MONTH'].apply(lambda m: 'Winter' if m in [12,1,2] else 'Spring' if m in [3,4,5] else 'Summer' if m in [6,7,8] else 'Fall')
    data['SEASON_SORT'] = data['MONTH'].apply(lambda m: '1.Winter' if m in [12,1,2] else '2.Spring' if m in [3,4,5] else '3.Summer' if m in [6,7,8] else '4.Fall')

    # 1. Avg delay top routes
    section("1 · Average Arrival Delay — Top 10 Busiest Routes")
    rd = data[data['CANCELLED']==0].groupby('ROUTE').agg(avg_delay=('ARRIVAL_DELAY','mean'), cnt=('ARRIVAL_DELAY','count')).reset_index()
    rdt = rd[rd['ROUTE'].isin(data['ROUTE'].value_counts().head(10).index)].sort_values('avg_delay', ascending=False)
    fig, ax = plt.subplots(figsize=(11,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    colors = ['#ef4444' if v>5 else '#10b981' for v in rdt['avg_delay']]
    bars = ax.barh(rdt['ROUTE'], rdt['avg_delay'], color=colors, edgecolor='white', height=0.65)
    for bar,v in zip(bars,rdt['avg_delay']): ax.text(v+(0.1 if v>=0 else -0.3), bar.get_y()+bar.get_height()/2, f'{v:.1f}m', va='center', fontsize=9)
    ax.axvline(0, color='#374151', linestyle='--', lw=1)
    ax.set_title("Avg Arrival Delay — Top 10 Busiest Routes\n(red >5 min delayed · green on-time or early)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Avg Arrival Delay (minutes)"); ax.invert_yaxis(); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("LAS→LAX most delayed (~12 min). JFK routes arrive early — transcontinental flights recover delays in air. Same corridor performs differently by direction.")
    st.markdown("---")

    # 2. Top routes cancellation rate
    section("2 · Top 10 Routes with Highest Cancellation Rate")
    rc = data.groupby('ROUTE').agg(cancel_rate=('CANCELLED','mean'), cnt=('CANCELLED','count')).reset_index()
    rc = rc[rc['cnt']>1000]; rc['cancel_pct'] = rc['cancel_rate']*100
    top10c = rc.nlargest(10,'cancel_pct')
    fig, ax = plt.subplots(figsize=(11,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bars = ax.barh(top10c['ROUTE'], top10c['cancel_pct'], color=sns.color_palette("Reds_r",len(top10c)), edgecolor='white', height=0.65)
    for bar,v in zip(bars,top10c['cancel_pct']): ax.text(v+0.05, bar.get_y()+bar.get_height()/2, f'{v:.1f}%', va='center', fontsize=9)
    ax.set_title("Top 10 Routes — Highest Cancellation Rate\n(routes with >1,000 flights)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Cancellation Rate (%)"); ax.invert_yaxis(); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("JFK→DCA leads at 10.6% — 1 in 10 flights cancelled. LGA in 6 of 10 routes — single runway extremely vulnerable. All routes are Northeast short-haul hit by winter storms.")
    st.markdown("---")

    # 3. Cancellation reason distribution
    section("3 · Cancellation Reasons Distribution")
    co = data[(data['CANCELLED']==1)&(data['CANCELLATION_REASON']!='Not Cancelled')]
    cc = co['CANCELLATION_REASON'].value_counts()
    fig, ax = plt.subplots(figsize=(10,4)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bars = ax.bar(cc.index, cc.values, color=['#3b82f6','#ef4444','#f59e0b','#10b981'][:len(cc)], edgecolor='white', width=0.6)
    for bar,v in zip(bars,cc.values): ax.text(bar.get_x()+bar.get_width()/2, v+80, f"{v:,}", ha='center', fontsize=10, fontweight='bold')
    ax.set_title("Cancellation Reasons Distribution (2015)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Cancellation Reason"); ax.set_ylabel("Number of Flights")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}')); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs(f"Weather causes ~54% of all {cc.sum():,} cancellations. Carrier issues second — entirely preventable internal failures. Security cancellations (22 total) virtually non-existent.")
    st.markdown("---")

    # 4. Cancellation rate by airline
    section("4 · Cancellation Rate by Airline")
    cr = data.groupby('AIRLINE')['CANCELLED'].mean().sort_values(ascending=False)*100
    fig, ax = plt.subplots(figsize=(11,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    cr.plot(kind='bar', ax=ax, color=['#ef4444' if v>3 else '#f59e0b' if v>1.5 else '#10b981' for v in cr.values], edgecolor='white', width=0.7)
    ax.set_title("Cancellation Rate by Airline (%)\n(red >3% · amber 1.5–3% · green <1.5%)", fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel("Cancellation Rate (%)"); ax.set_xlabel("Airline Code"); ax.tick_params(axis='x', rotation=30); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("Envoy Air (MQ) leads at ~5% — 3x the industry average. Hawaiian (HA) lowest at ~0.2% — island routes leave no alternative, creating strong incentive to never cancel.")
    st.markdown("---")

    # 5. Cancellation rate by month
    section("5 · Cancellation Rate by Month")
    crate = data.groupby('MONTH')['CANCELLED'].mean()*100
    fig, ax = plt.subplots(figsize=(11,4)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.plot(crate.index, crate.values, marker='o', color='#ef4444', linewidth=2.5, markersize=7, markerfacecolor='white', markeredgewidth=2)
    ax.fill_between(crate.index, crate.values, alpha=0.1, color='#ef4444')
    ax.set_xticks(range(1,13)); ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_title("Monthly Cancellation Rate (%)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Month"); ax.set_ylabel("Cancellation Rate (%)"); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("February peaks at ~4.7% — peak winter storm season. June secondary spike from summer thunderstorms. September–October safest at ~0.5%.")
    st.markdown("---")

    # 6. Route x month heatmap
    section("6 · Avg Arrival Delay — Route × Month Heatmap")
    top10r = data['ROUTE'].value_counts().head(10).index
    sub = data[data['ROUTE'].isin(top10r)&(data['CANCELLED']==0)]
    piv = sub.groupby(['ROUTE','MONTH'])['ARRIVAL_DELAY'].mean().unstack()
    mmap = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    piv.columns = [mmap[c] for c in piv.columns]
    fig, ax = plt.subplots(figsize=(14,6)); fig.patch.set_facecolor(BG)
    sns.heatmap(piv, annot=True, fmt='.1f', cmap='RdYlGn_r', linewidths=0.4, linecolor='white', cbar_kws={'label':'Avg Delay (min)'}, ax=ax)
    ax.set_title("Avg Arrival Delay by Route × Month", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Month"); ax.set_ylabel("Route")
    plt.tight_layout(); show(fig)
    obs("ORD→LGA red year-round — most persistently delayed corridor. LAX→SFO spikes to 24.8 min in December. October universally green — safest month on every major route.")
    st.markdown("---")

    # 7. Weather delay violin by season
    section("7 · Weather Delay Distribution by Season (Violin)")
    wf = data[(data['CANCELLED']==0)&(data['WEATHER_DELAY']>0)&(data['WEATHER_DELAY']<120)]
    fig, ax = plt.subplots(figsize=(11,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    sns.violinplot(data=wf, x='SEASON_SORT', y='WEATHER_DELAY', hue='SEASON_SORT', palette='coolwarm', inner='box', legend=False, ax=ax)
    mv = wf['WEATHER_DELAY'].mean()
    ax.axhline(mv, color='#ef4444', linestyle='--', lw=1.8, label=f"Industry mean: {mv:.1f} min")
    ax.set_title("Weather Delay Distribution by Season\n(only flights with weather delay > 0)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Season"); ax.set_ylabel("Weather Delay (minutes)"); ax.set_xticklabels(['Winter','Spring','Summer','Fall'])
    ax.legend(fontsize=9); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("Winter violin widest AND tallest — delays more frequent and more severe. Fall shows tightest, lowest distribution — quietest weather season for aviation.")
    st.markdown("---")

    # 8. Monthly cancellation trend
    section("8 · Monthly Cancellation Trend by Reason")
    mc = data[data['CANCELLED']==1].groupby(['MONTH','CANCELLATION_REASON']).size().unstack(fill_value=0)
    mc.index = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][:len(mc)]
    fig, ax = plt.subplots(figsize=(12,6)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    mc.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='white', width=0.75)
    ax.set_title("Monthly Cancellation Trend by Reason", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Number of Cancellations"); ax.set_xlabel("Month"); ax.tick_params(axis='x', rotation=30)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    ax.legend(title="Reason", bbox_to_anchor=(1.01,1), loc='upper left', fontsize=9); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("February dominates with 20,000+ cancellations driven by weather. Carrier cancellations stay flat year-round at ~2,500–3,500 — constant internal failures unaffected by season.")
    st.markdown("---")

    # 9. Cancellation by season
    section("9 · Cancellation Reasons by Season")
    sc = data[data['CANCELLED']==1].groupby(['SEASON','CANCELLATION_REASON']).size().unstack(fill_value=0)
    sc = sc.reindex([s for s in ['Winter','Spring','Summer','Fall'] if s in sc.index])
    fig, ax = plt.subplots(figsize=(10,5)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    sc.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='white', width=0.7)
    ax.set_title("Cancellation Reasons by Season", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Number of Cancellations"); ax.set_xlabel("Season"); ax.tick_params(axis='x', rotation=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{int(x):,}'))
    ax.legend(title="Reason", bbox_to_anchor=(1.01,1), loc='upper left', fontsize=9); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("Winter has 4x more cancellations than Fall — weather is the sole differentiator. Carrier cancellations equal across all seasons proving they are internal operational problems.")
    st.markdown("---")

    # 10. Arrival delay violin by airline
    section("10 · Arrival Delay Distribution by Airline (Violin)")
    top6 = data['AIRLINE'].value_counts().head(6).index
    sv = data[data['AIRLINE'].isin(top6)].copy(); sv = sv[sv['ARRIVAL_DELAY'].between(-30,120)]
    fig, ax = plt.subplots(figsize=(12,6)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    sns.violinplot(x='AIRLINE', y='ARRIVAL_DELAY', data=sv, hue='AIRLINE', palette='Set2', inner='box', legend=False, ax=ax)
    ax.axhline(0, color='#ef4444', linestyle='--', lw=1.8, label='On Time (0 min)')
    ax.set_title("Arrival Delay Distribution by Airline (Violin)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Airline Code"); ax.set_ylabel("Arrival Delay (minutes)"); ax.legend(fontsize=9); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout(); show(fig)
    obs("All airlines have medians below zero — most flights arrive early. Spirit (NK) and Frontier (F9) show wider upper bodies. Hawaiian (HA) and SkyWest (OO) show tightest shapes — most consistent performers.")

st.markdown("""<div style='text-align:center; color:#94a3b8; font-size:0.8rem; padding:3rem 0 1.5rem 0; border-top:1px solid #e2e8f0; margin-top:3rem;'>✈️ <strong>AirFly Insights</strong> · 2015 US Domestic Flight Data · Built with Streamlit · Pandas · Matplotlib · Seaborn</div>""", unsafe_allow_html=True)
