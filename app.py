import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Data Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stRadio label { 
        padding: 6px 10px; border-radius: 6px; display:block;
        transition: background 0.2s;
    }
    .page-title { font-size:2rem; font-weight:700; color:#1e3a5f; margin-bottom:4px; }
    .page-sub   { color:#64748b; font-size:1rem; margin-bottom:1.5rem; }
    .section-hdr {
        font-size:1.2rem; font-weight:600; color:#1e3a5f;
        border-left:5px solid #3b82f6; padding-left:10px;
        margin: 1.6rem 0 0.5rem 0;
    }
    .obs { background:#eff6ff; border-left:4px solid #3b82f6;
           padding:10px 14px; border-radius:5px;
           font-size:0.9rem; color:#1e3a5f; margin-top:8px; }
    .stat-box { background:#f8faff; border:1px solid #dbeafe;
                border-radius:8px; padding:14px 18px; }
    .stat-val { font-size:1.7rem; font-weight:700; color:#1e3a5f; }
    .stat-lbl { font-size:0.8rem; color:#64748b; }
    hr { border:none; border-top:1px solid #e2e8f0; margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ Flight Analysis")
    st.markdown("**2015 US Domestic Flights**")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        [
            "🏠  Overview",
            "📊  Milestone 2 — Delay Analysis",
            "🚫  Milestone 3 — Cancellations & Routes",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("📂 Reading `airline_preprocessed.csv` from the same folder as `app.py`.")

# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Loading dataset… this may take a moment for large files.")
def load_data():
    df = pd.read_csv("airline_preprocessed.csv", low_memory=False)
    df.columns = df.columns.str.strip()
    if 'ROUTE' not in df.columns and 'ORIGIN_AIRPORT' in df.columns:
        df['ROUTE'] = df['ORIGIN_AIRPORT'].astype(str) + '_' + df['DESTINATION_AIRPORT'].astype(str)
    for col in ['AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df

import os
data_ready = False
if os.path.exists("airline_preprocessed.csv"):
    try:
        data = load_data()
        data_ready = True
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        data_ready = False
else:
    with st.sidebar:
        st.error("airline_preprocessed.csv not found! Make sure it is in the same folder as app.py.")

def need_data():
    st.error("❌ `airline_preprocessed.csv` not found in the app folder. Place it in the same folder as `app.py` and restart the app.")
    st.stop()

def obs(txt):
    st.markdown(f'<div class="obs">💡 <b>Observation:</b> {txt}</div>', unsafe_allow_html=True)

def hdr(txt):
    st.markdown(f'<div class="section-hdr">{txt}</div>', unsafe_allow_html=True)

def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<p class="page-title">✈️ Flight Data Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Week 7 — Visual Report · 2015 US Domestic Flights</p>', unsafe_allow_html=True)

    if data_ready:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{len(data):,}</div><div class="stat-lbl">Total Flights</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{data["AIRLINE"].nunique()}</div><div class="stat-lbl">Airlines</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{data["ROUTE"].nunique():,}</div><div class="stat-lbl">Unique Routes</div></div>', unsafe_allow_html=True)
        with c4:
            pct = data['CANCELLED'].mean() * 100
            st.markdown(f'<div class="stat-box"><div class="stat-val">{pct:.1f}%</div><div class="stat-lbl">Cancellation Rate</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    ### 📖 Storyline
    | Milestone | Focus | Key Questions Answered |
    |-----------|-------|------------------------|
    | **Milestone 1** | Data Preparation | Cleaning, encoding, feature engineering on `flights.csv` |
    | **Milestone 2** | Delay Analysis | Which airlines/routes delay most? What causes delays? When? |
    | **Milestone 3** | Cancellations & Routes | Where do cancellations happen? Why? Does season matter? |

    Use the **sidebar** to navigate between milestones.  
    Every chart includes titles, axis labels, legends, and observation notes — as required for Week 7.
    """)


# ══════════════════════════════════════════════════════════════════════════════
#  MILESTONE 2 — DELAY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Milestone 2" in page:
    st.markdown('<p class="page-title">📊 Milestone 2 — Delay Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Exploring flight delays across airlines, routes, months, and delay causes.</p>', unsafe_allow_html=True)

    if not data_ready:
        need_data()

    # ── 1. Top 10 Airlines by Flight Volume ──────────────────────────────────
    hdr("1 · Top 10 Airlines by Flight Volume")
    top_airlines = data['AIRLINE'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax_bar = sns.barplot(x=top_airlines.values, y=top_airlines.index,
                         hue=top_airlines.index, palette="Blues_r",
                         legend=False, ax=ax)
    for i, v in enumerate(top_airlines.values):
        ax.text(v + 0.01 * max(top_airlines.values), i, f"{v:,}", va='center', fontsize=9)
    ax.set_title("Top 10 Airlines by Flight Volume", fontsize=13, fontweight='bold')
    ax.set_xlabel("Number of Flights"); ax.set_ylabel("Airline")
    plt.tight_layout()
    show(fig)
    obs("Southwest (WN) dominates with nearly double the volume of the next airline — a key factor in its operational risk profile.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 2. Busiest Months ─────────────────────────────────────────────────────
    hdr("2 · Busiest Months — Monthly Flight Volume Trend")
    monthly_flights = data.groupby('MONTH').size()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly_flights.index, monthly_flights.values,
            marker='o', color='steelblue', linewidth=2.5)
    ax.fill_between(monthly_flights.index, monthly_flights.values, alpha=0.15, color='steelblue')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_title("Monthly Flight Volume Trend", fontsize=13, fontweight='bold')
    ax.set_xlabel("Month"); ax.set_ylabel("Total Flights")
    plt.tight_layout()
    show(fig)
    obs("Summer months (Jun–Aug) carry the heaviest traffic. February dips to the lowest — fewer days plus winter disruption.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 3. Route Congestion vs Delay ──────────────────────────────────────────
    hdr("3 · Route Congestion vs Average Delay")
    route_stats = data.groupby('ROUTE').agg(
        ARRIVAL_DELAY=('ARRIVAL_DELAY', 'mean'),
        flight_count=('ROUTE', 'count')
    ).reset_index()
    route_stats = route_stats[route_stats['flight_count'] > 500]
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(route_stats['flight_count'], route_stats['ARRIVAL_DELAY'],
                    c=route_stats['ARRIVAL_DELAY'], cmap='RdYlGn_r',
                    s=20, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Avg Delay (min)')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title("Route Congestion vs Average Delay", fontsize=13, fontweight='bold')
    ax.set_xlabel("Number of Flights"); ax.set_ylabel("Average Arrival Delay (min)")
    plt.tight_layout()
    show(fig)
    obs("High-volume routes do not equal high delays — the busiest corridors are operationally efficient.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 4. Delay Distribution by Airline ─────────────────────────────────────
    hdr("4 · Delay Distribution by Top 5 Airlines")
    top5 = data['AIRLINE'].value_counts().head(5).index
    filtered = data[data['AIRLINE'].isin(top5)]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=filtered, x='AIRLINE', y='ARRIVAL_DELAY',
                palette='Set2', ax=ax, showfliers=False)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, label='On Time')
    ax.set_title("Delay Distribution by Top Airlines", fontsize=13, fontweight='bold')
    ax.set_xlabel("Airline"); ax.set_ylabel("Arrival Delay (Minutes)")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    show(fig)
    obs("Most median delays sit at or below zero — the majority of flights arrive on time or early. Whiskers show how far extreme delays can reach.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 5. Busiest Routes ─────────────────────────────────────────────────────
    hdr("5 · Top 10 Busiest Routes")
    top_routes = data['ROUTE'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_routes.plot(kind='barh', color='darkorange', ax=ax)
    ax.set_title("Top 10 Busiest Routes", fontsize=13, fontweight='bold')
    ax.set_xlabel("Number of Flights"); ax.set_ylabel("Route")
    ax.invert_yaxis()
    plt.tight_layout()
    show(fig)
    obs("SFO-LAX is the busiest corridor — short-haul high-frequency routes dominate the top 10.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 6. Delay Analysis — Average Delay Causes ──────────────────────────────
    hdr("6 · Delay Analysis — Average Delay Causes by Airline (Stacked Bar)")
    cause_cols = ['AIR_SYSTEM_DELAY','SECURITY_DELAY','WEATHER_DELAY',
                  'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY']
    delay_causes = data.groupby('AIRLINE')[cause_cols].mean().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    delay_causes.plot(kind='bar', stacked=True, ax=ax,
                      colormap='Set2', edgecolor='white', width=0.7)
    ax.set_title("Average Delay Causes by Airline", fontsize=13, fontweight='bold')
    ax.set_ylabel("Average Delay (Minutes)"); ax.set_xlabel("Airline")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="Delay Cause", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    show(fig)
    obs("Late Aircraft Delay is the dominant cause across almost every airline — a cascading effect where one delayed flight triggers the next.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 7. Delay Cause Comparison — 100% Stacked ──────────────────────────────
    hdr("7 · Delay Cause Comparison — % Breakdown by Airline")
    cause_labels = ['Carrier','Weather','NAS','Security','Late Aircraft']
    delay_causes2 = data[data['ARRIVAL_DELAY'] > 0].groupby('AIRLINE')[cause_cols].mean()
    delay_causes2.columns = cause_labels
    delay_pct = delay_causes2.div(delay_causes2.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(13, 6))
    delay_pct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2',
                   edgecolor='white', width=0.7)
    for bar_stack in ax.containers:
        for bar in bar_stack:
            h = bar.get_height()
            if h > 5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + h / 2,
                        f'{h:.0f}%',
                        ha='center', va='center', fontsize=7.5,
                        color='black', fontweight='bold')
    ax.set_title("Delay Cause Breakdown by Airline (% of Total Delay)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Percentage of Total Delay (%)"); ax.set_xlabel("Airline")
    ax.set_ylim(0, 105)
    ax.legend(title="Delay Cause", bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    show(fig)
    obs("Late Aircraft and Carrier delays together account for 70–80% of total delay across most airlines. Security delay is negligible (<1%) for all carriers.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 8. Weather Delay Seasonal Trend ───────────────────────────────────────
    hdr("8 · Weather Delay Seasonal Trend — Average Weather Delay by Month")
    weather_month = data.groupby('MONTH')['WEATHER_DELAY'].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(weather_month.index, weather_month.values,
            marker='s', color='tomato', linewidth=2.5)
    ax.fill_between(weather_month.index, weather_month.values, alpha=0.15, color='tomato')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_title("Average Weather Delay by Month", fontsize=13, fontweight='bold')
    ax.set_xlabel("Month"); ax.set_ylabel("Weather Delay (Minutes)")
    plt.tight_layout()
    show(fig)
    obs("Weather delays peak in winter (Dec–Feb) and again mid-summer (Jun–Jul thunderstorm season). Spring and Fall are the smoothest periods.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 9. Departure Delay Analysis ───────────────────────────────────────────
    hdr("9 · Departure Delay Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribution of Arrival Delay**")
        delay_arr = data['ARRIVAL_DELAY'].dropna()
        delay_arr = delay_arr[(delay_arr >= -60) & (delay_arr <= 180)]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(delay_arr, bins=80, color='steelblue', edgecolor='white')
        ax.axvline(delay_arr.mean(),   color='red',   linestyle='--', label=f'Mean: {delay_arr.mean():.1f} min')
        ax.axvline(delay_arr.median(), color='green', linestyle='--', label=f'Median: {delay_arr.median():.1f} min')
        ax.axvline(delay_arr.quantile(0.95), color='black', linestyle=':', label=f'95th %ile: {delay_arr.quantile(0.95):.0f} min')
        ax.set_title("Distribution of Arrival Delay", fontsize=11, fontweight='bold')
        ax.set_xlabel("Arrival Delay (minutes)"); ax.set_ylabel("Number of Flights")
        ax.legend(fontsize=8)
        plt.tight_layout()
        show(fig)
        st.caption(f"Early (<0): **{(delay_arr<0).mean()*100:.1f}%** | On-Time (0–15 min): **{((delay_arr>=0)&(delay_arr<=15)).mean()*100:.1f}%** | Severely delayed (>60 min): **{(delay_arr>60).mean()*100:.1f}%**")

    with col2:
        st.markdown("**Distribution of Departure Delay**")
        delay_dep = data['DEPARTURE_DELAY'].dropna()
        delay_dep = delay_dep[(delay_dep >= -60) & (delay_dep <= 180)]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(delay_dep, bins=80, color='steelblue', edgecolor='white')
        ax.axvline(delay_dep.mean(),   color='red',   linestyle='--', label=f'Mean: {delay_dep.mean():.1f} min')
        ax.axvline(delay_dep.median(), color='green', linestyle='--', label=f'Median: {delay_dep.median():.1f} min')
        ax.set_title("Departure Delay Distribution", fontsize=11, fontweight='bold')
        ax.set_xlabel("Delay (minutes)"); ax.set_ylabel("Number of Flights")
        ax.legend(fontsize=8)
        plt.tight_layout()
        show(fig)
        st.caption(f"Mean: **{delay_dep.mean():.1f} min** | Median: **{delay_dep.median():.1f} min**")

    obs(f"Most flights are on time or early. Mean >> Median shows extreme delays pull the average up. {(delay_arr<0).mean()*100:.1f}% of flights arrived early in 2015.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 10. Top 15 Airports — Departure Delay ────────────────────────────────
    hdr("10 · Top 15 Airports with Highest Average Departure Delay")
    airport_delay = data[data['CANCELLED'] == 0].groupby('ORIGIN_AIRPORT').agg(
        avg_delay=('DEPARTURE_DELAY', 'mean'),
        flight_count=('DEPARTURE_DELAY', 'count')
    ).reset_index()
    airport_delay = airport_delay[airport_delay['flight_count'] > 1000]
    top15 = airport_delay.nlargest(15, 'avg_delay')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top15['ORIGIN_AIRPORT'], top15['avg_delay'], color='tomato', edgecolor='white')
    ax.set_title("Top 15 Airports with Highest Average Departure Delay", fontsize=13, fontweight='bold')
    ax.set_xlabel("Avg Delay (minutes)"); ax.set_ylabel("Origin Airport")
    ax.invert_yaxis()
    plt.tight_layout()
    show(fig)
    obs("Mid-size congested airports show worse delays than major hubs — major hubs have better scheduling buffers and recovery capacity.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 11. Flight Distance vs Arrival Delay ──────────────────────────────────
    hdr("11 · Flight Distance vs Arrival Delay")
    sample = data[(data['ARRIVAL_DELAY'].between(-60, 180)) &
                  (data['CANCELLED'] == 0)].sample(10000, random_state=42)
    sample['DIST_BUCKET'] = pd.cut(sample['DISTANCE'],
        bins=[0, 500, 1000, 2000, 5000],
        labels=['Short (<500mi)', 'Medium (500-1000mi)',
                'Long (1000-2000mi)', 'Ultra (>2000mi)'])
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(data=sample, x='DISTANCE', y='ARRIVAL_DELAY',
                    hue='DIST_BUCKET', palette='coolwarm', alpha=0.4, s=15, ax=ax)
    z = np.polyfit(sample['DISTANCE'], sample['ARRIVAL_DELAY'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample['DISTANCE'].min(), sample['DISTANCE'].max(), 100)
    ax.plot(x_line, p(x_line), color='red', linewidth=2, linestyle='--', label='Trend Line')
    ax.axhline(0, color='black', linestyle=':', linewidth=1, label='On Time')
    ax.set_title("Flight Distance vs Arrival Delay\n(does flying further mean more delay?)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Flight Distance (miles)"); ax.set_ylabel("Arrival Delay (minutes)")
    ax.legend(title="Distance Category", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    show(fig)
    corr = sample['DISTANCE'].corr(sample['ARRIVAL_DELAY'])
    obs(f"Correlation between distance and arrival delay: **{corr:.3f}**. Longer flights tend to arrive closer to on-time — pilots can recover delays in the air on long routes. Short flights show the highest delay variance.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 12. Delay Severity Composition ───────────────────────────────────────
    hdr("12 · Delay Severity Composition by Airline")
    data_tmp = data.copy()
    data_tmp['delay_severity'] = pd.cut(
        data_tmp['ARRIVAL_DELAY'],
        bins=[-1000, 0, 15, 60, 10000],
        labels=['On Time', 'Minor', 'Moderate', 'Severe']
    )
    severity = (
        data_tmp.groupby('AIRLINE')['delay_severity']
        .value_counts(normalize=True)
        .unstack()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    severity.plot(kind='bar', stacked=True, ax=ax,
                  color=['#10b981','#3b82f6','#f59e0b','#ef4444'])
    ax.set_title("Delay Severity Composition by Airline", fontsize=13, fontweight='bold')
    ax.set_ylabel("Proportion"); ax.set_xlabel("Airline")
    ax.legend(title="Severity", bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    show(fig)
    obs("Hawaiian Airlines (HA) has the highest on-time proportion. Budget carriers like Spirit show a larger share of moderate delays.")


# ══════════════════════════════════════════════════════════════════════════════
#  MILESTONE 3 — CANCELLATIONS & ROUTES
# ══════════════════════════════════════════════════════════════════════════════
elif "Milestone 3" in page:
    st.markdown('<p class="page-title">🚫 Milestone 3 — Cancellations & Route Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Investigating where, why, and when flights are cancelled across routes, airlines, months, and seasons.</p>', unsafe_allow_html=True)

    if not data_ready:
        need_data()

    # Pre-compute SEASON
    data['SEASON'] = data['MONTH'].apply(lambda m:
        'Winter' if m in [12,1,2] else
        'Spring' if m in [3,4,5] else
        'Summer' if m in [6,7,8] else 'Fall')

    data['SEASON_SORT'] = data['MONTH'].apply(lambda m:
        '1.Winter' if m in [12,1,2] else
        '2.Spring' if m in [3,4,5] else
        '3.Summer' if m in [6,7,8] else '4.Fall')

    # ── 1. Arrival Delay vs Busiest Routes ───────────────────────────────────
    hdr("1 · Arrival Delay vs Busiest Origin — Avg Arrival Delay on Top 10 Busiest Routes")
    route_delay = data[data['CANCELLED'] == 0].groupby('ROUTE').agg(
        avg_delay=('ARRIVAL_DELAY', 'mean'),
        flight_count=('ARRIVAL_DELAY', 'count')
    ).reset_index()
    top10_busy = data['ROUTE'].value_counts().head(10).index
    route_delay_top = route_delay[route_delay['ROUTE'].isin(top10_busy)].sort_values('avg_delay', ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ['red' if v > 5 else 'green' for v in route_delay_top['avg_delay']]
    ax.barh(route_delay_top['ROUTE'], route_delay_top['avg_delay'],
            color=colors, edgecolor='white')
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title("Average Arrival Delay — Top 10 Busiest Routes", fontsize=13, fontweight='bold')
    ax.set_xlabel("Avg Arrival Delay (minutes)"); ax.set_ylabel("Route")
    ax.invert_yaxis()
    plt.tight_layout()
    show(fig)
    obs("Busiest routes are NOT the most delayed — high frequency routes are operationally well managed.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 2. Route Congestion vs Cancellation Rate ──────────────────────────────
    hdr("2 · Route Congestion vs Cancellation Rate — Top 10 Routes with Highest Cancellation Rate")
    route_cancel = data.groupby('ROUTE').agg(
        cancel_rate=('CANCELLED', 'mean'),
        flight_count=('CANCELLED', 'count')
    ).reset_index()
    route_cancel = route_cancel[route_cancel['flight_count'] > 1000]
    route_cancel['cancel_pct'] = route_cancel['cancel_rate'] * 100
    top10_cancel = route_cancel.nlargest(10, 'cancel_pct')
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = sns.color_palette("Reds_r", len(top10_cancel))
    bars = ax.barh(top10_cancel['ROUTE'], top10_cancel['cancel_pct'],
                   color=colors, edgecolor='white')
    for bar, val in zip(bars, top10_cancel['cancel_pct']):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)
    ax.set_title("Top 10 Routes with Highest Cancellation Rate\n(routes with >1000 flights only)",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Cancellation Rate (%)"); ax.set_ylabel("Route")
    ax.invert_yaxis()
    plt.tight_layout()
    show(fig)
    obs("Routes with the highest cancellation rates are predominantly northeast corridor routes — winter weather impact is concentrated in specific geographic regions.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 3. Cancellation Reason Distribution ──────────────────────────────────
    hdr("3 · Cancellation Reason Distribution")
    cancel_only = data[data['CANCELLED'] == 1]
    cancel_counts = cancel_only['CANCELLATION_REASON'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    palette = sns.color_palette("Set2", len(cancel_counts))
    ax.bar(cancel_counts.index, cancel_counts.values, color=palette, edgecolor='white')
    for i, v in enumerate(cancel_counts.values):
        ax.text(i, v + 50, f"{v:,}", ha='center', fontsize=10, fontweight='bold')
    ax.set_title("Cancellation Reasons Distribution", fontsize=13, fontweight='bold')
    ax.set_xlabel("Cancellation Reason"); ax.set_ylabel("Number of Flights")
    plt.tight_layout()
    show(fig)
    obs("Weather (B) is the single largest cancellation cause, accounting for over half of all cancellations. Security (D) is virtually negligible.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 4. Airlines with Highest Cancellation Rate ───────────────────────────
    hdr("4 · Airlines with Highest Cancellation Rate")
    cancel_rate = data.groupby('AIRLINE')['CANCELLED'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#ef4444' if v > 0.02 else '#f59e0b' if v > 0.01 else '#10b981'
              for v in cancel_rate.values]
    cancel_rate.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
    ax.set_title("Cancellation Rate by Airline", fontsize=13, fontweight='bold')
    ax.set_ylabel("Cancellation Rate"); ax.set_xlabel("Airline")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    show(fig)
    obs("Atlantic Southeast (EV) and SkyWest (OO) show the highest cancellation rates — regional carriers operating smaller aircraft in challenging conditions.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 5. Cancellation Rate vs Month ─────────────────────────────────────────
    hdr("5 · Cancellation Rate vs Month")
    cancellation_rate = data.groupby('MONTH')['CANCELLED'].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cancellation_rate.index, cancellation_rate.values,
            marker='o', color='tomato', linewidth=2.5)
    ax.fill_between(cancellation_rate.index, cancellation_rate.values,
                    alpha=0.15, color='tomato')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_title("Cancellation Rate by Month", fontsize=13, fontweight='bold')
    ax.set_xlabel("Month"); ax.set_ylabel("Cancellation Rate")
    plt.tight_layout()
    show(fig)
    obs("February has the highest cancellation rate — peak winter storm season. Summer months show the lowest rates despite carrying the highest traffic volume.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 6. Average Arrival Delay by Route × Month (Heatmap) ──────────────────
    hdr("6 · Average Arrival Delay by Route × Month (Heatmap)")
    top10_routes = data['ROUTE'].value_counts().head(10).index
    subset = data[data['ROUTE'].isin(top10_routes) & (data['CANCELLED'] == 0)]
    pivot = subset.groupby(['ROUTE', 'MONTH'])['ARRIVAL_DELAY'].mean().unstack()
    month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                 7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    pivot.columns = [month_map[c] for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                linewidths=0.5, cbar_kws={'label': 'Avg Delay (min)'}, ax=ax)
    ax.set_title("Avg Arrival Delay by Route × Month", fontsize=13, fontweight='bold')
    ax.set_xlabel("Month"); ax.set_ylabel("Route")
    plt.tight_layout()
    show(fig)
    obs("Summer months (Jun–Aug) consistently show the deepest red. SFO-LAX and LAX-SFO remain reliably green year-round — the most efficient corridor.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 7. Seasonal Trend Analysis — Weather Delay Violin ────────────────────
    hdr("7 · Seasonal Trend Analysis — Weather Delay Distribution by Season")
    weather_flights = data[
        (data['CANCELLED'] == 0) &
        (data['WEATHER_DELAY'] > 0) &
        (data['WEATHER_DELAY'] < 120)
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.violinplot(data=weather_flights, x='SEASON_SORT', y='WEATHER_DELAY',
                   hue='SEASON_SORT', palette='coolwarm', inner='box',
                   legend=False, ax=ax)
    mean_val = weather_flights['WEATHER_DELAY'].mean()
    ax.axhline(mean_val, color='red', linestyle='--',
               label=f"Overall mean: {mean_val:.1f} min")
    ax.set_title("Weather Delay Distribution by Season\n(flights with weather delay only)",
                 fontsize=13, fontweight='bold')
    ax.set_xlabel("Season"); ax.set_ylabel("Weather Delay (minutes)")
    ax.set_xticklabels(['Winter', 'Spring', 'Summer', 'Fall'])
    ax.legend()
    plt.tight_layout()
    show(fig)
    obs("Winter violin is widest and shifted highest — weather delays in winter are not only more frequent but more severe, with a longer upper tail than any other season.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 8. Monthly Cancellation Trend by Reason ───────────────────────────────
    hdr("8 · Monthly Cancellation Trend with Reason Breakdown")
    monthly_cancel = data[data['CANCELLED'] == 1].groupby(
        ['MONTH', 'CANCELLATION_REASON']).size().unstack(fill_value=0)
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    monthly_cancel.index = month_names[:len(monthly_cancel)]
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_cancel.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title("Monthly Cancellation Trend by Reason", fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Cancellations"); ax.set_xlabel("Month")
    ax.tick_params(axis='x', rotation=30)
    ax.legend(title="Reason", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    show(fig)
    obs("Weather cancellations spike in Jan–Feb. Carrier cancellations are consistent year-round — forming the stable base of every bar.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 9. Winter vs Summer Cancellation Comparison (Seasonal Stacked Bar) ───
    hdr("9 · Winter vs Summer Cancellation Comparison — Cancellation Reasons by Season")
    season_cancel = data[data['CANCELLED'] == 1].groupby(
        ['SEASON', 'CANCELLATION_REASON']).size().unstack(fill_value=0)
    season_order = [s for s in ['Winter', 'Spring', 'Summer', 'Fall']
                    if s in season_cancel.index]
    season_cancel = season_cancel.reindex(season_order)
    fig, ax = plt.subplots(figsize=(10, 5))
    season_cancel.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title("Cancellation Reasons by Season", fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Cancellations"); ax.set_xlabel("Season")
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title="Reason", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    show(fig)
    obs("Winter has 3× more weather cancellations than any other season. Summer has the fewest total cancellations despite the highest passenger volume.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 10. Arrival Delay Distribution by Airline (Violin) ───────────────────
    hdr("10 · Arrival Delay Distribution by Airline (Violin Plot)")
    top_airlines_6 = data['AIRLINE'].value_counts().head(6).index
    subset_violin = data[data['AIRLINE'].isin(top_airlines_6)]
    subset_violin = subset_violin[subset_violin['ARRIVAL_DELAY'].between(-30, 120)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x='AIRLINE', y='ARRIVAL_DELAY', data=subset_violin,
                   hue='AIRLINE', palette='Set2', inner='box',
                   legend=False, ax=ax)
    ax.axhline(0, color='red', linestyle='--', label='On Time')
    ax.set_title("Arrival Delay Distribution by Airline", fontsize=13, fontweight='bold')
    ax.set_xlabel("Airline"); ax.set_ylabel("Arrival Delay (Minutes)")
    ax.legend()
    plt.tight_layout()
    show(fig)
    obs("All airlines show median below 0 — most flights arrive early. Spirit and Frontier show wider upper bodies — higher concentration of moderate delays compared to Hawaiian and Alaska.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.8rem; padding:2rem 0 1rem 0;'>
✈️ Flight Data Analysis · 2015 US Domestic Flights · Streamlit + Pandas + Matplotlib + Seaborn
</div>
""", unsafe_allow_html=True)