import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AirFly Insights",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
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
                border-radius:8px; padding:14px 18px; text-align:center; }
    .stat-val { font-size:1.7rem; font-weight:700; color:#1e3a5f; }
    .stat-lbl { font-size:0.8rem; color:#64748b; }
    hr { border:none; border-top:1px solid #e2e8f0; margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ AirFly Insights")
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
    st.caption("Data: Kaggle Airlines Dataset · 5.8M flights")

# ── Helpers ───────────────────────────────────────────────────────────────────
def obs(txt):
    st.markdown(f'<div class="obs">💡 <b>Observation:</b> {txt}</div>', unsafe_allow_html=True)

def hdr(txt):
    st.markdown(f'<div class="section-hdr">{txt}</div>', unsafe_allow_html=True)

def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Data loader ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Loading dataset… please wait.")
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


    # Download from Google Drive
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", csv_dest, quiet=False)
    df = pd.read_csv(csv_dest, low_memory=False)

    # Fix column names
    df.columns = df.columns.str.strip().str.upper()

    # Fill delay nulls
    for col in ['AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Ensure ROUTE exists
    if 'ROUTE' not in df.columns:
        df['ROUTE'] = df['ORIGIN_AIRPORT'].astype(str) + '_' + df['DESTINATION_AIRPORT'].astype(str)

    # Ensure MONTH is integer
    df['MONTH'] = pd.to_numeric(df['MONTH'], errors='coerce').fillna(1).astype(int)

    # Fix CANCELLATION_REASON — remove "Not Cancelled" for cancelled-only charts
    df['CANCELLATION_REASON'] = df['CANCELLATION_REASON'].replace({
        'A': 'Airline/Carrier', 'B': 'Weather',
        'C': 'National Aviation System', 'D': 'Security'
    })

    # Save as parquet for next load
    try:
        df.to_parquet(dest, index=False)
    except Exception:
        pass

    return df

try:
    data = load_data()
    data.columns = data.columns.str.strip().str.upper()
    data_ready = True
except Exception as e:
    data_ready = False
    st.error(f"❌ Error loading data: {e}")

def need_data():
    st.error("❌ Data could not be loaded.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<p class="page-title">✈️ AirFly Insights Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Data Visualization and Analysis of 2015 US Domestic Airline Operations</p>', unsafe_allow_html=True)

    if data_ready:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{len(data):,}</div><div class="stat-lbl">Total Flights</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{data["AIRLINE"].nunique()}</div><div class="stat-lbl">Airlines</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box"><div class="stat-val">{data["ROUTE"].nunique():,}</div><div class="stat-lbl">Unique Routes</div></div>', unsafe_allow_html=True)
        with c4:
            pct = data["CANCELLED"].mean() * 100
            st.markdown(f'<div class="stat-box"><div class="stat-val">{pct:.1f}%</div><div class="stat-lbl">Cancellation Rate</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
### 📖 Project Storyline
| Milestone | Focus | Key Questions |
|-----------|-------|---------------|
| **Milestone 1** | Data Preparation | Cleaning, encoding, feature engineering |
| **Milestone 2** | Delay Analysis | Which airlines/routes delay most? What causes delays? When? |
| **Milestone 3** | Cancellations & Routes | Where do cancellations happen? Why? Does season matter? |

Use the **sidebar** to navigate between milestones.
""")

# ══════════════════════════════════════════════════════════════════════════════
#  MILESTONE 2 — DELAY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Milestone 2" in page:
    st.markdown('<p class="page-title">📊 Milestone 2 — Delay Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Exploring flight delays across airlines, routes, months, and delay causes.</p>', unsafe_allow_html=True)

    if not data_ready:
        need_data()

    # ── Chart 1: Top 10 Airlines by Flight Volume ─────────────────────────────
    hdr("1 · Top 10 Airlines by Flight Volume")
    top_airlines = data['AIRLINE'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=top_airlines.values, y=top_airlines.index,
                hue=top_airlines.index, palette="Blues_r", legend=False, ax=ax)
    for i, v in enumerate(top_airlines.values):
        ax.text(v + 0.01 * max(top_airlines.values), i, f"{v:,}", va='center', fontsize=9)
    ax.set_title("Top 10 Airlines by Flight Volume", fontsize=13, fontweight='bold')
    ax.set_xlabel("Number of Flights")
    ax.set_ylabel("Airline")
    plt.tight_layout()
    show(fig)
    obs("Southwest (WN) dominates with nearly double the volume of the next airline — operating over 1.2 million flights in 2015.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 2: Monthly Flight Volume Trend ──────────────────────────────────
    hdr("2 · Busiest Months — Monthly Flight Volume Trend")
    monthly_flights = data.groupby('MONTH').size()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly_flights.index, monthly_flights.values, marker='o', color='steelblue', linewidth=2.5)
    ax.fill_between(monthly_flights.index, monthly_flights.values, alpha=0.15, color='steelblue')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_title("Monthly Flight Volume Trend", fontsize=13, fontweight='bold')
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Flights")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    show(fig)
    obs("Summer months (Jun–Aug) carry the heaviest traffic peaking in July. February dips to the lowest — fewer calendar days and winter weather suppress demand.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 3: Route Congestion vs Delay ───────────────────────────────────
    hdr("3 · Route Congestion vs Average Delay")
    route_stats = data.groupby('ROUTE').agg(
        ARRIVAL_DELAY=('ARRIVAL_DELAY','mean'),
        flight_count=('ROUTE','count')
    ).reset_index()
    route_stats = route_stats[route_stats['flight_count'] > 500]
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(route_stats['flight_count'], route_stats['ARRIVAL_DELAY'],
                    c=route_stats['ARRIVAL_DELAY'], cmap='RdYlGn_r', s=20, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Avg Delay (min)')
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title("Route Congestion vs Average Delay", fontsize=13, fontweight='bold')
    ax.set_xlabel("Number of Flights")
    ax.set_ylabel("Average Arrival Delay (min)")
    plt.tight_layout()
    show(fig)
    obs("High-volume routes do NOT equal high delays — the busiest corridors are operationally efficient. Low-frequency routes show the most variance in delays.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 4: Delay Distribution by Top Airlines (Boxplot) ────────────────
    hdr("4 · Delay Distribution by Top 5 Airlines")
    top5 = data['AIRLINE'].value_counts().head(5).index
    filtered = data[data['AIRLINE'].isin(top5)]
    filtered = filtered[filtered['ARRIVAL_DELAY'].between(-30, 120)]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=filtered, x='AIRLINE', y='ARRIVAL_DELAY',
                hue='AIRLINE', palette='Set2', legend=False,
                flierprops=dict(marker='.', markersize=1), ax=ax)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, label='On Time')
    ax.set_title("Delay Distribution by Top 5 Airlines", fontsize=13, fontweight='bold')
    ax.set_xlabel("Airline")
    ax.set_ylabel("Arrival Delay (Minutes)")
    ax.legend()
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    show(fig)
    obs("Most median delays sit at or below zero — the majority of flights arrive on time or early. The boxes show where most delays are concentrated.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 5: Top 10 Busiest Routes ───────────────────────────────────────
    hdr("5 · Top 10 Busiest Routes")
    top_routes = data['ROUTE'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    top_routes.plot(kind='barh', color='darkorange', ax=ax)
    ax.set_title("Top 10 Busiest Routes", fontsize=13, fontweight='bold')
    ax.set_xlabel("Number of Flights")
    ax.set_ylabel("Route")
    ax.invert_yaxis()
    plt.tight_layout()
    show(fig)
    obs("SFO-LAX is the busiest corridor — short-haul high-frequency routes dominate the top 10. LAX appears in 7 out of 10 routes.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 6: Average Delay Causes by Airline (Stacked Bar — Minutes) ─────
    hdr("6 · Average Delay Causes by Airline (Minutes)")
    cause_cols = ['AIR_SYSTEM_DELAY','SECURITY_DELAY','WEATHER_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY']
    available_causes = [c for c in cause_cols if c in data.columns]
    delay_causes = data.groupby('AIRLINE')[available_causes].mean().head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    delay_causes.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='white', width=0.7)
    ax.set_title("Average Delay Causes by Airline", fontsize=13, fontweight='bold')
    ax.set_ylabel("Average Delay (Minutes)")
    ax.set_xlabel("Airline")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="Delay Cause", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    show(fig)
    obs("Late Aircraft Delay is the dominant cause across all airlines — a cascading effect where one delayed flight triggers the next throughout the day.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 7: Delay Cause % Breakdown ─────────────────────────────────────
    hdr("7 · Delay Cause % Breakdown by Airline")
    cause_labels_map = {
        'AIRLINE_DELAY':'Carrier','WEATHER_DELAY':'Weather',
        'AIR_SYSTEM_DELAY':'NAS','SECURITY_DELAY':'Security',
        'LATE_AIRCRAFT_DELAY':'Late Aircraft'
    }
    cols_available = [c for c in ['AIRLINE_DELAY','WEATHER_DELAY','AIR_SYSTEM_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY'] if c in data.columns]
    delay_causes2 = data[data['ARRIVAL_DELAY'] > 0].groupby('AIRLINE')[cols_available].mean()
    delay_causes2.columns = [cause_labels_map[c] for c in cols_available]
    delay_pct = delay_causes2.div(delay_causes2.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(13, 6))
    delay_pct.plot(kind='bar', stacked=True, ax=ax, colormap='Set2', edgecolor='white', width=0.7)
    for bar_stack in ax.containers:
        for bar in bar_stack:
            h = bar.get_height()
            if h > 5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + h/2,
                        f'{h:.0f}%', ha='center', va='center', fontsize=7.5, color='black', fontweight='bold')
    ax.set_title("Delay Cause Breakdown by Airline (% of Total Delay)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Percentage of Total Delay (%)")
    ax.set_xlabel("Airline")
    ax.set_ylim(0, 105)
    ax.legend(title="Delay Cause", bbox_to_anchor=(1.01, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    show(fig)
    obs("Late Aircraft and Carrier delays together account for 70–80% of total delay. Security delay is negligible (<1%) for all carriers.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 8: Weather Delay by Month ──────────────────────────────────────
    hdr("8 · Average Weather Delay by Month")
    weather_month = data.groupby('MONTH')['WEATHER_DELAY'].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(weather_month.index, weather_month.values, marker='s', color='purple', linewidth=2.5)
    ax.fill_between(weather_month.index, weather_month.values, alpha=0.15, color='purple')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_title("Average Weather Delay by Month", fontsize=13, fontweight='bold')
    ax.set_xlabel("Month")
    ax.set_ylabel("Weather Delay (Minutes)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    show(fig)
    obs("Weather delays peak in February (winter storms) and again in June–July (thunderstorm season). September–October are the most weather-stable months.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 9: Arrival & Departure Delay Histograms ────────────────────────
    hdr("9 · Arrival & Departure Delay Distributions")
    col1, col2 = st.columns(2)
    with col1:
        delay_arr = data['ARRIVAL_DELAY'].dropna()
        delay_arr = delay_arr[(delay_arr >= -60) & (delay_arr <= 180)]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(delay_arr, bins=80, color='steelblue', edgecolor='white')
        ax.axvline(delay_arr.mean(), color='red', linestyle='--', label=f'Mean: {delay_arr.mean():.1f} min')
        ax.axvline(delay_arr.median(), color='green', linestyle='--', label=f'Median: {delay_arr.median():.1f} min')
        ax.axvline(delay_arr.quantile(0.95), color='black', linestyle=':', label=f'95th %ile: {delay_arr.quantile(0.95):.0f} min')
        ax.set_title("Distribution of Arrival Delay", fontsize=11, fontweight='bold')
        ax.set_xlabel("Arrival Delay (minutes)")
        ax.set_ylabel("Number of Flights")
        ax.legend(fontsize=8)
        plt.tight_layout()
        show(fig)
        st.caption(f"Early (<0): {(delay_arr<0).mean()*100:.1f}% | On-Time (0-15min): {((delay_arr>=0)&(delay_arr<=15)).mean()*100:.1f}% | Severe (>60min): {(delay_arr>60).mean()*100:.1f}%")

    with col2:
        delay_dep = data['DEPARTURE_DELAY'].dropna()
        delay_dep = delay_dep[(delay_dep >= -60) & (delay_dep <= 180)]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(delay_dep, bins=80, color='coral', edgecolor='white')
        ax.axvline(delay_dep.mean(), color='red', linestyle='--', label=f'Mean: {delay_dep.mean():.1f} min')
        ax.axvline(delay_dep.median(), color='green', linestyle='--', label=f'Median: {delay_dep.median():.1f} min')
        ax.set_title("Distribution of Departure Delay", fontsize=11, fontweight='bold')
        ax.set_xlabel("Departure Delay (minutes)")
        ax.set_ylabel("Number of Flights")
        ax.legend(fontsize=8)
        plt.tight_layout()
        show(fig)
        st.caption(f"Mean: {delay_dep.mean():.1f} min | Median: {delay_dep.median():.1f} min")

    obs("Most flights are on time or early. Mean >> Median confirms a right-skewed distribution — extreme delays pull the average up significantly.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 10: Top 15 Airports by Departure Delay ─────────────────────────
    hdr("10 · Top 15 Airports with Highest Average Departure Delay")
    airport_delay = data[data['CANCELLED']==0].groupby('ORIGIN_AIRPORT').agg(
        avg_delay=('DEPARTURE_DELAY','mean'),
        flight_count=('DEPARTURE_DELAY','count')
    ).reset_index()
    airport_delay = airport_delay[airport_delay['flight_count'] > 1000]
    top15 = airport_delay.nlargest(15, 'avg_delay')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top15['ORIGIN_AIRPORT'], top15['avg_delay'], color='tomato', edgecolor='white')
    ax.set_title("Top 15 Airports with Highest Average Departure Delay", fontsize=13, fontweight='bold')
    ax.set_xlabel("Avg Delay (minutes)")
    ax.set_ylabel("Origin Airport")
    ax.invert_yaxis()
    plt.tight_layout()
    show(fig)
    obs("Mid-size congested airports show worse delays than major hubs — large airports have better ground crews, faster turnarounds and scheduling buffers.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 11: Flight Distance vs Arrival Delay ────────────────────────────
    hdr("11 · Flight Distance vs Arrival Delay")
    if 'DISTANCE' in data.columns:
        sample = data[(data['ARRIVAL_DELAY'].between(-60,180)) & (data['CANCELLED']==0)].sample(10000, random_state=42)
        sample['DIST_BUCKET'] = pd.cut(sample['DISTANCE'],
            bins=[0,500,1000,2000,5000],
            labels=['Short (<500mi)','Medium (500-1000mi)','Long (1000-2000mi)','Ultra (>2000mi)'])
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.scatterplot(data=sample, x='DISTANCE', y='ARRIVAL_DELAY',
                        hue='DIST_BUCKET', palette='coolwarm', alpha=0.4, s=15, ax=ax)
        z = np.polyfit(sample['DISTANCE'], sample['ARRIVAL_DELAY'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(sample['DISTANCE'].min(), sample['DISTANCE'].max(), 100)
        ax.plot(x_line, p(x_line), color='red', linewidth=2, linestyle='--', label='Trend Line')
        ax.axhline(0, color='black', linestyle=':', linewidth=1, label='On Time')
        ax.set_title("Flight Distance vs Arrival Delay", fontsize=13, fontweight='bold')
        ax.set_xlabel("Flight Distance (miles)")
        ax.set_ylabel("Arrival Delay (minutes)")
        ax.legend(title="Distance Category", bbox_to_anchor=(1.01,1), loc='upper left')
        plt.tight_layout()
        show(fig)
        corr = sample['DISTANCE'].corr(sample['ARRIVAL_DELAY'])
        obs(f"Longer flights tend to arrive closer to on-time (correlation: {corr:.3f}) — pilots can recover delays in the air on long routes. Short flights have highest delay variance.")
    else:
        st.warning("DISTANCE column not found in dataset.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 12: Delay Severity Composition ─────────────────────────────────
    hdr("12 · Delay Severity Composition by Airline")
    data_tmp = data.copy()
    data_tmp['delay_severity'] = pd.cut(data_tmp['ARRIVAL_DELAY'],
        bins=[-1000,0,15,60,10000], labels=['On Time','Minor','Moderate','Severe'])
    severity = data_tmp.groupby('AIRLINE')['delay_severity'].value_counts(normalize=True).unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6))
    severity.plot(kind='bar', stacked=True, ax=ax,
                  color=['#10b981','#3b82f6','#f59e0b','#ef4444'])
    ax.set_title("Delay Severity Composition by Airline", fontsize=13, fontweight='bold')
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Airline")
    ax.legend(title="Severity", bbox_to_anchor=(1.01,1), loc='upper left')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    show(fig)
    obs("Hawaiian Airlines (HA) has the highest on-time proportion. Budget carriers Spirit and Frontier show the largest share of moderate and severe delays.")

# ══════════════════════════════════════════════════════════════════════════════
#  MILESTONE 3 — CANCELLATIONS & ROUTES
# ══════════════════════════════════════════════════════════════════════════════
elif "Milestone 3" in page:
    st.markdown('<p class="page-title">🚫 Milestone 3 — Cancellations & Route Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Investigating where, why, and when flights are cancelled — and how routes perform.</p>', unsafe_allow_html=True)

    if not data_ready:
        need_data()

    # Add season columns
    data['SEASON'] = data['MONTH'].apply(lambda m:
        'Winter' if m in [12,1,2] else 'Spring' if m in [3,4,5] else
        'Summer' if m in [6,7,8] else 'Fall')
    data['SEASON_SORT'] = data['MONTH'].apply(lambda m:
        '1.Winter' if m in [12,1,2] else '2.Spring' if m in [3,4,5] else
        '3.Summer' if m in [6,7,8] else '4.Fall')

    # ── Chart 1: Avg Delay on Top 10 Busiest Routes ───────────────────────────
    hdr("1 · Average Arrival Delay — Top 10 Busiest Routes")
    route_delay = data[data['CANCELLED']==0].groupby('ROUTE').agg(
        avg_delay=('ARRIVAL_DELAY','mean'),
        flight_count=('ARRIVAL_DELAY','count')
    ).reset_index()
    top10_busy = data['ROUTE'].value_counts().head(10).index
    route_delay_top = route_delay[route_delay['ROUTE'].isin(top10_busy)].sort_values('avg_delay', ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ['#ef4444' if v > 5 else '#10b981' for v in route_delay_top['avg_delay']]
    ax.barh(route_delay_top['ROUTE'], route_delay_top['avg_delay'], color=colors, edgecolor='white')
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title("Average Arrival Delay — Top 10 Busiest Routes\n(red = avg delayed >5min, green = on time or early)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Avg Arrival Delay (minutes)")
    ax.set_ylabel("Route")
    ax.invert_yaxis()
    plt.tight_layout()
    show(fig)
    obs("Busiest routes are NOT the most delayed — high frequency routes are operationally well managed. JFK routes consistently arrive early due to long flight time for delay recovery.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 2: Top 10 Routes by Cancellation Rate ───────────────────────────
    hdr("2 · Top 10 Routes with Highest Cancellation Rate")
    route_cancel = data.groupby('ROUTE').agg(
        cancel_rate=('CANCELLED','mean'),
        flight_count=('CANCELLED','count')
    ).reset_index()
    route_cancel = route_cancel[route_cancel['flight_count'] > 1000]
    route_cancel['cancel_pct'] = route_cancel['cancel_rate'] * 100
    top10_cancel = route_cancel.nlargest(10, 'cancel_pct')
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = sns.color_palette("Reds_r", len(top10_cancel))
    bars = ax.barh(top10_cancel['ROUTE'], top10_cancel['cancel_pct'], color=colors, edgecolor='white')
    for bar, val in zip(bars, top10_cancel['cancel_pct']):
        ax.text(val+0.05, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontsize=9)
    ax.set_title("Top 10 Routes with Highest Cancellation Rate\n(routes with >1000 flights only)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Cancellation Rate (%)")
    ax.set_ylabel("Route")
    ax.invert_yaxis()
    plt.tight_layout()
    show(fig)
    obs("All top 10 routes are Northeast corridor short-haul connections. LGA appears in 6 out of 10 — its single-runway configuration makes it extremely vulnerable to weather disruption.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 3: Cancellation Reason Distribution ─────────────────────────────
    hdr("3 · Cancellation Reasons Distribution")
    cancel_only = data[(data['CANCELLED']==1) & (data['CANCELLATION_REASON'] != 'Not Cancelled')]
    cancel_counts = cancel_only['CANCELLATION_REASON'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    palette = ['#3b82f6','#ef4444','#f59e0b','#10b981'][:len(cancel_counts)]
    ax.bar(cancel_counts.index, cancel_counts.values, color=palette, edgecolor='white')
    for i, v in enumerate(cancel_counts.values):
        ax.text(i, v+50, f"{v:,}", ha='center', fontsize=10, fontweight='bold')
    ax.set_title("Cancellation Reasons Distribution", fontsize=13, fontweight='bold')
    ax.set_xlabel("Cancellation Reason")
    ax.set_ylabel("Number of Cancelled Flights")
    plt.tight_layout()
    show(fig)
    obs("Weather is the #1 cancellation cause (~54%). Carrier issues are second — these are entirely preventable internal failures. Security cancellations are virtually negligible at only 22 flights.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 4: Cancellation Rate by Airline ─────────────────────────────────
    hdr("4 · Cancellation Rate by Airline")
    cancel_rate = data.groupby('AIRLINE')['CANCELLED'].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#ef4444' if v>3 else '#f59e0b' if v>1.5 else '#10b981' for v in cancel_rate.values]
    cancel_rate.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
    ax.set_title("Cancellation Rate by Airline (%)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Cancellation Rate (%)")
    ax.set_xlabel("Airline")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    show(fig)
    obs("Envoy Air (MQ) has the highest cancellation rate (~5%) — nearly 3x the industry average. Hawaiian (HA) has the lowest — island routes leave no alternative when a flight is cancelled.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 5: Cancellation Rate by Month ───────────────────────────────────
    hdr("5 · Cancellation Rate by Month")
    cancellation_rate = data.groupby('MONTH')['CANCELLED'].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cancellation_rate.index, cancellation_rate.values, marker='o', color='tomato', linewidth=2.5)
    ax.fill_between(cancellation_rate.index, cancellation_rate.values, alpha=0.15, color='tomato')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax.set_title("Cancellation Rate by Month (%)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Month")
    ax.set_ylabel("Cancellation Rate (%)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    show(fig)
    obs("February peaks at ~4.7% — worst month for cancellations due to winter storms. September–October hit yearly lows (~0.5%) — the safest months to fly.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 6: Route × Month Heatmap ────────────────────────────────────────
    hdr("6 · Avg Arrival Delay by Route × Month (Heatmap)")
    top10_routes = data['ROUTE'].value_counts().head(10).index
    subset = data[data['ROUTE'].isin(top10_routes) & (data['CANCELLED']==0)]
    pivot = subset.groupby(['ROUTE','MONTH'])['ARRIVAL_DELAY'].mean().unstack()
    month_map = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                 7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    pivot.columns = [month_map[c] for c in pivot.columns]
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                linewidths=0.5, cbar_kws={'label':'Avg Delay (min)'}, ax=ax)
    ax.set_title("Avg Arrival Delay by Route × Month", fontsize=13, fontweight='bold')
    ax.set_xlabel("Month")
    ax.set_ylabel("Route")
    plt.tight_layout()
    show(fig)
    obs("ORD→LGA is red almost year-round — the most persistently delayed corridor. LAX→SFO spikes to 24.8 min in December — holiday surge on the busiest short-haul route. October is universally green.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 7: Weather Delay by Season (Violin) ─────────────────────────────
    hdr("7 · Weather Delay Distribution by Season")
    weather_flights = data[(data['CANCELLED']==0) & (data['WEATHER_DELAY']>0) & (data['WEATHER_DELAY']<120)]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.violinplot(data=weather_flights, x='SEASON_SORT', y='WEATHER_DELAY',
                   hue='SEASON_SORT', palette='coolwarm', inner='box', legend=False, ax=ax)
    mean_val = weather_flights['WEATHER_DELAY'].mean()
    ax.axhline(mean_val, color='red', linestyle='--', label=f"Overall mean: {mean_val:.1f} min")
    ax.set_title("Weather Delay Distribution by Season\n(flights with weather delay only)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Season")
    ax.set_ylabel("Weather Delay (minutes)")
    ax.set_xticklabels(['Winter','Spring','Summer','Fall'])
    ax.legend()
    plt.tight_layout()
    show(fig)
    obs("Winter violin is widest and shifted highest — weather delays in winter are not only more frequent but more severe, with a longer upper tail than any other season.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 8: Monthly Cancellation Trend by Reason ─────────────────────────
    hdr("8 · Monthly Cancellation Trend by Reason")
    monthly_cancel = data[data['CANCELLED']==1].groupby(
        ['MONTH','CANCELLATION_REASON']).size().unstack(fill_value=0)
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthly_cancel.index = month_names[:len(monthly_cancel)]
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_cancel.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title("Monthly Cancellation Trend by Reason", fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Cancellations")
    ax.set_xlabel("Month")
    ax.tick_params(axis='x', rotation=30)
    ax.legend(title="Reason", bbox_to_anchor=(1.01,1), loc='upper left')
    plt.tight_layout()
    show(fig)
    obs("February dominates with 20,000+ cancellations driven entirely by weather. Carrier cancellations (green) stay flat year-round at ~2,500–3,500 — a constant internal problem unaffected by season.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 9: Cancellation by Season ───────────────────────────────────────
    hdr("9 · Cancellation Reasons by Season")
    season_cancel = data[data['CANCELLED']==1].groupby(
        ['SEASON','CANCELLATION_REASON']).size().unstack(fill_value=0)
    season_order = [s for s in ['Winter','Spring','Summer','Fall'] if s in season_cancel.index]
    season_cancel = season_cancel.reindex(season_order)
    fig, ax = plt.subplots(figsize=(10, 5))
    season_cancel.plot(kind='bar', stacked=True, ax=ax, colormap='Set2')
    ax.set_title("Cancellation Reasons by Season", fontsize=13, fontweight='bold')
    ax.set_ylabel("Number of Cancellations")
    ax.set_xlabel("Season")
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title="Reason", bbox_to_anchor=(1.01,1), loc='upper left')
    plt.tight_layout()
    show(fig)
    obs("Winter has 4x more cancellations than Fall — weather (grey) is the sole driver. Carrier cancellations remain proportionally equal across all seasons proving they are an internal operational issue.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Chart 10: Arrival Delay by Airline (Violin) ────────────────────────────
    hdr("10 · Arrival Delay Distribution by Airline (Violin)")
    top6 = data['AIRLINE'].value_counts().head(6).index
    subset_v = data[data['AIRLINE'].isin(top6)]
    subset_v = subset_v[subset_v['ARRIVAL_DELAY'].between(-30, 120)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x='AIRLINE', y='ARRIVAL_DELAY', data=subset_v,
                   hue='AIRLINE', palette='Set2', inner='box', legend=False, ax=ax)
    ax.axhline(0, color='red', linestyle='--', label='On Time')
    ax.set_title("Arrival Delay Distribution by Airline", fontsize=13, fontweight='bold')
    ax.set_xlabel("Airline")
    ax.set_ylabel("Arrival Delay (Minutes)")
    ax.legend()
    plt.tight_layout()
    show(fig)
    obs("All airlines show median below 0 — most flights arrive early. Spirit and Frontier show wider upper bodies indicating higher concentration of moderate delays compared to Hawaiian and Alaska.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.8rem; padding:2rem 0 1rem 0;'>
✈️ AirFly Insights · 2015 US Domestic Flights · Streamlit + Pandas + Matplotlib + Seaborn
</div>
""", unsafe_allow_html=True)
