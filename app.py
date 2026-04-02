import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AirFly Insights", page_icon="✈️", layout="wide")

# ─────────────────────────────────────────────
# DATA LOADING (FIXED - NO CRASH)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Loading flight dataset...")
def load_data():
    import gdown

    FILE_ID = "1Xz4srzZ6mRK5GJJqyB3UY8WXQQCLdfrB"
    dest = "airline_preprocessed.parquet"

    if not os.path.exists(dest):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", dest, quiet=False)

    # 🔥 LOAD ONLY REQUIRED COLUMNS
    cols = [
        'AIRLINE','MONTH','ARRIVAL_DELAY','DEPARTURE_DELAY',
        'CANCELLED','CANCELLATION_REASON',
        'ORIGIN_AIRPORT','DESTINATION_AIRPORT',
        'AIR_SYSTEM_DELAY','SECURITY_DELAY',
        'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY',
        'DISTANCE'
    ]

    df = pd.read_parquet(dest, columns=cols)

    # 🔥 Reduce memory
    for col in df.select_dtypes(include=['float64','int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # Cleaning
    df.columns = df.columns.str.strip().str.upper()
    df['ROUTE'] = df['ORIGIN_AIRPORT'] + "_" + df['DESTINATION_AIRPORT']

    df['CANCELLATION_REASON'] = df['CANCELLATION_REASON'].replace({
        'A': 'Airline',
        'B': 'Weather',
        'C': 'NAS',
        'D': 'Security'
    })

    return df


data = load_data()

# 🔥 USER CONTROL (VERY IMPORTANT)
use_sample = st.sidebar.checkbox("⚡ Use Fast Mode (Recommended)", value=True)

if use_sample:
    data = data.sample(200000, random_state=42)

# ─────────────────────────────────────────────
# UI HEADER
# ─────────────────────────────────────────────
st.title("✈️ AirFly Insights Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Milestone 2", "Milestone 3"]
)

# ─────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────
if page == "Overview":

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Flights", f"{len(data):,}")
    col2.metric("Airlines", data["AIRLINE"].nunique())
    col3.metric("Routes", data["ROUTE"].nunique())
    col4.metric("Cancellation Rate", f"{data['CANCELLED'].mean()*100:.2f}%")

# ─────────────────────────────────────────────
# MILESTONE 2
# ─────────────────────────────────────────────
elif page == "Milestone 2":

    st.header("📊 Delay Analysis Dashboard")

    # 1. Top Airlines
    st.subheader("Top Airlines by Flight Volume")
    top_airlines = data['AIRLINE'].value_counts().head(10)

    fig, ax = plt.subplots()
    sns.barplot(x=top_airlines.values, y=top_airlines.index, ax=ax)

    for i, v in enumerate(top_airlines.values):
        ax.text(v, i, f"{v:,}", va='center')

    st.pyplot(fig)

    # 2. Monthly Trend
    st.subheader("Monthly Flight Trend")
    monthly = data.groupby('MONTH').size()

    fig, ax = plt.subplots()
    ax.plot(monthly.index, monthly.values, marker='o')
    st.pyplot(fig)

    # 3. Delay Histogram
    st.subheader("Arrival Delay Distribution")

    fig, ax = plt.subplots()
    sns.histplot(data['ARRIVAL_DELAY'], bins=50, kde=True, ax=ax)
    st.pyplot(fig)

    # 4. Boxplot
    st.subheader("Delay Distribution by Airline")

    fig, ax = plt.subplots()
    sns.boxplot(x='AIRLINE', y='ARRIVAL_DELAY', data=data.head(50000), ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 5. Heatmap
    st.subheader("Delay Heatmap (Airline vs Month)")

    pivot = data.pivot_table(
        values='ARRIVAL_DELAY',
        index='AIRLINE',
        columns='MONTH',
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(pivot, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ─────────────────────────────────────────────
# MILESTONE 3
# ─────────────────────────────────────────────
elif page == "Milestone 3":

    st.header("🚫 Cancellation & Route Analysis")

    # 1. Cancellation Reasons
    st.subheader("Cancellation Reasons")

    cancel_counts = data['CANCELLATION_REASON'].value_counts()

    fig, ax = plt.subplots()
    sns.barplot(x=cancel_counts.index, y=cancel_counts.values, ax=ax)
    st.pyplot(fig)

    # 2. Cancellation Rate by Month
    st.subheader("Cancellation Rate by Month")

    cancel_rate = data.groupby('MONTH')['CANCELLED'].mean()

    fig, ax = plt.subplots()
    ax.plot(cancel_rate.index, cancel_rate.values, marker='o')
    st.pyplot(fig)

    # 3. Route Analysis
    st.subheader("Top Routes")

    top_routes = data['ROUTE'].value_counts().head(10)

    fig, ax = plt.subplots()
    sns.barplot(x=top_routes.values, y=top_routes.index, ax=ax)
    st.pyplot(fig)
