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
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# DATA LOADER (OPTIMIZED + FIXED)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Loading dataset… please wait.")
def load_data():
    import gdown

    FILE_ID = "1Xz4srzZ6mRK5GJJqyB3UY8WXQQCLdfrB"
    dest = "airline_preprocessed.parquet"

    if not os.path.exists(dest):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", dest, quiet=False)

    # ✅ LOAD ONLY REQUIRED COLUMNS (CRITICAL)
    cols = [
        'AIRLINE','MONTH','ARRIVAL_DELAY','DEPARTURE_DELAY',
        'CANCELLED','CANCELLATION_REASON',
        'ORIGIN_AIRPORT','DESTINATION_AIRPORT',
        'AIR_SYSTEM_DELAY','SECURITY_DELAY',
        'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY',
        'DISTANCE'
    ]

    df = pd.read_parquet(dest, columns=cols)

    # ✅ MEMORY OPTIMIZATION
    for col in df.select_dtypes(include=['float64','int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # ✅ SAMPLE DATA (VERY IMPORTANT FOR STREAMLIT)
    df = df.sample(200000, random_state=42)

    df.columns = df.columns.str.strip().str.upper()

    # CREATE ROUTE COLUMN
    df['ROUTE'] = df['ORIGIN_AIRPORT'] + "_" + df['DESTINATION_AIRPORT']

    # CLEAN CANCELLATION LABELS
    df['CANCELLATION_REASON'] = df['CANCELLATION_REASON'].replace({
        'A': 'Airline/Carrier',
        'B': 'Weather',
        'C': 'NAS',
        'D': 'Security'
    })

    return df


# LOAD DATA
try:
    data = load_data()
    data_ready = True
except Exception as e:
    data_ready = False
    st.error(f"❌ Error loading data: {e}")


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("✈️ AirFly Insights Dashboard")

if not data_ready:
    st.stop()

# SIDEBAR NAV
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Milestone 2", "Milestone 3"]
)

# ─────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────
if page == "Overview":

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Flights", f"{len(data):,}")
    col2.metric("Airlines", data["AIRLINE"].nunique())
    col3.metric("Routes", data["ROUTE"].nunique())

# ─────────────────────────────────────────────────────────────
# MILESTONE 2
# ─────────────────────────────────────────────────────────────
elif page == "Milestone 2":

    st.subheader("Top Airlines by Flight Volume")

    top_airlines = data['AIRLINE'].value_counts().head(10)

    fig, ax = plt.subplots()
    sns.barplot(x=top_airlines.values, y=top_airlines.index, ax=ax)

    for i, v in enumerate(top_airlines.values):
        ax.text(v, i, f"{v:,}", va='center')

    st.pyplot(fig)


    st.subheader("Monthly Flight Trend")

    monthly = data.groupby('MONTH').size()

    fig, ax = plt.subplots()
    ax.plot(monthly.index, monthly.values, marker='o')

    st.pyplot(fig)


    st.subheader("Delay Distribution")

    fig, ax = plt.subplots()
    sns.histplot(data['ARRIVAL_DELAY'], bins=50, ax=ax)

    st.pyplot(fig)


# ─────────────────────────────────────────────────────────────
# MILESTONE 3
# ─────────────────────────────────────────────────────────────
elif page == "Milestone 3":

    st.subheader("Cancellation Reasons")

    cancel_counts = data['CANCELLATION_REASON'].value_counts()

    fig, ax = plt.subplots()
    sns.barplot(x=cancel_counts.index, y=cancel_counts.values, ax=ax)

    for i, v in enumerate(cancel_counts.values):
        ax.text(i, v, f"{v:,}", ha='center')

    st.pyplot(fig)


    st.subheader("Cancellation by Month")

    pivot = data.pivot_table(
        index='MONTH',
        columns='CANCELLATION_REASON',
        values='CANCELLED',
        aggfunc='sum'
    )

    fig, ax = plt.subplots()
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="coolwarm", ax=ax)

    st.pyplot(fig)


    st.subheader("Weather Delay Distribution")

    fig, ax = plt.subplots()
    sns.violinplot(x=data['MONTH'], y=data['WEATHER_DELAY'], ax=ax)

    st.pyplot(fig)
