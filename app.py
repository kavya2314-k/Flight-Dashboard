import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # 🔥 important for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AirFly Insights",
    page_icon="✈️",
    layout="wide"
)

# ─────────────────────────────────────────────
# DATA LOADER (FIXED FOR MEMORY)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Loading dataset...")
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

    df.columns = df.columns.str.strip().str.upper()

    # Create route column
    df['ROUTE'] = df['ORIGIN_AIRPORT'] + "_" + df['DESTINATION_AIRPORT']

    # Map cancellation reason
    df['CANCELLATION_REASON'] = df['CANCELLATION_REASON'].replace({
        'A': 'Airline',
        'B': 'Weather',
        'C': 'NAS',
        'D': 'Security'
    })

    return df


# LOAD DATA
data = load_data()

# 🔥 CRITICAL: SAMPLE DATA TO PREVENT CRASH
use_sample = st.sidebar.checkbox("Use Sample Data (Recommended)", value=True)

if use_sample:
    data = data.sample(150000, random_state=42)

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("✈️ AirFly Insights Dashboard")

page = st.sidebar.radio("Navigation", ["Overview", "Milestone 2", "Milestone 3"])

# ─────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────
if page == "Overview":

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Flights", f"{len(data):,}")
    col2.metric("Airlines", data["AIRLINE"].nunique())
    col3.metric("Routes", data["ROUTE"].nunique())

# ─────────────────────────────────────────────
# MILESTONE 2
# ─────────────────────────────────────────────
elif page == "Milestone 2":

    st.header("Milestone 2 Charts")

    plt.close('all')  # 🔥 clear old plots

    # Inject data into milestone file
    globals()['data'] = data

    # Run your file
    exec(open("milestone2.py").read())

    # Show all figures
    figs = [plt.figure(n) for n in plt.get_fignums()]

    for fig in figs:
        st.pyplot(fig)
        plt.close(fig)

# ─────────────────────────────────────────────
# MILESTONE 3
# ─────────────────────────────────────────────
elif page == "Milestone 3":

    st.header("Milestone 3 Charts")

    plt.close('all')

    globals()['data'] = data

    exec(open("milestone3.py").read())

    figs = [plt.figure(n) for n in plt.get_fignums()]

    for fig in figs:
        st.pyplot(fig)
        plt.close(fig)
