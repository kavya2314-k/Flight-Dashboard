import streamlit as st
import pandas as pd
import numpy as np
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
# DATA LOADER
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Loading dataset...")
def load_data():
    import gdown

    FILE_ID = "1Xz4srzZ6mRK5GJJqyB3UY8WXQQCLdfrB"
    dest = "airline_preprocessed.parquet"

    if not os.path.exists(dest):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", dest, quiet=False)

    cols = [
        'AIRLINE','MONTH','ARRIVAL_DELAY','DEPARTURE_DELAY',
        'CANCELLED','CANCELLATION_REASON',
        'ORIGIN_AIRPORT','DESTINATION_AIRPORT',
        'AIR_SYSTEM_DELAY','SECURITY_DELAY',
        'AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY',
        'DISTANCE'
    ]

    df = pd.read_parquet(dest, columns=cols)

    # Memory optimization
    for col in df.select_dtypes(include=['float64','int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    df.columns = df.columns.str.strip().str.upper()

    df['ROUTE'] = df['ORIGIN_AIRPORT'] + "_" + df['DESTINATION_AIRPORT']

    df['CANCELLATION_REASON'] = df['CANCELLATION_REASON'].replace({
        'A': 'Airline',
        'B': 'Weather',
        'C': 'NAS',
        'D': 'Security'
    })

    return df


# LOAD DATA
data = load_data()

# ─────────────────────────────────────────────
# OPTIONAL SAMPLING CONTROL
# ─────────────────────────────────────────────
use_sample = st.sidebar.checkbox("Use Sample Data (faster)", value=True)

if use_sample:
    data = data.sample(200000, random_state=42)

# ─────────────────────────────────────────────
# IMPORT MILESTONE FILES
# ─────────────────────────────────────────────
from milestone2 import all_charts as milestone2_charts
from milestone3 import all_charts as milestone3_charts

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

    st.header("Milestone 2 Visualizations")

    charts = milestone2_charts(data)

    for i, fig in enumerate(charts):
        st.subheader(f"Chart {i+1}")
        st.pyplot(fig)
        plt.close(fig)

# ─────────────────────────────────────────────
# MILESTONE 3
# ─────────────────────────────────────────────
elif page == "Milestone 3":

    st.header("Milestone 3 Visualizations")

    charts = milestone3_charts(data)

    for i, fig in enumerate(charts):
        st.subheader(f"Chart {i+1}")
        st.pyplot(fig)
        plt.close(fig)
