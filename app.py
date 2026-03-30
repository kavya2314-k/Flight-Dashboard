import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Required for Streamlit
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AirFly Insights",
    page_icon="✈️",
    layout="wide"
)

# ─────────────────────────────────────────────
# DATA LOADING (OPTIMIZED)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="⏳ Loading dataset...")
def load_data():
    import gdown

    FILE_ID = "1Xz4srzZ6mRK5GJJqyB3UY8WXQQCLdfrB"
    dest = "airline_preprocessed.parquet"

    if not os.path.exists(dest):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", dest, quiet=False)

    # Load only required columns (IMPORTANT)
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


# Load dataset
data = load_data()

# Sampling to prevent crash
use_sample = st.sidebar.checkbox("Use Sample Data (Recommended)", value=True)
if use_sample:
    data = data.sample(150000, random_state=42)

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("✈️ AirFly Insights Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Milestone 2", "Milestone 3"]
)

# ─────────────────────────────────────────────
# SCRIPT RUNNER (FOR MILESTONE FILES)
# ─────────────────────────────────────────────
def run_script(file_name):
    plt.close('all')

    try:
        # Inject variables so your old code works
        globals()['data'] = data
        globals()['df'] = data

        # Execute file
        exec(open(file_name).read())

        # Capture all figures
        figs = [plt.figure(n) for n in plt.get_fignums()]

        if not figs:
            st.warning(f"No charts found in {file_name}")

        for fig in figs:
            st.pyplot(fig)
            plt.close(fig)

    except FileNotFoundError:
        st.error(f"❌ File '{file_name}' not found. Upload it to repo.")
    except Exception as e:
        st.error(f"❌ Error in {file_name}: {e}")


# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────
if page == "Overview":

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Flights", f"{len(data):,}")
    col2.metric("Airlines", data["AIRLINE"].nunique())
    col3.metric("Routes", data["ROUTE"].nunique())


elif page == "Milestone 2":

    st.header("📊 Milestone 2 Charts")
    run_script("milestone2.py")


elif page == "Milestone 3":

    st.header("📊 Milestone 3 Charts")
    run_script("milestone3.py")
