import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import re

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AirFly Insights", page_icon="✈️", layout="wide")

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


data = load_data()

# Prevent crash
if st.sidebar.checkbox("Use Sample Data (Recommended)", value=True):
    data = data.sample(150000, random_state=42)

# ─────────────────────────────────────────────
# HELPER: RUN SCRIPT CONTENT SAFELY
# ─────────────────────────────────────────────
def run_inline_script(file_name):
    plt.close('all')

    try:
        with open(file_name, "r", encoding="utf-8") as f:
            code = f.read()

        # 🔥 REMOVE CSV LOADING LINES AUTOMATICALLY
        code = re.sub(r".*pd\.read_csv\(.*\).*", "", code)

        # Inject dataset
        global_vars = {
            "data": data,
            "df": data,
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns
        }

        exec(code, global_vars)

        figs = [plt.figure(n) for n in plt.get_fignums()]

        if not figs:
            st.warning(f"No charts found in {file_name}")

        for fig in figs:
            st.pyplot(fig)
            plt.close(fig)

    except Exception as e:
        st.error(f"❌ Error in {file_name}: {e}")


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
    st.header("📊 Milestone 2 Charts")
    run_inline_script("milestone2.py")

# ─────────────────────────────────────────────
# MILESTONE 3
# ─────────────────────────────────────────────
elif page == "Milestone 3":
    st.header("📊 Milestone 3 Charts")
    run_inline_script("milestone3.py")
