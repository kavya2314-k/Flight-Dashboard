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

    df = pd.read_parquet(dest)

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

# OPTIONAL SAMPLE (disable if you want exact results)
if st.sidebar.checkbox("Use Sample Data (faster)", value=True):
    data = data.sample(200000, random_state=42)

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
# MILESTONE 2 (AUTO RUN SCRIPT)
# ─────────────────────────────────────────────
elif page == "Milestone 2":

    st.header("Milestone 2 Charts")

    import matplotlib.pyplot as plt

    # Inject data into global namespace so script can use it
    globals()['data'] = data

    # Execute your milestone file
    exec(open("milestone2.py").read())

    # Show all generated plots
    figs = [plt.figure(n) for n in plt.get_fignums()]

    for fig in figs:
        st.pyplot(fig)
        plt.close(fig)

# ─────────────────────────────────────────────
# MILESTONE 3 (AUTO RUN SCRIPT)
# ─────────────────────────────────────────────
elif page == "Milestone 3":

    st.header("Milestone 3 Charts")

    import matplotlib.pyplot as plt

    globals()['data'] = data

    exec(open("milestone3.py").read())

    figs = [plt.figure(n) for n in plt.get_fignums()]

    for fig in figs:
        st.pyplot(fig)
        plt.close(fig)
