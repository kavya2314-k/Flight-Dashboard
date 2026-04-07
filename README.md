#  Airfly Insights – Airline Delay & Performance Dashboard

##  Project Overview

**Airfly Insights** is an interactive data analytics dashboard built using **Streamlit** to analyze airline performance and delay patterns. The project focuses on extracting meaningful insights from flight data to support better decision-making and operational understanding.

This application enables users to explore delay distributions, airline efficiency, route-level trends, and time-based performance metrics through an intuitive interface.

---

## Objectives

* Analyze airline delays across multiple categories
* Identify patterns in flight performance over time
* Compare airline efficiency across routes and months
* Build a scalable and efficient data pipeline for dashboard deployment

---

##  Tech Stack

* **Python**
* **Pandas & NumPy** – Data preprocessing and analysis
* **PyArrow** – Efficient data storage using Parquet
* **Streamlit** – Interactive dashboard development
* **Hugging Face Spaces** – Deployment platform

---


##  Data Preprocessing

The dataset underwent multiple optimization steps to ensure efficient deployment:

* Removed irrelevant and redundant columns
* Downcasted numerical data types (int64 → int32/int16)
* Converted categorical features for memory efficiency
* Handled missing values and cleaned inconsistencies
* Stored data in **Parquet format** for fast I/O

---

##  Features of the Dashboard

*  Airline-wise delay analysis
*  Route-based performance insights
*  Monthly trend visualization
*  Delay category breakdown:

  * Air System Delay
  * Security Delay
  * Airline Delay
  * Late Aircraft Delay
  * Weather Delay
*  Dynamic filtering and interactive exploration

---

##  Key Insights

* Certain airlines consistently outperform others in delay management
* Delay patterns vary significantly across months and routes
* Weather and late aircraft delays are major contributing factors

---

##  Performance Optimization

To ensure smooth deployment and avoid memory issues:

* Used **Parquet compression (ZSTD)**
* Applied **data type optimization**
* Implemented **Streamlit caching (`@st.cache_data`)**
* Reduced dataset size while preserving analytical value

---

##  Deployment

The application is deployed using **Hugging Face Spaces**, enabling easy access and scalability.

---


##  Author

**Kavya Sree K**

---


