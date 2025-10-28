# app.py
# Flood Pattern Data Mining & Forecasting - Streamlit Port
# Interactive Plotly charts + automatic median filling
# Author: ChatGPT (Streamlit upgrade for Lara Mae Vidal)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flood Pattern Mining & Forecasting", layout="wide")
st.title("🌊 Flood Pattern Data Mining & Forecasting App")

show_explanations = st.sidebar.checkbox("Show explanations", True)

# ------------------------------
# Helper Functions
# ------------------------------
def clean_numeric(series):
    """Convert messy numeric columns to clean floats."""
    return (
        series.astype(str)
        .str.replace(",", "")
        .str.replace("ft", "")
        .str.replace("m", "")
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )

def categorize_severity(water_level):
    if water_level < 1:
        return "Low"
    elif 1 <= water_level < 3:
        return "Medium"
    else:
        return "High"

def create_datetime_index(df):
    try:
        df['Date'] = pd.to_datetime(df[['Year','Month','Day']])
        df = df.set_index('Date')
        return df
    except Exception:
        return df

def fill_median(df):
    """Fill NaN and zero values in numeric columns with median."""
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
    exclude = ['Year', 'Month', 'Day']
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    for col in numeric_cols:
        median_val = df_filled.loc[df_filled[col] != 0, col].median()
        df_filled[col] = df_filled[col].replace(0, np.nan)
        df_filled[col] = df_filled[col].fillna(median_val)
    return df_filled

# ------------------------------
# Sidebar Tabs
# ------------------------------
tabs = st.tabs([
    "Data Upload",
    "Cleaning & EDA",
    "Clustering (KMeans)",
    "Flood Prediction (RandomForest)",
    "Flood Severity Classification",
    "Time Series (SARIMA)",
    "Tutorial"
])

# ------------------------------
# Data Upload
# ------------------------------
with tabs[0]:
    st.header("📂 Upload your dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        st.session_state['df_raw'] = df_raw
        st.success("✅ File uploaded successfully!")

# ------------------------------
# Cleaning & EDA
# ------------------------------
with tabs[1]:
    st.header("🧹 Data Cleaning & EDA")
    df_raw = st.session_state.get('df_raw', None)
    if df_raw is None:
        st.warning("Please upload a dataset first.")
    else:
        df = df_raw.copy()

        # Clean numeric columns
        for col in ['Water Level', 'No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']:
            if col in df.columns:
                df[col] = clean_numeric(df[col])

        # Fill median automatically
        df_filled = fill_median(df)

        # Store the processed dataset for ML models
        st.session_state['df_processed'] = df_filled

        # Show Raw Data
        st.subheader("📘 Raw Dataset (Before Median Filling)")
        st.dataframe(df.head(20), use_container_width=True)

        # Show Median-Filled Data
        st.subheader("📗 Processed Dataset (After Median Filling)")
        st.dataframe(df_filled.head(20), use_container_width=True)

        # Quick summary
        st.markdown("### 🧾 Quick Summary (Numeric Columns)")
        st.dataframe(df_filled.describe().round(2))

        if show_explanations:
            st.markdown("""
            **Explanation:**  
            - All numeric columns automatically cleaned and filled using median values.  
            - Zero and missing values are replaced with the column's median.  
            - This ensures models don’t fail due to missing or invalid data.
            """)

# ------------------------------
# KMeans Clustering
# ------------------------------
with tabs[2]:
    st.header("Clustering (KMeans)")
    df = st.session_state.get('df_processed', None)
    if df is None:
        st.warning("Do data cleaning first.")
    else:
        features = ['Water Level','No. of Families affected','Damage Infrastructure','Damage Agriculture']
        if not set(features).issubset(df.columns):
            st.error("Missing required columns for clustering.")
        else:
            k = st.slider("Number of clusters (k)", 2, 6, 3)
            X_cluster = df[features]
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cluster)
            df['Cluster'] = kmeans.labels_
            counts = df['Cluster'].value_counts().sort_index()
            st.write("Cluster counts:")
            st.write(counts)

            fig = px.scatter_3d(df, x='Water Level', y='No. of Families affected', z='Damage Infrastructure',
                                color='Cluster', hover_data=['Barangay','Municipality','Flood Cause'],
                                title="KMeans clusters (3D)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Cluster summary (numeric medians)")
            cluster_summary = df.groupby('Cluster')[features].median().round(2)
            st.dataframe(cluster_summary)

# ------------------------------
# Flood Prediction (RandomForest)
# ------------------------------
with tabs[3]:
    st.header("Flood Occurrence Prediction — RandomForest")
    df = st.session_state.get('df_processed', None)
    if df is None:
        st.warning("Do data cleaning first.")
    else:
        df['flood_occurred'] = (df['Water Level'] > 0).astype(int)

        month_dummies = pd.get_dummies(df['Month'].astype(str).fillna('Unknown'), prefix='Month')
        X_basic = pd.concat([df[['Water Level', 'No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']], month_dummies], axis=1)
        y = df['flood_occurred']

        Xtr, Xte, ytr, yte = train_test_split(X_basic, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        acc = accuracy_score(yte, ypred)

        st.subheader("📊 RandomForest Results")
        acc_table = pd.DataFrame({"Metric": ["Accuracy (test)"], "Value": [f"{acc:.4f}"]})
        st.table(acc_table)

        report = classification_report(yte, ypred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.markdown("### 📈 Classification Report")
        st.table(report_df)

        fi = pd.Series(model.feature_importances_, index=X_basic.columns).sort_values(ascending=False).head(10)
        st.subheader("Top feature importances")
        st.bar_chart(fi)

# ------------------------------
# Flood Severity Classification
# ------------------------------
with tabs[4]:
    st.header("🌊 Flood Severity Classification")
    df = st.session_state.get('df_processed', None)
    if df is None:
        st.warning("Please do data cleaning first.")
    else:
        df['Flood_Severity'] = df['Water Level'].apply(categorize_severity)
        st.subheader("📊 Severity Distribution")
        sev_counts = df['Flood_Severity'].value_counts().reset_index()
        sev_counts.columns = ['Severity Level', 'Count']
        st.table(sev_counts)

        base_feats = ['No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']
        month_d = pd.get_dummies(df['Month'].astype(str).fillna('Unknown'), prefix='Month')
        Xsev = pd.concat([df[base_feats], month_d], axis=1)
        ysev = df['Flood_Severity']

        Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(Xsev, ysev, test_size=0.3, random_state=42, stratify=ysev)
        model_sev = RandomForestClassifier(random_state=42)
        model_sev.fit(Xtr_s, ytr_s)
        ypred_s = model_sev.predict(Xte_s)
        acc_s = accuracy_score(yte_s, ypred_s)

        st.subheader("✅ Severity Model Results")
        st.table(pd.DataFrame({'Metric': ['Accuracy (test)'], 'Value': [f"{acc_s:.4f}"]}))
        report = classification_report(yte_s, ypred_s, output_dict=True, zero_division=0)
        st.markdown("### 📈 Classification Report (Low / Medium / High)")
        st.table(pd.DataFrame(report).transpose().round(3))

# ------------------------------
# Time Series (SARIMA)
# ------------------------------
with tabs[5]:
    st.header("Time Series Forecasting (SARIMA)")
    df = st.session_state.get('df_processed', None)
    if df is None:
        st.warning("Do data cleaning first.")
    else:
        df_temp = create_datetime_index(df)
        if not isinstance(df_temp.index, pd.DatetimeIndex):
            st.error("Add Year/Month/Day columns for time index.")
        else:
            ts = df_temp['Water Level'].resample('D').mean()
            ts_filled = ts.fillna(method='ffill').fillna(method='bfill')
            st.subheader("Time series preview (daily avg)")
            st.plotly_chart(px.line(ts_filled, title="Daily Average Water Level"), use_container_width=True)

            try:
                adf_result = adfuller(ts_filled.dropna())
                st.write(f"ADF Statistic: {adf_result[0]:.4f}")
                st.write(f"P-value: {adf_result[1]:.4f}")
            except Exception as e:
                st.error(f"ADF test failed: {e}")

            d = 1 if adf_result[1] > 0.05 else 0
            with st.spinner("Fitting SARIMA..."):
                try:
                    model_sarima = SARIMAX(ts_filled, order=(1, d, 1), seasonal_order=(1, 0, 1, 7), enforce_stationarity=False, enforce_invertibility=False)
                    results = model_sarima.fit(disp=False)
                    st.dataframe(results.summary().tables[1].as_html(), use_container_width=True)
                except Exception as e:
                    st.error(f"SARIMA fit failed: {e}")
                    results = None

            if results is not None:
                steps = st.slider("Forecast horizon (days)", 7, 365, 30)
                pred = results.get_forecast(steps=steps)
                pred_mean = pred.predicted_mean
                pred_ci = pred.conf_int()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts_filled.index, y=ts_filled, name='Observed'))
                fig.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean, name='Forecast'))
                fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:,0], fill=None, mode='lines'))
                fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:,1], fill='tonexty', name='95% CI', mode='lines'))
                fig.update_layout(title="SARIMA Forecast", xaxis_title="Date", yaxis_title="Water Level")
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Tutorial Tab
# ------------------------------
with tabs[6]:
    st.header("Tutorial & Walkthrough")
    st.markdown("""
    ### Quick Guide:
    1. **Upload Data** → CSV file.
    2. **Cleaning & EDA** → Automatic median filling for missing/zero numeric values.
    3. **KMeans / RandomForest / SARIMA** → All work off the filled dataset.
    """)
