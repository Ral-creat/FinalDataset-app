# app.py
# Flood Pattern Data Mining & Forecasting - Streamlit Port
# Interactive Plotly charts + automatic median filling + Excel export
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
import io

st.set_page_config(page_title="Flood Pattern Mining & Forecasting", layout="wide")
st.title("🌊 Flood Pattern Data Mining & Forecasting App")

show_explanations = st.sidebar.checkbox("Show explanations", True)

# ------------------------------
# Helper Functions
# ------------------------------
def clean_numeric(series):
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
# Tabs
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

        for col in ['Water Level', 'No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']:
            if col in df.columns:
                df[col] = clean_numeric(df[col])

        df_filled = fill_median(df)
        st.session_state['df_processed'] = df_filled

        st.subheader("📘 Raw Dataset (Before Median Filling)")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("📗 Processed Dataset (After Median Filling)")
        st.dataframe(df_filled.head(20), use_container_width=True)

        # ✅ Download as Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_filled.to_excel(writer, index=False, sheet_name='Processed_Data')
        processed_excel = output.getvalue()

        st.download_button(
            label="📥 Download Processed Dataset as Excel",
            data=processed_excel,
            file_name="processed_flood_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("### 🧾 Quick Summary (Numeric Columns)")
        st.dataframe(df_filled.describe().round(2))

        if show_explanations:
            st.markdown("""
            **Explanation:**  
            - Missing and zero values are automatically replaced by column medians.  
            - The processed dataset is now ready for modeling and can be downloaded as Excel.
            """)
 # ------------------------------
        # Monthly flood probability (fixed)
        # ------------------------------
        if 'Month' in df.columns:
            # create flood_occurred column if not exists
            if 'flood_occurred' not in df.columns:
                df['flood_occurred'] = (df['Water Level'].fillna(0) > 0).astype(int)

            st.subheader("Monthly Flood Probability")

            # month mapping
            month_map = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }

            # clean and convert month formats
            def clean_month(val):
                try:
                    val_str = str(val).strip().lower()
                    # numeric (1–12 or '01')
                    if val_str.isdigit():
                        num = int(val_str)
                        return month_map.get(num, np.nan)
                    # short text (jan, feb, mar…)
                    for num, name in month_map.items():
                        if val_str.startswith(name[:3].lower()):
                            return name
                    return np.nan
                except:
                    return np.nan

            df['Month_clean'] = df['Month'].apply(clean_month)
            df = df.dropna(subset=['Month_clean'])

            # compute monthly stats
            m_stats = df.groupby('Month_clean')['flood_occurred'].agg(['sum', 'count']).reset_index()
            m_stats['probability'] = (m_stats['sum'] / m_stats['count']).round(3)

            # keep months in correct order
            m_stats['Month_clean'] = pd.Categorical(
                m_stats['Month_clean'],
                categories=list(month_map.values()),
                ordered=True
            )
            m_stats = m_stats.sort_values('Month_clean')

            # bar chart
            fig = px.bar(
                m_stats,
                x='Month_clean',
                y='probability',
                title="Flood Probability by Month",
                text='probability'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_title="Month", yaxis_title="Flood Probability")

            st.plotly_chart(fig, use_container_width=True)

            if show_explanations:
                st.markdown("""
                **Explanation:**  
                This chart shows the chance of flooding per month.  
                - **Probability = Flood occurrences / Total records in that month**  
                Months with higher bars indicate higher flood risk periods.  
                """)
        # ------------------------------
        # Municipal flood probabilities
        # ------------------------------
        if 'Municipality' in df.columns:
            st.subheader("Flood probability by Municipality")
            mun = df.groupby('Municipality')['flood_occurred'].agg(['sum','count']).reset_index()
            mun['probability'] = (mun['sum'] / mun['count']).round(3)
            mun = mun.sort_values('probability', ascending=False)
            fig = px.bar(
                mun,
                x='Municipality',
                y='probability',
                title="Flood Probability by Municipality",
                text='probability'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_title="Municipality", yaxis_title="Flood Probability")
            st.plotly_chart(fig, use_container_width=True)
            if show_explanations:
                st.markdown("""
                **Explanation:**  
                This helps identify which municipalities historically experience more flooding,
                guiding local preparedness and response planning.
                """)
         # ------------------------------
        # Municipal flood probabilities
        # ------------------------------
        if 'Barangay' in df.columns:
            st.subheader("Flood probability by Barangay")
            mun = df.groupby('Barangay')['flood_occurred'].agg(['sum','count']).reset_index()
            mun['probability'] = (mun['sum'] / mun['count']).round(3)
            mun = mun.sort_values('probability', ascending=False)
            fig = px.bar(
                mun,
                x='Barangay',
                y='probability',
                title="Flood Probability by Barangay",
                text='probability'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_title="Barangay", yaxis_title="Flood Probability")
            st.plotly_chart(fig, use_container_width=True)
            if show_explanations:
                st.markdown("""
                **Explanation:**  
                This helps identify which Barangay historically experience more flooding,
                guiding local preparedness and response planning.
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


