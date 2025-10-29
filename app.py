# app.py
# Flood Pattern Data Mining & Forecasting - Streamlit
# Full version with Comparative Analysis tab
import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ================== LOAD LOCAL CSS ==================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")
# ====================================================

st.set_page_config(layout="wide", page_title="Flood Pattern Analysis Dashboard")

# ------------------------------
# Helpers: Cleaning & Preprocess
# ------------------------------
def clean_water_level(series):
    s = series.astype(str).str.replace(' ft.', '', regex=False)\
                         .str.replace(' ft', '', regex=False)\
                         .str.replace('ft', '', regex=False)\
                         .str.replace(' ', '', regex=False)\
                         .replace('nan', pd.NA)
    s = pd.to_numeric(s, errors='coerce')
    return s

def clean_damage_col(col):
    s = col.astype(str).str.replace(',', '', regex=False)
    s = s.str.replace(r'(\d)\.(\d)\.(\d)', lambda m: m.group(1)+m.group(2)+m.group(3), regex=True)
    s = pd.to_numeric(s, errors='coerce')
    return s

def _find_col(df, candidate_lower):
    for c in df.columns:
        if c.strip().lower() == candidate_lower:
            return c
    return None

def load_and_basic_clean(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    col_map = {
        'year': _find_col(df, 'year'),
        'month': _find_col(df, 'month'),
        'month_num': _find_col(df, 'month_num'),
        'day': _find_col(df, 'day'),
        'water_level': _find_col(df, 'water level'),
        'families': _find_col(df, 'no. of families affected'),
        'damage_infra': _find_col(df, 'damage infrastructure'),
        'damage_agri': _find_col(df, 'damage agriculture'),
        'municipality': _find_col(df, 'municipality'),
        'barangay': _find_col(df, 'barangay')
    }

    if col_map['year'] is not None: df['Year'] = df[col_map['year']]
    if col_map['month'] is not None: df['Month'] = df[col_map['month']].astype(str).str.strip()
    if col_map['month_num'] is not None: df['Month_Num'] = df[col_map['month_num']]
    if col_map['day'] is not None: df['Day'] = df[col_map['day']]
    if col_map['water_level'] is not None: df['Water Level'] = df[col_map['water_level']]
    if col_map['families'] is not None: df['No. of Families affected'] = df[col_map['families']]
    if col_map['damage_infra'] is not None: df['Damage Infrastructure'] = df[col_map['damage_infra']]
    if col_map['damage_agri'] is not None: df['Damage Agriculture'] = df[col_map['damage_agri']]
    if col_map['municipality'] is not None: df['Municipality'] = df[col_map['municipality']]
    if col_map['barangay'] is not None: df['Barangay'] = df[col_map['barangay']]

    if 'Month' in df.columns:
        df['Month'] = df['Month'].astype(str).str.strip().str.upper().replace({'NAN': pd.NA})
    if 'Month_Num' not in df.columns and 'Month' in df.columns:
        month_map = {'JANUARY':1,'FEBRUARY':2,'MARCH':3,'APRIL':4,'MAY':5,'JUNE':6,
                     'JULY':7,'AUGUST':8,'SEPTEMBER':9,'OCTOBER':10,'NOVEMBER':11,'DECEMBER':12}
        df['Month_Num'] = df['Month'].map(month_map)

    if 'Water Level' in df.columns:
        df['Water Level'] = clean_water_level(df['Water Level'])
        if df['Water Level'].notna().sum() > 0:
            df['Water Level'] = df['Water Level'].fillna(df['Water Level'].median())

    if 'No. of Families affected' in df.columns:
        df['No. of Families affected'] = pd.to_numeric(df['No. of Families affected'].astype(str).str.replace(',', ''), errors='coerce')
        if df['No. of Families affected'].notna().sum() > 0:
            df['No. of Families affected'] = df['No. of Families affected'].fillna(df['No. of Families affected'].median())

    for col in ['Damage Infrastructure', 'Damage Agriculture']:
        if col in df.columns:
            df[col] = clean_damage_col(df[col])
            df[col] = df[col].fillna(0)

    for c in ['Year','Month_Num','Day']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].ffill().bfill()

    return df

def create_datetime_index(df):
    tmp = df.copy()
    if 'Month' in tmp.columns and 'Month_Num' not in tmp.columns:
        month_map = {'JANUARY':1,'FEBRUARY':2,'MARCH':3,'APRIL':4,'MAY':5,'JUNE':6,
                     'JULY':7,'AUGUST':8,'SEPTEMBER':9,'OCTOBER':10,'NOVEMBER':11,'DECEMBER':12}
        tmp['Month_Num'] = tmp['Month'].astype(str).str.strip().str.upper().map(month_map)
    if not ({'Year', 'Month_Num', 'Day'}.issubset(tmp.columns)):
        return df
    tmp['Year'] = pd.to_numeric(tmp['Year'], errors='coerce')
    tmp['Month_Num'] = pd.to_numeric(tmp['Month_Num'], errors='coerce')
    tmp['Day'] = pd.to_numeric(tmp['Day'], errors='coerce')
    tmp = tmp.dropna(subset=['Year', 'Month_Num', 'Day']).copy()
    tmp['Year'] = tmp['Year'].astype(int)
    tmp['Month_Num'] = tmp['Month_Num'].astype(int)
    tmp['Day'] = tmp['Day'].astype(int)
    tmp['Date'] = pd.to_datetime({'year': tmp['Year'], 'month': tmp['Month_Num'], 'day': tmp['Day']}, errors='coerce')
    tmp = tmp.dropna(subset=['Date']).copy()
    tmp = tmp.set_index('Date').sort_index()
    return tmp

def categorize_severity(w):
    if pd.isna(w): return 'Unknown'
    try: w = float(w)
    except: return 'Unknown'
    if w <= 5: return 'Low'
    elif 5 < w <= 15: return 'Medium'
    else: return 'High'

# ------------------------------
# UI Layout
# ------------------------------
st.title("🌊 Flood Pattern Data Mining & Forecasting 🌊")
st.markdown("Upload your CSV and explore analyses: Cleaning, EDA, KMeans, RandomForest, SARIMA, Comparative Analysis.")

st.sidebar.header("Upload & Options")
uploaded_file = st.sidebar.file_uploader("Upload flood CSV", type=['csv','txt','xlsx'])
use_example = st.sidebar.checkbox("Use example dataset (if no upload)", value=False)
plotly_mode = st.sidebar.selectbox("Plot style", ["plotly (interactive)"], index=0)
show_explanations = st.sidebar.checkbox("Show explanations below outputs", value=True)

# Tabs
tabs = st.tabs([
    "Data Upload", "Data Cleaning & EDA", "Clustering (KMeans)",
    "Flood Prediction (RF)", "Flood Severity",
    "Time Series (SARIMA)", "Comparative Analysis", "Tutorial"
])

# ------------------------------
# Data Upload
# ------------------------------
with tabs[0]:
    st.markdown("## 📂 Data Upload & Overview")
    if uploaded_file is None and not use_example:
        st.info("Upload a CSV/Excel file or use example dataset.")
    else:
        if uploaded_file:
            file_name = uploaded_file.name
            if file_name.endswith('.xlsx'):
                df_raw = pd.read_excel(uploaded_file)
            else:
                df_raw = pd.read_csv(uploaded_file)
            st.success(f"Loaded **{file_name}** — {df_raw.shape[0]} rows, {df_raw.shape[1]} cols
            st.success(f"Loaded **{file_name}** — {df_raw.shape[0]} rows, {df_raw.shape[1]} cols")
        else:
            # Example dataset
            st.warning("Using example dataset")
            df_raw = pd.DataFrame({
                "Year":[2020,2020,2020,2021,2021],
                "Month":["JAN","FEB","MAR","JAN","FEB"],
                "Day":[1,1,1,1,1],
                "Water Level":[2.5, 7.8, 12.0, 3.2, 16.0],
                "No. of Families affected":[5, 10, 20, 8, 25],
                "Damage Infrastructure":[1000,5000,10000,2000,12000],
                "Damage Agriculture":[500,2000,5000,1000,7000],
                "Municipality":["Bunawan"]*5,
                "Barangay":["Brgy1","Brgy2","Brgy3","Brgy1","Brgy2"]
            })
            st.dataframe(df_raw)

# ------------------------------
# Data Cleaning & EDA
# ------------------------------
with tabs[1]:
    st.markdown("## 🧹 Data Cleaning & EDA")
    if uploaded_file or use_example:
        df = load_and_basic_clean(df_raw)
        st.markdown("### Raw Data Preview")
        st.dataframe(df.head(10))

        st.markdown("### Summary Statistics")
        st.dataframe(df.describe())

        # Water Level histogram
        if 'Water Level' in df.columns:
            fig = px.histogram(df, x='Water Level', nbins=20, title="Water Level Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # Municipality/Barangay counts
        if 'Municipality' in df.columns:
            fig2 = px.bar(df['Municipality'].value_counts().reset_index(), x='index', y='Municipality',
                          labels={'index':'Municipality','Municipality':'Count'}, title="Flood Records per Municipality")
            st.plotly_chart(fig2, use_container_width=True)

        if 'Barangay' in df.columns:
            fig3 = px.bar(df['Barangay'].value_counts().reset_index(), x='index', y='Barangay',
                          labels={'index':'Barangay','Barangay':'Count'}, title="Flood Records per Barangay")
            st.plotly_chart(fig3, use_container_width=True)

# ------------------------------
# Clustering (KMeans)
# ------------------------------
with tabs[2]:
    st.markdown("## 🔹 Clustering (KMeans)")
    if uploaded_file or use_example:
        cluster_cols = ['Water Level','No. of Families affected','Damage Infrastructure','Damage Agriculture']
        df_cluster = df[cluster_cols].fillna(df[cluster_cols].median())
        k = st.slider("Select number of clusters", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(df_cluster)
        df['Cluster'] = kmeans.labels_

        st.dataframe(df[['Water Level','No. of Families affected','Damage Infrastructure','Damage Agriculture','Cluster']].head(10))

        fig4 = px.scatter_3d(df, x='Water Level', y='No. of Families affected', z='Damage Infrastructure',
                             color='Cluster', title="3D Cluster Visualization")
        st.plotly_chart(fig4, use_container_width=True)

# ------------------------------
# Flood Prediction (Random Forest)
# ------------------------------
with tabs[3]:
    st.markdown("## 🌊 Flood Prediction (Random Forest)")
    if uploaded_file or use_example:
        features = ['Water Level','No. of Families affected','Damage Infrastructure','Damage Agriculture']
        df_rf = df.dropna(subset=features+['Cluster'])
        X = df_rf[features]
        y = df_rf['Cluster']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        st.markdown(f"### Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
        st.text(classification_report(y_test, y_pred))

# ------------------------------
# Flood Severity Categorization
# ------------------------------
with tabs[4]:
    st.markdown("## ⚠️ Flood Severity Levels")
    if uploaded_file or use_example:
        df['Severity'] = df['Water Level'].apply(categorize_severity)
        severity_counts = df['Severity'].value_counts()
        fig5 = px.bar(severity_counts, x=severity_counts.index, y=severity_counts.values,
                      labels={'x':'Severity','y':'Count'}, title="Flood Severity Distribution")
        st.plotly_chart(fig5, use_container_width=True)

# ------------------------------
# Time Series (SARIMA)
# ------------------------------
with tabs[5]:
    st.markdown("## 📈 Time Series Forecasting (SARIMA)")
    if uploaded_file or use_example:
        df_ts = create_datetime_index(df)
        if 'Water Level' in df_ts.columns:
            ts_data = df_ts['Water Level'].resample('M').mean()
            fig6 = px.line(ts_data, title="Monthly Avg Water Level Time Series")
            st.plotly_chart(fig6, use_container_width=True)

            # SARIMA Forecast
            p = st.number_input("AR order (p)", min_value=0, max_value=5, value=1)
            d = st.number_input("Difference order (d)", min_value=0, max_value=2, value=1)
            q = st.number_input("MA order (q)", min_value=0, max_value=5, value=1)
            P = st.number_input("Seasonal AR order (P)", min_value=0, max_value=2, value=1)
            D = st.number_input("Seasonal diff order (D)", min_value=0, max_value=1, value=1)
            Q = st.number_input("Seasonal MA order (Q)", min_value=0, max_value=2, value=1)
            m = st.number_input("Seasonal period (m)", min_value=1, max_value=12, value=12)

            if st.button("Run SARIMA Forecast"):
                model = SARIMAX(ts_data, order=(p,d,q), seasonal_order=(P,D,Q,m))
                results = model.fit(disp=False)
                forecast = results.get_forecast(steps=12)
                pred_ci = forecast.conf_int()
                forecast_index = pd.date_range(ts_data.index[-1]+pd.offsets.MonthBegin(), periods=12, freq='MS')

                fig7 = go.Figure()
                fig7.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines', name='Observed'))
                fig7.add_trace(go.Scatter(x=forecast_index, y=forecast.predicted_mean.values, mode='lines', name='Forecast'))
                fig7.add_trace(go.Scatter(x=forecast_index, y=pred_ci.iloc[:,0], mode='lines', line=dict(dash='dash'), name='Lower CI'))
                fig7.add_trace(go.Scatter(x=forecast_index, y=pred_ci.iloc[:,1], mode='lines', line=dict(dash='dash'), name='Upper CI'))
                st.plotly_chart(fig7, use_container_width=True)

# ------------------------------
# Comparative Analysis
# ------------------------------
with tabs[6]:
    st.markdown("## 📊 Comparative Analysis of ML Models")
    if uploaded_file or use_example:
        comp_results = pd.DataFrame({
            "Model":["Random Forest","KMeans Clustering","SARIMA Forecast"],
            "Accuracy/Score":[f"{accuracy_score(y_test, y_pred)*100:.2f}%", "N/A", "N/A"],
            "Notes":["Supervised, predicts cluster/severity","Unsupervised clustering","Time Series Forecast"]
        })
        st.table(comp_results)

# ------------------------------
# Tutorial Tab
# ------------------------------
with tabs[7]:
    st.markdown("## 📖 Tutorial & Guide")
    st.markdown("""
1. **Upload your dataset**: CSV or Excel with columns like `Year`, `Month`, `Day`, `Water Level`, etc.
2. **Check Cleaning & EDA**: View basic statistics, histograms, and distributions.
3. **Run KMeans Clustering**: Identify patterns and groupings in flood data.
4. **Random Forest Prediction**: Train and test ML models on your features.
5. **Categorize Severity**: See low, medium, high flood levels automatically.
6. **Time Series Forecast (SARIMA)**: Predict future monthly water levels.
7. **Comparative Analysis**: Compare Random Forest, KMeans, and SARIMA results in one table.
8. **Explore & Export**: Interact with charts and download processed data if needed.
""")
