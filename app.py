# app.py
# Flood Pattern Data Mining & Forecasting - Streamlit Port of floodpatternv2.ipynb
# Interactive Plotly charts + automatic explanations below each output
# Author: ChatGPT
# Usage: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# ------------------------------ Helpers ------------------------------
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

    # ---------- MEDIAN FIXED COLUMNS ----------
    if 'Water Level' in df.columns:
        df['Water Level'] = clean_water_level(df['Water Level'])
        df['Water Level'] = df['Water Level'].replace(0, pd.NA)
        if df['Water Level'].notna().sum() > 0:
            median_wl = df['Water Level'].median()
            df['Water Level'] = df['Water Level'].fillna(median_wl)

    if 'No. of Families affected' in df.columns:
        df['No. of Families affected'] = pd.to_numeric(df['No. of Families affected'].astype(str).str.replace(',', ''), errors='coerce')
        df['No. of Families affected'] = df['No. of Families affected'].replace(0, pd.NA)
        if df['No. of Families affected'].notna().sum() > 0:
            median_fam = df['No. of Families affected'].median()
            df['No. of Families affected'] = df['No. of Families affected'].fillna(median_fam)

    for col in ['Damage Infrastructure', 'Damage Agriculture']:
        if col in df.columns:
            df[col] = clean_damage_col(df[col])
            df[col] = df[col].replace(0, pd.NA)
            if df[col].notna().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

    for c in ['Year','Month_Num','Day']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            df[c] = df[c].ffill().bfill()

    return df

def create_datetime_index(df):
    tmp = df.copy()
    if 'Month_Num' not in tmp.columns and 'Month' in tmp.columns:
        month_map = {'JANUARY':1,'FEBRUARY':2,'MARCH':3,'APRIL':4,'MAY':5,'JUNE':6,
                     'JULY':7,'AUGUST':8,'SEPTEMBER':9,'OCTOBER':10,'NOVEMBER':11,'DECEMBER':12}
        tmp['Month_Num'] = tmp['Month'].astype(str).str.strip().str.upper().map(month_map)
    if not ({'Year','Month_Num','Day'}.issubset(tmp.columns)):
        return df
    tmp['Year'] = pd.to_numeric(tmp['Year'], errors='coerce')
    tmp['Month_Num'] = pd.to_numeric(tmp['Month_Num'], errors='coerce')
    tmp['Day'] = pd.to_numeric(tmp['Day'], errors='coerce')
    tmp = tmp.dropna(subset=['Year','Month_Num','Day'])
    tmp['Year'] = tmp['Year'].astype(int)
    tmp['Month_Num'] = tmp['Month_Num'].astype(int)
    tmp['Day'] = tmp['Day'].astype(int)
    tmp['Date'] = pd.to_datetime({'year': tmp['Year'], 'month': tmp['Month_Num'], 'day': tmp['Day']}, errors='coerce')
    tmp = tmp.dropna(subset=['Date'])
    tmp = tmp.set_index('Date').sort_index()
    return tmp

def categorize_severity(w):
    if pd.isna(w): return 'Unknown'
    try: w = float(w)
    except: return 'Unknown'
    if w <= 5: return 'Low'
    elif 5 < w <= 15: return 'Medium'
    else: return 'High'

# ------------------------------ Streamlit UI ------------------------------
st.set_page_config(layout="wide", page_title="Flood Pattern Dashboard")
st.title("🌊 Flood Pattern Data Mining & Forecasting 🌊")
st.sidebar.header("Upload CSV / Excel")
uploaded_file = st.sidebar.file_uploader("Upload your flood dataset", type=['csv','xlsx'])
use_example = st.sidebar.checkbox("Use example dataset", value=False)

# ------------------------------ Load Data ------------------------------
if uploaded_file is None and not use_example:
    st.info("Upload a CSV/Excel or toggle 'Use example dataset'")
    st.stop()
else:
    if uploaded_file:
        file_name = uploaded_file.name
        df_raw = pd.read_excel(uploaded_file) if file_name.endswith('.xlsx') else pd.read_csv(uploaded_file)
    else:
        # Example dummy dataset
        df_raw = pd.DataFrame({
            'Year':[2023]*6,
            'Month':['January','February','March','April','May','June'],
            'Day':[1,5,10,15,20,25],
            'Water Level':[0, 3, 0, 7, np.nan, 12],
            'No. of Families affected':[0,2,0,5,3,0],
            'Damage Infrastructure':[0, 5000, 0, 15000, np.nan, 20000],
            'Damage Agriculture':[0,10000,0,20000,np.nan,30000]
        })

# ------------------------------ Cleaning ------------------------------
df_clean = load_and_basic_clean(df_raw)
df_time = create_datetime_index(df_clean)
st.subheader("Raw Data + Cleaned Data")
st.dataframe(df_clean)

# ------------------------------ Severity Column ------------------------------
if 'Water Level' in df_clean.columns:
    df_clean['Severity'] = df_clean['Water Level'].apply(categorize_severity)

# ------------------------------ Clustering Example ------------------------------
st.subheader("KMeans Clustering (Water Level vs Families affected)")
X = df_clean[['Water Level','No. of Families affected']].values
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
df_clean['Cluster'] = kmeans.labels_
fig_cluster = px.scatter(df_clean, x='Water Level', y='No. of Families affected', color='Cluster', size='Water Level', hover_data=['Severity'])
st.plotly_chart(fig_cluster)

# ------------------------------ Random Forest Example ------------------------------
st.subheader("Random Forest Prediction (Severity)")
X_rf = df_clean[['Water Level','No. of Families affected','Damage Infrastructure','Damage Agriculture']]
y_rf = df_clean['Severity']
X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Classification Report")
st.text(classification_report(y_test, y_pred))

# ------------------------------ SARIMA Forecast Example ------------------------------
st.subheader("SARIMA Forecast (Water Level)")
if not df_time.empty:
    ts = df_time['Water Level'].resample('D').mean().fillna(method='ffill')
    model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=10)
    fc_values = forecast.predicted_mean
    fc_index = forecast.row_labels
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Observed'))
    fig_forecast.add_trace(go.Scatter(x=fc_index, y=fc_values, mode='lines', name='Forecast'))
    st.plotly_chart(fig_forecast)
