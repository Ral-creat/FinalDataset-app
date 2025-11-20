# app.py
# Flood Pattern Data Mining & Forecasting - Streamlit Port of floodpatternv2.ipynb
# Interactive Plotly charts + automatic explanations below each output
# Author: ChatGPT (converted for Streamlit) - enhanced uniform distribution & balancing options
# Usage: streamlit run app.py

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
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# ================== LOAD LOCAL CSS =================
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        # ignore if CSS not present
        pass

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

    if col_map['year'] is not None:
        df['Year'] = df[col_map['year']]
    if col_map['month'] is not None:
        df['Month'] = df[col_map['month']].astype(str).str.strip()
    if col_map['month_num'] is not None:
        df['Month_Num'] = df[col_map['month_num']]
    if col_map['day'] is not None:
        df['Day'] = df[col_map['day']]
    if col_map['water_level'] is not None:
        df['Water Level'] = df[col_map['water_level']]
    if col_map['families'] is not None:
        df['No. of Families affected'] = df[col_map['families']]
    if col_map['damage_infra'] is not None:
        df['Damage Infrastructure'] = df[col_map['damage_infra']]
    if col_map['damage_agri'] is not None:
        df['Damage Agriculture'] = df[col_map['damage_agri']]
    if col_map['municipality'] is not None:
        df['Municipality'] = df[col_map['municipality']]
    if col_map['barangay'] is not None:
        df['Barangay'] = df[col_map['barangay']]

    if 'Month' in df.columns:
        df['Month'] = df['Month'].astype(str).str.strip().str.upper().replace({'NAN': pd.NA})

    if 'Month_Num' not in df.columns and 'Month' in df.columns:
        month_map = {'JANUARY':1,'FEBRUARY':2,'MARCH':3,'APRIL':4,'MAY':5,'JUNE':6,
                     'JULY':7,'AUGUST':8,'SEPTEMBER':9,'OCTOBER':10,'NOVEMBER':11,'DECEMBER':12}
        df['Month_Num'] = df['Month'].map(month_map)

    if 'Water Level' in df.columns:
        df['Water Level'] = clean_water_level(df['Water Level'])
        if df['Water Level'].notna().sum() > 0:
            median_wl = df['Water Level'].median()
            df['Water Level'] = df['Water Level'].fillna(median_wl)

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

    before = len(tmp)
    tmp = tmp.dropna(subset=['Year', 'Month_Num', 'Day']).copy()
    dropped = before - len(tmp)
    if dropped > 0:
        st.info(f"Dropped {dropped} rows with missing Year/Month/Day parts which couldn't form valid dates.")

    if tmp.empty:
        return df

    tmp['Year'] = tmp['Year'].astype(int)
    tmp['Month_Num'] = tmp['Month_Num'].astype(int)
    tmp['Day'] = tmp['Day'].astype(int)

    tmp['Date'] = pd.to_datetime({'year': tmp['Year'], 'month': tmp['Month_Num'], 'day': tmp['Day']}, errors='coerce')

    before2 = len(tmp)
    tmp = tmp.dropna(subset=['Date']).copy()
    dropped2 = before2 - len(tmp)
    if dropped2 > 0:
        st.info(f"Dropped {dropped2} rows with invalid date combinations (e.g., Feb 30).")

    if tmp.empty:
        return df

    tmp = tmp.set_index('Date').sort_index()
    return tmp

def categorize_severity(w):
    if pd.isna(w):
        return 'Unknown'
    try:
        w = float(w)
    except:
        return 'Unknown'
    if w <= 5:
        return 'Low'
    elif 5 < w <= 15:
        return 'Medium'
    else:
        return 'High'

# ------------------------------
# Balancing helper
# ------------------------------
def balance_by_column(df, col, n_samples=None, method='upsample', random_state=42):
    """
    Balance df by column `col` categories (values or bins).
    method: 'upsample' or 'downsample'
    n_samples: if None, uses max (for upsample) or min (for downsample)
    Returns a new balanced dataframe (shuffled).
    """
    df = df.copy()
    counts = df[col].value_counts()
    if len(counts) == 0:
        return df
    if n_samples is None:
        n_samples = counts.max() if method == 'upsample' else counts.min()
    groups = []
    for val, cnt in counts.items():
        group = df[df[col] == val]
        if cnt == 0:
            continue
        if method == 'upsample' and cnt < n_samples:
            grp = resample(group, replace=True, n_samples=n_samples, random_state=random_state)
        elif method == 'downsample' and cnt > n_samples:
            grp = resample(group, replace=False, n_samples=n_samples, random_state=random_state)
        else:
            grp = group
        groups.append(grp)
    if not groups:
        return df
    balanced = pd.concat(groups).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return balanced

# ------------------------------
# UI Layout
# ------------------------------
st.title("🌊 BUNWAN AGUSAN DEL SUR FLOOD PATTERN REPORT 🌊")
st.markdown("Upload a CSV/Excel (e.g. FloodDataMDRRMO.csv) — explore cleaning, EDA, clustering, prediction & forecasting. Toggle uniform/balancing options in the sidebars.")

# Sidebar: file upload & options
st.sidebar.header("Upload & Options")
uploaded_file = st.sidebar.file_uploader("Upload flood CSV/Excel", type=['csv','txt','xlsx'])
use_example = st.sidebar.checkbox("Use example dataset (if no upload)", value=False)
plotly_mode = st.sidebar.selectbox("Plot style", ["plotly (interactive)"], index=0)
show_explanations = st.sidebar.checkbox("Show explanations below outputs", value=True)

# Additional balancing options in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Uniform & Balancing defaults**")
default_equalfreq_bins = st.sidebar.number_input("Default equal-frequency bins (for quick run)", min_value=3, max_value=20, value=6)
st.sidebar.markdown("---")

# Tabs (main)
tabs = st.tabs(["Data Upload", "Data Cleaning & EDA", "Clustering (KMeans)", "Flood Prediction (RF)", "Flood Severity", "Time Series (SARIMA)", "Model Comparison"])

# ------------------------------
# Data Upload Tab
# ------------------------------
with tabs[0]:
    st.markdown("<h2 class='main-title'>📂 Data Upload & Overview</h2>", unsafe_allow_html=True)

    if uploaded_file is None and not use_example:
        st.info("📤 Please upload a CSV/Excel to begin, or toggle **'Use example dataset'**.")
    else:
        if uploaded_file is not None:
            try:
                file_name = uploaded_file.name
                if file_name.endswith('.xlsx'):
                    df_raw = pd.read_excel(uploaded_file)
                else:
                    df_raw = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded **{file_name}** — **{df_raw.shape[0]:,} rows**, **{df_raw.shape[1]} columns**.")
            except Exception as e:
                st.error(f"❌ Failed to read file: {e}")
                st.stop()
        else:
            st.warning("⚠️ Using a **synthetic example dataset** for demo. Upload your real file for accurate results.")
            df_raw = pd.DataFrame({
                'Year': [2018, 2018, 2019, 2019, 2020, 2020],
                'Month': ['JANUARY', 'FEBRUARY', 'DECEMBER', 'FEBRUARY', 'MAY', 'NOVEMBER'],
                'Day': [10, 5, 12, 20, 1, 15],
                'Municipality': ['Bunawan'] * 6,
                'Barangay': ['Poblacion', 'Imelda', 'Poblacion', 'Mambalili', 'Bunawan Brook', 'Poblacion'],
                'Flood Cause': ['LPA', 'LPA', 'Easterlies', 'AURING', 'Shearline', 'LPA'],
                'Water Level': ['5 ft.', '8 ft', '12ft', '20ft', 'nan', '3 ft'],
                'No. of Families affected': [10, 20, 50, 200, 0, 5],
                'Damage Infrastructure': ['0', '0', '1,000', '5,000', '0', '0'],
                'Damage Agriculture': ['0', '0', '422.510.5', '10,000', '0', '0']
            })
            st.markdown("### 🧾 Example Data Preview")
            st.dataframe(df_raw.head(), use_container_width=True)

        st.markdown("### 📊 Dataset Overview")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("📅 Total Rows", f"{df_raw.shape[0]:,}")
        with info_col2:
            st.metric("📈 Total Columns", f"{df_raw.shape[1]}")

        with st.expander("🔍 View Raw Data (First 50 Rows)"):
            st.dataframe(df_raw.head(50), use_container_width=True)

        st.markdown("### 🧩 Column Names")
        col_df = pd.DataFrame({
            "Column Name": df_raw.columns,
            "Example Value": [str(df_raw[col].iloc[0]) if not df_raw[col].empty else "" for col in df_raw.columns]
        })
        st.table(col_df)

# ------------------------------
# Cleaning & EDA Tab
# ------------------------------
with tabs[1]:
    st.header("Data Cleaning & Exploratory Data Analysis (EDA)")
    if 'df_raw' not in locals():
        st.warning("Upload a dataset first in the Data Upload tab.")
    else:
        df = load_and_basic_clean(df_raw)
        # create flood_occurred if not present
        if 'flood_occurred' not in df.columns:
            df['flood_occurred'] = (df['Water Level'].fillna(0) > 0).astype(int)

        st.subheader("After basic cleaning (head):")
        st.dataframe(df.head(10))

        st.subheader("Summary statistics (numerical):")
        st.write(df.select_dtypes(include=[np.number]).describe())

 # Water Level distribution (Plotly)
        if 'Water Level' in df.columns:
            st.subheader("Water Level distribution")
            fig = px.histogram(
                df,
                x='Water Level',
                nbins=30,
                marginal="box",
                title="Distribution of Cleaned Water Level"
            )
            st.plotly_chart(fig, use_container_width=True)
            if show_explanations:
                st.markdown("""
                **Explanation:**  
                This histogram shows the distribution of `Water Level` after cleaning non-numeric characters
                and filling missing values with the median.  
                The boxplot margin highlights potential outliers.  
                Use this to detect skew and extreme flood events.
                """)

        # Monthly flood probability with equal-sample option
        if 'Month' in df.columns:
            st.subheader("Monthly Flood Probability")
            month_map = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }

            def clean_month(val):
                try:
                    val_str = str(val).strip().lower()
                    if val_str.isdigit():
                        num = int(val_str)
                        return month_map.get(num, np.nan)
                    for num, name in month_map.items():
                        if val_str.startswith(name[:3].lower()):
                            return name
                    return np.nan
                except:
                    return np.nan

            df['Month_clean'] = df['Month'].apply(clean_month)
            df = df.dropna(subset=['Month_clean'])
            equal_sample_month = st.checkbox("Equal-sample per month (sample each month to same size) for probability", value=False)
            if equal_sample_month:
                group_col = 'Month_clean'
                group_sizes = df.groupby(group_col).size()
                min_size = int(group_sizes.min()) if not group_sizes.empty else 0
                if min_size <= 0:
                    st.warning("Not enough samples per month to equalize.")
                else:
                    df_eq = df.groupby(group_col).apply(lambda g: g.sample(n=min_size, random_state=42)).reset_index(drop=True)
                    m_stats = df_eq.groupby('Month_clean')['flood_occurred'].agg(['sum', 'count']).reset_index()
                    m_stats['probability'] = (m_stats['sum'] / m_stats['count']).round(3)
                    m_stats['Month_clean'] = pd.Categorical(m_stats['Month_clean'], categories=list(month_map.values()), ordered=True)
                    m_stats = m_stats.sort_values('Month_clean')
                    fig = px.bar(m_stats, x='Month_clean', y='probability', title="Flood Probability by Month (equal-sample per month)", text='probability')
                    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                m_stats = df.groupby('Month_clean')['flood_occurred'].agg(['sum', 'count']).reset_index()
                m_stats['probability'] = (m_stats['sum'] / m_stats['count']).round(3)
                m_stats['Month_clean'] = pd.Categorical(m_stats['Month_clean'], categories=list(month_map.values()), ordered=True)
                m_stats = m_stats.sort_values('Month_clean')
                fig = px.bar(m_stats, x='Month_clean', y='probability', title="Flood Probability by Month", text='probability')
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(xaxis_title="Month", yaxis_title="Flood Probability")
                st.plotly_chart(fig, use_container_width=True)

            if show_explanations:
                st.markdown("""
                **Explanation:** Probability = Flood occurrences / Total records in that month.
                Equal-sample option uses the same sample count per month so bars are computed from equal-sized groups.
                """)

        # Municipality & Barangay with equal-sample options
        if 'Municipality' in df.columns:
            st.subheader("Flood probability by Municipality")
            equal_sample_muni = st.checkbox("Equal-sample per Municipality", key='eq_muni', value=False)
            if equal_sample_muni:
                group_col = 'Municipality'
                group_sizes = df.groupby(group_col).size()
                min_size = int(group_sizes.min()) if not group_sizes.empty else 0
                if min_size <= 0:
                    st.warning("Not enough municipality samples to equalize.")
                    mun = df.groupby('Municipality')['flood_occurred'].agg(['sum','count']).reset_index()
                else:
                    df_eq_m = df.groupby(group_col).apply(lambda g: g.sample(n=min_size, random_state=42)).reset_index(drop=True)
                    mun = df_eq_m.groupby('Municipality')['flood_occurred'].agg(['sum','count']).reset_index()
            else:
                mun = df.groupby('Municipality')['flood_occurred'].agg(['sum','count']).reset_index()
            mun['probability'] = (mun['sum'] / mun['count']).round(3)
            mun = mun.sort_values('probability', ascending=False)
            fig = px.bar(mun, x='Municipality', y='probability', title="Flood Probability by Municipality", text='probability')
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        if 'Barangay' in df.columns:
            st.subheader("Flood probability by Barangay")
            equal_sample_brgy = st.checkbox("Equal-sample per Barangay", key='eq_brgy', value=False)
            if equal_sample_brgy:
                group_col = 'Barangay'
                group_sizes = df.groupby(group_col).size()
                min_size = int(group_sizes.min()) if not group_sizes.empty else 0
                if min_size <= 0:
                    st.warning("Not enough barangay samples to equalize.")
                    brgy = df.groupby('Barangay')['flood_occurred'].agg(['sum','count']).reset_index()
                else:
                    df_eq_b = df.groupby(group_col).apply(lambda g: g.sample(n=min_size, random_state=42)).reset_index(drop=True)
                    brgy = df_eq_b.groupby('Barangay')['flood_occurred'].agg(['sum','count']).reset_index()
            else:
                brgy = df.groupby('Barangay')['flood_occurred'].agg(['sum','count']).reset_index()
            brgy['probability'] = (brgy['sum'] / brgy['count']).round(3)
            brgy = brgy.sort_values('probability', ascending=False)
            fig = px.bar(brgy, x='Barangay', y='probability', title="Flood Probability by Barangay", text='probability')
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)


# ------------------------------
# Clustering Tab (KMeans)
# ------------------------------
with tabs[2]:
    st.header("Clustering (KMeans)")
    if 'df' not in locals():
        st.warning("Do data cleaning first.")
    else:
        features = ['Water Level','No. of Families affected','Damage Infrastructure','Damage Agriculture']
        if not set(features).issubset(df.columns):
            st.error("Missing required columns for clustering.")
        else:
            st.subheader("KMeans clustering (k=3 default)")
            k = st.slider("Number of clusters (k)", 2, 6, 3)
            cluster_balance_toggle = st.checkbox("Balance data before clustering by WL quantile buckets", value=False)
            if cluster_balance_toggle:
                n_buckets_cluster = st.slider("Buckets to balance by (quantiles) for clustering", 3, 10, 5)
                df_tmp = df.copy()
                try:
                    df_tmp['WL_bucket'] = pd.qcut(df_tmp['Water Level'].fillna(df_tmp['Water Level'].median()), q=n_buckets_cluster, duplicates='drop').astype(str)
                    df_bal_cluster = balance_by_column(df_tmp, 'WL_bucket', method='upsample')
                except Exception:
                    df_bal_cluster = df.copy()
                    st.warning("Quantile bucket balancing failed; using original dataset for clustering.")
                X_cluster = df_bal_cluster[features].fillna(0)
            else:
                X_cluster = df[features].fillna(0)

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cluster)
            # map labels back to original df (best-effort - if we balanced, use balanced df's labels)
            if cluster_balance_toggle:
                df_bal_cluster['Cluster'] = kmeans.labels_
                counts = df_bal_cluster['Cluster'].value_counts().sort_index()
                st.write("Cluster counts (on balanced dataset):")
                st.write(counts)
                fig = px.scatter_3d(df_bal_cluster, x='Water Level', y='No. of Families affected', z='Damage Infrastructure',
                                    color='Cluster', hover_data=['Barangay','Municipality','Flood Cause'],
                                    title="KMeans clusters (3D) — balanced input")
            else:
                df['Cluster'] = kmeans.labels_
                counts = df['Cluster'].value_counts().sort_index()
                st.write("Cluster counts:")
                st.write(counts)
                fig = px.scatter_3d(df, x='Water Level', y='No. of Families affected', z='Damage Infrastructure',
                                    color='Cluster', hover_data=['Barangay','Municipality','Flood Cause'],
                                    title="KMeans clusters (3D)")

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Cluster summary (numeric medians)")
            if cluster_balance_toggle:
                cluster_summary = df_bal_cluster.groupby('Cluster')[features].median().round(2)
            else:
                cluster_summary = df.groupby('Cluster')[features].median().round(2)
            st.dataframe(cluster_summary)

# ------------------------------
# Flood Prediction (RandomForest) Tab
# ------------------------------
with tabs[3]:
    st.header("🌊 Flood Occurrence Prediction — RandomForest")

    if 'df' not in locals():
        st.warning("⚠️ Please run data cleaning first.")
    else:
        st.markdown("We train a **RandomForest** model to predict `flood_occurred`. Toggle balancing for training data below.")

        # Create target variable (1 = flood occurred)
        df['flood_occurred'] = (df['Water Level'] > 0).astype(int)

        # Balancing options
        st.markdown("### Optional: balance dataset before training")
        balance_toggle = st.checkbox("Balance dataset by Water Level quantile buckets (affects training data)", value=False)
        if balance_toggle:
            n_buckets = st.slider("Buckets to balance by (quantiles)", 3, 10, 5)
            method = st.selectbox("Balance method", ['upsample', 'downsample'])
            try:
                df_tmp = df.copy()
                df_tmp['WL_bucket'] = pd.qcut(df_tmp['Water Level'].fillna(df_tmp['Water Level'].median()), q=n_buckets, duplicates='drop').astype(str)
                df_bal = balance_by_column(df_tmp, 'WL_bucket', method=method)
                st.success(f"Balanced dataset created: {df_bal.shape[0]} rows ({method}).")
            except Exception as e:
                st.error(f"Balancing failed: {e}")
                df_bal = df.copy()
        else:
            df_bal = df.copy()

        # Feature engineering (numeric + month dummies)
        month_dummies = pd.get_dummies(df_bal['Month'].astype(str).fillna('Unknown'), prefix='Month')
        X_basic = pd.concat([
            df_bal[['Water Level', 'No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']].fillna(0),
            month_dummies
        ], axis=1)
        y = df_bal['flood_occurred']

        # Split data
        try:
            Xtr, Xte, ytr, yte = train_test_split(X_basic, y, test_size=0.3, random_state=42)
        except Exception:
            Xtr, Xte, ytr, yte = train_test_split(X_basic.fillna(0), y.fillna(0), test_size=0.3, random_state=42)

        # Train model
        model = RandomForestClassifier(random_state=42)
        try:
            model.fit(Xtr, ytr)
            ypred = model.predict(Xte)
            acc = accuracy_score(yte, ypred)
        except Exception as e:
            st.error(f"Model training failed: {e}")
            acc = 0.0
            ypred = np.zeros_like(yte)

        # Accuracy + report
        st.subheader("📊 RandomForest Evaluation")
        st.table(pd.DataFrame({"Metric": ["Accuracy (test)"], "Value": [f"{acc:.4f}"]}))

        try:
            report = classification_report(yte, ypred, output_dict=True, zero_division=0)
            st.markdown("### 📈 Classification Report")
            st.table(pd.DataFrame(report).transpose().round(3))
        except Exception:
            st.info("No classification report available.")

        if show_explanations:
            st.markdown("**🧠 Explanation:** RandomForest combines many decision trees to make predictions. Check class balance if accuracy seems odd.")

        # Feature importances
        try:
            fi = pd.Series(model.feature_importances_, index=X_basic.columns).sort_values(ascending=False).head(10)
            st.subheader("🔥 Top Feature Importances")
            st.bar_chart(fi)
        except Exception:
            pass

        # --- Monthly Flood Probability (Raw Data) ---
        st.subheader("📅 Monthly Flood Probabilities (actual data)")
        try:
            monthly_flood_counts = df.groupby('Month')['flood_occurred'].sum()
            monthly_total_counts = df.groupby('Month')['flood_occurred'].count()
            monthly_flood_probability = (monthly_flood_counts / monthly_total_counts).sort_values(ascending=False)
            st.dataframe(monthly_flood_probability.rename("Flood Probability").round(3))
            st.bar_chart(monthly_flood_probability)
        except Exception:
            st.info("Monthly probabilities can't be computed (missing Month).")

# ------------------------------
# Flood Severity Tab
# ------------------------------
with tabs[4]:
    st.header("🌊 Flood Severity Classification")

    if 'df' not in locals():
        st.warning("⚠️ Please perform data cleaning first.")
    else:
        df['Flood_Severity'] = df['Water Level'].apply(categorize_severity)

        st.subheader("📊 Severity Distribution")
        sev_counts = df['Flood_Severity'].value_counts().reset_index()
        sev_counts.columns = ['Severity Level', 'Count']
        st.table(sev_counts)

        base_feats = ['No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']
        month_d = pd.get_dummies(df['Month'].astype(str).fillna('Unknown'), prefix='Month')
        muni_d = pd.get_dummies(df['Municipality'].astype(str).fillna('Unknown'), prefix='Municipality') if 'Municipality' in df.columns else pd.DataFrame()
        brgy_d = pd.get_dummies(df['Barangay'].astype(str).fillna('Unknown'), prefix='Barangay') if 'Barangay' in df.columns else pd.DataFrame()

        st.markdown("### Optional: balance severity classes before training")
        balance_sev = st.checkbox("Balance severity classes (Low/Medium/High)", value=False)
        if balance_sev:
            method_sev = st.selectbox("Severity balance method", ['upsample', 'downsample'], key='sev_method')
            df_sev = df.copy()
            df_sev['Flood_Severity'] = df_sev['Flood_Severity'].astype(str)
            try:
                df_sev_bal = balance_by_column(df_sev, 'Flood_Severity', method=method_sev)
                st.write("Class counts after balancing:")
                st.table(df_sev_bal['Flood_Severity'].value_counts().reset_index().rename(columns={'index':'Severity','Flood_Severity':'Count'}))
            except Exception as e:
                st.error(f"Severity balancing failed: {e}")
                df_sev_bal = df.copy()
        else:
            df_sev_bal = df.copy()

        month_d = pd.get_dummies(df_sev_bal['Month'].astype(str).fillna('Unknown'), prefix='Month')
        muni_d = pd.get_dummies(df_sev_bal['Municipality'].astype(str).fillna('Unknown'), prefix='Municipality') if 'Municipality' in df_sev_bal.columns else pd.DataFrame()
        brgy_d = pd.get_dummies(df_sev_bal['Barangay'].astype(str).fillna('Unknown'), prefix='Barangay') if 'Barangay' in df_sev_bal.columns else pd.DataFrame()

        Xsev = pd.concat([df_sev_bal[base_feats].fillna(0), month_d, muni_d, brgy_d], axis=1)
        ysev = df_sev_bal['Flood_Severity']

        st.subheader("⚖️ Class Counts")
        class_counts = ysev.value_counts().reset_index()
        class_counts.columns = ['Flood Severity', 'Occurrences']
        st.table(class_counts)

        try:
            Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(Xsev, ysev, test_size=0.3, random_state=42, stratify=ysev)
            model_sev = RandomForestClassifier(random_state=42)
            model_sev.fit(Xtr_s, ytr_s)
            ypred_s = model_sev.predict(Xte_s)
            acc_s = accuracy_score(yte_s, ypred_s)

            st.subheader("✅ Severity Model Results")
            acc_table = pd.DataFrame({'Metric': ['Accuracy (test)'], 'Value': [f"{acc_s:.4f}"]})
            st.table(acc_table)

            report = classification_report(yte_s, ypred_s, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose().round(3)
            st.markdown("### 📈 Classification Report (Low / Medium / High)")
            st.table(report_df)

            if show_explanations:
                st.markdown("""
                **🧠 Explanation:** This multi-class RandomForest predicts flood severity levels — Low, Medium, or High.
                For production, consider SMOTE or class-weight adjustments to improve rare-class recall.
                """)

        except Exception as e:
            st.error(f"❌ Could not train severity model: {e}")

# ------------------------------
# Time Series (SARIMA)
# ------------------------------
with tabs[5]:
    st.header("Time Series Forecasting (SARIMA)")
    if 'df' not in locals():
        st.warning("Do data cleaning first.")
    else:
        st.markdown("This section resamples Water Level to daily average, checks stationarity, fits SARIMA, and shows forecasts.")

        df_temp = create_datetime_index(df)
        if not isinstance(df_temp.index, pd.DatetimeIndex):
            st.error("Dataset lacks usable Year/Month/Day to form a Date index. Add Year/Month/Day columns for forecasting.")
        else:
            ts = df_temp['Water Level'].resample('D').mean()
            ts_filled = ts.fillna(method='ffill').fillna(method='bfill')

            st.subheader("Time series preview (daily avg)")
            fig = px.line(ts_filled, title="Daily average Water Level")
            st.plotly_chart(fig, use_container_width=True)

            if show_explanations:
                st.markdown("**Explanation:** Resampled to daily averages and filled gaps for continuity before SARIMA.")

            st.subheader("Stationarity test (ADF)")
            try:
                adf_result = adfuller(ts_filled.dropna())
                st.write(f"ADF Statistic: {adf_result[0]:.4f}")
                st.write(f"P-value: {adf_result[1]:.4f}")
                st.write("If p-value > 0.05, the series is likely non-stationary — differencing recommended.")
            except Exception as e:
                st.error(f"ADF test failed: {e}")
                adf_result = (None, 1.0)

            d = 0
            if adf_result[1] > 0.05:
                d = 1
                ts_diff = ts_filled.diff().dropna()
                fig = px.line(ts_diff, title="First-order differenced series")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("ACF & PACF (help pick p/q values)")
            try:
                fig_acf = plt.figure(figsize=(10,4))
                plot_acf(ts_filled.dropna(), lags=40, ax=fig_acf.gca())
                st.pyplot(fig_acf)

                fig_pacf = plt.figure(figsize=(10,4))
                plot_pacf(ts_filled.dropna(), lags=40, ax=fig_pacf.gca())
                st.pyplot(fig_pacf)
            except Exception as e:
                st.error(f"Plot failed: {e}")

            st.subheader("Fit example SARIMA model")
            with st.spinner("Fitting SARIMA (may take a moment)..."):
                try:
                    order = (1, d, 1)
                    seasonal_order = (1, 0, 1, 7)
                    model_sarima = SARIMAX(
                        ts_filled,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    results = model_sarima.fit(disp=False)
                    summary_table = results.summary().tables[1]
                    import io
                    summary_df = pd.read_csv(io.StringIO(summary_table.as_csv()))
                    st.dataframe(summary_df, use_container_width=True)
                except Exception as e:
                    st.error(f"SARIMA fit failed: {e}")
                    results = None

            steps = st.slider("Forecast horizon (days)", 7, 365, 30)
            try:
                if results is not None:
                    pred = results.get_forecast(steps=steps)
                    pred_mean = pred.predicted_mean
                    pred_ci = pred.conf_int()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts_filled.index, y=ts_filled, name='Observed'))
                    fig.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean, name='Forecast'))
                    fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:,0], fill=None, mode='lines', line=dict(width=0)))
                    fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:,1], fill='tonexty', name='95% CI', mode='lines', line=dict(width=0)))
                    fig.update_layout(title="SARIMA Forecast", xaxis_title="Date", yaxis_title="Water Level")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("No SARIMA results available to forecast.")
            except Exception as e:
                st.error(f"Forecast failed: {e}")

# ------------------------------
# Model Comparison Tab
# ------------------------------
with tabs[6]:
    st.title("📊 Model Comparison Summary")
    st.markdown("""
    This section visually compares the three models used in the flood study.
    """)
    comparison_data = {
        "Model": ["K-Means Clustering", "Random Forest", "SARIMA"],
        "Purpose": [
            "Identify flood pattern clusters",
            "Predict flood occurrence / risk",
            "Forecast future water levels"
        ],
        "Metric": ["No. of Clusters", "Accuracy", "RMSE"],
        "Result": ["3 Clusters", "92%", "0.23"],
        "Notes": [
            "Groups areas with similar water behavior",
            "Accuracy on test split (example)",
            "Forecasting error (example)"
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)
    st.table(df_comparison)

    st.info("💡 Models focus on clustering, prediction, and forecasting — combine them for a fuller preparedness approach.")

    st.subheader("Visual Comparison of Each Model")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background-color:#E3F2FD;padding:20px;border-radius:15px;text-align:center;'>
            <h3>🌀 K-Means Clustering</h3>
            <p><b>Purpose:</b> Identify flood pattern clusters</p>
            <p><b>Result:</b> 3 Clusters (example)</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background-color:#E8F5E9;padding:20px;border-radius:15px;text-align:center;'>
            <h3>🌳 Random Forest</h3>
            <p><b>Purpose:</b> Predict flood occurrence</p>
            <p><b>Result:</b> 92% (example)</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background-color:#F3E5F5;padding:20px;border-radius:15px;text-align:center;'>
            <h3>📈 SARIMA</h3>
            <p><b>Purpose:</b> Forecast water levels</p>
            <p><b>Result:</b> RMSE 0.23 (example)</p>
        </div>
        """, unsafe_allow_html=True)

    perf_data = pd.DataFrame({
        "Model": ["K-Means", "Random Forest", "SARIMA"],
        "Performance": [3, 92, 0.23],
        "Metric": ["No. of Clusters", "Accuracy (%)", "RMSE"]
    })
    perf_data["Scaled Performance"] = perf_data["Performance"] / perf_data["Performance"].max() * 100
    fig = px.bar(
        perf_data,
        x="Model",
        y="Scaled Performance",
        color="Model",
        text="Performance",
        title="📊 Model Performance Comparison",
    )
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(yaxis_title="Scaled Performance (Normalized %)", xaxis_title="Model", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("App converted from Colab -> Streamlit. I added uniform/balancing options. Want SMOTE, model persistence, or downloadable reports? Say the word.")

























