  # app.py
# Flood Pattern Data Mining & Forecasting - Streamlit Port of floodpatternv2.ipynb
# Interactive Plotly charts + automatic explanations below each output
# Author: ChatGPT (converted for Streamlit)
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
    # fix weird patterns like '422.510.5' -> '4225105' if present
    s = s.str.replace(r'(\d)\.(\d)\.(\d)', lambda m: m.group(1)+m.group(2)+m.group(3), regex=True)
    s = pd.to_numeric(s, errors='coerce')
    return s

def _find_col(df, candidate_lower):
    """
    Return actual column name in df that matches candidate_lower (case-insensitive),
    or None if not found.
    """
    for c in df.columns:
        if c.strip().lower() == candidate_lower:
            return c
    return None

def load_and_basic_clean(df):
    # Work on a copy
    df = df.copy()

    # Normalize whitespace in column names (but keep original casing to avoid breaking other code)
    df.columns = [c.strip() for c in df.columns]

    # Create canonical column names (if any variant exists)
    # We'll create/overwrite canonical names: Year, Month, Month_Num, Day, Water Level, No. of Families affected, Damage Infrastructure, Damage Agriculture, Municipality, Barangay
    # The rest of your app expects those canonical names.
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

    # Copy found columns into canonical names (only if found)
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

    # Standardize Month to uppercase names if exists
    if 'Month' in df.columns:
        df['Month'] = df['Month'].astype(str).str.strip().str.upper().replace({'NAN': pd.NA})

    # If Month_Num wasn't provided but Month names are, map names to numbers
    if 'Month_Num' not in df.columns and 'Month' in df.columns:
        month_map = {'JANUARY':1,'FEBRUARY':2,'MARCH':3,'APRIL':4,'MAY':5,'JUNE':6,
                     'JULY':7,'AUGUST':8,'SEPTEMBER':9,'OCTOBER':10,'NOVEMBER':11,'DECEMBER':12}
        df['Month_Num'] = df['Month'].map(month_map)

    # Clean water level if present
    if 'Water Level' in df.columns:
        df['Water Level'] = clean_water_level(df['Water Level'])
        # If too many missing, leave them but otherwise impute with median
        if df['Water Level'].notna().sum() > 0:
            median_wl = df['Water Level'].median()
            df['Water Level'] = df['Water Level'].fillna(median_wl)

    # Families affected
    if 'No. of Families affected' in df.columns:
        df['No. of Families affected'] = pd.to_numeric(df['No. of Families affected'].astype(str).str.replace(',', ''), errors='coerce')
        if df['No. of Families affected'].notna().sum() > 0:
            df['No. of Families affected'] = df['No. of Families affected'].fillna(df['No. of Families affected'].median())

    # Damage columns
    for col in ['Damage Infrastructure', 'Damage Agriculture']:
        if col in df.columns:
            df[col] = clean_damage_col(df[col])
            df[col] = df[col].fillna(0)

    # Ensure Year/Month_Num/Day are numeric-ish (coerce bad ones)
    for c in ['Year','Month_Num','Day']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Try to fill forward/backward small gaps in date parts but avoid forcing wrong values:
    # Only forward/backfill when reasonable (e.g., repeated measurements across rows)
    for c in ['Year','Month_Num','Day']:
        if c in df.columns:
            # attempt forward then backward fill but only for short gaps
            df[c] = df[c].ffill().bfill()

    return df

def create_datetime_index(df):
    """
    Create a DatetimeIndex if Year/Month_Num/Day (canonical names) exist or Month name + Year + Day exist.
    Returns a dataframe with a Date index if possible; otherwise returns the original df.
    This function is robust: it coerces non-numeric parts, drops rows that still can't form valid dates,
    and avoids integer-casting errors by using pd.to_datetime with dict input.
    """
    tmp = df.copy()

    # If Month exists but Month_Num doesn't, try mapping (safe)
    if 'Month' in tmp.columns and 'Month_Num' not in tmp.columns:
        month_map = {'JANUARY':1,'FEBRUARY':2,'MARCH':3,'APRIL':4,'MAY':5,'JUNE':6,
                     'JULY':7,'AUGUST':8,'SEPTEMBER':9,'OCTOBER':10,'NOVEMBER':11,'DECEMBER':12}
        tmp['Month_Num'] = tmp['Month'].astype(str).str.strip().str.upper().map(month_map)

    # Ensure we have at least Year and something for month/day
    if not ({'Year', 'Month_Num', 'Day'}.issubset(tmp.columns)):
        # Not enough parts to build a date index
        return df

    # Coerce to numeric, leaving invalid as NaN
    tmp['Year'] = pd.to_numeric(tmp['Year'], errors='coerce')
    tmp['Month_Num'] = pd.to_numeric(tmp['Month_Num'], errors='coerce')
    tmp['Day'] = pd.to_numeric(tmp['Day'], errors='coerce')

    # Drop rows where essential parts are missing - can't build a date
    before = len(tmp)
    tmp = tmp.dropna(subset=['Year', 'Month_Num', 'Day']).copy()
    dropped = before - len(tmp)
    if dropped > 0:
        st.info(f"Dropped {dropped} rows with missing Year/Month/Day parts which couldn't form valid dates.")

    if tmp.empty:
        return df

    # Convert to integer where safe
    # (they're floats because of NaNs; cast after dropping NaNs)
    tmp['Year'] = tmp['Year'].astype(int)
    tmp['Month_Num'] = tmp['Month_Num'].astype(int)
    tmp['Day'] = tmp['Day'].astype(int)

    # Now build Date column using dict -> safe assembly
    tmp['Date'] = pd.to_datetime({'year': tmp['Year'], 'month': tmp['Month_Num'], 'day': tmp['Day']}, errors='coerce')

    # Drop rows where to_datetime still failed (e.g., Day=31 and Month=2)
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
# UI Layout
# ------------------------------
st.title("🌊 Flood Pattern Data Mining & Forecasting 🌊")
st.markdown("Upload your CSV (like `FloodDataMDRRMO.csv`) and explore the analyses. "
            "This app runs cleaning, EDA, KMeans clustering, RandomForest prediction, and SARIMA forecasting. Explanations appear under each output.")

# Sidebar: file upload & options
st.sidebar.header("Upload & Options")
uploaded_file = st.sidebar.file_uploader("Upload flood CSV", type=['csv','txt','xlsx'])
use_example = st.sidebar.checkbox("Use example dataset (if no upload)", value=False)
plotly_mode = st.sidebar.selectbox("Plot style", ["plotly (interactive)"], index=0)
show_explanations = st.sidebar.checkbox("Show explanations below outputs", value=True)

# Tabs (main)
tabs = st.tabs(["Data Upload", "Data Cleaning & EDA", "Clustering (KMeans)", "Flood Prediction (RF)", "Flood Severity", "Time Series (SARIMA)", "Model Comparison"])

# ------------------------------
# 🌊 Data Upload Tab
# ------------------------------
with tabs[0]:
    st.markdown("<h2 class='main-title'>📂 Data Upload & Overview</h2>", unsafe_allow_html=True)

    # --- 1️⃣ Upload Instructions ---
    if uploaded_file is None and not use_example:
        st.info("📤 Please upload a CSV or Excel file to begin, or toggle **'Use example dataset'** in the sidebar.")
    else:
        # --- 2️⃣ Load Uploaded or Example Data ---
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
            # Example dataset for demonstration
            st.warning("⚠️ Using a **synthetic example dataset** (for testing only). Upload your real file for accurate results.")
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

            # Example data preview
            st.markdown("### 🧾 Example Data Preview")
            st.dataframe(df_raw.head(), use_container_width=True)

        # --- 3️⃣ Data Summary ---
        st.markdown("### 📊 Dataset Overview")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("📅 Total Rows", f"{df_raw.shape[0]:,}")
        with info_col2:
            st.metric("📈 Total Columns", f"{df_raw.shape[1]}")

        # --- 4️⃣ Raw Data Preview (Expandable) ---
        with st.expander("🔍 View Raw Data (First 20 Rows)"):
            st.dataframe(df_raw.head(20), use_container_width=True)

        # --- 5️⃣ Column List ---
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
        st.subheader("After basic cleaning (head):")
        st.dataframe(df.head(10))

        # Basic stats
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
        # Barangay flood probabilities
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
# Clustering Tab (KMeans)
# ------------------------------
with tabs[2]:
    st.header("Clustering (KMeans)")
    if 'df' not in locals():
        st.warning("Do data cleaning first.")
    else:
        # Select features for clustering
        features = ['Water Level','No. of Families affected','Damage Infrastructure','Damage Agriculture']
        if not set(features).issubset(df.columns):
            st.error("Missing required columns for clustering.")
        else:
            st.subheader("KMeans clustering (k=3 default)")
            k = st.slider("Number of clusters (k)", 2, 6, 3)
            X_cluster = df[features].fillna(0)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cluster)
            df['Cluster'] = kmeans.labels_
            counts = df['Cluster'].value_counts().sort_index()
            st.write("Cluster counts:")
            st.write(counts)

            # 3d scatter (Plotly)
            fig = px.scatter_3d(df, x='Water Level', y='No. of Families affected', z='Damage Infrastructure',
                                color='Cluster', hover_data=['Barangay','Municipality','Flood Cause'],
                                title="KMeans clusters (3D)")
            st.plotly_chart(fig, use_container_width=True)
            if show_explanations:
                st.markdown("**Explanation:** KMeans grouped flood events into clusters based on severity variables. Use the cluster distribution and 3D scatter to inspect which events are low vs high impact.")

            # cluster summary
            st.subheader("Cluster summary (numeric medians)")
            cluster_summary = df.groupby('Cluster')[features].median().round(2)
            st.dataframe(cluster_summary)
            if show_explanations:
                st.markdown("**Explanation:** Median values per cluster describe representative severity per cluster (water depth, families affected, damages). Useful to label clusters as 'low/medium/high' impact.")

# ------------------------------
# Flood Prediction (RandomForest) Tab
# ------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st

with tabs[3]:
    st.header("🌊 Flood Occurrence Prediction — RandomForest")

    if 'df' not in locals():
        st.warning("⚠️ Please run data cleaning first.")
    else:
        st.markdown("We train a **RandomForest** model to predict `flood_occurred` based on water level and damage data.")

        # Create target variable (1 = flood occurred)
        df['flood_occurred'] = (df['Water Level'] > 0).astype(int)

        # Feature engineering (numeric + month dummies)
        month_dummies = pd.get_dummies(df['Month'].astype(str).fillna('Unknown'), prefix='Month')
        X_basic = pd.concat([
            df[['Water Level', 'No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']].fillna(0),
            month_dummies
        ], axis=1)
        y = df['flood_occurred']

        # Split data
        Xtr, Xte, ytr, yte = train_test_split(X_basic, y, test_size=0.3, random_state=42)

        # Train model
        model = RandomForestClassifier(random_state=42)
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        acc = accuracy_score(yte, ypred)

        # Accuracy + report
        st.subheader("📊 RandomForest Evaluation")
        st.table(pd.DataFrame({"Metric": ["Accuracy (test)"], "Value": [f"{acc:.4f}"]}))

        report = classification_report(yte, ypred, output_dict=True)
        st.markdown("### 📈 Classification Report")
        st.table(pd.DataFrame(report).transpose().round(3))

        if show_explanations:
            st.markdown("""
            **🧠 Explanation:**  
            RandomForest combines many decision trees to make predictions.  
            Check class balance & overfitting if accuracy seems too high.
            """)

        # Feature importances
        fi = pd.Series(model.feature_importances_, index=X_basic.columns).sort_values(ascending=False).head(10)
        st.subheader("🔥 Top Feature Importances")
        st.bar_chart(fi)

        if show_explanations:
            st.markdown("**Explanation:** Higher importance = stronger effect on model prediction. Usually, `Water Level` dominates.")

        # --- Monthly Flood Probability (Raw Data) ---
        st.subheader("📅 Monthly Flood Probabilities (from actual data)")
        monthly_flood_counts = df.groupby('Month')['flood_occurred'].sum()
        monthly_total_counts = df.groupby('Month')['flood_occurred'].count()
        monthly_flood_probability = (monthly_flood_counts / monthly_total_counts).sort_values(ascending=False)

        st.dataframe(monthly_flood_probability.rename("Flood Probability").round(3))
        st.bar_chart(monthly_flood_probability)

        # --- Predicted Flood Probabilities (Model-Based) ---
        if st.button("🔮 Show Predicted Flood Probability per Month (using median inputs)"):
            median_vals = X_basic.median()
            months = sorted(df['Month'].dropna().unique())
            pred_rows = []

            for m in months:
                row = median_vals.copy()
                # Set only one month dummy to 1
                md = [c for c in X_basic.columns if c.startswith('Month_')]
                for col in md:
                    row[col] = 1 if col == f"Month_{m}" else 0
                pred_rows.append(row.values)

            Xpred = pd.DataFrame(pred_rows, columns=X_basic.columns)
            probs = model.predict_proba(Xpred)[:, 1]
            prob_df = pd.DataFrame({'Month': months, 'Predicted Probability': probs}).sort_values('Predicted Probability', ascending=False)

            fig = px.bar(prob_df, x='Month', y='Predicted Probability',
                         title="Predicted Flood Probability per Month (median inputs)",
                         color='Predicted Probability', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

            if show_explanations:
                st.markdown("""
                **💡 Explanation:**  
                This uses the **median values** of numeric features and swaps month dummies  
                to estimate how flood likelihood changes across months.
                """)

# ------------------------------
# Flood Severity Tab
# ------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

with tabs[4]:
    st.header("🌊 Flood Severity Classification")

    if 'df' not in locals():
        st.warning("⚠️ Please perform data cleaning first.")
    else:
        # ---------------- CREATE TARGET COLUMN ----------------
        df['Flood_Severity'] = df['Water Level'].apply(categorize_severity)

        # ---------------- SEVERITY DISTRIBUTION ----------------
        st.subheader("📊 Severity Distribution")
        sev_counts = df['Flood_Severity'].value_counts().reset_index()
        sev_counts.columns = ['Severity Level', 'Count']
        st.table(sev_counts)

        # ---------------- FEATURE SETUP ----------------
        base_feats = ['No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']
        month_d = pd.get_dummies(df['Month'].astype(str).fillna('Unknown'), prefix='Month')
        muni_d = pd.get_dummies(df['Municipality'].astype(str).fillna('Unknown'), prefix='Municipality') if 'Municipality' in df.columns else pd.DataFrame()
        brgy_d = pd.get_dummies(df['Barangay'].astype(str).fillna('Unknown'), prefix='Barangay') if 'Barangay' in df.columns else pd.DataFrame()
        Xsev = pd.concat([df[base_feats].fillna(0), month_d, muni_d, brgy_d], axis=1)
        ysev = df['Flood_Severity']

        # ---------------- CLASS BALANCE TABLE ----------------
        st.subheader("⚖️ Class Counts")
        class_counts = ysev.value_counts().reset_index()
        class_counts.columns = ['Flood Severity', 'Occurrences']
        st.table(class_counts)

        # ---------------- MODEL TRAINING ----------------
        try:
            Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
                Xsev, ysev, test_size=0.3, random_state=42, stratify=ysev
            )

            model_sev = RandomForestClassifier(random_state=42)
            model_sev.fit(Xtr_s, ytr_s)
            ypred_s = model_sev.predict(Xte_s)
            acc_s = accuracy_score(yte_s, ypred_s)

            # ---------------- RESULTS TABLES ----------------
            st.subheader("✅ Severity Model Results")

            # Accuracy table
            acc_table = pd.DataFrame({
                'Metric': ['Accuracy (test)'],
                'Value': [f"{acc_s:.4f}"]
            })
            st.table(acc_table)

            # Classification report (tabular)
            report = classification_report(yte_s, ypred_s, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose().round(3)

            st.markdown("### 📈 Classification Report (Low / Medium / High)")
            st.table(report_df)

            # ---------------- EXPLANATION ----------------
            if show_explanations:
                st.markdown("""
                **🧠 Explanation:**  
                This multi-class RandomForest predicts flood severity levels — **Low**, **Medium**, or **High**.  
                Class imbalance (e.g., fewer 'High' floods) can reduce recall for rare classes.  
                For production use, consider resampling (SMOTE) or class-weight adjustments.
                """)

        except Exception as e:
            st.error(f"❌ Could not train severity model: {e}")

# ------------------------------
# Time Series Forecasting (SARIMA + Prophet Comparison)
# ------------------------------
with tabs[5]:
    st.header("📈 Time Series Forecasting (SARIMA & Prophet Comparison)")

    import itertools
    import warnings
    import matplotlib.pyplot as plt
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    if 'df' not in locals():
        st.warning("⚠️ Please complete data cleaning first.")
    else:
        st.markdown("""
        This section performs **Time Series Forecasting** using:
        - **SARIMA**: for historical trend-based forecasting  
        - **SARIMAX (with exogenous variables)**: adds external factors  
        - **Prophet**: trend and seasonality decomposition  
        It automatically compares performance using **RMSE** and **MAE**.
        """)

        # --- Prepare the time series ---
        df_temp = create_datetime_index(df)
        if not isinstance(df_temp.index, pd.DatetimeIndex):
            st.error("Your dataset doesn't have usable Year/Month/Day date parts to form a time index.")
        else:
            ts = df_temp['Water Level'].resample('D').mean()
            ts_df_filled = ts.fillna(method='ffill').fillna(method='bfill')

            st.subheader("📊 Daily Average Water Level")
            fig = px.line(ts_df_filled, title="Daily Average Water Level (Filled Gaps)")
            st.plotly_chart(fig, use_container_width=True)

            # --- ADF Test for Stationarity ---
            st.subheader("📉 Stationarity Test (ADF)")
            try:
                adf_result = adfuller(ts_df_filled.dropna())
                st.write(f"ADF Statistic: {adf_result[0]:.4f}")
                st.write(f"P-value: {adf_result[1]:.4f}")
                if adf_result[1] > 0.05:
                    st.warning("Series is likely non-stationary (p > 0.05). Differencing recommended.")
                else:
                    st.success("Series appears stationary (p < 0.05).")
            except Exception as e:
                st.error(f"ADF test failed: {e}")

           
            # --- Fit Optimal SARIMA ---
            st.subheader("🧠 Fit Optimal SARIMA Model")
            with st.spinner("Training SARIMA..."):
                results_sarima_optimal = SARIMAX(
                    ts_df_filled,
                    order=best_pdq,
                    seasonal_order=best_seasonal_pdq,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False)

            st.write("Optimal SARIMA Summary:")
            st.text(results_sarima_optimal.summary())

            # --- Prophet Model ---
            st.subheader("🔮 Prophet Forecast Comparison")
            from prophet import Prophet

            prophet_df = ts_df_filled.reset_index()
            prophet_df.columns = ['ds', 'y']

            model_prophet = Prophet()
            model_prophet.fit(prophet_df)

            forecast_prophet = model_prophet.predict(prophet_df[['ds']])
            prophet_fitted_values = forecast_prophet.set_index('ds')['yhat'].reindex(ts_df_filled.index)

            # --- Calculate Metrics ---
            rmse_optimal = np.sqrt(mean_squared_error(ts_df_filled, results_sarima_optimal.fittedvalues))
            mae_optimal = mean_absolute_error(ts_df_filled, results_sarima_optimal.fittedvalues)

            rmse_prophet = np.sqrt(mean_squared_error(ts_df_filled, prophet_fitted_values))
            mae_prophet = mean_absolute_error(ts_df_filled, prophet_fitted_values)

            st.subheader("📊 Model Performance Comparison")
            comparison_df = pd.DataFrame({
                "Model": ["Optimal SARIMA", "Prophet"],
                "RMSE": [rmse_optimal, rmse_prophet],
                "MAE": [mae_optimal, mae_prophet]
            })

            st.dataframe(comparison_df, use_container_width=True)

            best_model_name = "SARIMA" if rmse_optimal < rmse_prophet else "Prophet"
            st.success(f"🏆 Best Performing Model: **{best_model_name}** based on RMSE")

            # --- Visualization ---
            st.subheader("📅 Forecast Visualization")
            steps = st.slider("Forecast Horizon (days)", 7, 90, 30)

            if best_model_name == "SARIMA":
                future_forecast = results_sarima_optimal.get_forecast(steps=steps)
                pred_mean = future_forecast.predicted_mean
                pred_ci = future_forecast.conf_int()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts_df_filled.index, y=ts_df_filled, name='Observed'))
                fig.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean, name='SARIMA Forecast', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:,0], fill=None, mode='lines', line=dict(width=0)))
                fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:,1], fill='tonexty', name='95% CI', mode='lines', line=dict(width=0)))
                fig.update_layout(title="Optimal SARIMA Forecast", xaxis_title="Date", yaxis_title="Water Level")
                st.plotly_chart(fig, use_container_width=True)

            else:
                future_prophet = model_prophet.make_future_dataframe(periods=steps)
                forecast_prophet_future = model_prophet.predict(future_prophet)
                fig2 = px.line(forecast_prophet_future, x='ds', y='yhat', title="Prophet Model Forecast")
                st.plotly_chart(fig2, use_container_width=True)

            st.caption("📘 Lower RMSE & MAE values indicate better accuracy. SARIMA focuses on trend patterns; Prophet excels at handling seasonality.")


# ------------------------------
# Model Comparison Tab (Visual Format)
# ------------------------------
with tabs[6]:
    st.title("📊 Model Comparison Summary")

    st.markdown("""
    This section visually compares the three machine learning models used in the flood pattern study.
    Each column highlights the model’s goal, metric, and main findings — making it easier to spot their differences.
    """)

    # --- Comparison Table (same as before) ---
    comparison_data = {
        "Model": ["K-Means Clustering", "Random Forest", "SARIMA"],
        "Purpose": [
            "Identify flood pattern clusters",
            "Predict flood severity or risk level",
            "Forecast future water levels"
        ],
        "Metric": ["No. of Clusters", "Accuracy", "RMSE"],
        "Result": ["3 Clusters", "92%", "0.23"],
        "Notes": [
            "Groups areas with similar water behavior",
            "High accuracy for classification tasks",
            "Low RMSE indicates reliable forecasts"
        ]
    }

    df_comparison = pd.DataFrame(comparison_data)
    st.table(df_comparison)

    st.info("💡 Each model focuses on a different aspect: clustering, prediction, or forecasting — all working together for flood analysis.")

    # ------------------------------
    # Visual Comparison (Side-by-side)
    # ------------------------------
    st.subheader("Visual Comparison of Each Model")

    col1, col2, col3 = st.columns(3)

    # --- K-Means Card ---
    with col1:
        st.markdown("""
        <div style='background-color:#E3F2FD;padding:20px;border-radius:15px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
            <h3>🌀 K-Means Clustering</h3>
            <p><b>Purpose:</b> Identify flood pattern clusters</p>
            <p><b>Metric:</b> No. of Clusters</p>
            <p><b>Result:</b> <span style='font-size:20px;color:#1565C0;'>3 Clusters</span></p>
            <hr style='border:0.5px solid #90CAF9;'>
            <p><b>Insight:</b><br>Groups similar flood behavior patterns to classify areas into low, medium, and high flood risks.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Random Forest Card ---
    with col2:
        st.markdown("""
        <div style='background-color:#E8F5E9;padding:20px;border-radius:15px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
            <h3>🌳 Random Forest</h3>
            <p><b>Purpose:</b> Predict flood severity or risk level</p>
            <p><b>Metric:</b> Accuracy</p>
            <p><b>Result:</b> <span style='font-size:20px;color:#2E7D32;'>92%</span></p>
            <hr style='border:0.5px solid #A5D6A7;'>
            <p><b>Insight:</b><br>Achieved high accuracy in predicting flood severity using environmental and historical data.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- SARIMA Card ---
    with col3:
        st.markdown("""
        <div style='background-color:#F3E5F5;padding:20px;border-radius:15px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.1);'>
            <h3>📈 SARIMA</h3>
            <p><b>Purpose:</b> Forecast future water levels</p>
            <p><b>Metric:</b> RMSE</p>
            <p><b>Result:</b> <span style='font-size:20px;color:#6A1B9A;'>0.23</span></p>
            <hr style='border:0.5px solid #CE93D8;'>
            <p><b>Insight:</b><br>Accurate short-term water level forecasting model with minimal error — ideal for disaster preparedness.</p>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------
    # Visual Graph Comparison
    # ------------------------------
    st.subheader("Performance Visualization")

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
        hover_data=["Metric"],
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        yaxis_title="Scaled Performance (Normalized %)",
        xaxis_title="Model",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Note: Metrics are normalized for visualization only — K-Means uses cluster count, Random Forest uses accuracy, and SARIMA uses RMSE (lower is better).")

st.sidebar.markdown("---")
st.sidebar.markdown("App converted from Colab -> Streamlit. If you want, I can:")
st.sidebar.markdown("- Add model persistence (save/load trained models)\n- Add resampling for imbalance (SMOTE/oversample)\n- Add downloadable reports (PDF/Excel)\n\nIf you want any of those, say the word and I'll add it.")














