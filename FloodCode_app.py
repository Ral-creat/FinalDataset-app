import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Load dataset ---
file_path = "FLOOD DATA.csv"  # adjust if file path differs
df = pd.read_csv(file_path, encoding='latin1')

# --- Clean numeric columns ---
df['No. of Families affected'] = (
    df['No. of Families affected']
    .astype(str)
    .str.replace(',', '')
    .str.extract('(\d+)')
    .astype(float)
)

df['Estimated damage to Infrastructure'] = (
    df['Estimated damage to Infrastructure']
    .astype(str)
    .str.replace(',', '')
    .str.extract('(\d+\.?\d*)')
    .astype(float)
)

df['Estimated damage to agriculture'] = (
    df['Estimated damage to agriculture']
    .astype(str)
    .str.replace(',', '')
    .str.extract('(\d+\.?\d*)')
    .astype(float)
)

# --- 1. Flood occurrences per year ---
plt.figure(figsize=(8,5))
df['Year'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Flood Occurrences per Year")
plt.xlabel("Year")
plt.ylabel("Number of Floods")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- 2. Most affected Barangays ---
plt.figure(figsize=(10,5))
df.groupby('Baranggay')['No. of Families affected'].sum().sort_values(ascending=False).head(10).plot(
    kind='bar', color='salmon', edgecolor='black'
)
plt.title("Top 10 Most Affected Barangays (Families Affected)")
plt.ylabel("Families Affected")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- 3. Common Flood Causes ---
plt.figure(figsize=(10,5))
df['Flood Cause'].value_counts().head(10).plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title("Top 10 Common Flood Causes")
plt.ylabel("Occurrences")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- 4. Total estimated damages ---
plt.figure(figsize=(6,6))
plt.pie(
    [df['Estimated damage to Infrastructure'].sum(), df['Estimated damage to agriculture'].sum()],
    labels=['Infrastructure', 'Agriculture'],
    autopct='%1.1f%%',
    startangle=90,
    colors=['gold', 'lightcoral']
)
plt.title("Proportion of Total Estimated Damages")
plt.tight_layout()
plt.show()

# --- 5. Relation between affected families and total damages ---
plt.figure(figsize=(8,6))
plt.scatter(
    df['No. of Families affected'],
    df['Estimated damage to Infrastructure'] + df['Estimated damage to agriculture'],
    alpha=0.6,
    color='steelblue',
    edgecolor='black'
)
plt.title("Families Affected vs. Total Estimated Damage")
plt.xlabel("No. of Families Affected")
plt.ylabel("Total Estimated Damage (₱)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
