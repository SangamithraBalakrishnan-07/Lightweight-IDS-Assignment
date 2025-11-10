# eda_phase2.py

# --- Imports ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

sns.set(style="whitegrid")

# --- Output Directory ---
output_dir = "/Users/mithra/EDA_Results"
os.makedirs(output_dir, exist_ok=True)

# --- Load Dataset ---
df = pd.read_csv("/Users/mithra/Users/mithra/merged_dataset.csv", low_memory=False)
print("Data loaded successfully! Shape:", df.shape)

# --- Step 1: Fix Label/Target consistency ---
if 'target' not in df.columns:
    if 'Label' in df.columns:
        # Encode Label as binary target: 0 = Benign, 1 = Attack
        df['target'] = df['Label'].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)
        print("Target column created from Label column.")

# --- Step 2: Convert all numeric-like columns to numeric ---
for col in df.columns:
    if col not in ['Label', 'target']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("Converted all numeric-like columns to numeric type.")
print("Number of numeric columns detected:",
      len(df.select_dtypes(include=[np.number]).columns))

# --- Step 3: Class Distribution Plot ---
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df)
plt.title("Binary Class Distribution (0 = Benign, 1 = Attack)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "class_distribution_binary.png"))
plt.show()
print("Class Distribution plot saved to:", output_dir)

# --- Step 4: Feature Distributions (Sample Histograms) ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'target']
sample_cols = numeric_cols[:6]  # show first 6 numeric columns

df[sample_cols].hist(bins=50, figsize=(12, 8))
plt.suptitle("Sample Numeric Feature Distributions")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "numeric_feature_distributions.png"))
plt.show()
print("Feature distribution histograms saved to:", output_dir)

# --- Step 5: Correlation Heatmap (Top 20 Numeric Features) ---
subset = numeric_cols[:20]
plt.figure(figsize=(12, 10))
corr = df[subset].corr()
sns.heatmap(corr, cmap='coolwarm', linewidths=0.3)
plt.title("Feature Correlation Heatmap (Top 20 Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap_top20.png"))
plt.show()
print("Correlation heatmap saved to:", output_dir)

# --- Step 6: Feature Importance (Random Forest) ---

# Select only numeric features for model input
X = df.select_dtypes(include=[np.number]).drop(columns=['target'], errors='ignore')
y = df['target']

# --- Clean invalid or extreme numeric values ---
X = X.replace([np.inf, -np.inf], np.nan)            # Replace infinities with NaN
X = X.fillna(X.median(numeric_only=True))           # Fill NaNs with median
X = X.clip(lower=-1e9, upper=1e9)                   # Clip extreme values to a safe range
print("Cleaned infinities, NaNs, and extreme values.")
print("Training Random Forest on", X.shape[1], "numeric features...")

# --- Train Random Forest ---
rf = RandomForestClassifier(
    n_estimators=50, 
    random_state=42, 
    n_jobs=-1
)
rf.fit(X, y)

# --- Compute Top 20 Feature Importances ---
importances = rf.feature_importances_
indices = np.argsort(importances)[-20:]  # top 20 most important

plt.figure(figsize=(8, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title("Top 20 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()

# --- Save the Plot ---
plt.savefig(os.path.join(output_dir, "feature_importance_top20.png"))
plt.show()

print("Feature importance plot saved to:", output_dir)


# --- Step 7: Save EDA summary statistics ---
eda_summary = df.describe()
summary_path = os.path.join(output_dir, "eda_summary.csv")
eda_summary.to_csv(summary_path)
print("EDA summary saved to:", summary_path)

print("\nPhase 2 EDA completed successfully!")
print(f"All plots and summary saved in: {output_dir}")
