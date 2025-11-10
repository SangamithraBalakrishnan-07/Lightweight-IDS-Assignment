# Phase 3.2: Random Forest with Class Balancing (CSCI 590)

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load Dataset ---
print("Loading first 500,000 rows and essential columns...")
cols_to_use = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
               'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
               'Flow Byts/s', 'Flow Pkts/s',
               'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean',
               'Pkt Len Mean', 'Pkt Len Std', 'Label']

df = pd.read_csv("/Users/mithra/Users/mithra/merged_dataset.csv",
                 usecols=cols_to_use, nrows=500000, low_memory=False)
print("Loaded dataset shape:", df.shape)

# --- Step 2: Clean Numeric Columns ---
for col in df.columns:
    if col != 'Label':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.replace([np.inf, -np.inf, 'Infinity', 'inf', '-inf'], np.nan, inplace=True)
df.dropna(inplace=True)

num_cols = [c for c in df.columns if c != 'Label']
df[num_cols] = df[num_cols].astype(float)
mask = (df[num_cols] < 1e10).all(axis=1)
df = df.loc[mask]
print(f"Cleaned numeric columns. Shape after cleaning: {df.shape}")

# --- Step 3: Encode Label ---
df['Label'] = df['Label'].apply(lambda x: 0 if 'Benign' in str(x) else 1)
print("Label distribution after encoding:\n", df['Label'].value_counts())

# --- Step 4: Sample Data (50k for faster training) ---
print("\nSampling 50,000 rows safely...")
df_sample = df.sample(n=50000, random_state=42)
print("Sampled dataset shape:", df_sample.shape)
print("Sampled label distribution:\n", df_sample['Label'].value_counts())

# --- Step 5: Train-Test Split (80/20) ---
X = df_sample.drop(columns=['Label'])
y = df_sample['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

# --- Step 6: Model Setup with Class Balancing ---
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=2)

print("\n🚀 Starting Random Forest Grid Search (Balanced)...")
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("\nGrid Search Completed!")
print("Best Parameters:", grid_search.best_params_)

# --- Step 7: Evaluate Model ---
y_pred = best_rf.predict(X_test)
print("\n--- Evaluation on Test Data ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# --- Step 8: Save Results ---
os.makedirs("/Users/mithra/results", exist_ok=True)
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv("/Users/mithra/results/phase3_rf_gridsearch_results_balanced.csv", index=False)
print("Results saved to: /Users/mithra/results/phase3_rf_gridsearch_results_balanced.csv")

# --- Step 9: Confusion Matrix Plot ---
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix – Random Forest (Balanced)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/Users/mithra/results/confusion_matrix_rf_balanced.png")
plt.show()

# --- Step 10: Feature Importance Plot ---
importances = best_rf.feature_importances_
indices = np.argsort(importances)[-10:]
plt.figure(figsize=(6,4))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title("Top 10 Feature Importances – RF (Balanced)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("/Users/mithra/results/feature_importance_rf_balanced.png")
plt.show()

print("\nPhase 3.2 completed! All plots and metrics saved to /Users/mithra/results/")
