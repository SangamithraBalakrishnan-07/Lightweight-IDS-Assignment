
#Phase 3.3: Ensemble Learning with Voting Classifier (CSCI 590)

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, auc)

# --- Step 1: Load Balanced Data ---
print("Loading balanced 50k sample dataset...")
df = pd.read_csv("/Users/mithra/results/phase3_rf_gridsearch_results_balanced.csv", low_memory=False)
# If this file only contains gridsearch results, reload from merged CSV:
df = pd.read_csv("/Users/mithra/Users/mithra/merged_dataset.csv", nrows=500000)

# Select columns
cols_to_use = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
               'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
               'Flow Byts/s', 'Flow Pkts/s',
               'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean',
               'Pkt Len Mean', 'Pkt Len Std', 'Label']
df = df[cols_to_use]

# Clean numeric data
for col in df.columns:
    if col != 'Label':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.replace([np.inf, -np.inf, 'Infinity', 'inf', '-inf'], np.nan, inplace=True)
df.dropna(inplace=True)
df['Label'] = df['Label'].apply(lambda x: 0 if 'Benign' in str(x) else 1)

# Sample for manageable size
df = df.sample(n=50000, random_state=42)
print("Dataset shape:", df.shape)

# --- Step 2: Split Data (80/20) ---
X = df.drop(columns=['Label'])
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# --- Step 3: Feature Scaling (for SVM & LR) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- Step 4: Define Base Models with Balancing ---
rf = RandomForestClassifier(
    n_estimators=100, max_depth=20, min_samples_split=2,
    class_weight='balanced', random_state=42, n_jobs=-1)
lr = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
dt = DecisionTreeClassifier(max_depth=20, random_state=42, class_weight='balanced')


# --- Step 5: Ensemble Model (Soft Voting) ---
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('lr', lr), ('svm', svm), ('dt', dt)],
    voting='soft'
)

# --- Step 6: Train Ensemble ---
print("\nTraining Ensemble Voting Classifier...")
ensemble.fit(X_train_scaled, y_train)
print("Training Completed!")

# --- Step 7: Predictions & Evaluation ---
y_pred = ensemble.predict(X_test_scaled)
y_prob = ensemble.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n--- Evaluation on Test Data ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# --- Step 8: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix – Ensemble Voting Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

os.makedirs("/Users/mithra/results", exist_ok=True)
plt.savefig("/Users/mithra/results/confusion_matrix_ensemble_voting.png")
plt.show()

# --- Step 9: ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_val = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'Ensemble (AUC = {roc_auc_val:.3f})')
plt.plot([0,1], [0,1], color='red', linestyle='--')
plt.title("ROC Curve – Ensemble Voting Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("/Users/mithra/results/roc_curve_ensemble_voting.png")
plt.show()

# --- Step 10: Save Metrics ---
results = {
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1-score": [f1],
    "ROC-AUC": [roc_auc]
}
results_df = pd.DataFrame(results)
results_df.to_csv("/Users/mithra/results/ensemble_voting_metrics.csv", index=False)
print("\nEnsemble metrics saved to /Users/mithra/results/ensemble_voting_metrics.csv")
