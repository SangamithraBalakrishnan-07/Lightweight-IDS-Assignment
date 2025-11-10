
# Phase 3.3: Stacking Classifier Ensemble (Advanced)

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, auc)

# --- Load the same preprocessed data ---
print("Loading dataset...")
cols_to_use = ['Flow Duration','Tot Fwd Pkts','Tot Bwd Pkts',
               'TotLen Fwd Pkts','TotLen Bwd Pkts',
               'Flow Byts/s','Flow Pkts/s',
               'Fwd Pkt Len Mean','Bwd Pkt Len Mean',
               'Pkt Len Mean','Pkt Len Std','Label']

df = pd.read_csv("/Users/mithra/Users/mithra/merged_dataset.csv",
                 usecols=cols_to_use, nrows=500000, low_memory=False)

for col in df.columns:
    if col != 'Label':
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df['Label'] = df['Label'].apply(lambda x: 0 if 'Benign' in str(x) else 1)

# Optional sampling for faster training
df = df.sample(n=50000, random_state=42)
print("Loaded clean dataset:", df.shape)

# --- Split and Scale ---
X = df.drop(columns=['Label'])
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Define Base Learners ---
rf = RandomForestClassifier(n_estimators=100, max_depth=20, class_weight='balanced', random_state=42)
svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
dt = DecisionTreeClassifier(max_depth=20, class_weight='balanced', random_state=42)
lr = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')

base_estimators = [
    ('rf', rf),
    ('svm', svm),
    ('dt', dt)
]

# --- Define Stacking Model ---
meta_learner = LogisticRegression(max_iter=1000, solver='liblinear')
stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    stack_method='predict_proba',
    n_jobs=-1
)

# --- Train the Stacking Classifier ---
print("\nTraining Stacking Classifier Ensemble...")
stack.fit(X_train_scaled, y_train)
print("Training Completed!")

# --- Evaluate ---
y_pred = stack.predict(X_test_scaled)
y_prob = stack.predict_proba(X_test_scaled)[:, 1]

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

# --- Confusion Matrix ---
os.makedirs("/Users/mithra/results", exist_ok=True)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix – Stacking Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("/Users/mithra/results/confusion_matrix_stacking.png")
plt.show()

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_val = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='purple', lw=2, label=f'Stacking (AUC = {roc_auc_val:.3f})')
plt.plot([0,1],[0,1],'--',color='red')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Stacking Classifier Ensemble")
plt.legend()
plt.tight_layout()
plt.savefig("/Users/mithra/results/roc_curve_stacking.png")
plt.show()

# --- Save Results ---
results = {
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1-score": [f1],
    "ROC-AUC": [roc_auc]
}
results_df = pd.DataFrame(results)
results_df.to_csv("/Users/mithra/results/stacking_classifier_metrics.csv", index=False)
print("Stacking results saved to /Users/mithra/results/stacking_classifier_metrics.csv")
