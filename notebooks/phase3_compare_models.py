
# Phase 3: Model Comparison Visualization (CSCI 590)

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# --- Define Paths ---
base_dir = "/Users/mithra/results"
rf_path = os.path.join(base_dir, "phase3_rf_gridsearch_results_balanced.csv")
voting_path = os.path.join(base_dir, "ensemble_voting_metrics.csv")
stacking_path = os.path.join(base_dir, "stacking_classifier_metrics.csv")

# --- Load Model Metrics ---
data = []
def safe_load(path, model_name):
    try:
        df = pd.read_csv(path)
        df['Model'] = model_name
        data.append(df)
        print(f"Loaded metrics for {model_name}")
    except Exception as e:
        print(f"⚠️ Could not load {model_name} metrics: {e}")

safe_load(voting_path, "Voting Ensemble")
safe_load(stacking_path, "Stacking Ensemble")

# If you manually created baseline RF metrics, you can add them here:
# Or simulate a quick summary (based on your earlier output)
rf_metrics = pd.DataFrame({
    "Accuracy": [0.9327],
    "Precision": [0.0],
    "Recall": [0.0],
    "F1-score": [0.0],
    "ROC-AUC": [0.582],
    "Model": ["Random Forest"]
})
data.append(rf_metrics)

# --- Combine All Models ---
results = pd.concat(data, ignore_index=True)
results = results[["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]]
print("\nCombined Metrics:\n", results)

# --- Melt for Plotting ---
results_melted = results.melt(id_vars="Model", var_name="Metric", value_name="Score")

# --- Plot Comparison ---
sns.set(style="whitegrid")
plt.figure(figsize=(9, 5))
sns.barplot(x="Metric", y="Score", hue="Model", data=results_melted, palette="deep")
plt.title("Model Performance Comparison (RF vs Ensemble vs Stacking)")
plt.ylim(0, 1)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

os.makedirs(base_dir, exist_ok=True)
plt.savefig(os.path.join(base_dir, "model_comparison_barplot.png"))
plt.show()

print(f"\nComparison plot saved to: {os.path.join(base_dir, 'model_comparison_barplot.png')}")
