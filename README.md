### 📁 **Lightweight-IDS-Assignment**  
**Course:** CSCI 590 – Intrusion Detection Systems  
**Author:** *Sangamithra Balakrishnan, St. Francis Xavier University*  

---

## 📘 Project Overview  
This project implements a **Lightweight Intrusion Detection System (IDS)** using the **CICIDS 2018 dataset**. The system explores multiple machine learning approaches across three phases:

1. **Phase 1 – Data Preprocessing:**  
   - Loading, cleaning, merging, and encoding CSV files from the CICIDS 2018 dataset.  
   - Handling missing values and scaling numeric features.  
   - Output: `processed_dataset.csv`.

2. **Phase 2 – Exploratory Data Analysis (EDA):**  
   - Visualized feature distributions, correlations, and class imbalance.  
   - Identified key features contributing to network traffic classification.  
   - Output visualizations:  
     - `class_distribution_binary.png`  
     - `correlation_heatmap_top20.png`  
     - `feature_importance_top20.png`  
     - `numeric_feature_distributions.png`

3. **Phase 3 – Machine Learning Models:**  
   - **3.1 Random Forest (Baseline):** Tuned with GridSearchCV.  
   - **3.2 Ensemble Learning (Voting Classifier):** Combined RF, SVM, Logistic Regression, and Decision Tree.  
   - **3.3 Stacking Classifier:** Integrated multiple models using Logistic Regression as meta-learner.  
   - Final comparison: `model_comparison_barplot.png`.

---

## 🧠 Dataset Source  
**CICIDS 2018 Dataset:**  
Canadian Institute for Cybersecurity  
[https://www.unb.ca/cic/datasets/ids-2018.html](https://www.unb.ca/cic/datasets/ids-2018.html)

Use the “Processed Traffic Data for ML Algorithms” folder from the dataset, which contains CSV files for each day of captured traffic.
Note: The full merged dataset (~7 GB) was excluded from the GitHub repository due to file size limits.
---

## ⚙️ Folder Structure  

```
Lightweight-IDS-Assignment/
│
├── data/
│   └── merged_dataset.csv                 # Note: The full merged dataset (~7 GB) was excluded from the GitHub repository due to file size limits.
│
├── notebooks/
│   ├── data_preprocessing.py       # Data cleaning, encoding, saving
│   ├── eda_phase2.py             # EDA and feature visualization
│   ├── phase3_1_rf_tuning.py              # Random Forest baseline
│   ├── phase3_ensemble_learning.py      # Voting Ensemble
│   ├── phase3_stacking_classifier.py    # Stacking Ensemble
│   └── phase3_compare_models.py         # Comparison plots
│
├── results/
│   ├── phase3_rf_gridsearch_results_balanced.csv
│   ├── ensemble_voting_metrics.csv
│   ├── stacking_classifier_metrics.csv
│   ├── model_comparison_barplot.png
│   └── eda_summary.csv
│
├── README.md                              # This file
└── requirements.txt                       # Dependencies
```

---

## How to Run

### Step 1 – Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 – Run Phases
```bash
# Phase 1: Preprocessing
python notebooks/data_preprocessing.py

# Phase 2: EDA
python notebooks/eda_phase2.py

# Phase 3.1: Random Forest
python notebooks/phase3_1_rf_tuning.py

# Phase 3.2: Voting Ensemble
python notebooks/phase3_ensemble_learning.py

# Phase 3.3: Stacking Classifier
python notebooks/phase3_stacking_classifier.py

# Phase 3.5: Compare Models
python notebooks/phase3_compare_models.py
```

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|-----------|------------|--------|----------|
| Random Forest | 0.9327 | 0.0000 | 0.0000 | 0.0000 | 0.582 |
| Voting Ensemble | 0.9224 | 0.1544 | 0.0342 | 0.0560 | 0.589 |
| Stacking Ensemble | 0.9327 | 0.0000 | 0.0000 | 0.0000 | 0.583 |

See `results/model_comparison_barplot.png` for visual comparison.

---

## Example Output Images

| Figure | Description |
|---------|-------------|
| `class_distribution_binary.png` | Benign vs Attack distribution |
| `correlation_heatmap_top20.png` | Feature correlation heatmap |
| `feature_importance_top20.png` | Important features via Random Forest |
| `model_comparison_barplot.png` | Final model comparison |

---

## Discussion
- The models achieved high accuracy on benign traffic but struggled with attack detection due to dataset imbalance.  
- Future improvements include:
  - Using **SMOTE** or **ADASYN** oversampling  
  - Implementing **feature selection** for optimization  
  - Exploring **deep learning architectures** for time-series traffic data.

---

## References
1. CICIDS 2018 Dataset – Canadian Institute for Cybersecurity  
2. Scikit-learn Documentation – [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)  
3. Saharkhizan, M., et al. (2020). *“Improved Network Intrusion Detection using Random Forest Ensembles.”*

