# --- Imports ---
import os,glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Step 1: Load merged dataset ---
data_path = "/Users/mithra/Users/mithra/aws-cli/datasets/Processed Traffic Data for ML Algorithms"
csv_files = glob.glob(os.path.join(data_path, "*.csv"))

dfs = []
for f in csv_files:
    print("Loading:", os.path.basename(f))
    df_temp = pd.read_csv(f, low_memory=False)
    dfs.append(df_temp)

data = pd.concat(dfs, ignore_index=True)
print("Combined shape:", data.shape)

df = data.copy()

print("Saving merged dataset — this may take a few minutes...")
data.to_csv("/Users/mithra/Users/mithra/merged_dataset.csv", index=False)
print("✅ Merged dataset saved successfully!")

# --- Step 2: Drop irrelevant columns ---
cols_to_drop = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Timestamp']
df = data.drop(columns=cols_to_drop, errors='ignore')

# --- Step 3: Convert numeric columns ---
for col in df.columns:
    if col != 'Label':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Step 4: Handle missing and infinite values ---
thresh = int(df.shape[1] * 0.7)
df.dropna(thresh=thresh, inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Fill NaN only in numeric columns with median
num_cols = df.select_dtypes(include=['number']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
print("✅ Missing values handled successfully!")

# --- Step 5: Encode labels ---

from sklearn.preprocessing import LabelEncoder

# --- Detect categorical columns ---
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns detected:", cat_cols)

# --- Encode each categorical column ---
if cat_cols:
    le = LabelEncoder()
    for col in cat_cols:
        try:
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"Encoded: {col}")
        except Exception as e:
            print(f"⚠️ Skipping {col} due to: {e}")
else:
    print("✅ No categorical columns found — nothing to encode.")


df['target'] = df['Label'].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)
df.drop(columns=['Label'], inplace=True)


df['target'].value_counts()

# --- Step 6: Scale features ---
X = df.drop(columns=['target'])
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Step 7: Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\n✅ Preprocessing completed successfully!")
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
