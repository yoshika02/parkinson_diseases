import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# === Step 1: Load the cleaned dataset ===
file_path = "C:/Users/yoshi/Desktop/internship/Dataset/Merged_HandPD_Cleaned.csv"
df = pd.read_csv(file_path)

# === Step 2: Check for missing values ===
print("🧹 Missing values per column:\n", df.isnull().sum())

# === Step 3: Class balance check ===
print("\n🎯 Class distribution (MEANDER_CLASS_TYPE):")
print(df["MEANDER_CLASS_TYPE"].value_counts())

# === Step 4: Feature and target selection ===
drop_columns = ["ID_PATIENT", "MEANDER_CLASS_TYPE", "SPIRAL_CLASS_TYPE"]
X = df.drop(columns=drop_columns)
y = df["MEANDER_CLASS_TYPE"]

# === Step 5: Check for duplicate rows ===
duplicates = df.duplicated().sum()
print(f"\n🧬 Duplicate rows in dataset: {duplicates}")

# === Step 6: Train-test split (70-30) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === Step 7: Normalize features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Step 8: Model training ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# === Step 9: Evaluation ===
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# === Step 10: Output ===
print("\n✅ Accuracy on Test Set:", round(accuracy, 4))

# Print average metrics
print("\n📊 Average Metrics:")
print(f"Avg Precision: {round(report['weighted avg']['precision'], 4)}")
print(f"Avg Recall:    {round(report['weighted avg']['recall'], 4)}")
print(f"Avg F1-Score:  {round(report['weighted avg']['f1-score'], 4)}")

# Full report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))
