import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# === Step 1: Load data ===
file_path = "C:/Users/yoshi/Desktop/internship/Dataset/Merged_HandPD_Cleaned.csv"
df = pd.read_csv(file_path)

# === Step 2: Filter rows with consistent labels ===
df = df[df["MEANDER_CLASS_TYPE"] == df["SPIRAL_CLASS_TYPE"]].copy()
df["LABEL"] = df["MEANDER_CLASS_TYPE"] - 1  # 0 = Healthy, 1 = Parkinson

# === Step 3: Select MEANDER and SPIRAL features only ===
features = [col for col in df.columns if col.startswith("MEANDER_") or col.startswith("SPIRAL_")]
X = df[features].values
y = df["LABEL"].values

# === Step 4: Normalize features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 5: Train-test split (70-30) with stratification ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# === Step 6: Train logistic regression ===
lr = LogisticRegression()
lr.fit(X_train, y_train)

# === Step 7: Predict ===
y_pred = lr.predict(X_test)

# === Step 8: Evaluation ===
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_text = classification_report(y_test, y_pred)

# === Step 9: Print results ===
print("✅ Accuracy:", round(accuracy, 4))
print("📊 Average Precision (weighted):", round(report_dict['weighted avg']['precision'], 4))
print("📊 Average Recall (weighted):", round(report_dict['weighted avg']['recall'], 4))
print("📊 Average F1-score (weighted):", round(report_dict['weighted avg']['f1-score'], 4))

print("\n📋 Full Classification Report:\n", report_text)

# === Step 10: Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Healthy", "Parkinson"], yticklabels=["Healthy", "Parkinson"])
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
