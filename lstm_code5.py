import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load data
file_path = "C:/Users/yoshi/Desktop/internship/Dataset/Merged_HandPD_Cleaned.csv"
df = pd.read_csv(file_path)

# Use consistent labels only
df = df[df["MEANDER_CLASS_TYPE"] == df["SPIRAL_CLASS_TYPE"]].copy()
df["LABEL"] = df["MEANDER_CLASS_TYPE"] - 1

# Feature selection
features = [col for col in df.columns if col.startswith("MEANDER_") or col.startswith("SPIRAL_")]
X = df[features].values
y = df["LABEL"].values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for LSTM: (samples, time steps, features)
# Here we reshape each feature vector as a "sequence" of length = num_features
X_lstm = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Split
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y, test_size=0.2, random_state=42, stratify=y)

# Build LSTM Model
model = Sequential([
    LSTM(64, input_shape=(X_lstm.shape[1], 1)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X_train, y_train, epochs=25, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Predict
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Evaluate
print("LSTM Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples")
plt.title("LSTM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model
model.save("lstm_handpd_model.h5")
