import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load dataset
df = pd.read_csv("Chronic_Kidney_Dsease_data.csv")

df = df.drop(columns=["DoctorInCharge"], errors='ignore')
df = df.select_dtypes(include=['int64','float64'])

X = df.drop("Diagnosis", axis=1).values
y = df["Diagnosis"].values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load scaler + RF
scaler = pickle.load(open("models/scaler.pkl", "rb"))
rf_model = pickle.load(open("models/ckd_model.pkl", "rb"))

X_test_scaled = scaler.transform(X_test)

# RF prediction
rf_pred = rf_model.predict(X_test_scaled)

# -------- CNN --------
cnn_model = load_model("models/cnn_model.h5")

X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
cnn_pred = (cnn_model.predict(X_test_cnn) > 0.5).astype(int).flatten()

# -------- LSTM --------
lstm_model = load_model("models/lstm_model.h5")

X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
lstm_pred = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()

# -------- HYBRID --------
final_pred = (rf_pred + cnn_pred + lstm_pred) // 3

from sklearn.metrics import accuracy_score

print("Hybrid Accuracy:", accuracy_score(y_test, final_pred))