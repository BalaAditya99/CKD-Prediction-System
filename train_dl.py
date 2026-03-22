import pandas as pd
import numpy as np

df = pd.read_csv("Chronic_Kidney_Dsease_data.csv")

# Clean
df = df.drop(columns=["DoctorInCharge"], errors='ignore')
df = df.select_dtypes(include=['int64','float64'])

X = df.drop("Diagnosis", axis=1).values
y = df["Diagnosis"].values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= CNN =================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

cnn = Sequential([
    Conv1D(32, 2, activation='relu', input_shape=(X_train.shape[1],1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

cnn.fit(X_train_cnn, y_train, epochs=10)

_, cnn_acc = cnn.evaluate(X_test_cnn, y_test)

print("CNN Accuracy:", cnn_acc)
cnn.save("models/cnn_model.h5")

# ================= LSTM =================
from tensorflow.keras.layers import LSTM

X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

lstm = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[1])),
    Dense(1, activation='sigmoid')
])

lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

lstm.fit(X_train_lstm, y_train, epochs=10)

_, lstm_acc = lstm.evaluate(X_test_lstm, y_test)

print("LSTM Accuracy:", lstm_acc)

lstm.save("models/lstm_model.h5")

