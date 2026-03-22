import pandas as pd
import pickle

# Load dataset
df = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\ckdmain\Chronic_Kidney_Dsease_data.csv")

# Clean data
df = df.drop(columns=["DoctorInCharge"], errors='ignore')
df = df.select_dtypes(include=['int64','float64'])

# Split
X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
from sklearn.metrics import accuracy_score
print("RF Accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Save model
import os
os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/ckd_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))