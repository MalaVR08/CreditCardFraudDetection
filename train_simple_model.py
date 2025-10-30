import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv(r"data/creditcard.csv")

# Features and target
X = data.drop(columns=["Class"])
y = data["Class"]

# Scale all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance dataset with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Train RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_res, y_res)

# Save model and scaler
joblib.dump(rf, "src/fraud_model.pkl")
joblib.dump(scaler, "src/scaler.pkl")

print("✅ Model trained on 30 features (Time, V1–V28, Amount)")
