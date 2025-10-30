import streamlit as st
import joblib
import numpy as np

# ✅ Load model and scaler
model = joblib.load(r"src/fraud_model.pkl")
scaler = joblib.load(r"src/scaler.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter all transaction feature values (Time, V1–V28, Amount) to check if it’s Fraudulent or Legitimate.")

# ✅ Create input fields
st.subheader("🕒 Transaction Time and Amount")
time = st.number_input("Transaction Time (in seconds)", min_value=0, value=1000)
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)

st.subheader("📊 PCA Features (V1 to V28)")
v_features = []
cols = st.columns(4)  # 4 columns for cleaner layout
for i in range(1, 29):
    with cols[(i - 1) % 4]:
        val = st.number_input(f"V{i}", value=0.0, format="%.4f")
        v_features.append(val)

# ✅ Prepare input array
input_data = np.array([[time] + v_features + [amount]])
input_scaled = scaler.transform(input_data)

# ✅ Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]  # fraud probability

    if prediction[0] == 1:
        st.error(f"🚨 Fraudulent Transaction Detected! (Fraud Probability: {probability:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction. (Fraud Probability: {probability:.2f})")
