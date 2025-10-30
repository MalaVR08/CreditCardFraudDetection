import streamlit as st
import requests
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter all transaction feature values (Time, V1–V28, Amount) to check if it’s Fraudulent or Legitimate.")

# 🔹 Input fields
st.subheader("🕒 Transaction Time and Amount")
time = st.number_input("Transaction Time (in seconds)", min_value=0, value=1000)
amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)

st.subheader("📊 PCA Features (V1 to V28)")
v_features = []
cols = st.columns(4)
for i in range(1, 29):
    with cols[(i - 1) % 4]:
        val = st.number_input(f"V{i}", value=0.0, format="%.4f")
        v_features.append(val)

# ✅ Combine all features
features = [time] + v_features + [amount]

# ✅ Backend API URL
response = requests.post("https://creditcardfrauddetection-vn3z.onrender.com/predict", json={"features": features})


# ✅ Predict button
if st.button("🔍 Predict"):
    try:
        response = requests.post(API_URL, json={"features": features})
        if response.status_code == 200:
            result = response.json()
            prediction = result["predictions"][0]
            prob = None
            if result["probabilities"]:
                prob = result["probabilities"][0][1]  # fraud probability

            if prediction == 1:
                st.error(f"🚨 Fraudulent Transaction Detected! (Fraud Probability: {prob:.2f})")
            else:
                st.success(f"✅ Legitimate Transaction. (Fraud Probability: {prob:.2f})")
        else:
            st.error(f"❌ Server Error: {response.status_code}")
    except Exception as e:
        st.error(f"Connection Error: {e}")

