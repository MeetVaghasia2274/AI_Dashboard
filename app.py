import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hotel Booking Cancellation Predictor", layout="wide")

# --- Load model ---
model = joblib.load("hotel_model.pkl")
features = joblib.load("features.pkl")

st.title("🏨 Hotel Booking Cancellation Predictor")
st.write("Enter booking details to see if a reservation may be cancelled.")

# --- Sidebar inputs ---
user_input = {}
for f in features[:10]:
    user_input[f] = st.sidebar.text_input(f, "0")
X_input = pd.DataFrame([user_input])


if st.button("🔮 Predict"):
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]
    st.success(f"Prediction: {'Canceled' if pred==1 else 'Not Canceled'}")
    st.info(f"Probability: {proba:.2f}")


    # --- SHAP ---
    st.subheader("Feature importance (SHAP)")
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    X_trans = model.named_steps["preprocessor"].transform(X_input)
    shap_values = explainer.shap_values(X_trans)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_input, show=False)
    st.pyplot(fig)









