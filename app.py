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
# List of required columns
required_columns = [
    'reservation_status_date', 'arrival_date_month', 'deposit_type', 'hotel',
    'market_segment', 'country', 'arrival_date', 'reservation_status',
    'distribution_channel', 'customer_type', 'meal', 'reserved_room_type',
    'assigned_room_type'
]

# Initialize user input from Streamlit
user_input = {}
for f in required_columns:
    user_input[f] = st.sidebar.text_input(f, "Unknown")  # default string

# Convert to DataFrame
X_input = pd.DataFrame([user_input])

# Optional: handle numeric columns if any
numeric_cols = ['some_numeric_column1', 'some_numeric_column2']  # replace if needed
for col in numeric_cols:
    if col in X_input.columns:
        X_input[col] = pd.to_numeric(X_input[col], errors='coerce').fillna(0)

# Now safe to predict
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






