import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle

st.set_page_config(page_title="Seller Churn Predictor", layout="wide")

# Load model
with open("models/best_cv_result.pkl", "rb") as f:
    model = cloudpickle.load(f)

# Load features
feature_template = pd.read_csv("dataset/ML_feature_set.csv")
feature_columns = feature_template.drop(columns="churned").columns.tolist()

st.title("🛍️ Olist Seller Churn Prediction App")
st.markdown("Upload a CSV or enter manually to predict seller churn.")

mode = st.radio("Input method:", ["Upload CSV", "Manual"])

def predict(df):
    pred = model.predict(df)
    prob = model.predict_proba(df)[:, 1]
    return pred, prob

if mode == "Upload CSV":
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        if "churned" in df.columns:
            df = df.drop(columns="churned")
        df = df[feature_columns]
        if st.button("Predict"):
            y_pred, y_prob = predict(df)
            df["Churn"] = y_pred
            df["Probability"] = np.round(y_prob * 100, 2)
            st.write(df)
            st.download_button("Download", df.to_csv(index=False), "churn_results.csv")
else:
    inputs = {}
    for col in feature_columns:
        inputs[col] = st.number_input(col, value=float(feature_template[col].mean()))
    if st.button("Predict"):
        df = pd.DataFrame([inputs])
        y_pred, y_prob = predict(df)
        st.success(f"Prediction: {'Churned' if y_pred[0] else 'Active'} ({y_prob[0]*100:.2f}%)")
