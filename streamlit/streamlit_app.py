# streamlit_app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load trained model
with open('models/best_cv_result.pkl', 'rb') as f:
	model = pickle.load(f)

# Load feature template
feature_template = pd.read_csv('dataset/ML_feature_set.csv')
feature_columns = feature_template.drop(columns='churned').columns.tolist()

st.set_page_config(page_title="Seller Churn Predictor", layout="wide")

st.title("🛍️ Olist Seller Churn Prediction App")

st.markdown("Use this app to predict whether sellers are likely to churn based on their behavioral and transactional data.")

# --- Upload or Manual ---
input_mode = st.radio("Select input method:", ['Upload CSV file', 'Manual entry'])

def predict(df):
	prediction = model.predict(df)
	prob = model.predict_proba(df)[:, 1]
	return prediction, prob

if input_mode == 'Upload CSV file':
	uploaded_file = st.file_uploader("Upload CSV file with matching features:", type=['csv'])

	if uploaded_file:
		df_input = pd.read_csv(uploaded_file)

		# Drop 'churned' if it's included in the upload
		if 'churned' in df_input.columns:
			df_input = df_input.drop(columns='churned')

		# Check and align columns
		missing_cols = set(feature_columns) - set(df_input.columns)
		extra_cols = set(df_input.columns) - set(feature_columns)

		if missing_cols:
			st.error(f"Missing columns: {missing_cols}")
		else:
			if extra_cols:
				st.warning(f"Ignoring extra columns: {extra_cols}")
				df_input = df_input[feature_columns]
			else:
				st.success("Input file validated. Ready to predict.")

			if st.button("Predict"):
				y_pred, y_prob = predict(df_input)
				df_input['churn_prediction'] = y_pred
				df_input['churn_probability'] = y_prob
				st.write(df_input)
				st.download_button("Download Predictions", df_input.to_csv(index=False), "churn_predictions.csv")

else:
	st.markdown("### Manual Feature Input")
	input_data = {}

	for col in feature_columns:
		if 'review_pct_' in col or 'avg_' in col or 'std_' in col:
			input_data[col] = st.number_input(f"{col}", value=round(feature_template[col].mean(), 2))
		elif 'payment_type' in col:
			input_data[col] = st.selectbox(f"{col}", [0, 1])
		elif 'top_master_category' in col or 'cohort_month' in col:
			input_data[col] = st.text_input(f"{col}")
		else:
			input_data[col] = st.number_input(f"{col}", value=round(feature_template[col].mean(), 2))

	if st.button("Predict"):
		df_manual = pd.DataFrame([input_data])
		y_pred, y_prob = predict(df_manual)
		st.write(f"**Prediction:** {'Churned' if y_pred[0] == 1 else 'Active'}")
		st.write(f"**Probability of churn:** {y_prob[0]:.2%}")
