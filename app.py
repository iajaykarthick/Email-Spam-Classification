import os
import joblib

import numpy as np

import streamlit as st

from src.features.build_features import extract_features
from src.config import EXPERIMENT_DIR


models = [model_dir for model_dir in os.listdir(EXPERIMENT_DIR) if os.path.exists(os.path.join(EXPERIMENT_DIR, model_dir, "model.joblib"))]
default_model_index = models.index('sklearn_cart_w_best_params')
# Create a select box for the models
selected_model = st.selectbox("Select a model:", models, index=default_model_index)


model_dir = os.path.join(EXPERIMENT_DIR, selected_model)
model_path = os.path.join(model_dir, "model.joblib")
model = joblib.load(model_path)

text = st.text_area("Enter the text you want to classify:", height=250, max_chars=5000)

if text:
    features_dict = extract_features(text)
    features = np.array([value for key, value in features_dict.items()]).reshape(1, -1)
    features = features.reshape(1, -1) 
    
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.write("🚨 The text is likely spam! 🚨")
    else:
        st.write("✅ The text is likely not spam. ✅")