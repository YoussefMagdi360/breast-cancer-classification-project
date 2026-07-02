"""
Streamlit deployment app for the Breast Cancer Tumor Classification project.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Breast Cancer Tumor Classification",
    page_icon="🧬",
    layout="wide",
)


@st.cache_resource
def load_or_train_model():
    model_path = Path("models/model.joblib")

    if model_path.exists():
        saved = joblib.load(model_path)
        return saved["model"], saved["feature_names"], saved["target_names"]

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
        ]
    )
    model.fit(X, y)

    return model, list(X.columns), list(data.target_names)


data = load_breast_cancer(as_frame=True)
df = data.frame
model, feature_names, target_names = load_or_train_model()

st.title("Breast Cancer Tumor Classification")
st.write(
    "This app predicts whether a tumor is benign or malignant using the Breast Cancer Wisconsin dataset."
)

st.warning(
    "Educational project only. This app must not be used as a real medical diagnosis tool."
)

st.sidebar.header("Input Tumor Features")
st.sidebar.write("Default values are the dataset median values.")

median_values = data.data.median()

user_input = {}
for feature in feature_names:
    min_value = float(data.data[feature].min())
    max_value = float(data.data[feature].max())
    default_value = float(median_values[feature])
    user_input[feature] = st.sidebar.slider(
        label=feature,
        min_value=min_value,
        max_value=max_value,
        value=default_value,
    )

input_df = pd.DataFrame([user_input])

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Input Data")
    st.dataframe(input_df, use_container_width=True)

with right_col:
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    predicted_class = target_names[prediction]
    confidence = probabilities[prediction]

    st.subheader("Prediction")
    st.metric("Predicted Class", predicted_class)
    st.metric("Confidence", f"{confidence:.2%}")

    prob_df = pd.DataFrame(
        {
            "Class": target_names,
            "Probability": probabilities,
        }
    )
    st.bar_chart(prob_df.set_index("Class"))

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)
