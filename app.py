import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

MODEL_PATH = Path("artifacts/model.pkl")
REFERENCE_DATA_PATH = Path("artifacts/reference_data.csv")


def load_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please run train.py first to generate artifacts.")
        st.stop()
    return joblib.load(MODEL_PATH)


def load_reference_data() -> pd.DataFrame:
    if REFERENCE_DATA_PATH.exists():
        return pd.read_csv(REFERENCE_DATA_PATH)
    return pd.DataFrame()


def predict(model, payload: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.DataFrame([payload])
    prob = model.predict_proba(df)[0, 1]
    label = "High Risk" if prob >= 0.5 else "Lower Risk"
    return {"probability": float(prob), "label": label}


def layout_sidebar(defaults: Dict[str, Any]) -> Dict[str, Any]:
    st.sidebar.header("Patient Inputs")
    age = st.sidebar.slider("Age", 18, 90, int(defaults.get("age", 55)))
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, float(defaults.get("bmi", 28.0)))
    glucose = st.sidebar.slider("Avg Glucose Level", 60.0, 280.0, float(defaults.get("avg_glucose_level", 110.0)))
    hypertension = st.sidebar.selectbox("Hypertension", [0, 1], index=int(defaults.get("hypertension", 0)))
    heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1], index=int(defaults.get("heart_disease", 0)))
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], index=0 if defaults.get("gender", "Male") == "Male" else 1)
    ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"], index=0 if defaults.get("ever_married", "Yes") == "Yes" else 1)
    work_type = st.sidebar.selectbox(
        "Work Type", ["Private", "Self-employed", "Govt_job", "Never_worked"],
        index=["Private", "Self-employed", "Govt_job", "Never_worked"].index(defaults.get("work_type", "Private")),
    )
    residence = st.sidebar.selectbox("Residence", ["Urban", "Rural"], index=0 if defaults.get("Residence_type", "Urban") == "Urban" else 1)
    smoking_status = st.sidebar.selectbox(
        "Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"],
        index=["formerly smoked", "never smoked", "smokes", "Unknown"].index(defaults.get("smoking_status", "never smoked")),
    )

    return {
        "age": age,
        "bmi": bmi,
        "avg_glucose_level": glucose,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "gender": gender,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "smoking_status": smoking_status,
    }


def render_distributions(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No reference data available yet. Train first to view distributions.")
        return
    st.subheader("Reference Distributions")
    numeric_cols = ["age", "bmi", "avg_glucose_level"]
    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="HealthPredict - Stroke Risk", page_icon="🩺", layout="wide")
    st.title("HealthPredict: Stroke Risk Dashboard")
    st.markdown(
        """
        Real-time stroke risk scoring powered by a calibrated Random Forest and MLflow-tracked pipeline.
        Adjust patient features on the left to generate a risk probability.
        """
    )

    model = load_model()
    reference_df = load_reference_data()
    defaults = reference_df.iloc[0].to_dict() if not reference_df.empty else {}
    payload = layout_sidebar(defaults)

    with st.expander("Payload (debug)"):
        st.json(payload)

    if st.button("Predict", type="primary"):
        result = predict(model, payload)
        st.metric(label="Predicted Stroke Probability", value=f"{result['probability']:.2%}")
        st.success(f"Risk Category: {result['label']}")

    if not reference_df.empty:
        st.subheader("Population Snapshot")
        st.dataframe(reference_df.sample(min(5, len(reference_df))), use_container_width=True)

    render_distributions(reference_df)


if __name__ == "__main__":
    main()
