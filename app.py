from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="GlucoCheck | Diabetes Prediction", page_icon="+", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');
    :root { --ink:#17221f; --muted:#63716d; --green:#167451; --deep:#0e4b38; --line:#d8e5df; --paper:#f8fbf9; }
    .stApp { background:radial-gradient(circle at 10% 0%, #e7f7ee 0, transparent 35%), var(--paper); color:var(--ink); font-family:'DM Sans', sans-serif; }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:820px; padding:3rem 1.25rem 4rem; }
    h1,h2,h3 { font-family:'Manrope', sans-serif; color:var(--deep); }
    h1 { font-size:clamp(2rem, 5vw, 3.4rem); line-height:1.05; margin-bottom:.5rem; }
    .eyebrow { color:var(--green); font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:.75rem; }
    .intro { color:var(--muted); font-size:1.05rem; line-height:1.6; max-width:620px; margin-bottom:2rem; }
    .form-shell { background:rgba(255,255,255,.8); border:1px solid var(--line); border-radius:16px; padding:1.25rem 1.25rem .75rem; box-shadow:0 18px 50px rgba(23,70,53,.07); }
    [data-testid="stForm"] { border:0; padding:0; }
    div[data-testid="stFormSubmitButton"] button { background:var(--green); color:white; border:0; border-radius:10px; min-height:3rem; font-weight:700; width:100%; }
    div[data-testid="stFormSubmitButton"] button:hover { background:var(--deep); color:white; }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { border-color:var(--line); border-radius:9px; }
    .result { border-radius:14px; padding:1.25rem 1.4rem; margin-top:1.5rem; border:1px solid var(--line); background:white; }
    .result-title { font-family:'Manrope', sans-serif; font-size:1.35rem; font-weight:800; color:var(--deep); }
    .result-copy { color:var(--muted); margin-top:.35rem; }
    .footnote { color:var(--muted); font-size:.78rem; line-height:1.5; margin-top:1.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = Path(__file__).with_name("model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.markdown('<div class="eyebrow">Health screening tool</div>', unsafe_allow_html=True)
st.title("Diabetes prediction, made simple")
st.markdown('<p class="intro">Enter a few health details below. The trained machine-learning model will estimate whether the profile is more likely to be associated with diabetes.</p>', unsafe_allow_html=True)

model = load_model()

st.markdown('<div class="form-shell">', unsafe_allow_html=True)
with st.form("prediction_form"):
    st.subheader("Patient details")
    first_row = st.columns(2)
    with first_row[0]:
        gender = st.selectbox("Gender", ["Female", "Male"])
    with first_row[1]:
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=35.0, step=1.0)
    second_row = st.columns(2)
    with second_row[0]:
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda value: "Yes" if value else "No")
    with second_row[1]:
        heart_disease = st.selectbox("Heart disease", [0, 1], format_func=lambda value: "Yes" if value else "No")
    smoking_history = st.selectbox("Smoking history", ["No Info", "never", "former", "current", "ever", "not current"])
    third_row = st.columns(3)
    with third_row[0]:
        bmi = st.number_input("BMI", min_value=5.0, max_value=80.0, value=27.3, step=0.1)
    with third_row[1]:
        hba1c = st.number_input("HbA1c level", min_value=2.0, max_value=20.0, value=5.7, step=0.1)
    with third_row[2]:
        blood_glucose = st.number_input("Blood glucose", min_value=40, max_value=500, value=120, step=1)
    submitted = st.form_submit_button("Run prediction")
st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    patient = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": blood_glucose,
    }])
    prediction = int(model.predict(patient)[0])
    probability = float(model.predict_proba(patient)[0][1]) if hasattr(model, "predict_proba") else None
    if prediction:
        title = "Higher likelihood detected"
        copy = "The model flags this profile as more likely to be associated with diabetes."
    else:
        title = "Lower likelihood detected"
        copy = "The model does not flag this profile as likely to be associated with diabetes."
    probability_copy = f"Estimated model probability: {probability:.1%}" if probability is not None else ""
    st.markdown(f'<div class="result"><div class="result-title">{title}</div><div class="result-copy">{copy}</div><div class="result-copy">{probability_copy}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="footnote">This tool is for educational screening only and is not a medical diagnosis. Please consult a qualified healthcare professional for interpretation or advice.</div>', unsafe_allow_html=True)