import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit_shap as st_shap

from utils import ThresholdClassifier 

# app2.py uses file from Gdrive
def get_feature_names(preprocess, X_original):
    try:
        raw = preprocess.get_feature_names_out()
        return [n.split("__")[-1] for n in raw]
    except Exception:
        return list(X_original.columns)

def extract_shap_values(raw, class_idx=1, model_output_prob=True):
    if model_output_prob:
        if isinstance(raw, list):
            return raw[class_idx]
        if raw.ndim == 3:
            return raw[:, :, class_idx]
        return raw
    else:
        if isinstance(raw, list) and len(raw) == 2:
            return raw[class_idx]
        elif raw.ndim == 3:
            return raw[:, :, class_idx]
        return raw

def extract_expected_value(ev, class_idx=1, model_output_prob=True):
    if model_output_prob:
        if isinstance(ev, (list, np.ndarray)):
            return float(ev[class_idx])
        return float(ev)
    else:
        if isinstance(ev, (list, np.ndarray)):
            return float(ev[class_idx])
        return float(ev)

# page config
st.set_page_config(
    page_title="XAI Cardiovascular Disease Analysis",
    layout="wide"
)

# css
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 8px;
        padding-left: 20px;
        padding-right: 20px;
        font-size: 16px;
    }
    .big-font {
        font-size:22px !important;
        font-weight: bold;
    }
    /* Style for metrics to make them pop */
    div[data-testid="metric-container"] {
        background-color: #1e2127;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

url = "https://drive.google.com/file/d/16s3oVKKqvprcDakxUig6t9B1zekD3aBS/view?usp=sharing"

if not os.path.exists("framingham_CatBoost_model.joblib"):
    gdown.download(url, "framingham_CatBoost_model.joblib", quiet=False)


# load model & data
@st.cache_resource
def load_framingham_model():
    wrapper = joblib.load("framingham_CatBoost_model.joblib")
    X_train, X_test, y_train, y_test = joblib.load(
        "./framingham/framingham_train_test_data.joblib"
    )
    pipeline = wrapper.model.estimator
    preprocess = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    return wrapper, pipeline, preprocess, model, X_train, X_test, y_test

@st.cache_resource
def load_cleveland_model():
    wrapper = joblib.load("./cleveland/cleveland_CatBoost_model.joblib")
    X_train, X_test, y_train, y_test = joblib.load(
        "./cleveland/cleveland_train_test_data.joblib"
    )
    pipeline = wrapper.model.estimator
    preprocess = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    return wrapper, pipeline, preprocess, model, X_train, X_test, y_test

(
    framingham_wrapper, framingham_pipeline, framingham_preprocess,
    framingham_model, framingham_X_train, framingham_X_test, framingham_y_test
) = load_framingham_model()

(
    cleveland_wrapper, cleveland_pipeline, cleveland_preprocess,
    cleveland_model, cleveland_X_train, cleveland_X_test, cleveland_y_test
) = load_cleveland_model()

# navigation sidebar + input
st.sidebar.title("Navigation")
mode = st.sidebar.radio(
    "Choose Analysis Type",
    ["Prognosis (Framingham)", "Diagnosis (Cleveland)"]
)

st.sidebar.markdown("---")
input_mode = st.sidebar.radio(
    "Input Method",
    ["Manual Entry", "Test Data Explorer"]
)
st.sidebar.markdown("---")

def framingham_input_ui():
    st.sidebar.markdown("## Patient Information")
    male = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: ["Female", "Male"][x])
    age = st.sidebar.slider("Age", 20, 90, 45)
    education = st.sidebar.selectbox("Education", [1, 2, 3, 4], format_func=lambda x: ["0-11 Years", "High School Diploma, GED", "Some College, Vocational School", "College (BS, BA) degree or more"][x-1])
    currentSmoker = st.sidebar.selectbox("Current Smoker", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    cigsPerDay = st.sidebar.slider("Cigarettes Per Day", 0, 70, 0)
    BPMeds = st.sidebar.selectbox("Having Blood Pressure Medication", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    prevalentStroke = st.sidebar.selectbox("Prevalent Stroke", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    prevalentHyp = st.sidebar.selectbox("Prevalent Hypertension", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    diabetes = st.sidebar.selectbox("History of Diabetes", [0, 1], format_func=lambda x: ["No", "Yes"][x])
    totChol = st.sidebar.slider("Total Cholesterol", 100, 600, 200)
    sysBP = st.sidebar.slider("Systolic Blood Pressure (mmHg)", 80, 250, 120)
    diaBP = st.sidebar.slider("Diastolic Blood Pressure (mmHg)", 40, 150, 80)
    BMI = st.sidebar.slider("Body Mass Index (kg/m^2)", 10.0, 60.0, 25.0)
    heartRate = st.sidebar.slider("Heart Rate (bpm)", 40, 150, 75)
    glucose = st.sidebar.slider("Glucose (mg/dl)", 40, 400, 90)

    data = {
        "male": male, "age": age, "education": education, "currentSmoker": currentSmoker,
        "cigsPerDay": cigsPerDay, "BPMeds": BPMeds, "prevalentStroke": prevalentStroke,
        "prevalentHyp": prevalentHyp, "diabetes": diabetes, "totChol": totChol,
        "sysBP": sysBP, "diaBP": diaBP, "BMI": BMI, "heartRate": heartRate, "glucose": glucose,
    }
    return pd.DataFrame([data])

def cleveland_input_ui():
    st.sidebar.markdown("## Patient Information")
    age = st.sidebar.slider("Age", 20, 90, 55)
    sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: ["Female", "Male"][x])
    cp = st.sidebar.selectbox("Chest Pain Type", [1, 2, 3, 4], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x-1])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mmHg)", 80, 250, 130)
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    restecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x])
    thalach = st.sidebar.slider("Maximum Heart Rate (bpm)", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest", -3.0, 6.0, 1.0, step=0.1)
    slope = st.sidebar.selectbox("Slope", [1, 2, 3], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x-1])
    ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Exercise Thallium Scintigraphic Defects", [3, 6, 7], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][{3:0, 6:1, 7:2}[x]])
    
    data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
        "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal,
    }
    return pd.DataFrame([data])

# prog/diag option + input option (manual/test data)
actual_y = None

if mode == "Prognosis (Framingham)":
    wrapper = framingham_wrapper
    pipeline = framingham_pipeline
    preprocess = framingham_preprocess
    model = framingham_model
    sample_df = framingham_X_train.copy()
    dataset_name_str = "Framingham"
    X_train_bg = framingham_X_train
    beeswarm_path = "./framingham/shap_framingham_global_beeswarm.png"
    bar_path = "./framingham/shap_framingham_global_bar.png"

    if input_mode == "Manual Entry":
        input_df = framingham_input_ui()
    else:
        st.sidebar.markdown("## Test Data Selection")
        max_idx = len(framingham_X_test) - 1
        patient_idx = st.sidebar.slider("Select Patient Index", 0, max_idx, 0)
        input_df = framingham_X_test.iloc[[patient_idx]].copy()
        actual_y = framingham_y_test.iloc[patient_idx]

else: # Cleveland
    wrapper = cleveland_wrapper
    pipeline = cleveland_pipeline
    preprocess = cleveland_preprocess
    model = cleveland_model
    sample_df = cleveland_X_train.copy()
    dataset_name_str = "Cleveland"
    X_train_bg = cleveland_X_train
    beeswarm_path = "./cleveland/shap_cleveland_global_beeswarm.png"
    bar_path = "./cleveland/shap_cleveland_global_bar.png"

    if input_mode == "Manual Entry":
        input_df = cleveland_input_ui()
    else:
        st.sidebar.markdown("## Test Data Selection")
        max_idx = len(cleveland_X_test) - 1
        patient_idx = st.sidebar.slider("Select Patient Index", 0, max_idx, 0)
        input_df = cleveland_X_test.iloc[[patient_idx]].copy()
        actual_y = cleveland_y_test.iloc[patient_idx]


# main tabs
result_tab, global_tab, sample_tab = st.tabs(
    ["Prediction Result", "Global SHAP Analysis", "Sample Dataset"]
)

#local SHAP interpretation
def generate_shap_interpretation(explanation, input_df, probability, threshold):
    shap_values = explanation.values
    feature_names = explanation.feature_names
    feature_data = explanation.data

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values,
        "value": feature_data
    })
    shap_df["abs_shap"] = np.abs(shap_df["shap_value"])
    shap_df = shap_df.sort_values("abs_shap", ascending=False)
    top_features = shap_df.head(4)

    increasing = []
    decreasing = []
    
    FEATURE_LABELS = {
        "sysBP": "Systolic Blood Pressure", "diaBP": "Diastolic Blood Pressure",
        "totChol": "Total Cholesterol", "BMI": "Body Mass Index",
        "cigsPerDay": "Cigarettes Per Day", "thalach": "Maximum Heart Rate",
        "oldpeak": "ST Depression", "age": "Age", "cp": "Chest Pain Type",
        "exang": "Exercise Induced Angina", "slope": "ST Slope",
        "ca": "Number of Major Vessels Colored by Fluoroscopy",
        "thal": "Exercise Thallium Scintigraphic Defects",
        "education": "Education Level", "currentSmoker": "Current Smoker",
        "BPMeds": "Blood Pressure Medication", "prevalentStroke": "Prevalent Stroke",
        "prevalentHyp": "Prevalent Hypertension", "diabetes": "Diabetes History",
        "heartRate": "Heart Rate", "glucose": "Glucose Level",
        "restecg": "Resting ECG", "fbs": "Fasting Blood Sugar > 120 mg/dl",
        "male": "Gender (0: Female, 1: Male)", "sex": "Gender (0: Female, 1: Male)"
    }
    
    for _, row in top_features.iterrows():
        feature = FEATURE_LABELS.get(row["feature"], row["feature"])
        shap_val = row["shap_value"]
        try:
            patient_value = input_df.iloc[0][row["feature"]]
        except:
            patient_value = row["value"]

        text = f"{feature} (Shifted probability by {shap_val:+.3f})"
        if shap_val > 0:
            increasing.append(text)
        else:
            decreasing.append(text)

    explanation_text = ""
    if probability >= threshold:
        explanation_text += (
            f"The model predicts a **HIGH risk** of Cardiovascular Disease "
            f"with a probability of {probability:.1%} (Threshold > {threshold:.1%}). "
        )
    else:
        explanation_text += (
            f"The model predicts a **LOW risk** of Cardiovascular Disease "
            f"with a probability of {probability:.1%} (Threshold < {threshold:.1%}). "
        )

    if increasing:
        explanation_text += "\n\nThe following factors contributed MOST toward increasing risk:\n"
        for item in increasing:
            explanation_text += f"- {item}\n"

    if decreasing:
        explanation_text += "\nThe following factors helped reduce the predicted risk:\n"
        for item in decreasing:
            explanation_text += f"- {item}\n"

    strongest = shap_df.iloc[0]
    direction = "increased" if strongest["shap_value"] > 0 else "decreased"
    explanation_text += (
        f"\nOverall, the strongest single influence on this prediction "
        f"was **{FEATURE_LABELS.get(strongest['feature'], strongest['feature'])}**, which {direction} the probability significantly."
    )
    return explanation_text

# result tab
with result_tab:
    st.markdown('<p class="big-font">Prediction Result</p>', unsafe_allow_html=True)
    analyze = st.button("Analyze Patient Data")

    if analyze:
        probability = pipeline.predict_proba(input_df)[0][1]
        threshold = wrapper.threshold
        prediction = int(probability >= threshold)

        if input_mode == "Test Data Explorer":
            st.markdown("### Ground Truth vs. Prediction")
            
            actual_text = "High Risk (CHD)" if actual_y == 1 else "Low Risk"
            pred_text = "High Risk" if prediction == 1 else "Low Risk"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Actual Ground Truth", value=actual_text)
            with col2:
                # Color code the metric implicitly by matching/mismatching
                delta_color = "normal" if prediction == actual_y else "inverse"
                match_status = "Correct Match" if prediction == actual_y else "Misclassification"
                st.metric(label="Model Prediction", value=pred_text, delta=match_status, delta_color=delta_color)
            with col3:
                st.metric(label="Confidence (Probability)", value=f"{probability:.1%}", delta=f"Threshold: {threshold:.1%}", delta_color="off")
            
            st.markdown("---")
            
        else: # manual output
            if prediction == 1:
                st.error(f"High Risk Detected — Probability: {probability:.1%} (Decision Threshold: {threshold:.1%})")
            else:
                st.success(f"Low Risk Detected — Probability: {probability:.1%} (Decision Threshold: {threshold:.1%})")

        st.subheader("Patient Data")
        st.dataframe(input_df)


        st.subheader("Local SHAP Explanation with Waterfall Plot")

        transformed_input = preprocess.transform(input_df)
        if hasattr(transformed_input, 'values'):
            transformed_input = transformed_input.values
            
        feature_names = get_feature_names(preprocess, input_df)

        transformed_train = preprocess.transform(X_train_bg)
        if hasattr(transformed_train, 'values'):
            transformed_train = transformed_train.values
            
        background = shap.kmeans(transformed_train, 50)
        
        explainer = shap.TreeExplainer(
            model,
            data=background.data,
            model_output='probability',
            feature_perturbation='interventional'
        )
        
        raw_shap_values = explainer.shap_values(transformed_input)
        sv = extract_shap_values(raw_shap_values, class_idx=1, model_output_prob=True)
        ev = extract_expected_value(explainer.expected_value, class_idx=1, model_output_prob=True)

        explanation = shap.Explanation(
            values=sv[0],
            base_values=ev,
            data=transformed_input[0],
            feature_names=feature_names,
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig, clear_figure=True)

        interpretation = generate_shap_interpretation(
            explanation, input_df, probability, threshold
        )

        st.markdown("### Interpretation")
        st.info(interpretation)

        st.subheader("Force Plot")
        force_plot = shap.force_plot(
            ev,
            sv[0],
            transformed_input[0],
            feature_names=feature_names,
        )
        st_shap.st_shap(force_plot, height=300)

# global SHAP
with global_tab:
    st.markdown('<p class="big-font">Global SHAP Analysis</p>', unsafe_allow_html=True)
    st.info("These plots depict the average impact of features on the model's output in the *Probability Scale*.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Global Beeswarm Plot")
        try:
            st.image(beeswarm_path, use_container_width=True)
        except:
            st.warning("Global beeswarm image not found. Please ensure paths are correct.")

    with col2:
        st.subheader("Global Feature Importance")
        try:
            st.image(bar_path, use_container_width=True)
        except:
            st.warning("Global bar plot image not found. Please ensure paths are correct.")

# sample data tab
with sample_tab:
    st.markdown('<p class="big-font">Sample Dataset</p>', unsafe_allow_html=True)
    st.dataframe(sample_df.head(100), use_container_width=True)
    csv = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Sample Dataset", csv, "sample_dataset.csv", "text/csv")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center;'>
        <h4>Explainable AI for Cardiovascular Disease Analysis</h4>
        <p>This system provides:</p>
        <ul style='display: inline-block; text-align: left;'>
            <li>Cardiovascular prognosis using the Framingham dataset</li>
            <li>Heart disease diagnosis using the Cleveland dataset</li>
            <li>Threshold-based machine learning predictions</li>
            <li>Interactive SHAP explainability visualizations (Probability Scale)</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)