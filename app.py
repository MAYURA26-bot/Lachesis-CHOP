import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("obesity_model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Child Obesity Risk Predictor", layout="centered")
st.title("🧒 Childhood Obesity Risk Prediction")
st.markdown("Enter the child's details below:")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", min_value=5, max_value=18, value=10)
height = st.number_input("Height (in meters)", min_value=1.0, max_value=2.0, step=0.01)
weight = st.number_input("Weight (in kg)", min_value=10.0, max_value=100.0, step=0.5)
family_history = st.selectbox("Family history of overweight?", ["yes", "no"])
favc = st.selectbox("Consumes high-calorie food frequently (FAVC)?", ["yes", "no"])
fcvc = st.slider("Frequency of vegetable consumption (1 = rarely, 3 = daily)", 1, 3, 2)
ncp = st.slider("Number of main meals per day (NCP)", 1, 4, 3)
caec = st.selectbox("Snacking between meals (CAEC)", ["Sometimes", "Frequently", "Always"])
ch2o = st.slider("Daily water intake (CH2O)", 1, 3, 2)
scc = st.selectbox("Monitors calorie intake (SCC)?", ["yes", "no"])
faf = st.slider("Physical activity frequency (FAF)", 0, 3, 1)
tue = st.slider("Time spent using tech daily (TUE)", 0, 3, 2)
mtrans = st.selectbox("Primary transport mode (MTRANS)", ["Bike", "MotorBike", "Public Transportation", "Walking"])

# Create a DataFrame with the exact column names used in training
input_df = pd.DataFrame([{
    "age": age,
    "height": height,
    "weight": weight,
    "family_history_with_overweight": 1 if family_history == "yes" else 0,
    "favc": 1 if favc == "yes" else 0,
    "fcvc": fcvc,
    "ncp": ncp,
    "ch2o": ch2o,
    "scc": 1 if scc == "yes" else 0,
    "faf": faf,
    "tue": tue,

    # Gender (bool column, gender_Male is True for male, False for female)
    "gender_Male": True if gender == "Male" else False,

    # CAEC one-hot (bools)
    "caec_Always": caec == "Always",
    "caec_Frequently": caec == "Frequently",
    "caec_Sometimes": caec == "Sometimes",

    # MTRANS one-hot (booleans)
    "mtrans_Bike": mtrans == "Bike",
    "mtrans_Motorbike": mtrans == "Motorbike",
    "mtrans_Public_Transportation": mtrans == "Public Transportation",
    "mtrans_Walking": mtrans == "Walking"
}])

# Encode categorical fields using saved label encoders
#for col in input_df.columns:
#    if col in encoders:
#        input_df[col] = encoders[col].transform(input_df[col])

# Predict on button click
if st.button("Predict Obesity Risk"):
    prediction = model.predict(input_df)[0]

    # Create reverse map
    inverse_label_map = {v: k for k, v in encoders['nobeyesdad'].items()}
    
    predicted_label = inverse_label_map[int(prediction)]
    st.success(f"🏷️ Predicted Obesity Category: **{predicted_label}**")


# Show accuracy note
st.markdown("---")
st.markdown("📊 **Model Accuracy:** 88.5% (Prototype)")
st.caption("Note: Final accuracy may change once model is trained on child-specific dataset.")