import streamlit as st
import pandas as pd
import joblib

# Load the trained model
# @st.cache_resource ensures the model is only loaded once
@st.cache_resource
def load_model():
    return joblib.load('titanic_model.pkl')

model = load_model()

# App Title and Description
st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger details below to see if they would have survived the Titanic disaster.")

# Create input fields for the user
st.sidebar.header("Passenger Features")

# Inputs matching our training data features
pclass = st.sidebar.selectbox("Ticket Class (Pclass)", [1, 2, 3], help="1 = 1st, 2 = 2nd, 3 = 3rd")
sex = st.sidebar.radio("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 0.0, 100.0, 25.0)
fare = st.sidebar.slider("Fare (£)", 0.0, 500.0, 32.0)

# Convert sex back to the numerical format the model expects
sex_numeric = 0 if sex == "Male" else 1

# Create a dataframe for the prediction
input_data = pd.DataFrame({
    'pclass': [pclass],
    'sex': [sex_numeric],
    'age': [age],
    'fare': [fare]
})

# Display the user's input
st.subheader("Passenger Profile")
st.write(input_data)

# Prediction Button
if st.button("Predict Survival"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    st.subheader("Result:")
    if prediction[0] == 1:
        st.success(f"**Survived!** (Probability: {prediction_proba[0][1]*100:.2f}%)")
        st.balloons()
    else:
        st.error(f"**Did not survive.** (Probability: {prediction_proba[0][0]*100:.2f}%)")