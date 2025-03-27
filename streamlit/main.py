import streamlit as st
import pandas as pd
import requests

# Configuration
API_URL = 'https://mlops-app-675097283493.europe-west9.run.app/predict'
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

def predict_diabetes(data): 
    try: 
        converted_data = {
            "Pregnancies": float(data["Pregnancies"]),
            "Glucose": float(data["Glucose"]),
            "BloodPressure": float(data["BloodPressure"]),
            "SkinThickness": float(data["SkinThickness"]),
            "Insulin": float(data["Insulin"]),
            "BMI": float(data["BMI"]),
            "DiabetesPedigreeFunction": float(data["DiabetesPedigreeFunction"]),
            "Age": float(data["Age"])
        }
        response = requests.post(API_URL, json=converted_data, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def display_results(result): 
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", 
                "Diabetic 🔴" if result.get('prediction', 0) == 1 else "Healthy 🟢")
    
    with col2:
        prob = result.get('proba_diabetes', 0)
        st.metric("Probability", f"{prob:.1%}")
        st.progress(prob)
     
    st.text("Detailed Results:")
    st.json(result)

def main():
    st.title("🩺 Diabetes Prediction Dashboard")
    st.markdown(" This tool predicts diabetes risk using the Pima Indians Diabetes Dataset parameters. All measurements should be collected under standardized clinical conditions.")
     
    input_method = st.radio("Choose input method:", 
                          ["Single Patient", "Batch CSV Upload"])
    
    if input_method == "Single Patient":
        with st.form("single_patient_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                pregnancies = st.number_input(
                    "Number of Pregnancies", 
                    min_value=0, step=1,
                    help="Total number of pregnancies"
                )
                glucose = st.number_input(
                    "Glucose Concentration (mg/dL)", 
                    min_value=0,
                    help="Plasma glucose concentration from 2-hour oral glucose tolerance test"
                )
                bp = st.number_input(
                    "Diastolic Blood Pressure (mm Hg)", 
                    min_value=0,
                    help="Diastolic blood pressure measurement"
                )
                skin = st.number_input(
                    "Triceps Skinfold Thickness (mm)", 
                    min_value=0,
                    help="Thickness of skin fold at triceps"
                )
            
            with col2:
                insulin = st.number_input(
                    "2-Hour Serum Insulin (μU/mL)", 
                    min_value=0,
                    help="Insulin level 2 hours after glucose challenge"
                )
                bmi = st.number_input(
                    "Body Mass Index (kg/m²)", 
                    min_value=0.0, step=0.1,
                    help="Weight in kilograms divided by height in meters squared"
                )
                pedigree = st.number_input(
                    "Diabetes Pedigree Function", 
                    min_value=0.0, step=0.01,
                    help="Genetic risk score based on family history"
                )
                age = st.number_input(
                    "Age (years)", 
                    min_value=0, step=1,
                    help="Patient's age at time of examination"
                )
            submitted = st.form_submit_button("Predict")
            
            if submitted:
                patient_data = {
                    "Pregnancies": pregnancies,
                    "Glucose": glucose,
                    "BloodPressure": bp,
                    "SkinThickness": skin,
                    "Insulin": insulin,
                    "BMI": bmi,
                    "DiabetesPedigreeFunction": pedigree,
                    "Age": age
                }
                
                result = predict_diabetes(patient_data)
                if result:
                    display_results(result.get('result', {}))
    
    else:  # CSV Upload
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file:
            batch_data = pd.read_csv(uploaded_file)
            st.write("Preview:")
            st.dataframe(batch_data.head())
            
            if st.button("Predict Batch"):
                with st.spinner("Processing..."):
                    results = []
                    for _, row in batch_data.iterrows():
                        res = predict_diabetes(row.to_dict())
                        if res and 'result' in res:
                            results.append(res['result'])
                    
                    if results:
                        st.success("Predictions Complete!")
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df)
                        st.download_button(
                            "Download Results",
                            data=results_df.to_csv(index=False),
                            file_name="predictions.csv"
                        )

if __name__ == "__main__":
    main()