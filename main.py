from pydantic import BaseModel  
import numpy as np
import pandas as pd  
import joblib  
from flask import Flask, request, jsonify

# load the model
model = joblib.load('random_forest_model.pkl')

# Define the database schema with Pydantic
class DonneesEntree(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float   
    SkinThickness: float  
    Insulin: float 
    BMI: float  
    DiabetesPedigreeFunction: float  
    Age: float 

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    "root endpoint"
    return jsonify({"message": "Welcome to Diabetes Prediction API"})

@app.route("/predict", methods=["POST"])
def predict():
    "predict endpoint"
    if not request.json:
        return jsonify({"erreur": "Aucun JSON fourni"}), 400
    
    try:
        data = DonneesEntree(**request.json) # **request.json unpacks the json data into the DonneesEntree class
        data_df = pd.DataFrame([data.model_dump()]) # converts the data to a dictionary

        prediction = model.predict(data_df)
        proba = model.predict_proba(data_df)[:,1]

        res = data.model_dump()
        res["prediction"] = int(prediction[0])
        res["proba_diabetes"] = proba[0]

        return jsonify({"result": res})

    except Exception as e:
        return jsonify({"erreur": str(e)}), 400
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)