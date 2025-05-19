from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

router = APIRouter()

with open("model/stroke_model_2.pkl", "rb") as f:
    model = pickle.load(f)

# To see available methods for the loaded model object:
print(dir(model))


class PredictionModel(BaseModel):
    gender: str
    age:  Optional[float]
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level:  Optional[float]
    bmi: Optional[float]
    smoking_status: str

# TO run on swagger just use this url  ' http://127.0.0.1:8000/docs'

@router.post('/predict')
def prediction_stroke(inp: PredictionModel):
    try:
        # Handle missing or None values with defaults
        gender_value = 1 if getattr(inp, "gender", "male").lower() == 'male' else 0
        ever_married_value = 1 if getattr(inp, "ever_married", "yes").lower() == 'yes' else 0

        # One-hot encoding for work_type
        work_types = ["Never_worked", "Private", "Self-employed", "children"]
        work_type_values = [0, 0, 0, 0]
        work_type_input = getattr(inp, "work_type", "").replace(" ", "_").replace("-", "_")
        # Map input to match training feature names
        work_type_map = {
            "Never_worked": "Never_worked",
            "Private": "Private",
            "Self_employed": "Self-employed",
            "Self-employed": "Self-employed",
            "children": "children"
        }
        work_type_key = work_type_map.get(getattr(inp, "work_type", ""), "")
        if work_type_key in work_types:
            work_type_values[work_types.index(work_type_key)] = 1

        # One-hot encoding for Residence_type
        residence_type_urban = 1 if getattr(inp, "Residence_type", "").lower() == "urban" else 0

        # One-hot encoding for smoking_status
        smoking_statuses = ["formerly smoked", "never smoked", "smokes"]
        smoking_status_values = [0, 0, 0]
        smoking_status_input = getattr(inp, "smoking_status", "").lower()
        for idx, status in enumerate(smoking_statuses):
            if smoking_status_input == status:
                smoking_status_values[idx] = 1

        input_features = [
            gender_value,
            getattr(inp, "age", 0.0),
            getattr(inp, "hypertension", 0),
            getattr(inp, "heart_disease", 0),
            ever_married_value,
            getattr(inp, "avg_glucose_level", 0.0),
            getattr(inp, "bmi", 0.0),
            *work_type_values,
            residence_type_urban,
            *smoking_status_values
        ]

        feature_names = [
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "avg_glucose_level",
            "bmi",
            "work_type_Never_worked",
            "work_type_Private",
            "work_type_Self-employed",
            "work_type_children",
            "Residence_type_Urban",
            "smoking_status_formerly smoked",
            "smoking_status_never smoked",
            "smoking_status_smokes"
        ]

        # Convert input_features to DataFrame to match model's expected input
        input_df = pd.DataFrame([input_features], columns=feature_names)

        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]
        stroke_risk_percent = round(prediction_proba * 100, 2)

        personalized_suggestion = None
        if stroke_risk_percent >= 50.0:
            suggestions = []
            if getattr(inp, "hypertension", 0) == 1:
                suggestions.append("Manage your blood pressure with regular checkups and medication.")
            if getattr(inp, "heart_disease", 0) == 1:
                suggestions.append("Consult your cardiologist for heart health management.")
            if getattr(inp, "bmi", 0.0) > 30:
                suggestions.append("Consider a healthy diet and regular exercise to reduce BMI.")
            if smoking_status_values[2] == 1:
                suggestions.append("Quitting smoking can significantly reduce your stroke risk.")
            if getattr(inp, "avg_glucose_level", 0.0) > 140:
                suggestions.append("Monitor and control your blood glucose levels.")
            if not suggestions:
                suggestions.append("Consult your healthcare provider for a personalized stroke prevention plan.")
            personalized_suggestion = suggestions
        else:
            personalized_suggestion = ["Congratulations!, Your stroke risk is not high. Maintain a healthy lifestyle and regular checkups."]

        feature_importances = None
        if hasattr(model, "feature_importances_"):
            feature_names = [
                "gender",
                "age",
                "hypertension",
                "heart_disease",
                "ever_married",
                "avg_glucose_level",
                "bmi",
                "work_type_Never_worked",
                "work_type_Private",
                "work_type_Self-employed",
                "work_type_children",
                "Residence_type_Urban",
                "smoking_status_formerly smoked",
                "smoking_status_never smoked",
                "smoking_status_smokes"
            ]
            importances = list(model.feature_importances_)
            feature_importances = [
                {"feature": name, "importance": round(imp * 100, 2) }
                for name, imp in zip(feature_names,  importances)
            ]
            feature_importances.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "prediction": int(prediction),
            "stroke_chance_percent": stroke_risk_percent,
            "personalized_suggestion": personalized_suggestion,
            "feature_importances": feature_importances
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
