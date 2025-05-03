from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

router = APIRouter()


with open("model/stroke_model_1.pkl", "rb") as f:
    model = pickle.load(f)



class PredictionModel(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    avg_glucose_level: float
    bmi: float
    work_type_Never_worked: int
    work_type_Private: int
    work_type_Self_employed: int
    work_type_children: int
    Residence_type_Urban: int
    smoking_status_formerly_smoked: int
    smoking_status_never_smoked: int
    smoking_status_smokes: int

# TO run on swagger just use this url  ' http://127.0.0.1:8000/docs'

@router.post('/predict')
def prediction_stroke(inp: PredictionModel):
    try:

        gender_value = 1 if inp.gender.lower() == 'male' else 0
        ever_married_value = 1 if inp.ever_married.lower() == 'yes' else 0


        input_features = [
            gender_value,
            inp.age,
            inp.hypertension,
            inp.heart_disease,
            ever_married_value,
            inp.avg_glucose_level,
            inp.bmi,
            inp.work_type_Never_worked,
            inp.work_type_Private,
            inp.work_type_Self_employed,
            inp.work_type_children,
            inp.Residence_type_Urban,
            inp.smoking_status_formerly_smoked,
            inp.smoking_status_never_smoked,
            inp.smoking_status_smokes
        ]


        input_array = np.array([input_features])


        prediction = model.predict(input_array)[0]
        prediction_proba = model.predict_proba(input_array)[0][1]


        return {
            "prediction": int(prediction),
            "stroke_chance_percent": round(prediction_proba * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
