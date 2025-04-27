import pickle

with open('model/stroke_model.pkl','rb') as f:
    model=pickle.load(f)

with open('model/shap_explainer.pkl','rb') as f:
    explainer=pickle.load(f)

print('model and explainer load successfully')

import pandas as pd

sample_inp= pd.DataFrame([{
    'age': 67,
    'hypertension': 0,
    'heart_disease': 1,
    'avg_glucose_level': 228.69,
    'bmi': 36.6,
    'gender_Female': 1,
    'gender_Male': 0,
    'ever_married_Yes': 1,
    'work_type_Private': 1,
    'Residence_type_Urban': 1,
    'smoking_status_formerly smoked': 1
}],index=[0])

for col in model.feature_names_in_:
    if col not in sample_inp.columns:
        sample_inp[col]=0

sample_inp=sample_inp[model.feature_names_in_]

prediction=model.predict(sample_inp)

print("Prediction (0 = No Stroke, 1 = Stroke):", prediction[0])


