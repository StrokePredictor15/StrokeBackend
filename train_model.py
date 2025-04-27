import pandas as pd
import pickle
import shap
from sklearn.ensemble import RandomForestClassifier
import os

os.makedirs('model',exist_ok=True)

df=pd.read_csv(r"C:/Users/Admin/Desktop/stroke_dataset/healthcare-dataset-stroke-data.csv")

x=df[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level',
      'bmi','smoking_status']]

y=df['stroke']

X=pd.get_dummies(x)

model=RandomForestClassifier(n_estimators=100,random_state=45)
model.fit(X,y)

with open('model/stroke_model.pkl','wb') as f:
    pickle.dump(model,f)

explainer=shap.TreeExplainer(model)

with open('model/shap_explainer.pkl','wb') as f:
    pickle.dump(explainer,f)

print('model and explainer saved successfully')
