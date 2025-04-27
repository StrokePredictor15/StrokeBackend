import pickle


import pandas as pd


def fix_test_sample(test_sample_dict, required_features):
    # Convert to DataFrame
    sample_df = pd.DataFrame([test_sample_dict])

    # Add missing columns with 0
    for feature in required_features:
        if feature not in sample_df.columns:
            sample_df[feature] = 0

    # Remove extra columns
    sample_df = sample_df[required_features]

    return sample_df

required_features = [
 'gender',
 'age',
 'hypertension',
 'heart_disease',
 'ever_married',
 'avg_glucose_level',
 'bmi',
 'work_type_Never_worked',
 'work_type_Private',
 'work_type_Self-employed',
 'work_type_children',
 'Residence_type_Urban',
 'smoking_status_formerly smoked',
 'smoking_status_never smoked',
 'smoking_status_smokes'
]


# Load the trained model
with open("model/stroke_model_1.pkl", "rb") as f:
    loaded_model = pickle.load(f)


# Create sample **in One-Hot Encoded style**
sample_data = {
    'gender': 1,
    'age': 65,
    'hypertension': 1,
    'heart_disease': 0,
    'ever_married': 1,
    'avg_glucose_level': 228.69,
    'bmi': 36.6,
    'work_type_Private': 1,
    'Residence_type_Urban': 1,
    'smoking_status_formerly smoked': 1,
    'smoking_status_never smoked':0,
    'smoking_status_smokes':1

}

X_test_sample = fix_test_sample(sample_data, required_features)

# Now safe to predict
y_pred = loaded_model.predict(X_test_sample)

print("Predicted stroke (1) / no-stroke (0):", y_pred[0])


# Predict probabilities
y_pred_proba = loaded_model.predict_proba(X_test_sample)

# y_pred_proba is like: [[no-stroke probability, stroke probability]]
print("Predicted probabilities:", y_pred_proba)

# Get stroke chance (index 1)
stroke_probability = y_pred_proba[0][1]

print(f"Chance of stroke: {stroke_probability * 100:.2f}%")

import matplotlib.pyplot as plt

# Get feature importances
importances = loaded_model.feature_importances_

# Create a dataframe
feat_importances = pd.Series(importances, index=required_features)
feat_importances = feat_importances.sort_values(ascending=False)

# Plot
feat_importances.plot(kind='bar')
plt.title('Feature Importance')
plt.show()

