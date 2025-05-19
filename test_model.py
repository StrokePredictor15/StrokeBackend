import pickle
import pandas as pd


def fix_test_sample(test_sample_dict, required_features):
    """
    Ensures the test sample dict is converted to a DataFrame with all required features,
    filling missing features with 0 and ordering columns as required.
    """
    # Create DataFrame from dict
    sample_df = pd.DataFrame([test_sample_dict])

    # Add missing columns with 0
    missing_cols = [col for col in required_features if col not in sample_df.columns]
    for col in missing_cols:
        sample_df[col] = 0

    # Remove extra columns not in required_features
    sample_df = sample_df[[col for col in required_features]]

    # Ensure correct dtypes (float for numeric, int for binary/categorical)
    sample_df = sample_df.astype(float)

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
with open("model/stroke_model_2.pkl", "rb") as f:
    loaded_model = pickle.load(f)


# Create sample **in One-Hot Encoded style**
sample_data = {
    'gender': 1,
    'age': 99,
    'hypertension': 1,
    'heart_disease': 0,
    'ever_married': 1,
    'avg_glucose_level': 500.69,
    'bmi': 336.6,
    'work_type_Private': 1,
    'Residence_type_Urban': 1,
    'smoking_status_formerly smoked': 1,
    'smoking_status_never smoked':1,
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
#feat_importances.plot(kind='bar')
#plt.title('Feature Importance')
#plt.show()

# 🧠 LIME Explanation (Instance-Based)
import lime
import lime.lime_tabular
import numpy as np

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_test_sample),
    feature_names=required_features,  # <-- fix: required_features is a list, not .columns
    class_names=['No Stroke', 'Stroke'],
    mode='classification'
)


# Explain one prediction
i = 0
exp = explainer.explain_instance(X_test_sample.iloc[i], loaded_model.predict_proba)

# Instead of show_in_notebook (which requires IPython display), print as text and plot as figure
print(exp.as_list())
fig = exp.as_pyplot_figure()
plt.title('LIME Feature Contributions')
plt.tight_layout()
plt.show()
# exp.show_in_notebook(show_table=True)  # Remove or comment out this line

import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder
from xgboost import XGBClassifier  # Ensure XGBClassifier is imported
from sklearn.ensemble import RandomForestClassifier as RandomForest

# Load data from the specified CSV file
#data_path = r"C:\Users\molaw\Downloads\dataset\healthcare-dataset-stroke-data.csv"
data = X_test_sample  #pd.read_csv(data_path)

# Preprocessing: Encode categorical columns using LabelEncoder
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}

# Only encode columns that exist in the DataFrame
for col in categorical_columns:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

# When selecting features for X, only use columns that exist in data
feature_cols = [col for col in [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
] if col in data.columns]
X = data[feature_cols]
if 'stroke' in data.columns:
    y = data['stroke']
else:
    y = pd.Series([0]*len(data))  # Dummy target if not present

# Splitting the dataset
if len(X) > 1 and len(y) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    # Not enough data to split, use all as both train and test (for single-row test sample)
    X_train, X_test, y_train, y_test = X, X, y, y

def shap_plot(base_model, instance):
    # Use X_train, X_test, y_train, y_test from outer scope
    global X_train, X_test, y_train, y_test

    if len(X_test) == 0:
        raise ValueError("The test set is empty. Adjust the dataset or test_size parameter.")
    if instance < 0 or instance >= len(X_test):
        raise IndexError(f"Instance index {instance} is out of bounds. Valid range: 0 to {len(X_test) - 1}")

    # Fit the model
    model = base_model.fit(X_train, y_train)

    # SHAP explanation
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    shap.initjs()
    print(f"Sample number: {instance}")

    preds = model.predict(X_test)
    probability = model.predict_proba(X_test)

    print(f"Actual class: {y_test.iloc[instance]}")
    print(f"Predicted class: {preds[instance]}")

    # SHAP Plots
    shap.plots.bar(shap_values)   # global explanation
    shap.plots.bar(shap_values[instance], max_display=13)   # local explanation

    # force plot
    return shap.plots.force(shap_values[instance])

# Only run shap_plot if there is enough data to fit a model
if len(X_train) > 1 and len(y_train.unique()) > 1:
    shap_plot(RandomForest(), 0)  # Use a valid instance index within the range of X_test
else:
    print("Not enough data or target class variation to fit a model and run SHAP explanation. "
          "You need a real dataset (not just a single test sample) with at least two classes in the target column.")
