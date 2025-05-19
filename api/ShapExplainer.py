import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder
from xgboost import XGBClassifier  # Ensure XGBClassifier is imported

# Load data from the specified CSV file
data_path = r"C:\Users\molaw\Downloads\dataset\healthcare-dataset-stroke-data.csv"
data = pd.read_csv(data_path)

# Preprocessing: Encode categorical columns using LabelEncoder
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  # Encode and handle NaN as string
    label_encoders[col] = le  # Store the encoder for potential inverse transformation

# Assuming the dataset has columns 'feature1', 'feature2', and 'target'
# Update these column names based on the actual dataset structure
X = data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']]  # Replace with actual feature column names
y = data['stroke']  # Replace with the actual target column name

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def shap_plot(base_model, instance):
    """
    Description: SHAP Plot for feature importance, with local and global explanation.
    Created By: Fahim Muntasir
    Date: 7/12/23
    
    base_model: Algorithm used for prediction. Tree based algorithms are preferred.
    instance: Any random instance from the test set.
    Return: Barplot and Force plot for local and global explanation
    """
    if len(X_test) == 0:
        raise ValueError("The test set is empty. Adjust the dataset or test_size parameter.")
    if instance < 0 or instance >= len(X_test):
        raise IndexError(f"Instance index {instance} is out of bounds. Valid range: 0 to {len(X_test) - 1}")

    # Enable categorical support in XGBoost
    model = base_model.fit(X_train, y_train)

    explainer = shap.Explainer(model, X_train)   # this is the shap part
    shap_values = explainer(X_test)  # calculating feature importance score

    shap.initjs()           # this is for the plots to work
    print(f"Sample number: {instance}")

    preds = model.predict(X_test)
    probability = model.predict_proba(X_test)

    print(f"Actual class: {y_test.iloc[instance]}")
    print(f"Predicted class: {preds[instance]}")

    # SHAP Plots
    shap.plots.bar(shap_values)   # global expalanation
    shap.plots.bar(shap_values[instance], max_display=13)   # local explanation

    # force plot
    return shap.plots.force(shap_values[instance])

shap_plot(XGBClassifier(), 0)  # Use a valid instance index within the range of X_test
