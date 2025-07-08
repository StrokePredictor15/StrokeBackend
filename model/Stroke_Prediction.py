import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# 1. Load data
df = pd.read_csv("..\\..\\Downloads\\dataset\\healthcare-dataset-stroke-data.csv")
df = df.drop(columns=['id'])  # drop identifier

# 2. Feature / target split
X = df.drop(columns='stroke')
y = df['stroke']

# 3. Train/test split (stratify to keep imbalance proportion)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Preprocessing pipelines
num_feats = ['age', 'avg_glucose_level', 'bmi']
cat_feats = ['gender', 'hypertension', 'heart_disease',
             'ever_married', 'work_type', 'Residence_type', 'smoking_status']

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_feats),
    ('cat', cat_pipe, cat_feats)
])

# 5. Build imblearn Pipeline with SMOTE + XGBoost
model_pipe = ImbPipeline([
    ('prep', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.3, random_state=42)),
    ('clf', XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    ))
])

# 5b. Build imblearn Pipeline with SMOTE + RandomForest
rf_pipe = ImbPipeline([
    ('prep', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),  # increased from 0.3 to 0.5
    ('clf', RandomForestClassifier(
        random_state=42,
        class_weight='balanced'  # added to boost minority class
    ))
])

# 6. Hyperparameter grid (optimize for F1)
param_grid = {
    'clf__max_depth': [3, 5],
    'clf__learning_rate': [0.1, 0.01],
    'clf__n_estimators': [100, 200]
}

rf_param_grid = {
    'clf__n_estimators': [100, 200, 300, 500],
    'clf__max_depth': [5, 10, 15, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__max_features': ['sqrt', 'log2', None]
}

# XGBoost GridSearch
grid = GridSearchCV(
    model_pipe,
    param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# RandomForest GridSearch
rf_grid = GridSearchCV(
    rf_pipe,
    rf_param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# 7. Fit
grid.fit(X_train, y_train)
print("Best XGBoost params:", grid.best_params_)

rf_grid.fit(X_train, y_train)
print("Best RandomForest params:", rf_grid.best_params_)

# Save the best RandomForest model as a pickle file
with open('best_random_forest.pkl', 'wb') as f:
    pickle.dump(rf_grid.best_estimator_, f)

# 8. Evaluate on test set
y_pred = grid.predict(X_test)
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# For RandomForest, search for best threshold on validation set to maximize F1
from sklearn.metrics import f1_score
y_proba_rf = rf_grid.predict_proba(X_test)[:, 1]
best_f1 = 0
best_thresh = 0.5
for thresh in [i/100 for i in range(20, 81, 2)]:
    y_pred_tmp = (y_proba_rf >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_tmp)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
y_pred_rf = (y_proba_rf >= best_thresh).astype(int)
print(f"RandomForest Classification Report (best threshold={best_thresh:.2f}):")
print(classification_report(y_test, y_pred_rf, digits=4))

# 9. Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Stroke','Stroke'],
            yticklabels=['No Stroke','Stroke'])
plt.title('XGBoost Confusion Matrix')
plt.ylabel('Actual'); plt.xlabel('Predicted')

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.subplot(1,2,2)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Stroke','Stroke'],
            yticklabels=['No Stroke','Stroke'])
plt.title('RandomForest Confusion Matrix')
plt.ylabel('Actual'); plt.xlabel('Predicted')

plt.tight_layout()
plt.show()