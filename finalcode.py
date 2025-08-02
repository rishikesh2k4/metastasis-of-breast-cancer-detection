import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('/content/jvtm241.csv')

# Check class distribution
print("Class distribution:")
print(data['Tumour_Stage'].value_counts())

# Encode target column
le = LabelEncoder()
data['Tumour_Stage'] = le.fit_transform(data['Tumour_Stage'])

# Split into features and target
X = data.drop('Tumour_Stage', axis=1)
y = data['Tumour_Stage']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train initial Random Forest classifier (baseline)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Baseline Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Feature importance plot
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feat_importances, y=feat_importances.index)
plt.title('Feature Importances from Random Forest')
plt.show()

# Hyperparameter tuning for Random Forest using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best Random Forest params: {grid_search.best_params_}")
best_rf = grid_search.best_estimator_

# Evaluate best tuned Random Forest
y_pred_best_rf = best_rf.predict(X_test)
print("Tuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_best_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_best_rf, target_names=le.classes_))

# Cross-validation score for best Random Forest
cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy')
print(f"Cross-validated accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train and evaluate XGBoost as alternative model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Classification Report:\n", classification_report(y_test, y_pred_xgb, target_names=le.classes_))

# Predict for new sample
new_samples = [
    {'Age':63, 'Protein1':-0.523, 'Protein2':1.5526, 'Protein3':-0.1652, 'Protein4':0.276, 'ER status':1, 'PR status':1, 'HER2 status':0},
    {'Age':38, 'Protein1':-0.268450, 'Protein2':0.19515, 'Protein3':-1.024700, 'Protein4':0.101720, 'ER status':1, 'PR status':1, 'HER2 status':0}
]

for sample in new_samples:
    new_df = pd.DataFrame([sample])
    pred_idx_rf = best_rf.predict(new_df)[0]
    pred_label_rf = le.inverse_transform([pred_idx_rf])[0]

    pred_idx_xgb = xgb.predict(new_df)[0]
    pred_label_xgb = le.inverse_transform([pred_idx_xgb])[0]

    print(f"Sample: {sample}")
    print(f"Predicted Tumour Stage by Tuned RF: {pred_label_rf}")
    print(f"Predicted Tumour Stage by XGBoost: {pred_label_xgb}")
    print("-" * 40)
