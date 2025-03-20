➡import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time

# Load and clean data
data = pd.read_csv(r'C:\Portfolio\archive (2)\WA_Fn-UseC_-Telco-Customer-Churn.csv')
data = data.drop('customerID', axis=1)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)

# Features and target
features = ['MonthlyCharges', 'TotalCharges', 'tenure', 'Contract', 'OnlineSecurity', 'TechSupport', 'PaymentMethod']
target = 'Churn'
le = LabelEncoder()
for col in ['Contract', 'OnlineSecurity', 'TechSupport', 'PaymentMethod', 'Churn']:
    data[col] = le.fit_transform(data[col])

X = data[features]
y = data[target]

# Scale features for model training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=features)

# Train-Validation split (80% train, 20% validation, no test set)
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)
print("Train-Validation Split - Training Rows:", X_train.shape[0], "Validation Rows:", X_val.shape[0])

# GridSearchCV for SVM (on training set)
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.1],
    'kernel': ['rbf']
}
grid_search = GridSearchCV(SVC(class_weight={0: 1, 1: 3}, probability=True, random_state=42),
                          param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
print("\nTraining SVM with Cross-Validation...")
start_time_svm = time.time()
grid_search.fit(X_train, y_train)
end_time_svm = time.time()
print("SVM Training Time (CV):",
      "{:02d}:{:02d}".format(int((end_time_svm - start_time_svm) // 60), int((end_time_svm - start_time_svm) % 60)))

# Initialize models
svm_model = grid_search.best_estimator_
rf_model = RandomForestClassifier(random_state=42, class_weight={0: 1, 1: 3})

# Train SVM (for TV evaluation)
print("\nTraining SVM for Train-Validation...")
start_time_svm_tv = time.time()
svm_model.fit(X_train, y_train)
end_time_svm_tv = time.time()
print("SVM Training Time (TV):",
      "{:02d}:{:02d}".format(int((end_time_svm_tv - start_time_svm_tv) // 60), int((end_time_svm_tv - start_time_svm_tv) % 60)))

# Train Random Forest
print("\nTraining Random Forest...")
start_time_rf = time.time()
rf_model.fit(X_train, y_train)
end_time_rf = time.time()
print("Random Forest Training Time:",
      "{:02d}:{:02d}".format(int((end_time_rf - start_time_rf) // 60), int((end_time_rf - start_time_rf) % 60)))

# SVM Cross-Validation (on training set, threshold 0.35)
svm_cv_prob = svm_model.predict_proba(X_train)[:, 1]
svm_cv_pred = (svm_cv_prob >= 0.35).astype(int)
svm_cv_accuracy = round(accuracy_score(y_train, svm_cv_pred), 2)
svm_cv_churn_recall = round(recall_score(y_train, svm_cv_pred, pos_label=1), 2)
svm_cv_macro_recall = round(recall_score(y_train, svm_cv_pred, average='macro'), 2)
svm_cv_report = classification_report(y_train, svm_cv_pred, target_names=['No Churn', 'Churn'], output_dict=True)

# SVM Train-Validation (on validation set, threshold 0.35)
svm_val_prob = svm_model.predict_proba(X_val)[:, 1]
svm_val_pred = (svm_val_prob >= 0.35).astype(int)
svm_val_accuracy = round(accuracy_score(y_val, svm_val_pred), 2)
svm_val_churn_recall = round(recall_score(y_val, svm_val_pred, pos_label=1), 2)
svm_val_macro_recall = round(recall_score(y_val, svm_val_pred, average='macro'), 2)
svm_val_report = classification_report(y_val, svm_val_pred, target_names=['No Churn', 'Churn'], output_dict=True)

# Random Forest Cross-Validation (on training set)
rf_cv_pred = rf_model.predict(X_train)
rf_cv_accuracy = round(accuracy_score(y_train, rf_cv_pred), 2)
rf_cv_churn_recall = round(recall_score(y_train, rf_cv_pred, pos_label=1), 2)
rf_cv_macro_recall = round(recall_score(y_train, rf_cv_pred, average='macro'), 2)
rf_cv_report = classification_report(y_train, rf_cv_pred, target_names=['No Churn', 'Churn'], output_dict=True)

# Random Forest Train-Validation (on validation set)
rf_val_pred = rf_model.predict(X_val)
rf_val_accuracy = round(accuracy_score(y_val, rf_val_pred), 2)
rf_val_churn_recall = round(recall_score(y_val, rf_val_pred, pos_label=1), 2)
rf_val_macro_recall = round(recall_score(y_val, rf_val_pred, average='macro'), 2)
rf_val_report = classification_report(y_val, rf_val_pred, target_names=['No Churn', 'Churn'], output_dict=True)

# Churn Distribution (Validation Set)
churn_count = sum(y_val == 1)
no_churn_count = sum(y_val == 0)
total_count = len(y_val)
churn_dist_df = pd.DataFrame({
    "Churn": [f"{churn_count} ({churn_count/total_count*100:.1f}%)"],
    "No Churn": [f"{no_churn_count} ({no_churn_count/total_count*100:.1f}%)"],
    "Total": [total_count]
})
print("\nValidation Set Churn Distribution:")
print(churn_dist_df.to_string(index=False))

# SVM Model Metrics (Pivoted: Metrics as columns, Phases as rows)
svm_metrics_data = {
    "Phase": ["CV", "TV"],
    "Accuracy": [svm_cv_accuracy, svm_val_accuracy],
    "Churn Recall": [svm_cv_churn_recall, svm_val_churn_recall],
    "Macro Avg Recall": [svm_cv_macro_recall, svm_val_macro_recall]
}
svm_metrics_df = pd.DataFrame(svm_metrics_data)
print("\nSVM Model Metrics:")
print(svm_metrics_df.to_string(index=False))

# Random Forest Model Metrics (Pivoted: Metrics as columns, Phases as rows)
rf_metrics_data = {
    "Phase": ["CV", "TV"],
    "Accuracy": [rf_cv_accuracy, rf_val_accuracy],
    "Churn Recall": [rf_cv_churn_recall, rf_val_churn_recall],
    "Macro Avg Recall": [rf_cv_macro_recall, rf_val_macro_recall]
}
rf_metrics_df = pd.DataFrame(rf_metrics_data)
print("\nRandom Forest Model Metrics:")
print(rf_metrics_df.to_string(index=False))

# Classification Report (Pivoted for SVM: Metrics as columns, Phases as rows)
svm_class_report_data = {
    "Phase": ["CV", "TV"],
    "Precision (No Churn)": [
        round(svm_cv_report['No Churn']['precision'], 2),
        round(svm_val_report['No Churn']['precision'], 2)
    ],
    "Precision (Churn)": [
        round(svm_cv_report['Churn']['precision'], 2),
        round(svm_val_report['Churn']['precision'], 2)
    ],
    "Recall (No Churn)": [
        round(svm_cv_report['No Churn']['recall'], 2),
        round(svm_val_report['No Churn']['recall'], 2)
    ],
    "Recall (Churn)": [
        round(svm_cv_report['Churn']['recall'], 2),
        round(svm_val_report['Churn']['recall'], 2)
    ],
    "F1-Score (No Churn)": [
        round(svm_cv_report['No Churn']['f1-score'], 2),
        round(svm_val_report['No Churn']['f1-score'], 2)
    ],
    "F1-Score (Churn)": [
        round(svm_cv_report['Churn']['f1-score'], 2),
        round(svm_val_report['Churn']['f1-score'], 2)
    ],
    "Support (No Churn)": [
        int(svm_cv_report['No Churn']['support']),
        int(svm_val_report['No Churn']['support'])
    ],
    "Support (Churn)": [
        int(svm_cv_report['Churn']['support']),
        int(svm_val_report['Churn']['support'])
    ]
}
svm_class_report_df = pd.DataFrame(svm_class_report_data)
print("\nSVM Classification Report:")
print(svm_class_report_df.to_string(index=False))

# Classification Report (Pivoted for RF: Metrics as columns, Phases as rows)
rf_class_report_data = {
    "Phase": ["CV", "TV"],
    "Precision (No Churn)": [
        round(rf_cv_report['No Churn']['precision'], 2),
        round(rf_val_report['No Churn']['precision'], 2)
    ],
    "Precision (Churn)": [
        round(rf_cv_report['Churn']['precision'], 2),
        round(rf_val_report['Churn']['precision'], 2)
    ],
    "Recall (No Churn)": [
        round(rf_cv_report['No Churn']['recall'], 2),
        round(rf_val_report['No Churn']['recall'], 2)
    ],
    "Recall (Churn)": [
        round(rf_cv_report['Churn']['recall'], 2),
        round(rf_val_report['Churn']['recall'], 2)
    ],
    "F1-Score (No Churn)": [
        round(rf_cv_report['No Churn']['f1-score'], 2),
        round(rf_val_report['No Churn']['f1-score'], 2)
    ],
    "F1-Score (Churn)": [
        round(rf_cv_report['Churn']['f1-score'], 2),
        round(rf_val_report['Churn']['f1-score'], 2)
    ],
    "Support (No Churn)": [
        int(rf_cv_report['No Churn']['support']),
        int(rf_val_report['No Churn']['support'])
    ],
    "Support (Churn)": [
        int(rf_cv_report['Churn']['support']),
        int(rf_val_report['Churn']['support'])
    ]
}
rf_class_report_df = pd.DataFrame(rf_class_report_data)
print("\nRandom Forest Classification Report:")
print(rf_class_report_df.to_string(index=False))

# Feature Importance (Random Forest)
print("\nFeature Importance (Random Forest):")
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': [round(imp, 3) for imp in rf_model.feature_importances_]
}).sort_values(by='Importance', ascending=False)
print(feature_importance.to_string(index=False))

# Export All Results as CSVs (Columnar format)
churn_dist_df.to_csv(r'C:\Portfolio\archive (2)\telco_churn_distribution.csv', index=False)
svm_metrics_df.to_csv(r'C:\Portfolio\archive (2)\telco_svm_metrics_all.csv', index=False)
rf_metrics_df.to_csv(r'C:\Portfolio\archive (2)\telco_rf_metrics_all.csv', index=False)
svm_class_report_df.to_csv(r'C:\Portfolio\archive (2)\telco_svm_classification_report.csv', index=False)
rf_class_report_df.to_csv(r'C:\Portfolio\archive (2)\telco_rf_classification_report.csv', index=False)
feature_importance.to_csv(r'C:\Portfolio\archive (2)\telco_feature_importance.csv', index=False)

print("\nResults Exported to CSVs (All Columnar):")
print("- telco_churn_distribution.csv (Validation Set Churn Distribution)")
print("- telco_svm_metrics_all.csv (SVM Model Metrics: Metrics as Columns)")
print("- telco_rf_metrics_all.csv (Random Forest Model Metrics: Metrics as Columns)")
print("- telco_svm_classification_report.csv (SVM Classification Report: Metrics as Columns)")
print("- telco_rf_classification_report.csv (Random Forest Classification Report: Metrics as Columns)")
print("- telco_feature_importance.csv (Feature Importance)")