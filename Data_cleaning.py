from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load dataset
data = pd.read_csv(r'C:\Portfolio\archive (2)\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle missing values in TotalCharges
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)

# Convert Churn to binary
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Encode binary categorical columns manually
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    data[col] = data[col].map({'Yes': 1, 'No': 0})

# Drop 'No internet service' and 'No phone service' categories (redundant)
data.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)

# # Label encode multi-category columns to avoid multiple new columns
# multi_category_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
#                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
#                        'Contract', 'PaymentMethod']
#
# le = LabelEncoder()
# for col in multi_category_cols:
#     data[col] = le.fit_transform(data[col])

# Drop customerID (not useful)
data.drop(columns=['customerID'], inplace=True)

# Save cleaned data
data.to_csv(r"C:\Portfolio\archive (2)\cleaned_churn_data.csv", index=False)
print("Cleaned data saved without unnecessary columns.")
