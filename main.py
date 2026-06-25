import pandas as pd
data= pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# LOAD DATA

int(data.shape)
print(data.head())
print(data.info())

# DATA CLEANING

print("Missing values:")
print(data.isnull().sum())

print("\nDuplicate rows:")
print(data.duplicated().sum())

print(data["Churn"].value_counts())

data.drop("customerID", axis=1, inplace=True)

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

print(data["TotalCharges"].dtype)
print(data["TotalCharges"].isnull().sum())
data.dropna(inplace=True)

print(data.shape)
print(data.dtypes)

# ENCODES CATEGORICAL

from sklearn.preprocessing import LabelEncoder

binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService','PaperlessBilling', 'Churn']
le = LabelEncoder()

for col in binary_cols:
    data[col] = le.fit_transform(data[col])

data = pd.get_dummies(data, columns=['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod'])

# SPLIT AND SCALE DATA

