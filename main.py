import pandas as pd
data= pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# LOAD DATA

print(data.shape)
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = data.drop('Churn', axis= 1)
Y = data['Churn']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 42, stratify= Y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# TRAIN AND EVALUATE THE MODEL

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression(random_state=42)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))