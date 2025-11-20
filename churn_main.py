import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('Telco_Customer_Churn.csv')

# Drop ID column
df = df.drop("customerID", axis=1)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(0, inplace=True)

# Label Encoding for ALL categorical columns
le = LabelEncoder()

cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ------------------------ RANDOM FOREST (5 FEATURE MODEL) ------------------------

rf_features = ['tenure', 'InternetService', 'Contract', 'MonthlyCharges', 'TotalCharges']
X_rf = df[rf_features]
y_rf = df['Churn']

rf_model = RandomForestClassifier(n_estimators=100, random_state=101)
rf_model.fit(X_rf, y_rf)

dump(rf_model, 'random_forest_model.joblib')

print("Random Forest model saved successfully!")

# --------------------- LOGISTIC REGRESSION (19 FEATURE MODEL) ---------------------

lr_features = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'MonthlyCharges',
    'TotalCharges'
]

X_lr = df[lr_features]
y_lr = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)

# Logistic Regression model
log_model = LogisticRegression(max_iter=300)
log_model.fit(X_train, y_train)

dump(log_model, 'logistic_regression_model.joblib')

print("Logistic Regression model saved successfully!")
