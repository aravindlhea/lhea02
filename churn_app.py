import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os


# USER DATABASE (USERNAME + PASSWORD ONLY)

USER_DB = "users.csv"

# Create DB if not exists
if not os.path.exists(USER_DB):
    df = pd.DataFrame(columns=["username", "password"])
    df.to_csv(USER_DB, index=False)


def signup(username, password):
    users = pd.read_csv(USER_DB)

    if username in list(users["username"]):
        return False, "❌ Username already exists!"

    new_user = pd.DataFrame([[username, password]], columns=["username", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_DB, index=False)

    return True, "✅ Account created successfully! Please login using your username and password."



def login(username, password):
    users = pd.read_csv(USER_DB)

    for _, row in users.iterrows():
        if row["username"] == username and row["password"] == password:
            return True

    return False



# SIDEBAR LOGIN / SIGNUP UI
# ------------------------------------
# Session Memory Setup
# ------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None


# SIDEBAR LOGIN UI
st.sidebar.title("🔐 User Authentication")

auth_mode = st.sidebar.radio("Choose", ["Login", "Signup"])


# SIGNUP
if auth_mode == "Signup":
    st.sidebar.subheader("Create Account")
    new_user = st.sidebar.text_input("Username")
    new_pass = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Signup"):
        ok, msg = signup(new_user, new_pass)
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)


# LOGIN
if auth_mode == "Login":
    st.sidebar.subheader("Login")
    user = st.sidebar.text_input("Username")
    pwd = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        ok = login(user, pwd)
        if ok:
            st.session_state.logged_in = True
            st.session_state.username = user
            st.sidebar.success(f"Welcome {user} 👋")
        else:
            st.sidebar.error("❌ Login failed! Please check your username or password.")


# BLOCK APP UNTIL LOGIN
if not st.session_state.logged_in:
    st.warning("⚠ Please login to access the Churn Prediction features.")
    st.stop()


telecom_cust = pd.read_csv('Telco_Customer_Churn.csv')

# Load Model
rf_model = load('random_forest_model.joblib')

# Title
st.title("Customer Churn Prediction App")

home, tab1, tab2 = st.tabs(["🏠 Home", "🔵Customer Random Input", "🟢Customer Full Information"])


# 🏠 HOME PAGE TAB
with home:
    st.title("Customer Churn Prediction App")
    st.subheader("Welcome to the Churn Prediction Home Page 👋")

    st.write("""
    This application predicts **Customer Churn** using:
    - 🔵 **Random Forest Model** (5-input prediction)
    - 🟢 **Logistic Regression Model** (19-input prediction)

    Use the tabs above to test customer data.
    """)

    st.header("📊 Overall Churn Distribution")
    
    # Convert Yes/No → 1/0 only for graph
    churn_data = telecom_cust["Churn"].replace({"Yes": 1, "No": 0})

    labels = ["Stay (0)", "Churn (1)"]
    values = churn_data.value_counts().sort_index().values

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_xlabel("Customer Type")
    ax.set_ylabel("Count")
    ax.set_title("Overall Dataset Churn Distribution")

    st.pyplot(fig)

    st.header("ℹ️ Dataset Info")
    st.write(telecom_cust.head())

    st.success("Use the tabs above to start Predictions!")

# Input Section
st.header("Enter Customer Information")
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=1)
internet_service = st.selectbox("Internet Service", ('DSL', 'Fiber optic', 'No'))
contract = st.selectbox("Contract", ('Month-to-month', 'One year', 'Two year'))
monthly_charges = st.number_input("Monthly Charges", min_value=0, max_value=200, value=50)
total_charges = st.number_input("Total Charges", min_value=0, max_value=10000, value=0)

# Manual Label Encoding
label_mapping = {
    'DSL': 0,
    'Fiber optic': 1,
    'No': 2,
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2,
}
#internet_service = label_mapping[internet_service]
#contract = label_mapping[contract]

# Prediction
if st.button("Predict using Random Forest"):

        inputs = {
            "Tenure": tenure,
            "Internet Service": internet_service,
            "Contract": contract,
            "Monthly Charges": monthly_charges,
            "Total Charges": total_charges
        }

        # Encode ONLY here
        internet_service_enc = label_mapping[internet_service]
        contract_enc = label_mapping[contract]

        rf_pred = rf_model.predict([[tenure, internet_service_enc, contract_enc, monthly_charges, total_charges]])

        st.subheader("📝 Input Summary")
        st.table(pd.DataFrame(inputs.items(), columns=["Feature", "Value"]))

        st.subheader("Prediction Result")
        if rf_pred[0] == 0:
            st.success("This customer is likely to stay.")
        else:
            st.error("This customer is likely to churn.")

        st.subheader("📊 Prediction Probability")

        probs = rf_model.predict_proba([[tenure, internet_service_enc, contract_enc, monthly_charges, total_charges]])[0]

        labels = ['Stay (%)', 'Churn (%)']
        values = [probs[0] * 100, probs[1] * 100]

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%')
        ax.set_title("Customer Churn Probability")
        st.pyplot(fig)




                           #-----------LR---------------


log_model = load('logistic_regression_model.joblib')

st.title("Customer Churn Prediction App")

st.header("Enter Customer Information")

gender = st.selectbox("Gender (Male/Female): ",('Male','Female','Other'))
SeniorCitizen = st.number_input("Senior Citizen (0/1): ",min_value=0,max_value=1,value=1)
Partner = st.selectbox("Partner (Yes/No): ",('Yes','No'))
Dependents = st.selectbox("Dependents (Yes/No): ",('Yes','No'))
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
PhoneService = st.selectbox("Phone Service (Yes/No): ",('Yes','No'))
MultipleLines = st.selectbox("Multiple Lines : ",('Yes','No','No phone service'))
Internetservice = st.selectbox("Internet Service Type", ('DSL', 'Fiber optic', 'No'))
OnlineSecurity = st.selectbox("Online Security (Yes/No): ",('Yes','No'))
OnlineBackup =st.selectbox("Online Backup (Yes/No): ",('Yes','No'))
DeviceProtection = st.selectbox("Device Protection (Yes/No): ",('Yes','No'))
TechSupport = st.selectbox("Tech Support (Yes/No): ",('Yes','No'))
StreamingTV = st.selectbox("Streaming TV (Yes/No): ",('Yes','No'))
StreamingMovies = st.selectbox("Streaming Movies (Yes/No): ",('Yes','No'))
Contract = st.selectbox("Contract type", ('Month-to-month', 'One year', 'Two year'))
PaperlessBilling = st.selectbox("Paperless Billing (Yes/No): ",('Yes','No'))
PaymentMethod = st.selectbox("Payment Method (Electronic check/Mailed check/Bank transfer/Credit card): ",('Electronic check','Mailed check','Bank transfer','Credit card'))
Monthlycharges = st.number_input("Monthly Charge", min_value=0, max_value=200, value=50)
Totalcharges = st.number_input("Total Charge", min_value=0, max_value=10000, value=0)

# Manual Label Encoding
label_mapping = {
    'Male':0,
    'Female':1,
    'Other':2,
    'Yes':0,
    'No':1,
    'No phone service':2,
    'DSL': 0,
    'Fiber optic': 1,
    'No': 2,
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2,
    'Electronic check':0,
    'Mailed check':1,
    'Bank transfer':2,
    'Credit card':3
}
Internetservice = label_mapping[Internetservice]
Contract = label_mapping[Contract]
gender= label_mapping[gender]
Partner= label_mapping[Partner]
Dependents= label_mapping[Dependents]
PhoneService= label_mapping[PhoneService]
MultipleLines= label_mapping[MultipleLines]
OnlineSecurity= label_mapping[OnlineSecurity]
OnlineBackup= label_mapping[OnlineBackup]
DeviceProtection= label_mapping[DeviceProtection]
TechSupport= label_mapping[TechSupport]
StreamingTV= label_mapping[StreamingTV]
StreamingMovies= label_mapping[StreamingMovies]
PaperlessBilling= label_mapping[PaperlessBilling]
PaymentMethod= label_mapping[PaymentMethod]

features = [
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    Internetservice,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    Monthlycharges,
    Totalcharges
]


if st.button("Predict using Logistic Regression"):
    lr_output = log_model.predict([features])   # Now features = 19 values

    st.subheader("Prediction Result")
    if lr_output[0] == 0:
        st.success("This customer is likely to stay.")
    else:
        st.error("This customer is likely to churn.")

    # Prediction Graph
    st.subheader("📊 Prediction Output Graph")

    labels = ['Stay', 'Churn']
    values = [1, 0] if lr_output[0] == 0 else [0, 1]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Prediction")
    ax.set_title("Logistic Regression Prediction Result")
    st.pyplot(fig)
