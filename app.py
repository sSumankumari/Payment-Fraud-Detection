import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved XGBoost model
model = joblib.load('xgboost_fraud_detection_model.joblib')

# Define the prediction function
def predict_fraud(data):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]
    return prediction, probability

# Set the page configuration, including title and layout
st.set_page_config(
    page_title="Payment Fraud Detection System",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for better styling
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content h4 {
        color: #007bff;
    }
    .main .block-container {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #007bff;
        color: #ffffff;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with detailed instructions
st.sidebar.header("App Instructions")
st.sidebar.write("""
#### Follow the steps below to use the Payment Fraud Detection System:

1. **Input Transaction Details:**
   - Enter the step number.
   - Select the transaction type.
   - Input the transaction amount.
   - Provide the old and new balance details for both origin and destination accounts.

2. **Click the 'Predict' Button:**
   - The system will predict whether the transaction is fraudulent.

3. **Review Prediction Result:**
   - See the predicted result and probability.
""")

# Collecting user inputs
st.title("💳 Payment Fraud Detection System")

st.markdown("#### Input Transaction Details")

def user_input_features():
    step = st.number_input("Step", min_value=0, step=1)

    # Dropdown for transaction type, one-hot encoded
    transaction_types = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    type = st.selectbox("Type", transaction_types)

    # Create one-hot encoded feature columns
    type_features = {t: 0 for t in transaction_types}
    type_features[type] = 1

    amount = st.number_input("Amount", min_value=0.0)

    old_balance_origin = st.number_input("Old Balance Origin", min_value=0.0)
    new_balance_origin = st.number_input("New Balance Origin", min_value=0.0)
    old_balance_destination = st.number_input("Old Balance Destination", min_value=0.0)
    new_balance_destination = st.number_input("New Balance Destination", min_value=0.0)
    isFlaggedFraud = st.selectbox("Is Flagged Fraud", ["No", "Yes"]) == "Yes"

    # Calculate derived features
    balance_diff = new_balance_origin - old_balance_origin
    amount_to_balance_ratio = amount / (old_balance_origin + 1)  # Adding 1 to avoid division by zero

    # Prepare the data dictionary with full words and one-hot encoding
    data = {
        'step': step,
        'type_CASH_IN': type_features['CASH_IN'],
        'type_CASH_OUT': type_features['CASH_OUT'],
        'type_DEBIT': type_features['DEBIT'],
        'type_PAYMENT': type_features['PAYMENT'],
        'type_TRANSFER': type_features['TRANSFER'],
        'amount': amount,
        'old_balance_origin': old_balance_origin,
        'new_balance_origin': new_balance_origin,
        'balance_diff': balance_diff,
        'amount_to_balance_ratio': amount_to_balance_ratio,
        'old_balance_destination': old_balance_destination,
        'new_balance_destination': new_balance_destination,
        'isFlaggedFraud': int(isFlaggedFraud)
    }
    return data

# Define the correct order of features as expected by the model
feature_order = [
    'step',
    'type_CASH_IN',
    'type_CASH_OUT',
    'type_DEBIT',
    'type_PAYMENT',
    'type_TRANSFER',
    'amount',
    'old_balance_origin',
    'new_balance_origin',
    'balance_diff',
    'amount_to_balance_ratio',
    'old_balance_destination',
    'new_balance_destination',
    'isFlaggedFraud'
]

# Get user input
input_data = user_input_features()

st.header("Transaction Details")
# Ensure the DataFrame is in the correct column order
st.write(pd.DataFrame([dict(zip(feature_order, [input_data[feature] for feature in feature_order]))]))

if st.button("🔍 Predict"):
    prediction, probability = predict_fraud(list(input_data.values()))
    st.header("Prediction Result")
    if probability[0] >= 0.5:
        st.error(f"This transaction is predicted to be fraudulent with a probability of {probability[0]:.2f}.")
    else:
        st.success(f"This transaction is predicted to be non-fraudulent with a probability of {probability[0]:.2f}.")

# Sidebar with model information
st.sidebar.header("Model Information")
st.sidebar.write("### Model Details")
st.sidebar.write(
    "The model used is an XGBoost classifier trained to detect fraudulent transactions based on various transaction features.")

st.sidebar.write("### Feature Information")
st.sidebar.write("""
- **Step**: Represents the time step in the dataset.
- **Type**: The type of transaction (one-hot encoded).
- **Amount**: The transaction amount.
- **Old Balance Origin**: The original balance before the transaction.
- **New Balance Origin**: The new balance after the transaction.
- **Balance Diff**: The difference between new and old balance origin.
- **Amount to Balance Ratio**: The ratio of amount to the old balance origin.
- **Old Balance Destination**: The balance of the destination account before the transaction.
- **New Balance Destination**: The balance of the destination account after the transaction.
- **Is Flagged Fraud**: Indicates if the transaction was flagged as fraudulent.
""")

st.sidebar.write("### Contact")
st.sidebar.write("For any questions or feedback, please connect with me on [LinkedIn](https://www.linkedin.com/in/sumaninsights/).")
