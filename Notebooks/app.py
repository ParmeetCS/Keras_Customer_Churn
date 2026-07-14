import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

import os

# Get directory of current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.h5")
preprocessor_path = os.path.join(BASE_DIR, "preprocessor.pkl")

# Load model
model = load_model(model_path)

with open(preprocessor_path, "rb") as f:
    preprocessor = pickle.load(f)

st.title("📊 Customer Churn Prediction (IBM Telco Dataset Based)")
st.write("Predict whether a customer is likely to churn based on demographics, services, contract details, and financial metrics.")

with st.form("prediction_form"):
    
    # ----------------------------------------------------
    # Section 1: Demographic Information
    # ----------------------------------------------------
    st.subheader("👤 Demographic Information")
    st.markdown("*General profile of the customer.*")
    col_dem1, col_dem2 = st.columns(2)
    with col_dem1:
        gender = st.selectbox(
            "Gender", 
            ["Male", "Female"],
            help="The customer's gender."
        )
        senior = st.selectbox(
            "Senior Citizen", 
            ["No", "Yes"],
            help="Indicates if the customer is 65 or older."
        )
    with col_dem2:
        partner = st.selectbox(
            "Partner", 
            ["Yes", "No"],
            help="Indicates if the customer has a partner."
        )
        dependents = st.selectbox(
            "Dependents", 
            ["No", "Yes"],
            help="Indicates if the customer lives with any dependents (children, parents, etc.)."
        )
    
    st.markdown("---")
    
    # ----------------------------------------------------
    # Section 2: Services Subscribed
    # ----------------------------------------------------
    st.subheader("📞 Services Subscribed")
    st.markdown("*Services and features the customer has registered for.*")
    col_serv1, col_serv2 = st.columns(2)
    
    with col_serv1:
        phone = st.selectbox(
            "Phone Service", 
            ["Yes", "No"],
            help="Indicates if the customer subscribes to home phone service."
        )
        if phone == "Yes":
            multiple = st.selectbox(
                "Multiple Lines", 
                ["No", "Yes"],
                help="Indicates if the customer subscribes to multiple telephone lines."
            )
        else:
            multiple = "No phone service"
            
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            help="Indicates if the customer subscribes to Internet service with the company."
        )

    with col_serv2:
        if internet == "No":
            security = backup = protection = support = tv = movies = "No internet service"
            st.info("No internet service selected. Internet-related add-on services are disabled.")
        else:
            security = st.selectbox(
                "Online Security", 
                ["No", "Yes"],
                help="Indicates if the customer subscribes to an additional online security service."
            )
            backup = st.selectbox(
                "Online Backup", 
                ["No", "Yes"],
                help="Indicates if the customer subscribes to an additional online backup service."
            )
            protection = st.selectbox(
                "Device Protection", 
                ["No", "Yes"],
                help="Indicates if the customer subscribes to an additional device protection plan."
            )
            support = st.selectbox(
                "Tech Support", 
                ["No", "Yes"],
                help="Indicates if the customer subscribes to an additional tech support plan with reduced wait times."
            )
            tv = st.selectbox(
                "Streaming TV", 
                ["No", "Yes"],
                help="Indicates if the customer uses their internet service to stream TV programming."
            )
            movies = st.selectbox(
                "Streaming Movies", 
                ["No", "Yes"],
                help="Indicates if the customer uses their internet service to stream movies."
            )

    st.markdown("---")
    
    # ----------------------------------------------------
    # Section 3: Contract & Billing Details
    # ----------------------------------------------------
    st.subheader("📄 Contract & Billing")
    st.markdown("*Terms of agreement and payment preferences.*")
    col_con1, col_con2 = st.columns(2)
    
    with col_con1:
        tenure = st.slider(
            "Tenure (Months)", 
            0, 72, 12,
            help="Indicates the total number of months that the customer has been with the company."
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
            help="The contract term type for the customer."
        )
        
    with col_con2:
        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
            help="Indicates if the customer has chosen paperless billing."
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            help="How the customer pays their bill."
        )

    st.markdown("---")

    # ----------------------------------------------------
    # Section 4: Financial & Valuation Metrics
    # ----------------------------------------------------
    st.subheader("💵 Financial & Valuation Metrics")
    st.markdown("*Charges and estimated lifetime value.*")
    col_fin1, col_fin2 = st.columns(2)
    
    with col_fin1:
        monthly = st.number_input(
            "Monthly Charges",
            value=65.0,
            help="The customer’s current monthly charge for all their services."
        )
        total = st.number_input(
            "Total Charges",
            value=780.0,
            help="The customer's total charges calculated up to the end of the quarter."
        )
        
    with col_fin2:
        cltv = st.slider(
            "CLTV",
            2000,
            6500,
            4500,
            help="Customer Lifetime Value. The higher the value, the more valuable the customer."
        )

    predict = st.form_submit_button("Predict")

if predict:

    data = pd.DataFrame([{
        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone,
        "Multiple Lines": multiple,
        "Internet Service": internet,
        "Online Security": security,
        "Online Backup": backup,
        "Device Protection": protection,
        "Tech Support": support,
        "Streaming TV": tv,
        "Streaming Movies": movies,
        "Contract": contract,
        "Paperless Billing": paperless,
        "Payment Method": payment,
        "Monthly Charges": monthly,
        "Total Charges": total,
        "CLTV": cltv
    }])

    X = preprocessor.transform(data)

    probability = model.predict(X)[0][0]

    st.metric("Churn Probability", f"{probability*100:.2f}%")

    if probability >= 0.5:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    with st.expander("Input Data"):
        st.dataframe(data)