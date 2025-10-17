# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 19:16:09 2025

@author: Benjamin
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load your trained model
loaded_Model = pickle.load(open('C:/Users/Benjamin/Desktop/Peter/Loan_status_data.sav', 'rb'))

# Prediction function
def Loan_status_prediction(Gender, Married, Education, Self_Employed,
                           ApplicantIncome, CoapplicantIncome, LoanAmount, Property_Area):
    
    # Create DataFrame using input parameters
    Loan = pd.DataFrame([{
        'Gender': Gender,
        'Married': Married,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Property_Area': Property_Area
    }])
    
    # Predict using the loaded model
    predicted_Loan = loaded_Model.predict(Loan)
    
    return predicted_Loan[0]


# Streamlit app main function
def main():
    st.title("🏦 LOAN STATUS PREDICTION APP")

    # Input fields for all features
    Gender = st.text_input('Gender (e.g., 1 for Male, 0 for Female)')
    Married = st.text_input('Married (1 = Yes, 0 = No)')
    Education = st.text_input('Education (1 = Graduate, 0 = Not Graduate)')
    Self_Employed = st.text_input('Self Employed (1 = Yes, 0 = No)')
    ApplicantIncome = st.text_input('Applicant Income (e.g., 3000)')
    CoapplicantIncome = st.text_input('Coapplicant Income (e.g., 0.0)')
    LoanAmount = st.text_input('Loan Amount (e.g., 66)')
    Property_Area = st.text_input('Property Area (0 = Rural, 1 = Semiurban, 2 = Urban)')

    if st.button('🔍 Predict Loan Status'):
        try:
            # Convert all inputs to numeric types
            Gender = int(Gender)
            Married = int(Married)
            Education = int(Education)
            Self_Employed = int(Self_Employed)
            ApplicantIncome = float(ApplicantIncome)
            CoapplicantIncome = float(CoapplicantIncome)
            LoanAmount = float(LoanAmount)
            Property_Area = int(Property_Area)

            # Call prediction function
            result = Loan_status_prediction(Gender, Married, Education, Self_Employed,
                                            ApplicantIncome, CoapplicantIncome, LoanAmount, Property_Area)

            # Display prediction result
            if result == 1:
                st.success("The predicted loan status is: **LOAN APPROVED**")
            else:
                st.error("The predicted loan status is: **LOAN REJECTED**")

        except ValueError:
            st.error("Please enter valid numeric values for all inputs.")


if __name__ == '__main__':
    main()
