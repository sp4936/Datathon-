import streamlit as st
import numpy as np
from joblib import load

st.title('Compliance Classification')

# Load the model using joblib
loaded_model = load('rfc_model.pkl')

# Dictionary mapping labels to categories
label_to_category = {
    0: 'Credit card',
    1: 'Credit reporting',
    2: 'Debt collection',
    3: 'Mortgages and loan',
    4: 'Retail banking'
}

# Function to classify the complaint and return the category
def classify_complaint(complaint_text):
    prediction = loaded_model.predict([complaint_text])[0]
    category = label_to_category[prediction]
    return category

def main():
    st.subheader('Prediction model')

    # Text input for the complaint
    complaint_text = st.text_area("Enter your complaint", "", height = 250)

    if st.button('Classify'):
        if complaint_text:
            category = classify_complaint(complaint_text)
            st.write("Predicted Category:", category)
        else:
            st.error("Please enter a complaint before classifying.")

if __name__ == '__main__':
    main()