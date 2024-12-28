import streamlit as st
import pickle
import numpy as np
import os


def load_model():
    current_dir = os.path.dirname(__file__) 
    model_path = os.path.join(current_dir, 'best_model.pkl')  
    with open(model_path, 'rb') as model:
        data = pickle.load(model)
    return data
   
data = load_model()

classifier = data['model']
product_le = data['product']
marital_status_le = data['marital']
designation_le = data['designation']



def show_classifer_page():
    st.set_page_config(page_title='Holiday Package Prediction', layout='centered')
    
    st.markdown(
        """
        <style>
            .main {text-align: center;}
            div.stButton > button {background-color: #4CAF50; color: white; border-radius: 10px; width: 100%; height: 50px;}
            div.stButton > button:hover {background-color: #45a049;}
            .stSelectbox {margin-bottom: 20px;}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Title and Description
    st.title('Holiday Package Prediction')
    st.markdown(
        """
        ### Welcome to our Holiday Package Prediction Tool!
        Please provide the following details to predict the best holiday package for you.
        """
    )
    
    # Form Layout
    with st.form(key='prediction_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            product_choice = st.selectbox('Select Product Type', ('Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'Premium'))
            marital_status_choice = st.selectbox('Marital Status', ('Single', 'Married', 'Divorced', 'Unmarried'))
            age = st.slider('Age', 18, 60)
            
        with col2:
            designation_choice = st.selectbox('Designation', ('Executive', 'Senior Manager', 'AVP', 'VP', 'Manager'))
            passport_choice = st.selectbox('Passport Status', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
            income = st.slider('Income (SGD)', 0, 1000000)
        
        num_followups = st.slider('Number of Follow-ups', 0, 10)
        
        submit_button = st.form_submit_button(label='Predict')
    
    # Prediction Logic
    if submit_button:
        X = np.array([[product_choice, marital_status_choice, designation_choice, passport_choice, num_followups, age, income]])
        X[:, 0] = product_le.transform([X[:, 0]])
        X[:, 1] = marital_status_le.transform([X[:, 1]])
        X[:, 2] = designation_le.transform([X[:, 2]])
        X = X.astype(float)

        prediction = classifier.predict(X)
        result = 'buy' if prediction == 1 else 'not buy'

        st.write(f'Based on the details you provided, the customer will {result} the holiday package.')


show_classifer_page()





