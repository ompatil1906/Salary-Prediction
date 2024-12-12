import streamlit as st
import joblib
import numpy as np


st.title('Salary Prediction App')


st.divider()


st.write('With this app, you can get estimations for the salaries of the company employee.')

years=st.number_input('Enter the years at Company',value=1,step=1,min_value=0)
jobrate=st.number_input('Enter the job rate',value=3.5,step=0.5,min_value=0.0)

X = np.array([[years, jobrate]])

model=joblib.load("linearmodel.pkl")

st.divider()

predict=st.button("Press the button for salary prediction")

st.divider()

if predict:

    st.balloons()

    prediction=model.predict(X)[0]

    st.write(f'Salary Prediction is {prediction:,.2f}')

else:
    'Please press the button for app to make to prediction'