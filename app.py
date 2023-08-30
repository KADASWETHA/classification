import streamlit as st
import os
import pandas as pd
import joblib as jb

heading_style = '''
<div style="color:red;" align='center'>
<h1>Loan Amount Prediction System</h1>
</div>
'''
def return_df(Gender,
    Married,
    Dependents,
    Education,
    Self_Employed,
	ApplicantIncome,
	CoapplicantIncome,
	LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    Property_Area):
    kbn={
    'Gender':[Gender],
    'Married':[Married],
    'Dependents':[Dependents],
    'Education':[Education],
    'Self_Employed':[Self_Employed],
	'ApplicantIncome':[ApplicantIncome],
	'CoapplicantIncome':[CoapplicantIncome],
    'LoanAmount':[LoanAmount],
    'Loan_Amount_Term':[Loan_Amount_Term],
    'Credit_History':[Credit_History],
    'Property_Area':[Property_Area]
    }   
    final_df=pd.DataFrame(kbn)
    return final_df


def base_model():
    bmodel=jb.load(os.path.join('finalised_rf_model.pkl'))
    return bmodel

st.markdown(heading_style, unsafe_allow_html=True)
Gender=st.selectbox('Select your gender',['Male','Female'])
Married=st.selectbox('Married ?',['Yes','No'])
Dependents=st.slider('Dependents',0,3,0)
Education=st.selectbox('Ed',['Graduate','Not Graduate'])
Self_Employed=st.selectbox('self',['Yes','No'])
ApplicantIncome=st.number_input('appki', min_value=0)
CoapplicantIncome=st.number_input('coap', min_value=0)
LoanAmount=st.number_input('laon am',min_value=0)
Loan_Amount_Term=st.number_input('lmt',min_value=0)
Credit_History=st.slider('creh',0,1,0)
Property_Area=st.selectbox('property',['Urban','Rural','Semiurban'])
df=return_df(Gender,
    Married,
    Dependents,
    Education,
    Self_Employed,
	ApplicantIncome,
    CoapplicantIncome,
	LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    Property_Area)
if st.button('Submit'):
	model=base_model()
	preds=model.predict(df)
	predictions=preds[0]
	if predictions=='Y':
		st.write('Approved')
	elif predictions=='N':
		st.write('Not Approved')
