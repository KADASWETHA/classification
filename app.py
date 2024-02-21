import streamlit as st
import os
import pandas as pd
import joblib as jb
glowing_text_style = '''
    <style>
        .glowing-text {
            font-family: 'Arial Black', sans-serif;
            font-size: 33px;
            text-align: center;
            animation: glowing 2s infinite;
        }
        
        @keyframes glowing {
            0% { color: #FF9933; } /* Saffron color */
            10% { color: #FFD700; } /* Gold color */
            20% { color: #FF1493; } /* Deep Pink */
            30% { color: #00FF00; } /* Lime Green */
            40% { color: #FF4500; } /* Orange Red */
            50% { color: #9400D3; } /* Dark Violet */
            60% { color: #00BFFF; } /* Deep Sky Blue */
            70% { color: #FF69B4; } /* Hot Pink */
            80% { color: #ADFF2F; } /* Green Yellow */
            90% { color: #1E90FF; } /* Dodger Blue */
            100% { color: #FF9933; } /* Saffron color */
        }
    </style>
'''


st.markdown(glowing_text_style, unsafe_allow_html=True)
st.markdown(f'<p class="glowing-text">Loan Amount Prediction</p>', unsafe_allow_html=True)

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


Gender=st.selectbox('Select your gender',['Male','Female'])
Married=st.selectbox('Are you Married ?',['Yes','No'])
Dependents=st.slider('Count of Dependents',0,3,0)
Education=st.selectbox('Educational status',['Graduate','Not Graduate'])
Self_Employed=st.selectbox('Employment Status',['Yes','No'])
ApplicantIncome=st.number_input('What is your Applicant Income ?', min_value=0)
CoapplicantIncome=st.number_input('How much is your Co-Applicant Income ?', min_value=0)
LoanAmount=st.number_input('What is your loan amount ?',min_value=0)
Loan_Amount_Term=st.number_input('What is your loan amount term ?',min_value=0)
Credit_History=st.slider('What is your credit history ?',0,1)
Property_Area=st.selectbox('Waht is your property area ?',['Urban','Rural','Semiurban'])
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
