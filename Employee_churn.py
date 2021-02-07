import streamlit as st
import pickle
import pandas as pd

st.sidebar.title('Employee Churn Prediction')
#html_temp = """
#<div style="background-color:tomato;padding:10px">
#<h2 style="color:white;text-align:center;">Streamlit ML App </h2>
#</div>"""
#st.markdown(html_temp,unsafe_allow_html=True)

with open('variables.pkl', 'rb') as f:
    variables = pickle.load(f)


satisfaction_level=st.sidebar.slider("What is satisfaction level of employee:",0.0,1.0,step=0.1)
last_evaluation=st.sidebar.slider("What is last_evaluation", 0.0,1.0,step=0.1)
number_project=st.sidebar.slider("What is the total number of project included", 2,7,step=1)
average_montly_hours=st.sidebar.slider("What is the average_montly_hours", 96,310,step=50)
time_spend_company=st.sidebar.slider('How mnay years working',2,10, 1)
Work_accident=st.sidebar.selectbox('Had any accident',('Yes','No'))
promotion_last_5years=st.sidebar.radio ("Had promotion_last_5 years?", (1,0))
Departments=st.sidebar.selectbox ("Which Department?", ('sales','accounting','hr','technical','support','management','IT','product mng','marketing','RandD'))
salary=st.sidebar.selectbox("What is your salary level", ('Low','Medium','High')) 

model_name=st.selectbox("Select your model:",("GBoost","Random_Forest",'KNN'))
if model_name=="Gboost":
	model=pickle.load(open("model_gbooster","rb"))
	st.success("You selected {} model".format(model_name))
elif model_name=='Random_Forest':
	model=pickle.load(open("model_ranfor","rb"))
	st.success("You selected {} model".format(model_name))
else:
  model=pickle.load(open('model_knn','rb'))
  st.success('You selected {} model'.format(model_name))


my_dict = {
    "satisfaction_level": satisfaction_level,
    'last_evaluation':last_evaluation, 
    'number_project' :number_project,
    'average_montly_hours':average_montly_hours,
    'time_spend_company':time_spend_company,
    'Work_accident' :Work_accident,
    'promotion_last_5years':promotion_last_5years,
    'Departments salary':Departments,
    'salary':salary
    
}

df = pd.DataFrame.from_dict([my_dict])

columns=variables

df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)


st.header("Please chack the data you entered")
st.table(df)

st.subheader("Press predcit if configuration is okay")
if st.button("Predict"):
    prediction=model.predict(df)
    st.success("The churn prediction is {}. ".format(int(prediction)))

#prediction = model.predict(df)

#st.success("The estimated price of your car is €{}. ".format(int(prediction[0])))