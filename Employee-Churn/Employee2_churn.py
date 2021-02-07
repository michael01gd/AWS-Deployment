import streamlit as st
import pickle
import pandas as pd

st.sidebar.title('Employee Churn Prediction')



satisfaction_level=st.sidebar.slider("What is satisfaction level of employee:",0.0,1.0,step=0.1)
last_evaluation=st.sidebar.slider("What is last_evaluation", 0.0,1.0,step=0.1)
number_project=st.sidebar.slider("What is the total number of project included", 2,7,step=1)
average_montly_hours=st.sidebar.slider("What is the average_montly_hours", 96,310,step=50)
time_spend_company=st.sidebar.slider('How many years working',2,10, 1)
Work_accident=st.sidebar.selectbox('Had any accident',('Yes','No'))
promotion_last_5years=st.sidebar.radio ("Had promotion_last_5 years?", (1,0))
Departments=st.sidebar.selectbox ("Which Department?", ('sales','accounting','hr','technical','support','management','IT','product mng','marketing','RandD'))
salary=st.sidebar.selectbox("What is your salary level", ('Low','Medium','High')) 

model_name=st.selectbox("Select your model:",("GBooster","Random_Forest",'KNN'))
if model_name=="Gbooster":
	model=pickle.load(open("model_booster","rb"))
	st.success("You selected {} model".format(model_name))
elif model_name=='Random_Forest':
    model=pickle.load(open('model_ranfor','rb'))
    st.success('You selected {} model'.format(model_name))
else:
	model=pickle.load(open("model_knn","rb"))
	st.success("You selected {} model".format(model_name))


my_dict = {
    "satisfaction_level": satisfaction_level,
    'last_evaluation':last_evaluation, 
    'number_project' :number_project,
    'average_montly_hours':average_montly_hours,
    'time_spend_company':time_spend_company,
    'Work_accident' :Work_accident,
    'promotion_last_5years':promotion_last_5years,
    'Departments':Departments,
    'salary':salary    
}

df = pd.DataFrame.from_dict([my_dict])

columns=["satisfaction_level",'last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','Departments','salary']

df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)

st.header("The configuration of your churn data is below")
st.table(df)

st.subheader("Press predict if configuration is okay")
if st.button("Predict"):
    prediction=model.predict(df)
    st.success("The estimated churn state is {}. ".format(int(prediction[0])))

#prediction = model.predict(df)

#st.success("The estimated price of your car is €{}. ".format(int(prediction[0])))