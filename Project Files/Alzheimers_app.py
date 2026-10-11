#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')


# In[2]:


best_xgb=pickle.load(open('Alzheimers_project.pkl','rb'))
scaler = pickle.load(open('st_scaler.pkl','rb'))


# In[3]:


st.title("🧠 Alzheimer's Disease Risk Prediction")


# In[4]:


def user_input_parameter():

    st.sidebar.header("👤 Patient Information")

    Age = st.sidebar.number_input('Age',60,100,70)
    Gender = st.sidebar.selectbox('Gender',[0,1])
    Ethnicity = st.sidebar.selectbox('Ethnicity',[0,1,2,3])
    EducationLevel = st.sidebar.selectbox('Education Level',[0,1,2,3])
    BMI = st.sidebar.number_input('BMI',10.0,50.0,25.0)


    st.sidebar.header("🚬 Lifestyle")

    Smoking = st.sidebar.selectbox('Smoking',[0,1])
    AlcoholConsumption = st.sidebar.number_input('Alcohol Consumption',0.0)
    PhysicalActivity = st.sidebar.number_input('Physical Activity',0.0)
    DietQuality = st.sidebar.number_input('Diet Quality',0.0)
    SleepQuality = st.sidebar.number_input('Sleep Quality',0.0)


    st.sidebar.header("🩺 Medical History")

    FamilyHistoryAlzheimers = st.sidebar.selectbox('Family History of Alzheimer',[0,1])
    CardiovascularDisease = st.sidebar.selectbox('Cardiovascular Disease',[0,1])
    Diabetes = st.sidebar.selectbox('Diabetes',[0,1])
    Depression = st.sidebar.selectbox('Depression',[0,1])
    HeadInjury = st.sidebar.selectbox('Head Injury',[0,1])
    Hypertension = st.sidebar.selectbox('Hypertension',[0,1])


    st.sidebar.header("❤️ Clinical Measurements")

    SystolicBP = st.sidebar.number_input('Systolic BP',80,250,120)
    DiastolicBP = st.sidebar.number_input('Diastolic BP',40,150,80)
    CholesterolTotal = st.sidebar.number_input('Total Cholesterol',50.0)
    CholesterolLDL = st.sidebar.number_input('LDL Cholesterol',10.0)
    CholesterolHDL = st.sidebar.number_input('HDL Cholesterol',10.0)
    CholesterolTriglycerides = st.sidebar.number_input('Triglycerides',10.0)


    st.sidebar.header("🧠 Cognitive Assessment")

    MMSE = st.sidebar.number_input('MMSE Score',0.0,30.0,15.0)
    FunctionalAssessment = st.sidebar.number_input('Functional Assessment',0.0)
    ADL = st.sidebar.number_input('ADL Score',0.0)


    st.sidebar.header("⚠ Symptoms")

    MemoryComplaints = st.sidebar.selectbox('Memory Complaints',[0,1])
    BehavioralProblems = st.sidebar.selectbox('Behavioral Problems',[0,1])
    Confusion = st.sidebar.selectbox('Confusion',[0,1])
    Disorientation = st.sidebar.selectbox('Disorientation',[0,1])
    PersonalityChanges = st.sidebar.selectbox('Personality Changes',[0,1])
    DifficultyCompletingTasks = st.sidebar.selectbox('Difficulty Completing Tasks',[0,1])
    Forgetfulness = st.sidebar.selectbox('Forgetfulness',[0,1])

    # Feature Engineering
    if BMI < 18.5:
        BMI_Category = 0
    elif BMI < 25:
        BMI_Category = 1
    elif BMI < 30:
        BMI_Category = 2
    else:
        BMI_Category = 3

    if Age <= 70:
        Age_Group = 0
    elif Age <= 80:
        Age_Group = 1
    else:
        Age_Group = 2

    dict1 = {
        'Age':Age,
        'Gender':Gender,
        'Ethnicity':Ethnicity,
        'EducationLevel':EducationLevel,
        'BMI':BMI,
        'Smoking':Smoking,
        'AlcoholConsumption':AlcoholConsumption,
        'PhysicalActivity':PhysicalActivity,
        'DietQuality':DietQuality,
        'SleepQuality':SleepQuality,
        'FamilyHistoryAlzheimers':FamilyHistoryAlzheimers,
        'CardiovascularDisease':CardiovascularDisease,
        'Diabetes':Diabetes,
        'Depression':Depression,
        'HeadInjury':HeadInjury,
        'Hypertension':Hypertension,
        'SystolicBP':SystolicBP,
        'DiastolicBP':DiastolicBP,
        'CholesterolTotal':CholesterolTotal,
        'CholesterolLDL':CholesterolLDL,
        'CholesterolHDL':CholesterolHDL,
        'CholesterolTriglycerides':CholesterolTriglycerides,
        'MMSE':MMSE,
        'FunctionalAssessment':FunctionalAssessment,
        'MemoryComplaints':MemoryComplaints,
        'BehavioralProblems':BehavioralProblems,
        'ADL':ADL,
        'Confusion':Confusion,
        'Disorientation':Disorientation,
        'PersonalityChanges':PersonalityChanges,
        'DifficultyCompletingTasks':DifficultyCompletingTasks,
        'Forgetfulness':Forgetfulness,
        'BMI_Category':BMI_Category,
        'Age_Group':Age_Group
    }

    return pd.DataFrame(dict1, index=[0])


# In[5]:


df = user_input_parameter()

button = st.button("Predict")

if button:

    # Scale numerical features
    df[['Age','BMI','MMSE','ADL','FunctionalAssessment']] = scaler.transform(
        df[['Age','BMI','MMSE','ADL','FunctionalAssessment']]
    )

    prediction = best_xgb.predict(df)

    probability = best_xgb.predict_proba(df)

    # Alzheimer's probability (Class 1)
    alz_prob = probability[0][1] * 100

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("🔴 High Risk of Alzheimer's Disease")
    else:
        st.success("🟢 Low Risk of Alzheimer's Disease")

    st.subheader("Alzheimer's Risk Probability")

    st.progress(float(probability[0][1]))

    st.write(f"Risk Probability : **{alz_prob:.2f}%**")

