import streamlit as st
import pandas as pd
import numpy as  np
import joblib
import os

from req.titanic_model import titanic

st.write("""
# This is a classifier app for titanic and cars
""")

# Creating slots in side bar
slot1 = st.sidebar.empty()
slot2 = st.sidebar.empty()
slot3 = st.sidebar.empty()
slot4 = st.sidebar.empty()
slot5 = st.sidebar.empty()
slot6 = st.sidebar.empty()
slot7 = st.sidebar.empty()
slot8 = st.sidebar.empty()

predict_button = st.sidebar.button("Predict")

# creating slots in The main window


# Funtion for titanic input
# @st.cache(suppress_st_warning = True)
def titanic_input():
    name = slot1.text_input("Name", value="Enter Name With TITLE: Mr.")
    age = slot2.number_input("Age", min_value=0, max_value=100, step=None, value=30)
    sex = slot3.selectbox("Sex", options=["male", "female"])
    pclass = slot4.number_input("PClass", min_value=1, max_value=3, step=1, value=1)
    sibsp = slot5.number_input("Siblings Onboard", min_value=0, max_value=5, step=1, value=0)
    parch = slot6.number_input("Parents Onboard", min_value=0, max_value=5, step=1, value=0)
    fare = slot7.number_input("Fare", min_value=0, max_value=None, step=None, value=0)
    embarked = slot8.selectbox("Embarked", options = ["C", "Q", "S"])

    dat = {"Pclass" : pclass,
            "Name" : name,
            "Sex" : sex,
            "Age" : age,
            "SibSp" : sibsp,
            "Parch" : parch,
            "Fare" : fare,
            "Embarked" : embarked}
    dat = pd.DataFrame(dat, index = [0])

    st.write("""
    # Titanic Survival Prediction

    Enter the passenger details in the sidebar to view the predictions
    """)

    st.subheader("User Input features")
    st.write(dat)

    dat_altered = dat
    dat_altered["PassengerId"] = "0"
    dat_altered["Cabin"] = "0"
    dat_altered["Ticket"] = "0"
    model = titanic()
    data = model.data_preprocessing(dat_altered)
    # "### DATA", data

    
    model_select = st.radio("Select Prediction Model", options = ["Decision Tree", "Support Vector Machine"])

    # Build the model
    model.build_decision_tree()
    model.build_svc()
    dtree, svc = model.predict_survival(data)


    if model_select == "Decision Tree":
        if predict_button:
            message = f"#### {name} unfortunately will not Survive" if dtree[0] == 0 else f"{name} will Survive"
            st.write("### DECISION TREE PREDICTION")
            st.write(message)
    elif model_select == "Support Vector Machine":
        if predict_button:
            message = f"#### {name} unfortunately will not Survive" if svc[0] == 0 else f"{name} will Survive"
            st.write("### SUPPORT VECTOR MACHINE PREDICTION")
            st.write(message)


def cars_input():
    cars = []
    buying = slot1.selectbox("Buying Value", options = ["vhigh", "low", "high", "med"])
    maint = slot2.selectbox("Maintanance", options = ["vhigh", "low", "high", "med"])
    doors = slot3.number_input("Number of Doors", min_value = 2, max_value = 5, value = 4, step = 1)
    persons = slot4.selectbox("Number of Persons", options = [2, 4, 5])
    lug_boot = slot5.selectbox("Lugguage Boot Space", options = ["big", "small", "med"])
    safety = slot6.selectbox("Safety", options = ["low", "high", "med"])

    dat = {"buying" : buying,
            "maint" : maint,
            "doors" : doors,
            "persons" : persons,
            "lug_boot" : lug_boot,
            "safety" : safety}

    dat = pd.DataFrame(dat, index = [0])

    buying = 3 if buying == "vhigh" else buying
    buying = 2 if buying == "low" else buying
    buying = 1 if buying == "high" else buying
    buying = 0 if buying == "med" else buying

    maint = 3 if maint == "vhigh" else maint
    maint = 2 if maint == "low" else maint
    maint = 1 if maint == "high" else maint
    maint = 0 if maint == "med" else maint

    doors = 1 if doors == 4 else doors
    doors = 0 if doors == 5 else doors

    persons = 1 if persons == 4 else persons
    persons = 0 if persons == 5 else persons

    lug_boot = 2 if lug_boot == "big" else lug_boot
    lug_boot = 1 if lug_boot == "small" else lug_boot
    lug_boot = 0 if lug_boot == "med" else lug_boot

    safety = 2 if safety == "low" else safety
    safety = 1 if safety == "high" else safety
    safety = 0 if safety == "med" else safety

    st.write("""
    # Cars Evaluation

    Enter the details of your car to view the predictions
    """)

    st.subheader("User Input features")
    st.write(dat)

    arr = np.array([buying, maint, doors, persons, lug_boot, safety])

    file_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(file_path, "dtree_cars.sav")

    dtree_cars = joblib.load(file_path)

    if predict_button:
        st.write("""
        ## Decision Tree Prediction
        """)
        prediction_cars = dtree_cars.predict(arr.reshape(1,-1))
        message = "### The Car is "
        message = message + "***Un Acceptable***" if prediction_cars == "unacc" else message
        message = message + "***Un Acceptable***" if prediction_cars == "acc" else message
        message = message + "***Un Acceptable***" if prediction_cars == "good" else message
        message = message + "***Un Acceptable***" if prediction_cars == "v-good" else message

        st.write(message)
        


data_select = st.radio("Select the Classifier", options = ["Titanic Survival Classifier", "Cars safety Clsssifier"])

if data_select == "Titanic Survival Classifier":
    titanic_input()
elif data_select == "Cars safety Clsssifier":
    cars_input()

