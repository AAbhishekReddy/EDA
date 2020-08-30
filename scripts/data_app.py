import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from titanic_req.titanic_model import titanic

# Global variables
model_select_preview = "Classification Models"

def cars_prediction(mainwindow_slots, sidebar_slots):
    buying = sidebar_slots[0].selectbox("Buying Value", options = ["vhigh", "low", "high", "med"])
    maint = sidebar_slots[1].selectbox("Maintanance", options = ["vhigh", "low", "high", "med"])
    doors = sidebar_slots[2].number_input("Number of Doors", min_value = 2, max_value = 5, value = 4, step = 1)
    persons = sidebar_slots[3].selectbox("Number of Persons", options = [2, 4, 5])
    lug_boot = sidebar_slots[4].selectbox("Lugguage Boot Space", options = ["big", "small", "med"])
    safety = sidebar_slots[5].selectbox("Safety", options = ["low", "high", "med"])

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

    if predict_button:
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

# Creating the support functions
def classification(mainwindow_slots, sidebar_slots):
    

    data_select = st.radio("Select the data that you want to apply the model on", options = ["Titanic Dataset", "Car Evaluation Dataset"])
    if data_select == "Titanic Dataset":
        model_titanic = titanic()
        dtree, svc, name = model_titanic.input(mainwindow_slots, sidebar_slots, predict_button)

        if predict_button:
            model_select = st.radio("Select the model", options = ["Decision Tree", "Support Vector Machine"])

            if model_select == "Decision Tree":
                message = f"{name} unfortunately will **_not_ Survive**" if dtree[0] == 0 else f"{name} **_will_ Survive**"
                st.subheader("DECISION TREE PREDICTION")
                st.markdown(message)
            else:
                message = f"{name} unfortunately will **_not_ Survive**" if svc[0] == 0 else f"{name} **_will_ Survive**"
                st.subheader("SUPPORT VECTOR MACHINE PREDICTION")
                st.markdown(message)

    else:
        cars_prediction(mainwindow_slots, sidebar_slots)


def regression(mainwindow_slots, sidebar_slots):
    beer_name = sidebar_slots[0].text_input("Beer Name:", value = "British Empire")
    review_aroma = sidebar_slots[1].number_input("Aroma Review", min_value = 1, max_value = 5)
    review_appearence = sidebar_slots[2].number_input("Aroma Appereance", min_value = 1, max_value = 5)
    review_palate = sidebar_slots[3].number_input("Palate Appereance", min_value = 1, max_value = 5)
    review_taste = sidebar_slots[4].number_input("Taste Appereance", min_value = 1, max_value = 5)
    beer_abv = sidebar_slots[5].number_input("Beer ABV", min_value = 1, max_value = 100)

    beers = {"beer_name": beer_name,
             "review_aroma": review_aroma,
             "review_palate": review_palate,
             "review_taste": review_taste,
             "review_appereance": review_appearence,
             "beer_abv": beer_abv}
    
    beers = pd.DataFrame(beers, index = [0])

    st.subheader("""
        Beer Review Prediction

        Enter the passenger details in the sidebar to view the predictions
        """)

    if predict_button:
        st.subheader("User Input features")
        st.dataframe(beers)

        arr = np.array([review_aroma, review_palate, review_taste])

        file_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(file_path, "linear_regression.sav")

        beer_lr = joblib.load(file_path)

        prediction_beers = beer_lr.predict(arr.reshape(1,-1))
        st.write("### Linear Regression Prediction")
        message = f"The overall rating for {beer_name} is: {prediction_beers[0]}" 
        st.write(message)



if __name__ == "__main__":
    # Creating the side bar empty fields.
    sidebar_slots = []

    for i in range(10):
        sidebar_slots.append(st.sidebar.empty())

    predict_button = sidebar_slots[8].button("Predict") 

    # Creating the main page Elements
    st.write("""
    # This is an Interactive app to play around with various Data Models
    """)

    # Main window slots:
    # 1. Select the type of Models ("Classification or Regeression")
    # 2. Select the data set in the options.
    # 3. Display the objective of the Models
    # 4. Display the use Input data.
    # 5. Display the option for the models in the same data.
    # 6. Display the output or the prediction.

    # Creating a radio button for slecting the type of models.
    st.write("## Select the type of models that you want to play with below")
    model_select = st.radio("Select the models of your choice", options = ["Classification Models", "Regression Models"])

    # Creating the mainwindow slots.
    mainwindow_slots = []

    for i in range(10):
        mainwindow_slots.append(st.empty())

    if model_select == "Classification Models":
        classification(mainwindow_slots, sidebar_slots)

        
    else:
        regression(mainwindow_slots, sidebar_slots)

