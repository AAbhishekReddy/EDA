import streamlit as st

from titanic_req.titanic_model import titanic


def classification(mainwindow_slots, sidebar_slots):
    data_select = mainwindow_slots[0].radio("Select the data that you want to apply the model on", options = ["Titanic Dataset", "Car Evaluation Dataset"])
    model_titanic = titanic(sidebar_slots)
    model_titanic.input(mainwindow_slots, sidebar_slots)

def regression(mainwindow_slots, sidebar_slots):
    st.write("Regression")