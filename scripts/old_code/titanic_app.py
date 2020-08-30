import streamlit as st
import pandas as pd
from PIL import Image

from req.model import titanic

st.beta_set_page_config(page_title = "TITANIC", page_icon = "🚢", layout = "centered", initial_sidebar_state = "expanded")

titanic_image = st.image(Image.open(r"E:\CES\git\EDA\images\titanic_sinking.jpg"), use_column_width = True)

st.write("""
# Welcome to Titanic Survival

## Did you ***SURVIVE*** or not?????
Input your details in the side bar to view the prediction
""")

st.sidebar.header("User Input Parameters")

def user_input():
    name = st.sidebar.text_input("Name", value="Enter Name With TITLE: Mr.")
    age = st.sidebar.number_input("Age", min_value=0, max_value=100, step=None, value=30)
    sex = st.sidebar.selectbox("Sex", options=["male", "female"])
    pclass = st.sidebar.number_input("PClass", min_value=1, max_value=3, step=1, value=1)
    sibsp = st.sidebar.number_input("Siblings Onboard", min_value=0, max_value=5, step=1, value=0)
    parch = st.sidebar.number_input("Parents Onboard", min_value=0, max_value=5, step=1, value=0)
    fare = st.sidebar.number_input("Fare", min_value=0, max_value=None, step=None, value=0)
    embarked = st.sidebar.selectbox("Embarked", options = ["C", "Q", "S"])

    data = {"Pclass" : pclass,
            "Name" : name,
            "Sex" : sex,
            "Age" : age,
            "SibSp" : sibsp,
            "Parch" : parch,
            "Fare" : fare,
            "Embarked" : embarked}
    data = pd.DataFrame(data, index = [0])

    return data

dat = user_input()

st.subheader("User Input features")
st.write(dat)

dat_altered = dat
dat_altered["PassengerId"] = "0"
dat_altered["Cabin"] = "0"
dat_altered["Ticket"] = "0"
model = titanic()
data = model.data_preprocessing(dat_altered)
"### DATA", data

model_select = st.radio("Select Prediction Model", options = ["Decision Tree", "Support Vector Machine"])

# Build the model
model.build_decision_tree()
model.build_svc()
dtree, svc = model.predict_survival(data)

if model_select == "Decision Tree":
    "### DECISION TREE PREDICTION", dtree
elif model_select == "Support Vector Machine":
    "### SUPPORT VECTOR MACHINE PREDICTION", svc
st.write("0 -> Dead : 1 -> Alive")

# st.balloons()
