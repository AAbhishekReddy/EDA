import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import streamlit as st

import os

class titanic():

    def __init__(self):
        self.file_path = os.path.dirname(os.path.realpath(__file__))
        self.file_path = os.path.join(self.file_path, "titanic-train.csv")
        self.dat = pd.read_csv(self.file_path)
        self.dat_copy = self.dat
        self.decision_tree = None
        self.svc = None
        self.dat = self.data_preprocessing(self.dat)

    def input(self, mainwindow_slots, sidebar_slots, predict_button):
        name = sidebar_slots[0].text_input("Name", value="Enter Name With TITLE: Mr.")
        age = sidebar_slots[1].number_input("Age", min_value=0, max_value=100, step=None, value=30)
        sex = sidebar_slots[2].selectbox("Sex", options=["male", "female"])
        pclass = sidebar_slots[3].number_input("PClass", min_value=1, max_value=3, step=1, value=1)
        sibsp = sidebar_slots[4].number_input("Siblings Onboard", min_value=0, max_value=5, step=1, value=0)
        parch = sidebar_slots[5].number_input("Parents Onboard", min_value=0, max_value=5, step=1, value=0)
        fare = sidebar_slots[6].number_input("Fare", min_value=0, max_value=None, step=None, value=0)
        embarked = sidebar_slots[7].selectbox("Embarked", options = ["C", "Q", "S"])

        dat = {"Pclass" : pclass,
                "Name" : name,
                "Sex" : sex,
                "Age" : age,
                "SibSp" : sibsp,
                "Parch" : parch,
                "Fare" : fare,
                "Embarked" : embarked}
        dat = pd.DataFrame(dat, index = [0])

        st.subheader("""
        Titanic Survival Prediction

        Enter the passenger details in the sidebar to view the predictions
        """)

        if predict_button:
            st.subheader("User Input features")
            st.dataframe(dat)

        dat_altered = dat
        dat_altered["PassengerId"] = "0"
        dat_altered["Cabin"] = "0"
        dat_altered["Ticket"] = "0"
        data = self.data_preprocessing(dat_altered)
        # "### DATA", data

        # Build the model
        self.build_decision_tree()
        self.build_svc()
        dtree, svc = self.predict_survival(data)
        return dtree, svc, name
    
        

    def build_svc(self):
        y_train = self.dat["Survived"]
        x_train = self.dat.drop(columns = ["Survived"])

        self.svc = SVC(kernel="linear", C=0.025, random_state=80)
        self.svc.fit(x_train, y_train)

    def build_decision_tree(self):
        y_train = self.dat["Survived"]
        x_train = self.dat.drop(columns = ["Survived"])

        self.decision_tree = DecisionTreeClassifier()
        self.decision_tree.fit(x_train, y_train)

    def impute_age(self, columns):
        Age = columns[0]
        Pclass = columns[1]
        if pd.isnull(Age):
            if Pclass == 1:
                return 37
            elif Pclass == 2:
                return 29
            else:
                return 24
        else:
            return Age

    def data_preprocessing(self, new_input):
        flag = 0
        if not new_input.equals(self.dat_copy):
            new_input = self.dat_copy.append(new_input)
            flag = 1
        # Creating a new column TITLE
        new_input["Title"] = new_input.Name.str.extract("([A-Za-z]+)\.", expand = False)
        # Imputing Age
        new_input['Age'] = new_input[['Age','Pclass']].apply(self.impute_age,axis=1)
        # Handling outliers
        new_input.loc[new_input["Age"] > 66, "Age"] = 66
        new_input.loc[new_input["Fare"] > 70, "Fare"] = 70.0
        # Adding an age band
        new_input["AgeBand"] = pd.cut(new_input.Age, 5)
        new_input["AgeBand"] = new_input["AgeBand"].astype("category").cat.codes
        # Adding the fare range 
        new_input["FareRange"] = pd.cut(new_input.Age, 5)
        new_input["FareRange"] = new_input["FareRange"].astype("category").cat.codes
        # Correcting the titles
        new_input.Title = new_input.Title.replace(["Mlle, Ms"], "Miss")
        new_input.Title = new_input.Title.replace("Mme", "Mrs")
        new_input.Title = new_input.Title.replace("Ms", "Miss")
        new_input.Title = new_input.Title.replace(['Lady', 'Countess','Capt', 'Col',\
            'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], "Rare")
        # Converting Title to ordinal values
        new_input.Title = new_input.Title.astype("category")
        new_input.Title = new_input.Title.cat.codes
        # Creating a new Column Family
        new_input["Family"] = new_input["Parch"] + new_input["SibSp"]
        # Filling na in Embarked
        new_input.Embarked.fillna("S", inplace = True)
        # Removing the uneccessary columns
        new_input = new_input.drop(columns = ["PassengerId", "Parch", "SibSp", "Cabin", "Ticket", "Name", "Fare", "Age"])
        new_input = pd.get_dummies(new_input, columns = ["Sex", "Embarked", "Pclass"])

        if flag == 1:
            new_input = new_input.iloc[-1]
            new_input = pd.DataFrame(new_input).T
            new_input = new_input.drop(columns = ["Survived"])
        
        return new_input

    def predict_survival(self, data):
        survival_dtree = self.decision_tree.predict(data)
        survival_svc = self.svc.predict(data)

        return survival_dtree, survival_svc
