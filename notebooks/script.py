import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Data Dictionary

# Variable	Definition	                                    Key
# survival	Survival	                                    0 = No, 1 = Yes
# pclass	Ticket class	                                1 = 1st, 2 = 2nd, 3 = 3rd
# sex	    Sex
# Age	    Age in years	
# sibsp	    # of siblings / spouses aboard the Titanic
# parch	    # of parents / children aboard the Titanic
# ticket	Ticket number
# fare	    Passenger fare
# cabin	    Cabin number
# embarked	Port of Embarkation	                            C = Cherbourg, Q = Queenstown, S = Southampton

titanic_train = pd.read_csv("train.csv")

titanic_train.describe()
titanic_train.columns
titanic_train["Cabin"].isnull().sum()

# Dropping uneccessary columns.
titanic_train = titanic_train.drop(columns = ["PassengerId", "Name", "Cabin", "Ticket"])

# Dropping the rows with nan values.
titanic_train = titanic_train.dropna()

# EDA
boxplot = titanic_train.boxplot(column = ["Parch"])

# Correlation matrix
corr_matrix = titanic_train.corr()
sn.heatmap(corr_matrix, annot = True)
plt.show()

# Removing the outliers
titanic_train.loc[titanic_train["Age"] > 66, "Age"] = 66.0
titanic_train.loc[titanic_train["Fare"] > 70, "Fare"] = 70.0

sn.pairplot(titanic_train[["Survived", "Pclass", "Age", "Fare"]], hue = "Survived", size = 3)
