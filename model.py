import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

base_data = pd.read_csv("DSP_1.csv")
base_data.columns

cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
data = base_data[cols].copy()

data["Age"].fillna((data["Age"].mean()), inplace=True) # wypełni nam brakujące informacje średnią
data.dropna(subset=['Embarked'], inplace=True)

encoder = LabelEncoder()
data.loc[:,"Sex"] = encoder.fit_transform(data.loc[:,"Sex"])
data.loc[:,"Embarked"] = encoder.fit_transform(data.loc[:,"Embarked"])

y = data.iloc[:,0]
x = data.iloc[:,1:8]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

def model(X_train, y_train):
    
    forest = RandomForestClassifier (n_estimators=20, random_state=0)
    forest.fit(X_train,y_train)
    print("Random Forest: {0}".format(forest.score(X_train,y_train)))
    
    return forest

forest = model(X_train,y_train)

my_data =[
            [
             1,  #"Pclass"
             1,  #"Sex", Sex 0 = Female, 1 = Male
             50,  #"Age", Age
             0,  #"SibSp"
             0,  #"Parch"
             0,  #"Fare", 
             2,  #"Embarked"
    ]
]

print(forest.predict(my_data))

my_data =[
            [
             1,  #"Pclass"
             0,  #"Sex", Sex 0 = Female, 1 = Male
             20,  #"Age", Age
             1,  #"SibSp"
             0,  #"Parch"
             0,  #"Fare", 
             2,  #"Embarked"
    ]
]

print(forest.predict(my_data))

filename = "model.sv"
pickle.dump(forest, open(filename,'wb'))