import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

base_data = pd.read_csv("DSP_2.csv");
base_data.columns

cols = ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope","HeartDisease"]
data = base_data[cols].copy()

data["RestingBP"].replace(0, np.nan).ffill() #zamieniam 0 na NAN a nastepnie wypełniam średnią
data["RestingBP"].fillna((data["RestingBP"].mean()), inplace=True)

data["Cholesterol"].replace(0, np.nan).ffill()
data["Cholesterol"].fillna((data["Cholesterol"].mean()), inplace=True)

encoder = LabelEncoder()
data.loc[:,"Sex"] = encoder.fit_transform(data.loc[:,"Sex"])
data.loc[:,"ChestPainType"] = encoder.fit_transform(data.loc[:,"ChestPainType"])
data.loc[:,"RestingECG"] = encoder.fit_transform(data.loc[:,"RestingECG"])
data.loc[:,"ExerciseAngina"] = encoder.fit_transform(data.loc[:,"ExerciseAngina"])
data.loc[:,"ST_Slope"] = encoder.fit_transform(data.loc[:,"ST_Slope"])


y = data.iloc[:,11]
x = data.iloc[:,0:11]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

def model(X_train, y_train):
    
    forest = RandomForestClassifier (n_estimators=20, random_state=0)
    forest.fit(X_train,y_train)
    print("Random Forest: {0}".format(forest.score(X_train,y_train)))
    
    return forest

forest = model(X_train,y_train)

my_data = [
    [
    64,
    1,
    0,
    200,
    6,
    0,
    0,
    250,
    1,
    3.4,
    2
    ]


]
print(forest.predict(my_data))

my_data = [
    [
    49,
    0,
    0,
    140,
    185,
    0,
    0,
    130,
    0,
    0,
    2
    ]


]
print(forest.predict(my_data))

filename = "model.sv"
pickle.dump(forest, open(filename,'wb'))