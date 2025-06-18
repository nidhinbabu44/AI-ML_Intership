import pandas as pd
import matplotlib.pyplot as mt
from sklearn import linear_model
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data.csv")

X= df[["height"]]
Y = df[["weight"]]

mt.scatter(X,Y)
mt.xlabel("Height in CM")
mt.ylabel("Weight in KG")
mt.show()

X_train,X_text,Y_train,Ytest = train_test_split(X,Y, test_size=0.2,random_state=42)
print(X_train)

# Training Start
lr_model = linear_model.LinearRegression()
lr_model.fit(X_train,Y_train)
# Training Over----

joblib.dump(lr_model,"my_model.pkl")

print("Training Completed")
