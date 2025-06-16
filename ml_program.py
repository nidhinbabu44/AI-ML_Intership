import pandas as pd
import matplotlib.pyplot as mt

df = pd.read_csv("data.csv")

x= df[["height"]]
y = df[["weight"]]

mt.scatter(x,y)
mt.xlabel("Height in CM")
mt.ylabel("Weight in KG")
mt.show()


