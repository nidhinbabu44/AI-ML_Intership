import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kn
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv ("diabetes.csv")

X = df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
Y=df[["Outcome"]]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

knn_model = kn(n_neighbors=3)

knn_model.fit(X_train,Y_train) # Algorithm

decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, Y_train)



Y_prediction = knn_model.predict(X_test)
print("accuracy score of knn", accuracy_score(Y_test,Y_prediction)*100)

Y_prediction = decision_tree_model.predict(X_test)
print("accuracy score of dec tree", accuracy_score(Y_test,Y_prediction)*100)

patient_data = pd.DataFrame([[5,166,72,19,175,25.8,0.587,51]])

result = knn_model.predict(patient_data)
print(result) 

decision_tree_model_result =decision_tree_model.predict(patient_data)

if result [0] ==1:
    print("Diabetic Person")
else:
    print("Not a Diabetic Person")