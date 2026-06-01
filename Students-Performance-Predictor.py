import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df=pd.read_csv("Student_marks.csv")

x=df[["Hours"]]
y=df["Marks"]

x_train,x_test,y_train,y_test=train_test_split(
x,y, test_size=0.2 ,random_state=42)

model=LinearRegression()

model.fit(x_train,y_train)

score=model.score(x_test,y_test)

print("score ",score)

hours=float(input("enter study hours"))

prediction=model.predict([[hours]])

print(prediction[0])