# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Vishal S
RegisterNumber:  212223110063
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
```

## Output:

## Dataset:
![second experiment ml 3](https://github.com/vishal23000591/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147139719/3f0732d1-0169-4aa5-b465-1d598060b3cc)


## Y predicted:
![second experiment ml 1](https://github.com/vishal23000591/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147139719/dbefb09b-919f-4101-a541-7b00b74e8fec)


## Training set:![
![second experiment ml 2](https://github.com/vishal23000591/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147139719/170e88ec-5625-43be-9b07-837a93eb2372)



## Values of MSE,MAE,RMSE:
![second experiment ml 4](https://github.com/vishal23000591/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147139719/a137269c-d926-4a8c-8eea-ab2de7c356d6)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
