# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load the dataset into a DataFrame and explore its contents to understand the data structure.

2.Separate the dataset into independent (X) and dependent (Y) variables, and split them into training and testing sets.

3.Create a linear regression model and fit it using the training data.

4.Predict the results for the testing set and plot the training and testing sets with fitted lines. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Akash A 
RegisterNumber: 212225230008 


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
df=pd.read_csv(r"C:\Users\acer\Downloads\student_scores.csv")
df.head(10)

plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
y

from sklearn.model_selection import train_test_split
x_train,x_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,Y_train)
x_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['Hours'],df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.plot(x_train,lr.predict(x_train),color='red')
lr.coef_
lr.intercept_

y_pred=lr.predict(x_test)
mse=mean_squared_error(Y_test,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(Y_test,y_pred)
r2=r2_score(Y_test,y_pred)
print("MSE : ",mse)
print("RMSE : ",rmse)
print("MAE : ",mae)
print("R2 : ",r2)
```

## Output:
<img width="197" height="400" alt="Screenshot 2026-04-29 101936" src="https://github.com/user-attachments/assets/940fa6a4-05cd-4a44-b3bf-43f23cb18bbb" />

<img width="661" height="807" alt="Screenshot 2026-04-29 102017" src="https://github.com/user-attachments/assets/0f14159e-b76e-4073-a12d-425b1904ccef" />

<img width="632" height="433" alt="Screenshot 2026-04-29 102050" src="https://github.com/user-attachments/assets/1df15380-2a9e-4efe-924a-fbf28cdfdf1f" />

<img width="206" height="74" alt="Screenshot 2026-04-29 102117" src="https://github.com/user-attachments/assets/2f083ceb-e49b-4360-b81c-457e2f1a2eca" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
