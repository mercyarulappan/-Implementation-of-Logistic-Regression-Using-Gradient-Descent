# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.start the program

2.Import the necessary python packages

3.Read the dataset.

4.Define X and Y array.

5.Define a function for costFunction,cost and gradient.

6.Define a function to plot the decision boundary and predict the Regression value

7.End the program 

## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: MERCY A
RegisterNumber:  212223110027
*/

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
dataset=pd.read_csv('Placement.csv')
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset["gender"]=dataset["gender"].astype("category")
dataset["ssc_b"]=dataset["ssc_b"].astype("category")
dataset["hsc_b"]=dataset["hsc_b"].astype("category")
dataset["degree_t"]=dataset["degree_t"].astype("category")
dataset["workex"]=dataset["workex"].astype("category")
dataset["specialisation"]=dataset["specialisation"].astype("category")
dataset["status"]=dataset["status"].astype("category")
dataset["hsc_s"]=dataset["hsc_s"].astype("category")
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset.info()
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
theta = np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta, X, y, alpha, num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y)/m
        theta -= alpha * gradient
    return theta 

theta = gradient_descent(theta, X, y,alpha = 0.01,num_iterations = 1000)

def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print("Predicted Result:",y_prednew)
print(theta)
```

## Output:
![Screenshot 2024-10-09 113147](https://github.com/user-attachments/assets/be7aa0a8-5214-439d-b50e-10c91f331af1)

![Screenshot 2024-10-09 113153](https://github.com/user-attachments/assets/e3656dbe-f269-4726-b278-c692ad84e10c)

![Screenshot 2024-10-09 113159](https://github.com/user-attachments/assets/5204723f-c563-49e1-9b4d-36e1114ce2f5)

![Screenshot 2024-10-09 113206](https://github.com/user-attachments/assets/3b81be0b-05c8-45da-9c0a-38532a8e59cd)

![Screenshot 2024-10-09 113532](https://github.com/user-attachments/assets/4e82c239-950c-4b1b-ac67-e53ace8b0e42)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

