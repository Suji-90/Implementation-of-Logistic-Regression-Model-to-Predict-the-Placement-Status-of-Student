## NAME: SUJITHRA K
## REGISTER NUMBER:212223040212

# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

## Placement data

![432667952-3f736cf6-8fd6-4d1e-bed4-9579777714b8](https://github.com/user-attachments/assets/f735ac3d-c356-4508-8a7d-126b41ccadb2)

## Salary data

![432668096-5fcd898b-f2bb-4465-b76e-1ef902d4c7de](https://github.com/user-attachments/assets/66cabba2-c2a6-4631-9c42-43e2f9c72ce2)


## Checking null function


![432668352-4c4b52ac-d1a4-4066-9f03-f76a60990ffb](https://github.com/user-attachments/assets/81ec8174-79f3-4d2f-b3d2-cefa2f362f87)


## Data duplicate

![432668465-53ac73a8-cd3b-4ead-a7c9-4a148e1484ed](https://github.com/user-attachments/assets/a983a485-1b94-4aac-9f37-be93fc3faaa0)


## Print data

![432668614-a54bedf5-194a-4144-b93b-622f47915418](https://github.com/user-attachments/assets/ea830f9a-274f-4d25-adc5-2b3cd7c94979)


## Data status

![432668970-fe74ffef-7c77-4353-b4b1-945febc2ce17](https://github.com/user-attachments/assets/a95d5a96-6155-4ad9-a4ee-489ffb88c0f3)


## Y-prediction array

![432669082-05862a3d-e717-4d86-8f1c-585d9cbccc29](https://github.com/user-attachments/assets/c3f39d50-c1b8-4856-9a06-0505d11998ad)


## Accuracy value

![432670049-000f19f4-153f-4af2-8d1a-f6ad61a79d2b](https://github.com/user-attachments/assets/b36bb894-dd2b-4685-9f57-5a62db9d4644)


## Confusion array

![432670051-0ee3d663-7857-472f-b600-2bc1fd3f95b3](https://github.com/user-attachments/assets/4ea546b5-9bdc-468b-a55d-cf4c54c5275d)


## Classification report

![432670174-293cc3b1-5726-464b-aa78-d9e5b5f48e50](https://github.com/user-attachments/assets/26f723f1-2e50-4f1b-a73e-83d52708a4c6)


## Prediction of LR

![432670368-f16a6a4c-afe3-450f-93e3-ab9772f6d561](https://github.com/user-attachments/assets/f3f18fac-3474-409b-a99f-25f213cd3deb)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
