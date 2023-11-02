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

7.Apply new unknown values
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: BASKARAN V
RegisterNumber:  212222230020



# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Read The File
dataset=pd.read_csv('Placement_Data_Full_Class.csv')
dataset
dataset.head(10)
dataset.tail(10)
# Dropping the serial number and salary column
dataset=dataset.drop(['sl_no','ssc_p','workex','ssc_b'],axis=1)
dataset
dataset.shape
dataset.info()

dataset["gender"]=dataset["gender"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset.info()

dataset["gender"]=dataset["gender"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset.info()
dataset
# selecting the features and labels
x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y
# dividing the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
dataset.head()
y_train.shape
x_train.shape
# Creating a Classifier using Sklearn
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0,solver='lbfgs',max_iter=1000).fit(x_train,y_train)
# Printing the acc
clf=LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
# Predicting for random value
clf.predict([[0,0,1,90,0,1,90,1,80]])
*/
```

## Output:
![Screenshot 2023-11-02 232718](https://github.com/BaskaranV15/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118703522/39e0d5f2-eb63-44ab-8050-41f24b3c0587)

![Screenshot 2023-11-02 232745](https://github.com/BaskaranV15/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118703522/0864efb4-e97a-4757-a03e-f4dc0f32bf0c)

![Screenshot 2023-11-02 232853](https://github.com/BaskaranV15/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118703522/92990a6d-bb80-4d79-95e9-7e6269bcc252)

![Screenshot 2023-11-02 232901](https://github.com/BaskaranV15/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118703522/155de535-353e-4aa4-a30b-f6f7a95f75f6)

![Screenshot 2023-11-02 232913](https://github.com/BaskaranV15/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118703522/a44b5339-3356-4304-bd55-80588d9af167)

![Screenshot 2023-11-02 232933](https://github.com/BaskaranV15/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118703522/4c5100a6-c94d-4b63-8f87-44af244694c3)

![Screenshot 2023-11-02 232954](https://github.com/BaskaranV15/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118703522/328a5efa-1065-4263-803a-61bd78a72f8b)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
