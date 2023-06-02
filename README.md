# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Split data into training set and testing set.
3. Use CountVectorizer to extract features.
4. Import SVC and predict y values.
5. Find the accuracy of model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Keerthika N
RegisterNumber: 212221230049
*/
```
```
import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('spam.csv')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.fit_transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
* data.head():

![1](https://github.com/KeerthikaNagarajan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427089/866fddc6-81e3-4a29-ba5c-0f4d1d153837)

* data.info():

![2](https://github.com/KeerthikaNagarajan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427089/c25f65f4-6757-4300-aa75-c89b63993866)

* data.isnull().sum():

![3](https://github.com/KeerthikaNagarajan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427089/073714e2-5ecf-4c7b-b8f0-63b82f615d2c)

* Y_prediction value:
![4](https://github.com/KeerthikaNagarajan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427089/cb81c8bb-640e-450c-8ea8-32c95f2b8aa9)

* Accuracy value:

![5](https://github.com/KeerthikaNagarajan/Implementation-of-SVM-For-Spam-Mail-Detection/assets/93427089/b237de89-7480-484d-9bc7-55985a66f092)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
