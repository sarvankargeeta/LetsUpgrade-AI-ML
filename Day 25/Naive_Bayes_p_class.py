# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:51:39 2020

@author: Lenovo
"""


import pandas as pd
dataset=pd.read_csv("E:/Rutuja/LockDown/AI and ML/Day 25 Notes/train.csv")

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import *

le=preprocessing.LabelEncoder()

le.fit(dataset['Sex'])

print(le.classes_)
 
dataset["Sex"]=le.transform(dataset["Sex"])
 
y=dataset['Pclass']

X=dataset.drop(["Pclass","PassengerId","Name","Ticket","Cabin","Embarked"],axis=1)

y.count()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

X_train.head()

clf=BernoulliNB()

y_pred=clf.fit(X_train,y_train).predict(X_test)