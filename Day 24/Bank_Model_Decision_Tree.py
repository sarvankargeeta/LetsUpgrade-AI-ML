# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:58:42 2020

@author: Lenovo
"""

import pandas as pd

dataset= pd.read_excel("E:/Rutuja/LockDown/AI and ML/Day 21 -Assignment and Notes/dataset/Bank_Personal_Loan_Modelling.xlsx", sheet_name=1)

dataset1=dataset.drop(["ID","ZIP Code"],axis=1)

dataset2=dataset1.dropna()

dataset3 = dataset2.drop_duplicates()

from sklearn.ensemble import RandomForestClassifier

import numpy as np

dataset2["CCAvg"]=np.round(dataset2["CCAvg"])

rf_model = RandomForestClassifier(n_estimators=1000,max_features=2,oob_score=True)

feature=['Age', 'Experience', 'Income', 'Family', 'CCAvg','Education', 'Mortgage','Securities Account','CD Account','Online','CreditCard']

rf_model.fit(X=dataset3[feature],y=dataset3["Personal Loan"])

print("OOB Accuracy:\n", rf_model.oob_score)

for feature, imp in zip(feature,rf_model.feature_importances_):
    print(feature,imp);
    
from sklearn import tree

tree_model=tree.DecisionTreeClassifier(max_depth=6,max_leaf_nodes=10)

predictors=pd.DataFrame([dataset3["Education"],dataset3["CCAvg"],dataset3["Income"]]).T

tree_model.fit(X=predictors, y=dataset3["Personal Loan"])
with open("Dtree.dot",'w')as f:
    f=tree.export_graphviz(tree_model,feature_names=["Education","CCAvg","Income"],out_file=f)
    