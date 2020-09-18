# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:58:42 2020

@author: Lenovo
"""

import pandas as pd

from sklearn import preprocessing

dataset4= pd.read_csv("E:/Rutuja/LockDown/AI and ML/Day 24 Notes and Assignment/general_data.csv")

le=preprocessing.LabelEncoder()

dataset4.columns

dataset4["Attrition"] = le.fit_transform(dataset4["Attrition"])

dataset4["BusinessTravel"] = le.fit_transform(dataset4["BusinessTravel"])

dataset4["Department"] = le.fit_transform(dataset4["Department"])

dataset4["EducationField"] = le.fit_transform(dataset4["EducationField"])

dataset4["Gender"] = le.fit_transform(dataset4["Gender"])

dataset4["MaritalStatus"] = le.fit_transform(dataset4["MaritalStatus"])

dataset4["JobRole"] = le.fit_transform(dataset4["JobRole"])

dataset5=dataset4.drop(["EmployeeCount","EmployeeID","Over18","StandardHours"],axis=1)

from sklearn.ensemble import RandomForestClassifier

dataset6=dataset5.dropna()

dataset7=dataset6.drop_duplicates()

rf_model=RandomForestClassifier(n_estimators=1000, max_features=2, oob_score=True)

features=['Age','BusinessTravel','Department', 'DistanceFromHome','Education', 'EducationField', 'Gender','JobLevel','JobRole','MaritalStatus','MonthlyIncome','NumCompaniesWorked','PercentSalaryHike','StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear','YearsAtCompany','YearsSinceLastPromotion','YearsWithCurrManager']

rf_model.fit(X=dataset7[features],y=dataset7["Attrition"])

print("OOB Accuracy:\n", rf_model.oob_score)

for feature, imp in zip(features,rf_model.feature_importances_):
    print(feature,imp);
    
from sklearn import tree

tree_model=tree.DecisionTreeClassifier(max_depth=6,max_leaf_nodes=10)

predictors=pd.DataFrame([dataset7["Age"],dataset7["MonthlyIncome"],dataset7["TotalWorkingYears"]]).T

tree_model.fit(X=predictors, y=dataset7["Attrition"])
with open("Dtree2.dot",'w')as f:
    f=tree.export_graphviz(tree_model,feature_names=["TotalWorkingYears","Age","MonthlyIncome"],out_file=f)
    