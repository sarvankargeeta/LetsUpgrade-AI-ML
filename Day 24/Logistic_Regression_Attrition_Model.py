# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:25:41 2020

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

import statsmodels.api as sm


Y = dataset7.Attrition

#print(dataset7.columns)

X = dataset7[['Age','BusinessTravel', 'Department', 'DistanceFromHome','Education', 'EducationField', 'Gender', 'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked','PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears','TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion','YearsWithCurrManager']]

X1=sm.add_constant(X)

Logistic_Attrition = sm.Logit(Y, X1)

result=Logistic_Attrition.fit()

print(result.summary())