# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:00:22 2020

@author: Lenovo
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected =True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

attrition = attrition.drop(['Attrition_numerical'],axis=1)
attrition_num = attrition[numerical]

train, test,target_train,target_val = train_test_split(attrition_final, target, train_size=0.80, random_size=0)
