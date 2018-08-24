# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 00:50:44 2018

@author: Sarvesh Dubey
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

dataset=pd.read_csv('Salary_Data.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#Feature Scaling



#fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting the test set result
y_pred=regressor.predict(x_test)

#feature plotting
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary Vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
