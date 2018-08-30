# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 01:46:26 2018

@author: Sarvesh Dubey
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
#to set the image size
plt.rcParams['figure.figsize']=(20.0,10.0)
 
data=pd.read_csv('headbrain.csv')
print(data.shape)
data.head()

X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values

#print(X)
#print(Y)

mean_x=np.mean(X)
mean_y=np.mean(Y)

m=len(X)

numer=0
denom=0
for i in range(m):
    numer+=(X[i]-mean_x)*(Y[i]-mean_y)
    denom+=(X[i]-mean_x)**2
b1=numer/denom
b0=mean_y-(b1*mean_x)

print(b1,b0)

rmse=0
for i in range(m):
    y_pred=b0+b1*X[i]
    rmse+=(Y[i]-y_pred)**2
    
rmse=np.sqrt(rmse/m)
#print(rmse)

sst=0
ssr=0

for i in range(m):
    y_pred=b0+b1*X[i]
    sst+=(Y[i]-mean_y)**2
    ssr+=(Y[i]-y_pred)**2
r2=1-(ssr)/sst
print(r2)
