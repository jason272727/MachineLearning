# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:09:04 2020

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
boston=datasets.load_boston()
x=pd.DataFrame(boston.data,columns=boston.feature_names)
y=boston['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=1)
lm=LinearRegression()
lm.fit(x_train,y_train)
pred_target=lm.predict(x_test)
plt.scatter(y_test,pred_target)
plt.plot([0,50],[0,50],'r')
plt.rcParams["axes.unicode_minus"]=False
plt.rcParams["font.sans-serif"]="Microsoft JhengHei"
plt.xlabel('原始房價')
plt.ylabel('預測房價')
plt.show()
print('--------------------模型績效------------------------')
pred_test=lm.predict(x_test)
pred_train=lm.predict(x_train)
train_MSE=np.mean((y_train-pred_train)**2)
test_MSE=np.mean((y_test-pred_test)**2)
print('訓練資料MSE:',train_MSE)
print('測試資料MSE:',test_MSE)
print('訓練資料正確率:',lm.score(x_train,y_train))
print('測試資料正確率:',lm.score(x_test,y_test))
print('=================Outlier================')
plt.scatter(pred_train,(pred_train-y_train),color='b',label='Train_data')
plt.scatter(pred_test,(pred_test-y_test),color='r',label='test_data')
plt.hlines(y=0, xmin=0, xmax=50)
plt.legend()
plt.show()