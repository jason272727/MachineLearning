# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:55:19 2020

@author: User
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
titanic=pd.read_csv('C:/Users/User/Desktop/pypractice/titanic.csv')
label_encoder=preprocessing.LabelEncoder()
encoded_class=label_encoder.fit_transform(titanic['PClass'])
titanic['Age']=titanic['Age'].fillna(value=titanic['Age'].median())
x=pd.DataFrame([encoded_class,titanic['Age'],titanic['SexCode']]).T
x.columns=['PClass','Age','SexCode']
y=titanic['Survived']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33,random_state=9)
logistic=LogisticRegression()
logistic.fit(xtrain,ytrain)
print(logistic.coef_)#可利用迴歸係數探討上述欄位，哪些影響生存率較重，以過濾一些欄位，提升準確率。
pred_survived=logistic.predict(xtest)
print('Train_data正確率:',logistic.score(xtrain,ytrain))
print('Test_data正確率:',logistic.score(xtest,ytest))
print('===================預測欲得知資料方法=================')
pred_data=logistic.predict(np.array([[1,60,1],[3,20,0],[2,50,0],[1,20,0]]))#陣列資料內容順序跟第16行一樣
#pred_data=logistic.predict(pd.DataFrame([[1,20,1],[3,20,0],[2,50,0],[1,20,0]]))
#pred_data=logistic.predict([[1,20,1],[3,20,0],[2,50,0],[1,20,0]])
print(pred_data)
print('====================================================')
print(pd.crosstab(pred_survived,ytest))
print((239+69)/(239+70+16+69))