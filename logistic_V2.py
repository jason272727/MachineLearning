# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:56:28 2020

@author: User
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
print('=====================STEP1=先找出影響率最大的的欄位======================')
titanic=pd.read_csv('C:/Users/User/Desktop/pypractice/titanic.csv')
label_encoder=preprocessing.LabelEncoder()
titanic['Age']=titanic['Age'].fillna(value=titanic['Age'].median())
encoded_class=label_encoder.fit_transform(titanic['PClass'])
x=pd.DataFrame([encoded_class,titanic['Age'],titanic['SexCode']]).T
x.columns=['PClass','Age','SexCode']
y=titanic['Survived']
logistic=LogisticRegression()
logistic.fit(x,y)
print(logistic.coef_)
print('====================STEP2=根據上述結果過濾影響不大之欄位=================')
x=pd.DataFrame([encoded_class,titanic['SexCode']]).T
x.columns=['PClass','SexCode']
y=titanic['Survived']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33,random_state=9)
logistic=LogisticRegression()
logistic.fit(xtrain,ytrain)
print('train_data正確率',logistic.score(xtrain,ytrain))
print('test_data正確率',logistic.score(xtest,ytest))
print('==================STEP3=開始預測==============')
pred_survived=logistic.predict(xtest)
print(pd.crosstab(pred_survived,ytest))
print('正確率:',((275+71)/(275+83+5+71)))
pred_data=logistic.predict([[1,0],[1,1],[2,0],[2,1],[3,0],[3,1]])
print(pred_data)