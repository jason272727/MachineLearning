# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 23:54:54 2020

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing,neighbors
from sklearn.model_selection import train_test_split,cross_val_score
iris=pd.read_csv('C:/Users/User/Desktop/pypractice/iris.csv')
label_encoder=preprocessing.LabelEncoder()
encoded_target=label_encoder.fit_transform(iris['target'])
x=pd.DataFrame([iris['sepal_length'],iris['sepal_width'],iris['petal_length'],iris['petal_width']]).T
y=pd.DataFrame([encoded_target]).T
y.columns=['target']
accuries=[]
k_values=np.arange(1,round(0.2*len(x)+1))
print('===============交叉驗證 K值最佳化===============')
for k in k_values:
    knn=neighbors.KNeighborsClassifier(n_neighbors=k)
    score=cross_val_score(knn,x,y,scoring='accuracy',cv=10)
    accuries.append(score.mean())
plt.plot(k_values,accuries)
print('============')
k=13
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33,random_state=1)
knn=neighbors.KNeighborsClassifier(n_neighbors=k)
knn.fit(xtrain,ytrain)
print('test_data正確率:',knn.score(xtest,ytest))
print('預測資料:',knn.predict(xtest))
print('原始資料:',ytest.values.T)
