# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:57:35 2020

@author: User
"""

import pandas as pd
from sklearn import preprocessing,tree
from sklearn.model_selection import train_test_split
from graphviz import Source
iris=pd.read_csv('C:/Users/User/Desktop/pypractice/iris.csv')
label_encoder=preprocessing.LabelEncoder()
encoded_target=label_encoder.fit_transform(iris['target'])
x=pd.DataFrame([iris['sepal_length'],iris['sepal_width'],iris['petal_length'],iris['petal_width']]).T
y=pd.DataFrame([encoded_target]).T
y.columns=['target']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=51)
lmtree=tree.DecisionTreeClassifier()
lmtree.fit(xtrain,ytrain)
print('train_data正確率:',lmtree.score(xtrain,ytrain))
print('test_data正確率:',lmtree.score(xtest,ytest))
with open('C:/Users/User/Desktop/pypractice/iristree_1217.dot','w') as f:
    f=tree.export_graphviz(lmtree,feature_names=['sepal_length','sepal_width','petal_length','petal_width'],out_file=f)

s=Source.from_file('C:/Users/User/Desktop/pypractice/iristree_1217.dot')
s.view()