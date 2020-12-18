# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 12:25:18 2020

@author: User
"""

import pandas as pd
from sklearn import preprocessing,tree
from sklearn.model_selection import train_test_split
from graphviz import Source
titanic=pd.read_csv('C:/Users/User/Desktop/pypractice/titanic.csv')
label_encoder=preprocessing.LabelEncoder()
encoded_calss=label_encoder.fit_transform(titanic['PClass'])
x=pd.DataFrame([encoded_calss,titanic['SexCode']]).T
x.columns=['PClass','SexCode']
y=titanic['Survived']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=9)
lmtree=tree.DecisionTreeClassifier(max_depth=8)
lmtree.fit(xtrain,ytrain)
print('正確率:',lmtree.score(xtest,ytest))
with open('C:/Users/User/Desktop/pypractice/1217_tree1.dot','w') as f:
    f=tree.export_graphviz(lmtree,feature_names=['Class','Gender'],out_file=f)
path='C:/Users/User/Desktop/pypractice/1217_tree1.dot'
s=Source.from_file(path)
s.view()


