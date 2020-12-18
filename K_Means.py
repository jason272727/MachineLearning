# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 22:05:31 2020

@author: User
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing,cluster
import sklearn.metrics as sm
iris=pd.read_csv('C:/Users/User/Desktop/pypractice/iris.csv')
label_encoder=preprocessing.LabelEncoder()
encoded_target=label_encoder.fit_transform(iris['target'])
x=pd.DataFrame([iris['sepal_length'],iris['sepal_width'],iris['petal_length'],iris['petal_width']]).T
y=pd.DataFrame([encoded_target]).T
y.columns=['target']
colmap=np.array(['r','g','b'])
k=3
k_means=cluster.KMeans(n_clusters=k,random_state=9)
k_means.fit(x)
print('正確率:',sm.accuracy_score(y, k_means.labels_))
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(x['petal_length'],x['petal_width'],color=colmap[encoded_target])
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('Original Data')
plt.subplot(1,2,2)
plt.scatter(x['petal_length'],x['petal_width'],color=colmap[k_means.labels_])
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.title('K-Means')
plt.show()
print('=====分類結果=====')
print(sm.confusion_matrix(y,k_means.labels_))