# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:12:40 2020

@author: praji
"""


import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../Data/iris.csv')
#show sample data
df.head()

from sklearn.model_selection import train_test_split

features = df.drop('target',axis=1).values
label = df['target'].values

X_train,X_test, y_train,y_test = train_test_split(features,label,test_size=0.2)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train).reshape(-1,1)
y_test = torch.LongTensor(y_test).reshape(-1,1)

#using pytorch 
from torch.utils.data import TensorDataset, DataLoader

data = df.drop('target',axis=1).values
labels = df['target'].values
iris = TensorDataset(torch.FloatTensor(data),torch.FloatTensor(labels))
iris_loader = DataLoader(iris,batch_size = 50, shuffle=True)


