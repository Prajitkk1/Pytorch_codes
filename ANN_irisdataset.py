# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:26:29 2020

@author: praji
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

class Model(nn.Module):
    def __init__(self, in_features=4,h1=8,h2=8,out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    

model = Model()
from sklearn.model_selection import train_test_split
df = pd.read_csv('../Data/iris.csv')

features = df.drop('target',axis=1).values
label = df['target'].values

X_train,X_test, y_train,y_test = train_test_split(features,label,test_size=0.2)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

epochs=100
losses = list()
for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred,y_train)
    losses.append(loss)
    #back propogation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#validation
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval,y_test)
    
#saving model
torch.save(model.state_dict(),'iris_model.pt')

#loading
new_model = Model()
new_model.load_state_dict(torch.load('iris_model.pt'))
new_model.eval()


