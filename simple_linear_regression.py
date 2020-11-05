# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 11:52:02 2020

@author: praji
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
        #create a dataset
X = torch.linspace(1,50,50).reshape(-1,1)
        #error
e = torch.randint(-8,9,(50,1),dtype=torch.float)
        #creating Y with small error
y = 2*X + 1 + e

plt.scatter(X.numpy(),y.numpy())

        #creating Model
model = nn.Linear(1,1)
#print(model.weight)
#print(model.bias)
criterion = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(),lr=0.001)
epochs = 50
losses = []

for i in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred,y)
    losses.append(loss)
    optimiser.zero_grad()
    print(loss.item())
    loss.backward()
    optimiser.step()

plt.plot(y_pred.detach().numpy())
plt.scatter(X,y)