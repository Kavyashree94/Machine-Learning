#!/usr/bin/env python
# coding: utf-8

# Code Below provides Implementation about Univariate Linear Regression

# #Model
# #Fwb(x) = w*x+b
# 
# #Cost Function 
# #J_w_b = 1/(2*m)*(Summation(Fwb(x) - y)^2) Where Summation runs for m features
# 
# #Gradient Descent for Univaraite Linear Regression
# #Repeat below steps untill convergence
# #tmp_w = w- alpha* Differentiation of J_w_b w.r.t w
# #tmp_b = b- alpha* Differentiation of J_temp_w_b w.r.t b
# #w = tmp_w
# #b = tmp_b
# 
# 
# dJ/dw = -2/n *(Sum(x(y-ypred)))
# dJ/db = -2/n *(Sum(y-ypred))
# 
# 
# 
# An epoch means one full pass through the entire training dataset.
# 
# If your dataset has 1000 samples, one epoch means the model has seen all 1000 samples once.

# In[5]:


import numpy as np


# In[16]:


#Input
HouseDimension = np.array([1,2,3,4])
HousePrices    = np.array([100,200,200,100])
w = 10
b = 10
m = HouseDimension.size #Number of features
learning_rate = 0.01
epochs = 100

# Gradient Descent
for epoch in range(epochs):
    # Predictions
    y_pred = w * HouseDimension + b

    # Compute gradients
    dw = (-2 / m) * np.sum(HouseDimension * (HousePrices- y_pred))
    db = (-2 / m) * np.sum(HousePrices - y_pred)

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    # Optional: Print loss and parameters
    if epoch % 100 == 0:
        loss = np.mean((HousePrices - y_pred) ** 2)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

# Final parameters
print("\nFinal model:")
print(f"w = {w:.4f}, b = {b:.4f}")


# In[ ]:




