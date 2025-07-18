#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries

# In[137]:


import numpy as np
import matplotlib.pyplot as plt


# Training Data: x_train and y_train variables. The data is stored in one-dimensional NumPy arrays.

# In[138]:


# x_Adv_train is the input variable (Price of an advertizement in rupees)
# y_Rev_train is the target (Price of sales revence of the item in rupees)
x_Adv_train = np.array([1000.0, 2000.0])
y_Rev_train = np.array([2000.0, 3000.0])


# Using scatter() function in the matplotlib library

# In[139]:


plt.scatter(x_Adv_train, y_Rev_train, marker='o', c='b')
plt.title("Predict Sales Revence from Advertizement")
plt.ylabel('Sales Revenue')
plt.xlabel('Advertizement Cost')
plt.show()


# Assign w and b values for cost function J(x)

# Calculate Function F w,b(X)

# In[140]:


def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros_like(x)
    for i, x_i in enumerate(x):
        f_wb[i] = w * x_i + b
        print(f_wb[i])
    
    return f_wb


# Calculate cost function for F(x)= wx+b

# In[141]:


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0.0

    for i, x_i in enumerate(x):
        f_wb = w * x_i + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost

    return cost_sum / (2 * m)


# Calculate J(x) and Plot data to print output

# In[159]:


w = 1
b = 1000
tmp_f_wb = compute_model_output(x_Adv_train, w, b)

tmp_J_w = compute_cost(x_Adv_train, y_Rev_train, w, b)


# In[162]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# First subplot - training
plt.subplot(1, 2, 1)
plt.plot(x_Adv_train, tmp_f_wb, c='y', label='Our Prediction')
plt.scatter(x_Adv_train, y_Rev_train, marker='o', c='b', label='Actual Values')
plt.title('Training Data')
plt.xlabel('Advertising Spend')
plt.ylabel('Revenue')
plt.legend()
plt.grid(True)
print('Cost function Result',tmp_J_w)

