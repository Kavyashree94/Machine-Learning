#!/usr/bin/env python
# coding: utf-8

# Multivariable Linear Regression
# Fwb(X) = W.X +B

# In[21]:


import numpy as np

House_Dimensions = np.array([[2, 5, 2, 4], [6, 9, 8, 4], [9, 9, 1, 5]])
House_Prices = np.array([4, 5, 1])

w = np.array([1.,2.,3.,4.])
b = 10

print(House_Dimensions.shape)
print(w.shape)

y_pred = np.dot(House_Dimensions, w) + b   
print("Predicted Y:",y_pred)

#Cost

m = House_Dimensions.shape[0]
cost = 0.0
for i in range(m):                                
    f_wb_i = np.dot(House_Dimensions[i], w) + b           #(n,)(n,) = scalar (see np.dot)
    cost = cost + (f_wb_i - House_Prices[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    

print("COST Function result:",cost)


# In[22]:


# Hyperparameters
learning_rate = 0.000001
epochs = 1000

# Gradient Descent Loop
for epoch in range(epochs):
    # Predict: ŷ = Xw + b
    y_pred = House_Dimensions.dot(w) + b  # Shape: (3,)

    # Compute gradients
    dw = (-2 / m) * House_Dimensions.T.dot(House_Prices - y_pred)  # Shape: (4,)
    db = (-2 / m) * np.sum(House_Prices - y_pred)   # Scalar

    # Update weights and bias
    w -= learning_rate * dw
    b -= learning_rate * db

    # Optional: Print every 100 epochs
    if epoch % 100 == 0:
        loss = np.mean((House_Prices - y_pred) ** 2)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Weights = {w}, Bias = {b:.4f}")

# Final result
print("\nTrained Weights:", w)
print("Trained Bias:", b)


# In[ ]:




