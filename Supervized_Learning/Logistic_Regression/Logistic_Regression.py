#!/usr/bin/env python
# coding: utf-8

# In[11]:


import math
import numpy as np
X = [1,2,3,4,5,6,7,8,9,10]
# Convert list to numpy array for math operations
X = np.array(X)
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # Binary target
result = 1 / (1 + np.exp(-X))
print(result)

import matplotlib.pyplot as plt

# Plot sigmoid transformation for each feature column
plt.figure(figsize=(10, 4))

# Plot first feature
plt.plot(X, result, 'bo')
plt.title('Feature 1: Original vs Sigmoid')
plt.xlabel('Original X[:,0]')
plt.ylabel('Sigmoid(X[:,0])')


plt.tight_layout()
plt.show()



# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Convert X to 2D array (10, 1)
X = np.array(X).reshape(-1, 1)
y = np.array(y)

print(X)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Create values for smooth curve
x_range = np.linspace(min(X), max(X), 300).reshape(-1, 1)
probs = model.predict_proba(x_range)[:, 1]  # Probability of class 1

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='black', label='Data')
plt.plot(x_range, probs, color='blue', linewidth=2, label='Logistic Regression Curve')
plt.axhline(0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
plt.xlabel('Feature')
plt.ylabel('Probability')
plt.title('Logistic Regression Fit (1D Feature)')
plt.legend()
plt.grid(True)
plt.show()


# In[17]:


# Predict on test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[18]:


# Step 5: Calculate decision boundary (where probability = 0.5)
boundary = -model.intercept_[0] / model.coef_[0][0]

# Step 6: Plot everything
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='black', label='Data Points')
plt.plot(x_range, probs, color='blue', label='Sigmoid Curve')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold = 0.5')
plt.axvline(boundary, color='green', linestyle='--', label=f'Decision Boundary ≈ {boundary:.2f}')
plt.xlabel('Feature Value (X)')
plt.ylabel('Predicted Probability')
plt.title('Logistic Regression with Decision Boundary (1D)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




