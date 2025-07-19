#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np    # it is an unofficial standard to use np for numpy
import time
a = np.zeros(4);                
print(a)

a = np.arange(0, 4, 0.5);              
print(a)

a = np.array([5,4,3,2]);  
print(a)

a= np.arange(10)
print(a)

c = a[2:7:1];     print("a[2:7:1] = ", c)

c = a[2:7:2];     print("a[2:7:2] = ", c)

# access all elements index 3 and above
c = a[3:];        print("a[3:]    = ", c)

# access all elements below index 3
c = a[:3];        print("a[:3]    = ", c)

# access all elements
c = a[:];         print("a[:]     = ", c)

# negative incides
c = a[-1:];         print("a[-1:]     = ", c)

# negative incides
c = a[-2:];         print("a[-2:]     = ", c)


# In[20]:


a= np.arange(10)
print(a)
b= np.arange(10)
print(b)
print(f"a+b: {a + b}")
print(f"a*b: {a * b}")


# In[22]:


def my_dot(a, b): 
    """
   Compute the dot product of two vectors
 
    Args:
      a (ndarray (n,)):  input vector 
      b (ndarray (n,)):  input vector with same dimension as a
    
    Returns:
      x (scalar): 
    """
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x

np.random.seed(1)
a = np.random.rand(10000000)  # very large arrays
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #remove these big arrays from memory


# In[35]:


a = np.zeros((2, 5)) 
print(a)

a = np.array([[120], [100], [101]]);
print(a)

a = np.arange(6).reshape(3, 2)
print(a)






