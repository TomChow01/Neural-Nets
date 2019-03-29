
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:



X = [[0.1,0.2,0.3],[0.5,0.6,0.7]]
Y = [0.1,0.3]


W = np.array([0.0,0.0,0.0])
B = 0.0


# In[3]:


X = np.asarray(X)
Y = np.asarray(Y)


# In[4]:


def f(w,b,x):
    return 1.0/(1.0+np.exp(-(np.dot(x,w)+b)))


# In[5]:


def error(w,b):
    error = 0.0
    for i in range(len(Y)):
        fx = f(w,b,X[i])
        error += 0.5*(fx-Y[i])**2
    return error


# In[6]:


def grad_b(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y)*fx*(1-fx)


# In[7]:


def grad_w(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y)*fx*(1-fx)*x


# In[8]:


def grad_descent():
    w,b,eta,max_epochs = [-2.0,-2.0,-2.0],-2.0,1.0,100000
    for i in range(max_epochs):
        dw,db = [0.0,0.0,0.0],0.0
        for j in range(len(X)):
            dw += grad_w(w,b,X[j],Y[j])
            db += grad_b(w,b,X[j],Y[j])
        w = w-eta*dw
        b = b-eta*db
    return w,b
        


# In[9]:


grad_descent()


# In[10]:


W,B = grad_descent()


# In[11]:


f(W,B,X)


# In[12]:


import matplotlib.pyplot as plt

