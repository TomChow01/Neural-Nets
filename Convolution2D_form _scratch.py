
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[133]:


def convolution2d(image, kernel):
    m, n = kernel.shape
    if (m == n):
        h,w = image.shape
        h = h - m + 1
        w = w - m + 1
        new_image = np.zeros((h,w))
        for i in range(h):
            for j in range(w):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    return new_image


# In[134]:


def import_img(name):
    image = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
    shape = image.shape
    print(shape)
    return image
    


# In[135]:


def show(image):
    plt.imshow(image,cmap='gray')
    plt.show()


# In[151]:


def kernel(n):
    matrix = [[int(input()) for j in range(n)] for i in range(n)]
    kernel = np.asarray(matrix)
    return kernel
            

