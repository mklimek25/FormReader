#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pickle


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from tensorflow import keras


# In[5]:


from tensorflow.keras.datasets import mnist


# In[6]:


(xtrain, ytrain), (xtest, ytest) = mnist.load_data()


# In[7]:


xtest.shape


# In[8]:


from tensorflow.keras.utils import to_categorical


# In[9]:


y_cat_test = to_categorical(ytest, 10)
y_cat_train = to_categorical(ytrain, 10)


# In[10]:


y_cat_train[0]


# In[11]:


xtrain = xtrain/xtrain.max()


# In[12]:


xtest = xtest/xtest.max()


# In[13]:


type(xtest)


# In[14]:


scaled_image = xtrain[0]


# In[15]:


print(scaled_image)


# In[16]:


plt.imshow(scaled_image, cmap='gray')


# In[17]:


xtrain = xtrain.reshape(60000, 28,28,1)


# In[18]:


xtest = xtest.reshape(10000, 28, 28, 1)


# In[19]:


from tensorflow.keras.models import Sequential


# In[20]:


from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


# In[21]:


model = Sequential()

# Convolutional Layer
model.add(Conv2D(filters=32, kernel_size = (4,4), input_shape=(28,28,1), activation='relu'))


# In[22]:


# 2D to 1D
model.add(Flatten())
# DENSE LAYER
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# model.summary()

# In[23]:


model.summary()


# In[23]:


model.fit(xtrain, y_cat_train, epochs=2)


# In[24]:


model.metrics_names


# In[25]:


model.evaluate(xtest, y_cat_test)


# In[62]:


len(xtest)


# In[60]:


# below is where you can test new images 
predictions = model.predict_classes(xtest)


# In[68]:


print(xtest.shape)


# In[67]:


xtest[0]


# In[43]:


model.save('trying_this_now')


# In[70]:


help(model.predict)


# In[72]:


from tensorflow.keras.models import load_model


# In[87]:


loaded_model = load_model("trying_this_now")


# In[90]:


predictions = loaded_model.predict_classes(xtest)


# In[91]:


predictions


# In[ ]:




