#!/usr/bin/env python
# coding: utf-8

# # Fashion MNIST via TensorFlow #

# ## Preliminaries ##

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization

import numpy as np
import matplotlib.pyplot as plt
import _pickle
import gzip

with gzip.open('./data/fashion-mnist.pkl.gz', 'rb') as fp:
    (train_images, train_labels, test_images, test_labels) = _pickle.load(fp)


# In[2]:


train_images_reshape = train_images.reshape(train_images.shape[0],
                                            28, 28, 1)
test_images_reshape = test_images.reshape(test_images.shape[0],
                                         28, 28, 1)


# ## Simplistic Network ##

# In[3]:


model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])


# In[4]:


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[5]:


model.fit(train_images, train_labels,
          validation_data=(test_images, test_labels),
          epochs=100)


# In[7]:


keras.backend.image_data_format()


# ## Model 2 ##

# In[8]:


model2 = Sequential([
    Conv2D(32, [3, 3], input_shape=(28, 28, 1), activation='relu'),
    BatchNormalization(axis=-1),
    Conv2D(32, [3, 3], activation='relu'),
    BatchNormalization(axis=-1),
    AveragePooling2D(pool_size=(2,2)),
    Dropout(0.25),
    Conv2D(64, [3, 3], activation='relu'),
    BatchNormalization(axis=-1),
    Conv2D(64, [3, 3], activation='relu'),
    BatchNormalization(axis=-1),
    AveragePooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])


# In[9]:


model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])


# In[14]:


train_labels_cat.shape


# In[15]:


model2.fit(train_images_reshape,
          train_labels,
          validation_data=(test_images_reshape, test_labels),
          epochs=100,
          batch_size=25)


# Personal best of
# * loss: 0.1303
# * acc: 0.9526
# * val_loss: 0.2367
# * val_acc: 0.9252
# 
# with a NN over 25 epochs:
# * 2d CNN 32 (5, 5) relu
# * normalization, axis=-1
# * 2d CNN 32 (5, 5) relu
# * normalization, axis=-1
# * 2d max pooling (2, 2)
# * dropout p = 0.25
# * 2d CNN 64 (5, 5) relu
# * normalization, axis=-1
# * 2d CNN 64 (5, 5) relu
# * normalization, axis=-1
# * 2d max pooling (2, 2)
# * dropout p = 0.25
# * flatten
# * dense 512 relu
# * normalization
# * dropout p = 0.5
# * dense 10 softmax

# This model is clearly overfitting, going to try some regularization

# ## Model 3 ##

# In[17]:


model3 = Sequential([
    BatchNormalization(axis=-1, input_shape=(28, 28, 1)),
    Conv2D(64, [3, 3],
           activation='relu',
          bias_initializer='RandomNormal',
          kernel_initializer='random_uniform'),
    AveragePooling2D(pool_size=(2,2)),
    Conv2D(512, [3, 3],
           activation='relu'),
    BatchNormalization(axis=-1),
    AveragePooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128,
          activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])


# In[18]:


model3.compile(loss='sparse_categorical_crossentropy',
               optimizer='adam',
              metrics=['accuracy'])


# In[19]:


model3.fit(train_images_reshape, train_labels,
          validation_data=(test_images_reshape, test_labels),
          epochs = 100,
          batch_size = 25)


# In[ ]:




