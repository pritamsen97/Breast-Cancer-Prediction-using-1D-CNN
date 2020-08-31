#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow-gpu==2.0.0-rc0')


# In[2]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D,Dropout,BatchNormalization,Flatten,Dense,MaxPool1D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

print(tf.__version__)


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


from sklearn import datasets,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[8]:


cancer=datasets.load_breast_cancer()


# In[9]:


print(cancer.DESCR)


# In[10]:


X=pd.DataFrame(data=cancer.data,columns=cancer.feature_names)
X.head()


# In[11]:


X.shape


# In[12]:


y=cancer.target


# In[13]:


y


# In[14]:


cancer.target_names


# In[15]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y)


# In[16]:


X_train.shape


# In[17]:


X_test.shape


# In[18]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# In[19]:


X_train=X_train.reshape(455,30,1)
X_test=X_test.reshape(114,30,1)


# In[20]:


epochs=50
model=Sequential()
model.add(Conv1D(filters=32,kernel_size=2,activation='relu',input_shape=(30,1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters=64,kernel_size=2,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))


# In[21]:


model.summary()


# In[22]:


model.compile(optimizer=Adam(lr=0.00005),loss='binary_crossentropy',metrics=['accuracy'])


# In[23]:


history=model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test),verbose=1)


# In[26]:


def plot_learningCurve(history,epoch):
  epoch_range=range(1,epoch+1)
  plt.plot(epoch_range,history.history['accuracy'])
  plt.plot(epoch_range,history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train','Val'],loc='upper left')
  plt.show()

  epoch_range=range(1,epoch+1)
  plt.plot(epoch_range,history.history['loss'])
  plt.plot(epoch_range,history.history['val_loss'])
  plt.title('Model loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(['Train','Val'],loc='upper left')
  plt.show()


# In[25]:


plot_learningCurve(history,epochs)

