#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras import regularizers, optimizers
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
import numpy as np


# In[4]:


import pandas as pd
testmd=pd.read_csv(r'D:/ML data/csv files/skin cancer/2019/ISIC_2019_Test_Metadata.csv')
trainmd=pd.read_csv(r'D:/ML data/csv files/skin cancer/2019/ISIC_2019_Training_Metadata.csv')
testmd.head(),trainmd.head()


# In[5]:


class_site=set(trainmd["anatom_site_general"])
print(class_site)
len(class_site)


# In[6]:


def fun1(site):
    if(site=="anterior torso"):
        return 0
    elif(site=="head/neck"):
        return 1
    elif(site=="lateral torso"):
        return 2
    elif(site=='lower extremity'):
        return 3
    elif(site=="oral/genital"):
        return 4
    elif(site=="palms/soles"):
        return 5
    elif(site=="posterior torso"):
        return 6
    elif(site=="upper extremity"):
        return 7
    else:
        return site
trainmd["anatom_site_general"]=[fun1(i) for i in list(trainmd["anatom_site_general"])]
testmd["anatom_site_general"]=[fun1(i) for i in list(testmd["anatom_site_general"])]
trainmd["anatom_site_general"][0:10],testmd["anatom_site_general"][0:10]


# In[7]:


trainmd=trainmd.drop('lesion_id',axis=1)


# In[ ]:





# In[ ]:





# In[8]:


def fun(gender):
    if(gender=="male"):
        return 1
    elif(gender=="female"):
        return 0
    else:
        return gender
    
gen=list(trainmd["sex"])
mod_gen=[]
for s in gen:
    mod_gen.append(fun(s))
trainmd["sex"]=mod_gen

gen1=list(testmd["sex"])
mod_gen1=[]
for s1 in gen1:
    mod_gen1.append(fun(s1))
testmd["sex"]=mod_gen1


trainmd.head(),testmd.head()


# In[9]:


trainmd.isna().sum(),testmd.isna().sum()


# In[10]:


trainmd["sex"]=trainmd["sex"].fillna(round(trainmd["sex"].median()))
trainmd["age_approx"]=trainmd["age_approx"].fillna(trainmd["age_approx"].median())
trainmd["anatom_site_general"]=trainmd["anatom_site_general"].fillna(round(trainmd["anatom_site_general"].median()))

testmd["sex"]=testmd["sex"].fillna(round(testmd["sex"].median()))
testmd["age_approx"]=testmd["age_approx"].fillna(testmd["age_approx"].median())
testmd["anatom_site_general"]=testmd["anatom_site_general"].fillna(round(testmd["anatom_site_general"].median()))


# In[11]:


trainmd["sex"]=trainmd["sex"].apply(int)
trainmd["age_approx"]=trainmd["age_approx"].astype(int)
trainmd["anatom_site_general"]=trainmd["anatom_site_general"].astype(int)

testmd["sex"]=testmd["sex"].apply(int)
testmd["age_approx"]=testmd["age_approx"].astype(int)
testmd["anatom_site_general"]=testmd["anatom_site_general"].astype(int)


# In[12]:


trainmd.isna().sum(),testmd.isna().sum()


# In[13]:


trainmd[["age_approx","anatom_site_general","sex"]]=trainmd[["age_approx","anatom_site_general","sex"]].astype(int)
testmd[["age_approx","anatom_site_general","sex"]]=testmd[["age_approx","anatom_site_general","sex"]].astype(int)


# In[14]:


trainmd.info(),testmd.info()


# In[15]:


trainmd.head(),testmd.head()


# In[16]:


print(trainmd.head())


# In[17]:


print(trainmd.head())


# In[20]:


trainmd.head(10),testmd.head(10)


# In[21]:


pca_train_features=pd.read_csv("D:/ML data/csv files/skin cancer/2019/pca_train_features1.csv")
pca_train_features.drop(pca_train_features.columns[[0]], axis = 1, inplace = True)
pca_train_features.drop(pca_train_features.columns[[0]], axis = 1, inplace = True)
pca_train_features.head()


# In[22]:


print(pca_train_features.head())


# In[16]:


pca_train_features.shape


# In[17]:


traindf=pd.read_csv(r'D:/ML data/csv files/skin cancer/2019/ISIC_2019_Training_GroundTruth.csv')
traindf[["MEL","NV","BCC","AK","BKL","DF","VASC","SCC","UNK"]] = traindf[["MEL","NV","BCC","AK","BKL","DF","VASC","SCC","UNK"]].astype(int)
traindf.head()


# In[18]:


pca_test_features=pd.read_csv("D:/ML data/csv files/skin cancer/2019/pca_test_features1.csv")
pca_test_features["age_approx"]=testmd["age_approx"]
pca_test_features["anatom_site_general"]=testmd["anatom_site_general"]
pca_test_features["sex"]=testmd["sex"]
pca_test_features.drop(pca_test_features.columns[[0]], axis = 1, inplace = True)
pca_test_features.head()


# In[19]:


pca_test_features.shape


# In[20]:


mn=min(pca_train_features.min())
mx=max(pca_train_features.max())

def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


# In[21]:


# assign array and range
site=trainmd["anatom_site_general"]
site=list(site)
normalized_site = normalize(site,mn,mx)

trainmd["anatom_site_general"]=normalized_site


age=trainmd["age_approx"]
age=list(age)
normalized_age = normalize(age,mn,mx)

trainmd["age_approx"]=normalized_age


# In[22]:


age1=testmd["age_approx"]
age1=list(age1)
normalized_age1 = normalize(age1,mn,mx)

testmd["age_approx"]=normalized_age1


site1=testmd["anatom_site_general"]
site1=list(site1)
normalized_site1 = normalize(site1,mn,mx)

testmd["anatom_site_general"]=normalized_site1


# In[ ]:





# In[23]:


train_x=pca_train_features
train_x.shape


# In[24]:


train_y=traindf[["MEL","NV","BCC","AK","BKL","DF","VASC","SCC","UNK"]]


# In[25]:


train_y.shape


# In[26]:


X_test=pca_test_features
X_test.shape


# In[27]:


# train_x       #train_x ready


# In[28]:


# train_y       #train_y ready


# In[29]:


# X_test        #test_x ready


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(train_x, train_y, test_size = 0.10, random_state = 0)


# In[32]:


X_train.shape,X_val.shape


# In[33]:


X_train=np.expand_dims(X_train, axis=2)


# In[34]:


X_val=np.expand_dims(X_val, axis=2)


# In[35]:


X_test=np.expand_dims(X_test, axis=2)


# # Training part
# 

# In[41]:


from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.merge import concatenate
from keras.regularizers import l2


# In[42]:


model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3,kernel_regularizer=l2(0.01), activation='relu', input_shape=(1002,1)))   #1002,1
model.add(BatchNormalization())
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
# model.add(BatchNormalization())
model.add(Dropout(0.25))
# model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2)) 
# model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(9, activation='softmax'))


# In[43]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[44]:


# model.fit(X_train, Y_train, epochs=1, verbose=1)


# In[45]:


from sklearn.model_selection import KFold, StratifiedKFold


# In[46]:


#fitting data

model.fit(
    X_train, Y_train,
#     steps_per_epoch=22797//32,    #total images in training dataset//batch_size
    validation_data=(X_val,Y_val),
    shuffle=True,
    epochs=5
)

