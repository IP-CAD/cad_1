#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing our cancer dataset
df1 = pd.read_csv('data2.csv')


# In[4]:


df1.head()


# In[5]:


df1.drop('id', axis = 1, inplace = True)


# In[6]:


df1


# In[85]:


X = df1.iloc[:, 1:32].values
Y = df1.iloc[:, 0].values


# In[88]:


print("Cancer data set dimensions : {}".format(df1.shape))


# In[89]:


df1.groupby('diagnosis').size()

#Visualization of data
df1.groupby('diagnosis').hist(figsize=(12, 12))


# In[97]:


df1.groupby('diagnosis').concavity_worst.hist()


# In[98]:


df1.isnull().sum()
df1.isna().sum()


# In[92]:


#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
# X = labelencoder_Y.fit_transform(X)


# In[93]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[94]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[95]:


from sklearn.metrics import confusion_matrix

#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
tp, fn, fp, tn = confusion_matrix(Y_test,Y_pred,labels=[1,0]).reshape(-1)
acc=(tn+tp)/(tn+fp+tp+fn)
se=tp/(tp+fn)
sp=tn/(tn+fp)
pr=tp/(tp+fp)
print('acc',acc)

#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
tp, fn, fp, tn = confusion_matrix(Y_test,Y_pred,labels=[1,0]).reshape(-1)
acc=(tn+tp)/(tn+fp+tp+fn)
se=tp/(tp+fn)
sp=tn/(tn+fp)
pr=tp/(tp+fp)
print('acc',acc)

