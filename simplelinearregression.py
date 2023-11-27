#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Salary_Data.csv")
dataset.head()


# In[3]:


x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


# In[5]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


# In[6]:


from sklearn.linear_model import LinearRegression
Regression=LinearRegression()
Regression.fit(x_train,y_train)


# In[7]:


y_pred=Regression.predict(x_test)
y_pred


# In[8]:


y_test


# In[10]:


plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,Regression.predict(x_train),color="blue")
plt.title("Salary vs Experience")
plt.xlabel("Salary")
plt.ylabel("Experience")
plt.show()


# In[12]:


plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,Regression.predict(x_train),color="blue")
plt.title("Salary vs Experience")
plt.xlabel("Salary")
plt.ylabel("Experience")
plt.show()


# In[ ]:




