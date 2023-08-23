#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("C:/Users/UDHAYA KUMAR . R/Desktop/Oasis/archive (5).zip")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[11]:


df.head()


# In[18]:


df.drop("Unnamed: 0", axis=1, inplace=True)


# In[19]:


df.head()


# In[20]:


sns.pairplot(df)
plt.show()


# In[24]:


x = df['Sales']
y = df['TV']
plt.scatter(x,y)

plt.xlabel('Sales')
plt.ylabel('TV')
plt.title('Sales over TV advertisment')
plt.show()


# In[26]:


x = df['Sales']
y = df['Newspaper']

plt.scatter(x,y)

plt.xlabel('Sales')
plt.ylabel('Newspaper')
plt.title('Sales over Newspaper advertisment')
plt.show()


# In[27]:


x = df['Sales']
y = df['Radio']

plt.scatter(x,y)

plt.xlabel('Sales')
plt.ylabel('Radio')
plt.title('Sales over Radio advertisment')
plt.show()


# In[28]:


df.corr()


# In[34]:


plt.figure(figsize=(8,6))
sns.heatmap(df.corr(),annot=True)

plt.show()


# In[46]:


X = df.iloc[:,:-1]


# In[47]:


y = df['Sales']


# In[49]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)


# In[51]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train,y_train)


# In[54]:


#Prediciting the test results

y_pred = lr.predict(X_test)


# In[55]:


print(lr.coef_)


# In[56]:


print(lr.intercept_)


# In[57]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[60]:


result = pd.DataFrame({'Actual Outcome':y_test , 'Predicted_values':y_pred})
result.head()


# In[ ]:





# In[ ]:




