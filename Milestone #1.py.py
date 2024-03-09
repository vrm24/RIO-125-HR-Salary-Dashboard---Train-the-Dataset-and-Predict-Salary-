#!/usr/bin/env python
# coding: utf-8

# # 1. Importing dataset

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#importing dataset
df=pd.read_csv('salary_data.csv')


# In[4]:


#READ DATA
df.head()


# In[5]:


df.tail()


# In[6]:


#CHECK DATA INFORMATION
df.info()


# In[7]:


#DESCRIBE DATA
df.describe()


# In[8]:


#CHECK SHAPE OF DATA
df.shape


# In[9]:


#CHECK DATA COLUMNS
df.columns


# In[10]:


#CHECK UNIQUE VALUES
df.nunique()
     


# # 2. EDA

# In[11]:


df.hist(figsize=(20,15))
plt.show()


# In[13]:


#Generates histogram specifying age column in the dataset
plt.hist(df['age'])
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(np.arange(20,100,5))
plt.rcParams['figure.figsize'] = (10,10)
plt.title('Age Distribution')
plt.show()


# In[14]:


#generates a bar plot that effectively visualizes the distribution of education levels in the dataset
df['education'].value_counts().plot(kind = 'bar')
plt.xlabel('Education')
plt.ylabel('Count')
plt.rcParams['figure.figsize'] = (10,10)
plt.title('Education level')
plt.show()


# In[15]:


# creates a bar plot showing the count of each category in the 'sex' column of the dataset
sns.countplot(x=df['sex'])
plt.title('gender count')


# In[16]:


#generates a pie chart to visualize the distribution of marital status categories in the dataset
df['marital-status'].value_counts().plot(kind = 'pie',subplots=True, autopct='%1.1f%%')
plt.rcParams['figure.figsize'] = (10,10)
plt.title('marital status')
plt.show()


# In[17]:


# generates a pie chart to visualize the distribution of relationship categories in the dataset
df['relationship'].value_counts().plot(kind = 'pie',subplots=True, autopct='%1.1f%%')
plt.rcParams['figure.figsize'] = (10,10)
plt.title('Relationship')
plt.show()


# In[18]:


# generates a count plot to visualize the distribution of different occupations in the dataset
sns.countplot(x=df['occupation'])
plt.xticks(rotation=90)       #xticks = sets the tickmarks on x axis
plt.title('occupation status')
plt.show


# In[ ]:




