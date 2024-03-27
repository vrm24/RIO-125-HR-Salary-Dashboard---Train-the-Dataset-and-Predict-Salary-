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


# In[2]:


#importing dataset
df=pd.read_csv('salary_data.csv')


# In[3]:


#READ DATA
df.head()


# In[4]:


df.tail()


# In[5]:


#CHECK DATA INFORMATION
df.info()


# In[6]:


#DESCRIBE DATA
df.describe()


# In[7]:


#CHECK SHAPE OF DATA
df.shape


# In[8]:


#CHECK DATA COLUMNS
df.columns


# In[9]:


#CHECK UNIQUE VALUES
df.nunique()
     


# # 2. EDA

# In[10]:


df.hist(figsize=(20,15))
plt.show()


# In[11]:


#Generates histogram specifying age column in the dataset
plt.hist(df['age'])
plt.xlabel('Age')
plt.ylabel('Count')
plt.xticks(np.arange(20,100,5))
plt.rcParams['figure.figsize'] = (10,10)
plt.title('Age Distribution')
plt.show()


# In[12]:


#generates a bar plot that effectively visualizes the distribution of education levels in the dataset
df['education'].value_counts().plot(kind = 'bar')
plt.xlabel('Education')
plt.ylabel('Count')
plt.rcParams['figure.figsize'] = (10,10)
plt.title('Education level')
plt.show()


# In[13]:


# creates a bar plot showing the count of each category in the 'sex' column of the dataset
sns.countplot(x=df['sex'])
plt.title('gender count')


# In[14]:


#generates a pie chart to visualize the distribution of marital status categories in the dataset
df['marital-status'].value_counts().plot(kind = 'pie',subplots=True, autopct='%1.1f%%')
plt.rcParams['figure.figsize'] = (10,10)
plt.title('marital status')
plt.show()


# In[15]:


# generates a pie chart to visualize the distribution of relationship categories in the dataset
df['relationship'].value_counts().plot(kind = 'pie',subplots=True, autopct='%1.1f%%')
plt.rcParams['figure.figsize'] = (10,10)
plt.title('Relationship')
plt.show()


# In[16]:


# generates a count plot to visualize the distribution of different occupations in the dataset
sns.countplot(x=df['occupation'])
plt.xticks(rotation=90)       #xticks = sets the tickmarks on x axis
plt.title('occupation status')
plt.show


# # 3.DATA PREPROCESSING
# 

# # # 3.1 HANDLING MISSING VALUES 

# In[18]:


#check null values
df.isna().sum()


# In[19]:


#CHECK DATA TYPES
df.dtypes


# REPLACED MISSING VALUES WITH MEAN

# In[20]:


for i in ['capital-gain','capital-loss', 'hours-per-week']:
      df[i]=df[i].fillna(df[i].mean())


# In[21]:


#RECHECK NULL VALUES
df.isna().sum()


# # # 3.2 CORRELATION

# In[22]:


corrmatrix=df.corr()
plt.subplots(figsize=(10,4))
sns.heatmap(corrmatrix,vmin=0.2,vmax=0.9,annot=True,cmap='Blues')


# # 3.3 REMOVING OR MODIFYING OUTLIERS
# 

# OUTLIER DETECTION

# In[23]:


num_col = df.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(20,30))

for i, variable in enumerate(num_col):
                     plt.subplot(5,5,i+1)
                     plt.boxplot(df[variable],whis=1.5)
                     plt.tight_layout()
                     plt.title(variable)


# MODIFYING OUTLIERS
# 
# 

# In[25]:


for i in num_col:
    Q1=df[i].quantile(0.25) # 25th quantile
    Q3=df[i].quantile(0.75)  # 75th quantile
    IQR=Q3-Q1
    Lower_Whisker = Q1 - 1.5*IQR 
    Upper_Whisker = Q3 + 1.5*IQR
    df[i] = np.clip(df[i], Lower_Whisker, Upper_Whisker) 


# In[27]:


plt.figure(figsize=(20,30))

for i, variable in enumerate(num_col):
                     plt.subplot(5,5,i+1)
                     plt.boxplot(df[variable],whis=1.5)
                     plt.tight_layout()
                     plt.title(variable)


# In[29]:


from sklearn import preprocessing 
label= preprocessing.LabelEncoder()  
df['workclass']=label.fit_transform(df['workclass'])
df['education']=label.fit_transform(df['education'])
df['occupation']=label.fit_transform(df['occupation'])
df['sex']=label.fit_transform(df['sex'])
#df['salary']=label.fit_transform(df['salary'])
df['race']=label.fit_transform(df['race'])
df['native-country']=label.fit_transform(df['native-country'])
df['marital-status']=label.fit_transform(df['marital-status'])
df['relationship']=label.fit_transform(df['relationship'])


# In[30]:


df


# # Standard Scaling

# In[31]:


X=df.drop(columns=['salary'],axis=1)
y=df['salary']


# In[32]:


X.describe()


# In[33]:


X.columns


# In[34]:


#import library
from sklearn.preprocessing import StandardScaler


# In[35]:


#fitting the model
scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[46]:


type(X)


# In[48]:


X = np.random.rand(32561, 13)  # Example random data
columns = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'hours-per-week', 'native-country']

# Check the shape of X and columns
print("Shape of X:", X.shape)
print("Number of Columns:", len(columns))

X = X[:, :10]  # Selecting only the first 10 columns of X

# Now, create the DataFrame
df = pd.DataFrame(X, columns=columns)


# In[50]:


X=pd.DataFrame(X,columns = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'hours-per-week', 'native-country'])


# In[51]:


X


# In[52]:


from sklearn.model_selection import train_test_split


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# # Logistic regression
# 

# In[54]:


#import logistic regression
from sklearn.linear_model import LogisticRegression
lor=LogisticRegression()
lor_model=lor.fit(X_train,y_train)
y_predict_lor=lor_model.predict(X_test)


# In[55]:


y_predict_lor


# confusion matrix
# 
# 

# In[56]:


#import confusion matrix
from sklearn.metrics import confusion_matrix
     


# In[57]:


#confusion matrix
confusion_matrix(y_test,y_predict_lor)


# precision,accuracy and recall score
# 
# 

# In[58]:


#import precision score and recall score
from sklearn.metrics import precision_score,recall_score


# In[59]:


#precision score
precision_score(y_test,y_predict_lor,pos_label=1,average='micro')


# In[60]:


#recallscore
recall_score(y_test,y_predict_lor,pos_label=1,average='micro')


# In[61]:


#check accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict_lor)
     


# In[62]:


R1=accuracy_score(y_test,y_predict_lor)*100
R1
     


# In[ ]:




