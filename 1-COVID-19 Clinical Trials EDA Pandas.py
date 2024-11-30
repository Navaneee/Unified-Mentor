#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Objective of the project :

# The objective is to explore the dataset to gain insights into the characteristics of COVID-19 clinical trials.


#Importing Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


# Load the dataset
df = pd.read_csv('C:/Users/ANAVADYA/Downloads/COVID clinical trials.csv')


# In[7]:


#Exploratory Data Analysis

df.head()


# In[8]:


# Columns of Dataset
df.columns


# In[9]:


# No. of rows and columns
df.shape


# In[11]:


# Check the columns and data types
df.info()


# In[12]:


# Summary statistics for numerical columns
print(df.describe())


# In[13]:


# Summary statistics for categorical columns
print(df.describe(include='object'))


# In[14]:


# Categorical Features
df.select_dtypes(include = 'object').columns


# In[15]:


# Categorical Features
df.select_dtypes(include = 'object').columns


# In[ ]:


# Numerical Features
df.select_dtypes(exclude = 'object').columns


# In[17]:


# Detecting (Percentage) Missing Data
missing_data = df.isnull().mean() * 100
missing_data

# Visualize data without calculating
def visualize_data(data , caption = '' , ylabel = 'Percentage of Missing Data'):
   sns.set(rc={'figure.figsize' : (15,8.27)}) # set figure size
   plt.xticks(rotation=90) # make ticks vertical
   fig = sns.barplot(x = data.keys()[ :min(40 , len(data))].tolist() , y = data.values[ : min(40 , len(data))].tolist()).set_title(caption) # set title to the image and plot it or the highest 40
   plt.ylabel(ylabel) # set labels
   plt.show()

visualize_data(missing_data , 'Percentage of missing data in each feature')


# In[18]:


# Check for missing values
print(df.isnull().sum())


# In[19]:


df = df.drop(columns=['Acronym', 'Study Documents']) #  dropping columns


# In[20]:


df['Results First Posted'].fillna('Unknown', inplace=True)


# In[21]:


df


# In[22]:


# # Univariate Analysis

# Analyzing each column individually to understand the distribution and key characteristics.

# # Status Distribution

# Analyzing the status of clinical trials
print(df['Status'].value_counts())
df['Status'].value_counts().plot(kind='bar', title='Status of Clinical Trials')


# In[23]:


# Analyzing the gender of clinical trials
print(df['Gender'].value_counts())
df['Gender'].value_counts().plot(kind='bar', title='Gender of Clinical Trials')


# In[24]:


# # Phase Distribution
# Understanding the distribution of trial phases.
print(df['Phases'].value_counts())
df['Phases'].value_counts().plot(kind='bar', title='Phases of Clinical Trials')


# In[25]:


# Analyzing the Funded bys of clinical trials
print(df['Funded Bys'].value_counts())
df['Funded Bys'].value_counts().plot(kind='bar', title='Distribution of Funded Bys')


# In[26]:


# Analyzing the Study type distribution
print(df['Study Type'].value_counts())
df['Study Type'].value_counts().plot(kind='bar', title='Distribution of Study Type')


# In[27]:


# # Bivariate Analysis

# Exploring relationships between different variables.

# # Status vs. Phases

# Exploring how trial phases are distributed across different statuses.
status_phase = pd.crosstab(df['Status'], df['Phases'])
print(status_phase)
status_phase.plot(kind='bar', stacked=True, title='Status vs.Phases')


# In[28]:


# # Time Series Analysis

# Analyzing the trends over time, such as the number of trials started over the months.

# Convert date columns to datetime
df['Start Date'] = pd.to_datetime(df['Start Date'],errors='coerce')
df['Primary Completion Date'] = pd.to_datetime(df['Primary Completion Date'], errors='coerce')


# In[29]:


# Plot the number of trials started over time
df['Start Date'].dt.to_period('M').value_counts().sort_index().plot(kind='line', title='Trials Started Over Time')


# In[30]:


# # Conclusion

# Most trials target adult populations.\
# Most trials were interventional Study type.\
# Phase 2 have higher number than others in Phase Distribution.\
# The most of the trials are in completed phase.\
# There's a steady increase in the number of trials over time

