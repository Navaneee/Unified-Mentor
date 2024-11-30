#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df =pd.read_csv("C:/Users/ANAVADYA/Downloads/Medicines Usage, Side Effects and Substitutes.csv")
df.head()


# In[3]:


df.columns


# In[4]:


df.shape


# In[5]:


missing_vals = df.isnull().sum() / len(df)
missing_vals


# In[6]:


missing_more_15 = missing_vals[missing_vals > 0.15]
missing_more_15


# In[7]:


list(missing_more_15.index)


# In[8]:


missing_vals[missing_vals <= 0.15]


# In[9]:


df1 = df.drop(list(missing_more_15.index) , axis = 'columns')
df1.head()


# In[10]:


df1 = df1.drop_duplicates()


# In[11]:


df1['name'] = df1['name'].str.lower()
df1['name'] = df1['name'].str.strip()


# In[12]:


df1['name'].value_counts()


# In[13]:


df1.isnull().sum()


# In[14]:


df1['use0'] = df1['use0'].str.lower()
df1['use0'] = df1['use0'].str.strip()


# In[15]:


df1['use0'].value_counts()


# In[16]:


top_treatments = list(df1['use0'].value_counts().head(15).index)
top_treatments


# In[17]:


df1['use0'].value_counts().head(15).sum()


# In[18]:


df1.shape[0] - df1['use0'].value_counts().head(15).sum()


# In[20]:


def filter_high_uses(x):
    return x in top_treatments


# In[21]:


df2 = df1[df1['use0'].apply(filter_high_uses)]
df2.head()


# In[22]:


df2.isnull().sum()


# In[23]:


df3 = df2.dropna()
df3.head()


# In[24]:


df2.shape[0] - df3.shape[0]


# In[25]:


df3.shape


# In[26]:


df3.isnull().sum()


# In[27]:


df3.head()


# In[28]:


df4 = df3.drop('id' , axis = 'columns')


# In[29]:


df4 = df4.applymap(lambda x : x.lower())
df4 = df4.applymap(lambda x : x.strip())


# In[30]:


df4.head()


# In[31]:


list(df4.columns)


# In[33]:


for col in list(df4.columns):
    print(col , len(df4[col].unique()))


# In[34]:


df4['Habit Forming'].value_counts().plot.bar()


# In[35]:


df4['sideEffect0'].value_counts()


# In[36]:


df4['sideEffect0'].value_counts().head(10).plot.bar()


# In[37]:


df4['sideEffect1'].value_counts().head(10).plot.bar()


# In[38]:


df4['sideEffect2'].value_counts().head(10).plot.bar()


# In[39]:


df4.head()


# In[44]:


substitute0_sideeffects = {}
for sub,sideffect in df4.groupby('substitute0')['sideEffect0']:
    print('Substtitute' , sub)
    print('Sideeffect' , sideffect)
    substitute0_sideeffects[sub] = sideffect

print(substitute0_sideeffects)


# In[45]:


st0_sideeffects = pd.DataFrame(substitute0_sideeffects).fillna(0)
st0_sideeffects


# In[ ]:




