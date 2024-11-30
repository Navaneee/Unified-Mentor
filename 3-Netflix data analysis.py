#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


pip install wordcloud


# In[ ]:


from wordcloud import WordCloud


# In[ ]:


# importing the dataset
data = pd.read_csv('C:/Users/ANAVADYA/Downloads/netflix1.csv')


# In[ ]:


# Display the first few rows of the dataset
print(data.head())


# In[ ]:


# Data Cleaning
#Identify and handle missing data, correct data types, and drop duplicates.
# Check for missing values
print(data.isnull().sum())


# In[ ]:


data[data.duplicated()]
# Drop duplicates if any
data.drop_duplicates(inplace=True)


# In[ ]:


# Drop rows with missing critical information
data.dropna(subset=['director','country'], inplace=True)


# In[ ]:


# Convert 'date_added' to datetime
data['date_added'] = pd.to_datetime(data['date_added'])


# In[ ]:


# Show data types to confirm changes
print(data.dtypes)


# In[ ]:


# # EDA
# # 1. Content Type Distribution (Movies vs. TV Shows)
# Count the number of Movies and TV Shows
type_counts = data['type'].value_counts()


# In[ ]:


# Plot the distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=type_counts.index, y=type_counts.values,  palette='Set2')
plt.title('Distribution of Content by Type')
plt.xlabel('Type')
plt.ylabel('Count')
plt.show()


# In[ ]:


# # 2. Most Common Genres
# Split the 'listed_in' column and count genres
data['genres'] = data['listed_in'].apply(lambda x: x.split(','))
all_genres = sum(data['genres'], [])
genre_counts = pd.Series(all_genres).value_counts().head(10)


# In[ ]:


# Plot the most common genres
plt.figure(figsize=(10, 6))
sns.barplot(x=genre_counts.values,  y=genre_counts.index, palette='Set3')
plt.title('Most Common Genres on Netflix')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()


# In[ ]:


# # 3. Content Added Over Time
# Extract year and month from 'date_added'
y=genre_counts.index,
data['year_added'] = data['date_added'].dt.year
data['month_added'] = data['date_added'].dt.month


# In[ ]:


# Plot content added over the years
plt.figure(figsize=(12, 6))
sns.countplot(x='year_added', data=data, palette='coolwarm')
plt.title('Content Added Over Time')
plt.xlabel('Year')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# # 4. Top 10 Directors with the Most Titles
# Count titles by director
top_directors = data['director'].value_counts().head(10)


# In[ ]:


# Plot top directors
plt.figure(figsize=(10, 6))
sns.barplot(x=top_directors.values, y=top_directors.index, palette='Blues_d')
plt.title('Top 10 Directors with the Most Titles')
plt.xlabel('Number of Titles')
plt.ylabel('Director')
plt.show()


# In[ ]:


# # 5. Word Cloud of Movie Titles
# Generate word cloud
movie_titles = data[data['type'] == 'Movie']['title']
wordcloud= WordCloud(width=800, height=400, background_color='black').generate(' '.join(movie_titles))


# In[ ]:


# Plot word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[ ]:


# # Step 5: Conclusion and Insights

# 1. Cleaned the data by handling missing values, removing duplicates, and
#  converting data types.
# 2. Explored the data through various visualizations such as bar plots and word
#  clouds.
# 3. Analyzed content trends over time, identified popular genres, and highlighted
#  top directors.


# In[ ]:





# In[ ]:




