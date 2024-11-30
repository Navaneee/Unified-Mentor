#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)


# In[2]:


df = pd.read_csv('C:/Users/ANAVADYA/Downloads/world_population.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isna().sum()


# In[6]:


print(f"amount of duplicates: {df.duplicated().sum()}")


# In[7]:


df.columns


# In[8]:


df.drop(['CCA3', 'Capital'], axis=1, inplace=True)


# In[9]:


df.head()


# In[10]:


custom_palette = ['#0b3d91', '#e0f7fa', '#228b22', '#1e90ff', '#8B4513', '#D2691E',
'#DAA520', '#556B2F']


# In[11]:


custom_palette


# In[12]:


countries_by_continent = df['Continent'].value_counts().reset_index()


# In[13]:


countries_by_continent


# In[14]:


import plotly.express as px
import pandas as pd

# Example DataFrame
countries_by_continent = pd.DataFrame({
    'Continent': ['Asia', 'Europe', 'Africa', 'Oceania', 'Americas'],
    'count': [48, 44, 54, 14, 35]
})

# Define custom color palette
custom_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

# Create the bar chart
fig = px.bar(
    countries_by_continent,
    x='Continent',
    y='count',
    color='Continent',
    text='count',
    title='Number of Countries by Continent',
    color_discrete_sequence=custom_palette
)
# Show the plot
fig.show()



# In[15]:


# Customize the layout
fig.update_layout(
    xaxis_title='Continents',
    yaxis_title='Number of Countries',
    plot_bgcolor='rgba(0,0,0,0)',  # Set the background color to transparent
    font_family='Arial',          # Set font family
    title_font_size=20            # Set title font size
)

# Show the plot
fig.show()


# In[16]:


continent_population_percentage = df.groupby('Continent')['World Population Percentage'].sum().reset_index()


# In[18]:


# Create the pie chart
fig = go.Figure(data=[go.Pie(
    labels=continent_population_percentage['Continent'],
    values=continent_population_percentage['World Population Percentage']
)])

# Update layout
fig.update_layout(
    title='World Population Percentage by Continent',
    template='plotly',
    paper_bgcolor='rgba(255,255,255,0)',  # Set the paper background color to transparent
    plot_bgcolor='rgba(255,255,255,0)'    # Set the plot background color to transparent
)

# Update pie chart colors
fig.update_traces(
    marker=dict(colors=custom_palette, line=dict(color='#FFFFFF', width=1))
)

# Show the plot
fig.show()


# In[19]:


import plotly.graph_objects as go
import pandas as pd

# Example DataFrame
continent_population_percentage = pd.DataFrame({
    'Continent': ['Asia', 'Africa', 'Europe', 'Oceania', 'Americas'],
    'World Population Percentage': [59.5, 17.2, 9.6, 0.5, 13.2]
})

# Define custom color palette
custom_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

# Create the pie chart
fig = go.Figure(data=[
    go.Pie(
        labels=continent_population_percentage['Continent'],
        values=continent_population_percentage['World Population Percentage']
    )
])

# Update layout
fig.update_layout(
    title='World Population Percentage by Continent',
    template='plotly',
    paper_bgcolor='rgba(255,255,255,0)',  # Set the paper background color to transparent
    plot_bgcolor='rgba(255,255,255,0)'   # Set the plot background color to transparent
)

# Update pie colors
fig.update_traces(
    marker=dict(
        colors=custom_palette,
        line=dict(color='#FFFFFF', width=1)
    )
)

# Show the plot
fig.show()


# In[20]:


df_melted = df.melt(
    id_vars=['Continent'],
    value_vars=[
        '2022 Population', '2020 Population', '2015 Population',
        '2010 Population', '2000 Population', '1990 Population',
        '1980 Population', '1970 Population'
    ],
    var_name='Year',
    value_name='Population'
)
# Convert 'Year' to a more suitable format
df_melted['Year'] = df_melted['Year'].str.split().str[0].astype(int)

# Aggregate population by continent and year
population_by_continent = df_melted.groupby(['Continent', 'Year'])['Population'].sum().reset_index()


# In[21]:


print(population_by_continent)


# In[22]:


import plotly.express as px

fig = px.line(
    population_by_continent,
    x='Year',
    y='Population',
    color='Continent',
    title='Population Trends by Continent Over Time',
    labels={'Population': 'Population', 'Year': 'Year'}
)

fig.update_layout(
    template='plotly_white',
    xaxis_title='Year',
    yaxis_title='Population',
    title_font_size=20
)

fig.show()


# In[ ]:


#World Population Comparison: 1970 to 2020


# In[23]:


import plotly.express as px

# List of features for which to create the choropleth maps
features = ['1970 Population', '2020 Population']

# Loop through the features and create a choropleth for each
for feature in features:
    fig = px.choropleth(
        df,
        locations='Country/Territory',
        locationmode='country names',
        color=feature,
        hover_name='Country/Territory',
        template='plotly_white',
        title=feature
    )
    fig.show()


# In[24]:


growth = (df.groupby(by='Country/Territory')['2022 Population'].sum() - 
          df.groupby(by='Country/Territory')['1970 Population'].sum()).sort_values(ascending=False).head(8)
import plotly.express as px

fig = px.bar(
    growth,
    x=growth.index,
    y=growth.values,
    title='Top 8 Countries by Population Growth (1970 to 2022)',
    labels={'x': 'Country/Territory', 'y': 'Population Growth'},
    color=growth.values,
    color_continuous_scale='Viridis'
)

fig.update_layout(
    template='plotly_white',
    xaxis_title='Country/Territory',
    yaxis_title='Population Growth',
    title_font_size=20
)

fig.show()


# In[25]:


import plotly.express as px

fig = px.bar(
    x=growth.index,
    y=growth.values,
    text=growth.values,
    color=growth.values,
    color_continuous_scale='Viridis',  # Adding a color scale
    title='Growth Of Population From 1970 to 2022 (Top 8)',
    template='plotly_white'
)

# Customize layout
fig.update_layout(
    xaxis_title='Country/Territory',
    yaxis_title='Population Growth',
    title_font_size=20
)

# Show the plot
fig.show()


# In[26]:


import plotly.express as px

# Group by 'Country/Territory' and get the top 8 populated countries for 1970 and 2022
top_8_populated_countries_1970 = df.groupby('Country/Territory')['1970 Population'].sum().sort_values(ascending=False).head(8)
top_8_populated_countries_2022 = df.groupby('Country/Territory')['2022 Population'].sum().sort_values(ascending=False).head(8)

# Create a dictionary to store the data for both years
features = {
    'top_8_populated_countries_1970': top_8_populated_countries_1970,
    'top_8_populated_countries_2022': top_8_populated_countries_2022
}

# Loop through each feature and create a bar chart
for feature_name, feature_data in features.items():
    year = feature_name.split('_')[-1]  # Extract the year from the feature name
    
    # Create the bar chart for each year
    fig = px.bar(
        x=feature_data.index,
        y=feature_data.values,
        text=feature_data.values,
        color=feature_data.values,
        title=f'Top 8 Most Populated Countries ({year})',
        template='plotly_white'
    )

    # Update layout
    fig.update_layout(
        xaxis_title='Country/Territory',
        yaxis_title='Population',
        title_font_size=20
    )

    # Show the plot
    fig.show()


# In[27]:


##World Population Growth Rates: The Fastest Growing Countries


# In[28]:


sorted_df_growth = df.sort_values(by='Growth Rate', ascending=False)

top_fastest = sorted_df_growth.head(6)
top_slowest = sorted_df_growth.tail(6)


# In[29]:


top_fastest


# In[30]:


import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go


# In[31]:


def plot_population_trends(countries, df, custom_palette):
    # Calculate the number of rows needed
    n_cols = 2
    n_rows = (len(countries) + n_cols - 1) // n_cols  # Number of rows needed

    # Create subplots
    fig = sp.make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=countries, 
        horizontal_spacing=0.1, vertical_spacing=0.1
    )

    for i, country in enumerate(countries, start=1):
        # Filter data for the selected country
        country_df = df[df['Country/Territory'] == country]

        # Melt the DataFrame to have a long format
        country_melted = country_df.melt(
            id_vars=['Country/Territory'],
            value_vars=[
                '2022 Population', '2020 Population', '2015 Population',
                '2010 Population', '2000 Population', '1990 Population',
                '1980 Population', '1970 Population'
            ],
            var_name='Year',
            value_name='Population'
        )


# In[32]:


import plotly.express as px
import plotly.subplots as sp

# Function to plot population trends
def plot_population_trends(countries, df, custom_palette):
    # Calculate the number of rows needed
    n_cols = 2  # Fixed number of columns
    n_rows = (len(countries) + n_cols - 1) // n_cols  # Calculate number of rows required

    # Create subplots
    fig = sp.make_subplots(
        rows=n_rows, cols=n_cols, subplot_titles=countries, 
        horizontal_spacing=0.1, vertical_spacing=0.1
    )

    for i, country in enumerate(countries, start=1):
        # Filter data for the selected country
        country_df = df[df['Country/Territory'] == country]

        # Melt the DataFrame to have a long format
        country_melted = country_df.melt(
            id_vars=['Country/Territory'],
            value_vars=[
                '2022 Population', '2020 Population', '2015 Population',
                '2010 Population', '2000 Population', '1990 Population',
                '1980 Population', '1970 Population'
            ],
            var_name='Year',
            value_name='Population'
        )

        # Convert 'Year' to a more suitable format (handle non-numeric values)
        country_melted['Year'] = country_melted['Year'].str.split().str[0].astype(int)

        # Create a line plot for each country
        line_fig = px.line(
            country_melted, x='Year', y='Population',
            color='Country/Territory',
            labels={'Population': 'Population', 'Year': 'Year'},
            color_discrete_sequence=custom_palette
        )

        # Update the line plot to fit the subplot
        row = (i - 1) // n_cols + 1  # Calculate the row index
        col = (i - 1) % n_cols + 1   # Calculate the column index

        for trace in line_fig.data:
            fig.add_trace(trace, row=row, col=col)

    # Update the layout of the subplots
    fig.update_layout(
        title='Population Trends of Selected Countries Over Time',
        template='plotly_white',
        font_family='Arial',
        title_font_size=20,
        showlegend=False,
        height=600 * n_rows,  # Adjust height based on the number of rows
        width=800  # Set a fixed width for consistency
    )

    # Update line properties for all traces
    fig.update_traces(line=dict(width=3))

    # Update axis labels
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Population')

  




# In[33]:



# Show the plot
fig.show()


# In[34]:


# Assuming 'top_fastest' is a DataFrame with 'Country/Territory' and 'Growth Rate' columns
# Sort 'top_fastest' by 'Growth Rate' in descending order
fastest = top_fastest[['Country/Territory', 'Growth Rate']].sort_values(by='Growth Rate', ascending=False).reset_index(drop=True)
# Display the sorted DataFrame
fastest


# In[35]:


def plot_population_trends(countries, df, custom_palette):
    n_cols = 2
    n_rows = (len(countries) + n_cols - 1) // n_cols

    fig = sp.make_subplots(rows=n_rows, cols=n_cols, subplot_titles=countries,
                           horizontal_spacing=0.1, vertical_spacing=0.1)

    for i, country in enumerate(countries, start=1):
        # Filter data for the selected country
        country_df = df[df['Country/Territory'] == country]

        # Melt the DataFrame to have a long format
        country_melted = country_df.melt(id_vars=['Country/Territory'],
                                         value_vars=['2022 Population', '2020 Population', '2015 Population',
                                                     '2010 Population', '2000 Population', '1990 Population',
                                                     '1980 Population', '1970 Population'],
                                         var_name='Year', value_name='Population')

        # Convert 'Year' to a more suitable format
        country_melted['Year'] = country_melted['Year'].str.split().str[0].astype(int)

        # Print the last population value (for example)
        last_population = country_melted.iloc[-1]['Population']
        print(f"Last population for {country} in {country_melted.iloc[-1]['Year']}: {last_population}")

        # Create a line plot for each country
        line_fig = px.line(country_melted, x='Year', y='Population',
                           color='Country/Territory', labels={'Population': 'Population', 'Year': 'Year'},
                           color_discrete_sequence=custom_palette)

        # Update the line plot to fit the subplot
        row = (i - 1) // n_cols + 1
        col = (i - 1) % n_cols + 1
        for trace in line_fig.data:
            fig.add_trace(trace, row=row, col=col)

    # Update the layout of the subplots
    fig.update_layout(
        title='Population Trends of Selected Countries Over Time',
        template='plotly_white',
        font_family='Arial',
        title_font_size=20,
        showlegend=False,
        height=600 * n_rows,  # Adjust height for bigger plots
    )

    fig.update_traces(line=dict(width=3))
    fig.update_xaxes(title_text='Year')
    fig.update_yaxes(title_text='Population')

    # Show the plot
    fig.show()


# Custom color palette (you can change it as per your preference)
custom_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Assuming 'df' is already your DataFrame containing the population data
plot_population_trends(['Moldova', 'Poland', 'Niger', 'Syria', 'Slovakia', 'DR Congo'], df, custom_palette)


# In[36]:


#World Population Growth Rates: The Slowest Growing Countries


# In[37]:


slowest = top_slowest[['Country/Territory', 'Growth Rate']].sort_values(by='Growth Rate', ascending=False).reset_index(drop=True)
slowest


# In[38]:


custom_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Call the plot_population_trends function with the specified countries
plot_population_trends(
    ['Latvia', 'Lithuania', 'Bulgaria', 'American Samoa', 'Lebanon', 'Ukraine'],
    df,  # Ensure df contains the data you want to use
    custom_palette
)


# In[39]:


#Land Area by Country


# In[40]:


# Ensure the column name is correctly formatted by checking for hidden spaces or formatting issues
land_by_country = df.groupby('Country/Territory')['Area (km²)'].sum().sort_values(ascending=False)

# Get the top 5 countries with the most land area
most_land = land_by_country.head(5)

# Get the bottom 5 countries with the least land area
least_land = land_by_country.tail(5)

# Print the results
print("Top 5 countries with the most land area:")
print(most_land)

print("\nTop 5 countries with the least land area:")
print(least_land)


# In[41]:


import plotly.graph_objects as go
import plotly.subplots as sp

# Create subplots
fig = sp.make_subplots(
    rows=1, cols=2, subplot_titles=("Countries with Most Land", "Countries with Least Land")
)

# Plot countries with the most land
fig.add_trace(
    go.Bar(x=most_land.index, y=most_land.values, name='Most Land', marker_color=custom_palette[0]),
    row=1, col=1
)

# Plot countries with the least land
fig.add_trace(
    go.Bar(x=least_land.index, y=least_land.values, name='Least Land', marker_color=custom_palette[1]),
    row=1, col=2
)

# Update layout
fig.update_layout(
    title_text="Geographical Distribution of Land Area by Country",
    showlegend=False,
    template='plotly_white'
)

# Update y-axes for both subplots
fig.update_yaxes(title_text="Area (km2)", row=1, col=1)
fig.update_yaxes(title_text="Area (km2)", row=1, col=2)

# Show the plot
fig.show()


# In[42]:


#Land Area Per Person by Country


# In[43]:


# Calculate the Area per Person for each country
df['Area per Person'] = df['Area (km²)'] / df['2022 Population']

# Group by 'Country/Territory' and sum the 'Area per Person' for each country
country_area_per_person = df.groupby('Country/Territory')['Area per Person'].sum()

# Get the top 5 countries with the most land available per person
most_land_available = country_area_per_person.sort_values(ascending=False).head(5)

# Get the bottom 5 countries with the least land available per person
least_land_available = country_area_per_person.sort_values(ascending=False).tail(5)

# Print the results
print("Top 5 countries with the most land available per person:")
print(most_land_available)

print("\nTop 5 countries with the least land available per person:")
print(least_land_available)


# In[44]:


# Create subplots
fig = sp.make_subplots(
    rows=1, cols=2, 
    subplot_titles=("Countries with Most Land Available Per Capita", 
                    "Countries with Least Land Available Per Capita")
)

# Plot countries with the most land available per person
fig.add_trace(
    go.Bar(
        x=most_land_available.index, 
        y=most_land_available.values,
        name='Most Land Available Per Capita', 
        marker_color=custom_palette[2]
    ), 
    row=1, col=1
)

# Plot countries with the least land available per person
fig.add_trace(
    go.Bar(
        x=least_land_available.index, 
        y=least_land_available.values,
        name='Least Land Available Per Capita', 
        marker_color=custom_palette[3]
    ), 
    row=1, col=2
)

# Update layout
fig.update_layout(
    title_text="Distribution of Available Land Area by Country Per Capita",
    showlegend=False,
    template='plotly_white'
)

# Update y-axes titles
fig.update_yaxes(title_text="Land Available Per Person (km²)", row=1, col=1)
fig.update_yaxes(title_text="Land Available Per Person (km²)", row=1, col=2)

# Show the plot
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:




