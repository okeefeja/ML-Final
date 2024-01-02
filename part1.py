import pandas as pd
import matplotlib.pyplot as plt
import folium

directory1 = '/Users/jamesokeefe/Desktop/Machine-Learning-Final-Assignment/Dublinbikes-Usage-Data/Pre-Pandemic/'
directory2 = '/Users/jamesokeefe/Desktop/Machine-Learning-Final-Assignment/Dublinbikes-Usage-Data/During-Pandemic/'
directory3 = '/Users/jamesokeefe/Desktop/Machine-Learning-Final-Assignment/Dublinbikes-Usage-Data/Post-Pandemic/'

file_names1 = [
    f'{directory1}Dublinbikes-2018-Q3-Usage-Data',
    f'{directory1}Dublinbikes-2018-Q4-Usage-Data',
    f'{directory1}Dublinbikes-2019-Q1-Usage-Data',
    f'{directory1}Dublinbikes-2019-Q2-Usage-Data',
    f'{directory1}Dublinbikes-2019-Q3-Usage-Data',
    f'{directory1}Dublinbikes-2019-Q4-Usage-Data',
    f'{directory1}Dublinbikes-2020-Q1-Usage-Data',
]

file_names2 = [
    f'{directory2}Dublinbikes-2020-Q2-Usage-Data',
    f'{directory2}Dublinbikes-2020-Q3-Usage-Data',
    f'{directory2}Dublinbikes-2020-Q4-Usage-Data',
    f'{directory2}Dublinbikes-2021-Q1-Usage-Data',
    f'{directory2}Dublinbikes-2021-Q2-Usage-Data',
    f'{directory2}Dublinbikes-2021-Q3-Usage-Data',
    f'{directory2}Dublinbikes-2021-Q4-Usage-Data',
    f'{directory2}January-2022',
    f'{directory2}February-2022'
]

file_names3 = [
    f'{directory3}March 2022',
    f'{directory3}April 2022',
    f'{directory3}May 2022',
    f'{directory3}June 2022',
    f'{directory3}July 2022',
    f'{directory3}August 2022',
    f'{directory3}September 2022',
    f'{directory3}October 2022',
    f'{directory3}November 2022',
    f'{directory3}December 2022',
    f'{directory3}January 2023',
    f'{directory3}February 2023',
    f'{directory3}March 2023',
    f'{directory3}April 2023',
    f'{directory3}May 2023',
    f'{directory3}June 2023',
    f'{directory3}July 2023',
    f'{directory3}August 2023',
    f'{directory3}September 2023',
    f'{directory3}October 2023',
    f'{directory3}November 2023',
    f'{directory3}December 2023'
]

# Read and combine CSV files into separate DataFrames for each set
dfs1 = [pd.read_csv(file) for file in file_names1]
dfs2 = [pd.read_csv(file) for file in file_names2]
dfs3 = [pd.read_csv(file) for file in file_names3]

# Combine DataFrames for each set
combined_df1 = pd.concat(dfs1, ignore_index=True)
combined_df2 = pd.concat(dfs2, ignore_index=True)
combined_df3 = pd.concat(dfs3, ignore_index=True)

# Convert 'TIME' column to datetime format for all sets
combined_df1['TIME'] = pd.to_datetime(combined_df1['TIME'])
combined_df2['TIME'] = pd.to_datetime(combined_df2['TIME'])
combined_df3['TIME'] = pd.to_datetime(combined_df3['TIME'])

# Filter the data for stations that are open
open_stations1 = combined_df1[combined_df1['STATUS'].str.contains('Open|OPEN')]
open_stations2 = combined_df2[combined_df2['STATUS'].str.contains('Open|OPEN')]
open_stations3 = combined_df3[combined_df3['STATUS'].str.contains('Open|OPEN')]

# Calculate the daily bike usages
open_stations1.loc[open_stations1['BIKE_STANDS'] != 0, 'PERCENT_AVAILABLE_BIKE_STANDS'] = (open_stations1['AVAILABLE_BIKE_STANDS'] / open_stations1['BIKE_STANDS']) * 100
open_stations2.loc[open_stations2['BIKE_STANDS'] != 0, 'PERCENT_AVAILABLE_BIKE_STANDS'] = (open_stations2['AVAILABLE_BIKE_STANDS'] / open_stations2['BIKE_STANDS']) * 100
open_stations3.loc[open_stations3['BIKE_STANDS'] != 0, 'PERCENT_AVAILABLE_BIKE_STANDS'] = (open_stations3['AVAILABLE_BIKE_STANDS'] / open_stations3['BIKE_STANDS']) * 100

# Group by date and calculate the average percentage of available bike stands for each day for all sets
daily_average1 = open_stations1.groupby(open_stations1['TIME'].dt.date)['PERCENT_AVAILABLE_BIKE_STANDS'].mean()
daily_average2 = open_stations2.groupby(open_stations2['TIME'].dt.date)['PERCENT_AVAILABLE_BIKE_STANDS'].mean()
daily_average3 = open_stations3.groupby(open_stations3['TIME'].dt.date)['PERCENT_AVAILABLE_BIKE_STANDS'].mean()

# Plotting the line graph with formatted dates
plt.figure(figsize=(12, 6))

# Plotting data from the first set of files in one color as a line
plt.plot(
    daily_average1.index,
    daily_average1.values,
    color='skyblue',
    label='Pre-Pandemic - Available Bike Stands (Open Stations)'
)

# Plotting data from the second set of files in another color as a line
plt.plot(
    daily_average2.index,
    daily_average2.values,
    color='salmon',
    label='During Pandemic - Available Bike Stands (Open Stations)'
)

# Plotting data from the third set of files in another color as a line
plt.plot(
    daily_average3.index,
    daily_average3.values,
    color='green',
    label='Post-Pandemic - Available Bike Stands (Open Stations)'
)

plt.xlabel('Date')
plt.ylabel('Percentage')
plt.title('Daily Average Percentage of Available Bike Stands for Open Stations')
plt.xticks(rotation=45)
plt.legend()  
plt.tight_layout()
plt.show()











# Combine the open station data from all sets
all_open_stations = pd.concat([open_stations1, open_stations2, open_stations3])

# Calculate the average usage percentage for each station across all time periods
station_data = all_open_stations.groupby('STATION_ID').agg({
    'LATITUDE': 'mean',  # Assuming latitude column is named 'LAT'
    'LONGITUDE': 'mean',  # Assuming longitude column is named 'LON'
    'PERCENT_AVAILABLE_BIKE_STANDS': 'mean'  # Replace with your percentage column name
}).reset_index()

# Create a map centered around Dublin
m = folium.Map(location=[53.349805, -6.26031], zoom_start=13)

# Add markers for each station and color them based on usage percentage
for index, row in station_data.iterrows():
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=10,
        popup=f"Station: {row['STATION_ID']}, Usage: {row['PERCENT_AVAILABLE_BIKE_STANDS']:.2f}%",  # Adjust popup text
        fill=True,
        color='red' if row['PERCENT_AVAILABLE_BIKE_STANDS'] < 60 else 'blue',  # Example color change based on usage percentage
    ).add_to(m)

# Display the map
m.save('spatial_analysis_map.html')