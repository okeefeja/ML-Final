import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


directory1 = '/Users/jamesokeefe/Desktop/Machine-Learning-Final-Assignment/Dublinbikes-Usage-Data/Pre-Pandemic/'
directory2 = '/Users/jamesokeefe/Desktop/Machine-Learning-Final-Assignment/Dublinbikes-Usage-Data/During-Pandemic/'
directory3 = '/Users/jamesokeefe/Desktop/Machine-Learning-Final-Assignment/Dublinbikes-Usage-Data/Post-Pandemic/'

average_temperature_data = '/Users/jamesokeefe/Desktop/Machine-Learning-Final-Assignment/Average-Temperature-Data.csv'
average_rain_data = '/Users/jamesokeefe/Desktop/Machine-Learning-Final-Assignment/Average-Rain-Data.csv'

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
pre_pandemic_data = combined_df1[combined_df1['STATUS'].str.contains('Open|OPEN')]
pandemic_data = combined_df2[combined_df2['STATUS'].str.contains('Open|OPEN')]
post_pandemic_data = combined_df3[combined_df3['STATUS'].str.contains('Open|OPEN')]

# Calculate the daily bike usages
pre_pandemic_data.loc[pre_pandemic_data['BIKE_STANDS'] != 0, 'PERCENT_AVAILABLE_BIKE_STANDS'] = (pre_pandemic_data['AVAILABLE_BIKE_STANDS'] / pre_pandemic_data['BIKE_STANDS']) * 100
pandemic_data.loc[pandemic_data['BIKE_STANDS'] != 0, 'PERCENT_AVAILABLE_BIKE_STANDS'] = (pandemic_data['AVAILABLE_BIKE_STANDS'] / pandemic_data['BIKE_STANDS']) * 100
post_pandemic_data.loc[post_pandemic_data['BIKE_STANDS'] != 0, 'PERCENT_AVAILABLE_BIKE_STANDS'] = (post_pandemic_data['AVAILABLE_BIKE_STANDS'] / post_pandemic_data['BIKE_STANDS']) * 100

# Group by date and calculate the average percentage of available bike stands for each day for all sets
pre_pandemic_bike_usage = pre_pandemic_data.groupby(pre_pandemic_data['TIME'].dt.date)['PERCENT_AVAILABLE_BIKE_STANDS'].mean().reset_index()
pandemic_bike_usage = pandemic_data.groupby(pandemic_data['TIME'].dt.date)['PERCENT_AVAILABLE_BIKE_STANDS'].mean().reset_index()
post_pandemic_bike_usage = post_pandemic_data.groupby(post_pandemic_data['TIME'].dt.date)['PERCENT_AVAILABLE_BIKE_STANDS'].mean().reset_index()

pre_pandemic_bike_usage['TIME'] = pd.to_datetime(pre_pandemic_bike_usage['TIME'])
pandemic_bike_usage['TIME'] = pd.to_datetime(pandemic_bike_usage['TIME'])
post_pandemic_bike_usage['TIME'] = pd.to_datetime(post_pandemic_bike_usage['TIME'])





# Get the dailly average temperature in Dublin for each of the three pandemic periods
# -----------------------------------------------------------------------------------
# Load the CSV data into a pandas DataFrame
data = pd.read_csv(average_temperature_data)

# Convert the 'Date' column to datetime format for date comparisons
data['DATE'] = pd.to_datetime(data['DATE'])

# Filter data based on conditions: date after July 1st, 2018 and location name
filtered_data = data[(data['DATE'] >= '2018-07-01') & (data['NAME'] == 'DUBLIN PHOENIX PARK, EI')]

# Extract the daily average temperature for the filtered records
daily_average_temperature = filtered_data['TAVG'].tolist()

average_temperature_dates = filtered_data['DATE']

# Create a DataFrame for 'daily_average_temperature' and 'DATE' for easier matching
temperature_df = pd.DataFrame({'DATE': average_temperature_dates, 'daily_average_temperature': daily_average_temperature})

# Merge 'daily_average_temperature' with 'open_stations' datasets based on dates
pre_pandemic_bike_usage = pd.merge(pre_pandemic_bike_usage, temperature_df, left_on='TIME', right_on='DATE', how='left')
pandemic_bike_usage = pd.merge(pandemic_bike_usage, temperature_df, left_on='TIME', right_on='DATE', how='left')
post_pandemic_bike_usage = pd.merge(post_pandemic_bike_usage, temperature_df, left_on='TIME', right_on='DATE', how='left')









# Get the dailly average amount of rain in Dublin for each of the three pandemic periods
# -------------------------------------------------------------------------------------
# Load the CSV data into a pandas DataFrame
rain_data = pd.read_csv(average_rain_data)

# Convert the 'date' column to datetime format for date comparisons
rain_data['date'] = pd.to_datetime(rain_data['date'])

# Filter data based on conditions: date on or after July 1, 2018
filtered_rain_data = rain_data[rain_data['date'] >= '2018-07-01']

# Extract the daily average rain for the filtered records
daily_average_rain = filtered_rain_data['rain'].tolist()

rain_dates = filtered_rain_data['date']

# Create a DataFrame for 'daily_average_rain' and 'date' for easier matching
rain_df = pd.DataFrame({'date': rain_dates, 'daily_average_rain': daily_average_rain})

# Merge 'daily_average_rain' with 'open_stations' datasets based on dates
pre_pandemic_bike_usage = pd.merge(pre_pandemic_bike_usage, rain_df, left_on='TIME', right_on='date', how='left')
pandemic_bike_usage = pd.merge(pandemic_bike_usage, rain_df, left_on='TIME', right_on='date', how='left')
post_pandemic_bike_usage = pd.merge(post_pandemic_bike_usage, rain_df, left_on='TIME', right_on='date', how='left')






test_data = post_pandemic_bike_usage[['TIME', 'PERCENT_AVAILABLE_BIKE_STANDS', 'daily_average_temperature', 'daily_average_rain']]
test_labels = np.random.randint(0, 2, size=(100,))

# Assuming test_data is your DataFrame for testing in a similar format to the training data
test_data = test_data[['TIME', 'PERCENT_AVAILABLE_BIKE_STANDS', 'daily_average_temperature', 'daily_average_rain']]

# Select relevant columns for the features
features = ['PERCENT_AVAILABLE_BIKE_STANDS', 'daily_average_temperature', 'daily_average_rain']
data_test = test_data[features]

# Convert DataFrame to numpy array
data_array_test = data_test.to_numpy()

# Define timesteps and features
timesteps = 10  
num_features = data_test.shape[1]

# Prepare sequences for test data
sequences_test = []
targets_test = []

for i in range(len(data_array_test) - timesteps):
    sequences_test.append(data_array_test[i:i + timesteps])
    targets_test.append(data_array_test[i + timesteps][0])  

# Convert lists to numpy arrays for test data
X_test = np.array(sequences_test)
y_test = np.array(targets_test)




# Prepare training and testing datasets
train_data = pre_pandemic_bike_usage[['TIME', 'PERCENT_AVAILABLE_BIKE_STANDS', 'daily_average_temperature', 'daily_average_rain']]
train_labels = np.random.randint(0, 2, size=(100,))

features = ['PERCENT_AVAILABLE_BIKE_STANDS', 'daily_average_temperature', 'daily_average_rain']
data = train_data[features]

# Convert DataFrame to numpy array
data_array = data.to_numpy()

# Prepare sequences
sequences = []
targets = []

for i in range(len(data_array) - timesteps):
    sequences.append(data_array[i:i + timesteps])
    targets.append(data_array[i + timesteps][0])  

# Convert lists to numpy arrays
X_train = np.array(sequences)
y_train = np.array(targets)






# Define the model architecture
model = Sequential()
model.add(LSTM(units=8, input_shape=(timesteps, num_features)))
model.add(Dense(units=1))  

# Print model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=2, batch_size=8, validation_data=(X_test, y_test))




# # Plot training & validation loss values
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.show()







predictions = model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs. Predicted')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.show()