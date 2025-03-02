# --------------------------------------------------------
#                 Import required libraries
# --------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold


# --------------------------------------------------------
#                     Loading dataset
# --------------------------------------------------------

# Dataset file name: "aviation_dataset.csv"
df = pd.read_csv("airline_profit_data.csv")


# --------------------------------------------------------
#                   Data preprocessing
# --------------------------------------------------------

# Selecting the key columns from the given Dataset
key_columns = ['Delay (Minutes)',
               'Actual Departure Time',
               'Aircraft Utilization (Hours/Day)',
               'Maintenance Downtime (Hours)',
               'Cost per ASK',
               'Revenue (USD)',
               'Load Factor (%)',
               'Fleet Availability (%)',
               'Fuel Efficiency (ASK)',
               'Profit (USD)']

# Create a new DataFrame with only the key columns
data_key = df[key_columns].copy()

# Convert 'Actual Departure Time' to datetime objects within the DataFrame
data_key['Actual Departure Time'] = pd.to_datetime(data_key['Actual Departure Time'], format='%Y-%m-%d %H:%M:%S')

# Calculate the day of the week and map it to float values
day_mapping = {'Monday': 1.0, 'Tuesday': 2.0, 'Wednesday': 3.0, 'Thursday': 4.0, 'Friday': 5.0, 'Saturday': 6.0, 'Sunday': 7.0}
data_key['Day of Week'] = data_key['Actual Departure Time'].dt.day_name().map(day_mapping)

# Convert 'Delay (Minutes)' to 'Delay (Hours)'
data_key['Delay (Hours)'] = data_key['Delay (Minutes)'] / 60

# Drop the original 'Delay (Minutes)' column if it's no longer needed
data_key.drop('Delay (Minutes)', axis=1, inplace=True)

# Round the float values to 5 values after decimal
data_key = data_key.round(5)

# Display the first few rows of the new DataFrame
# print(data_key.head())


# --------------------------------------------------------
#                 Visualizing Distributions
# --------------------------------------------------------

# Histograms
data_key.hist(bins=20, figsize=(18, 12))
plt.show()

# Box plots
plt.figure(figsize=(10, 8))

num_cols = data_key.shape[1]  #getting the number of cols to plot
num_rows = (num_cols + 1) // 2 # getting the ceiling of half to ensure we have enough rows

for i, column in enumerate(data_key.columns, 1):
    plt.subplot(num_rows, 2, i)
    sns.boxplot(data_key[column])
    plt.title(f'Box plot of {column}')
plt.tight_layout()
plt.show()


# --------------------------------------------------------
#              Analyzing Pairwise Relationships
# --------------------------------------------------------

# Pair plot
sns.pairplot(data_key)
plt.show()


# --------------------------------------------------------
#              Correlation Matrix and Heatmap
# --------------------------------------------------------

# Correlation matrix
corr_matrix = data_key.corr()

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# --------------------------------------------------------
#          Train the Neural network and evaluate
# --------------------------------------------------------

# Split the data into features and target variable
X = data_key.drop(['Profit (USD)', 'Actual Departure Time'], axis=1) # Drop 'Actual Departure Time'
y = data_key['Profit (USD)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Neural Network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Make predictions
y_pred_nn = model.predict(X_test_scaled)

# Evaluate the model
nn_mse = mean_squared_error(y_test, y_pred_nn)
nn_r2 = r2_score(y_test, y_pred_nn)

print(f"Neural Network - Mean Squared Error: {nn_mse}")
print(f"Neural Network - R-squared: {nn_r2}")

# --------------------------------------------------------
#                         Result
# --------------------------------------------------------
# Neural Network - Mean Squared Error: 1166.8263072690024
# Neural Network - R-squared: 0.9999964217529854