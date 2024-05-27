
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the preprocessed dataset
data = pd.read_csv('data/preprocessed_data.csv')

data_cleaned = data.drop(columns=['TimeElapsed_max', 'Hour_', 'Patient_ID'])

# Calculate the correlation matrix
correlation_matrix = data_cleaned.corr()

# Extract the correlation of each feature with 'EventualSepsisLabel'
correlation_with_sepsis = correlation_matrix['EventualSepsisLabel'].drop('EventualSepsisLabel')

# Plot the correlation values
plt.figure(figsize=(10, 8))
correlation_with_sepsis.plot(kind='bar')
plt.title('Correlation of Features with EventualSepsisLabel')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.show()


# Separate data into two groups based on EventualSepsisLabel
sepsis_data = data_cleaned[data_cleaned['EventualSepsisLabel'] == 1]
no_sepsis_data = data_cleaned[data_cleaned['EventualSepsisLabel'] == 0]

# List of features to plot
features_to_plot = data_cleaned.columns[:-1]

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))

# Plot data for patients with sepsis
for feature in features_to_plot:
    axes[0].plot(sepsis_data.index, sepsis_data[feature], label=feature, alpha=0.6)
axes[0].set_title('Patients with Sepsis')
axes[0].set_xlabel('Index')
axes[0].set_ylabel('Feature Values')
axes[0].legend()

# Plot data for patients without sepsis
for feature in features_to_plot:
    axes[1].plot(no_sepsis_data.index, no_sepsis_data[feature], label=feature, alpha=0.6)
axes[1].set_title('Patients without Sepsis')
axes[1].set_xlabel('Index')
axes[1].set_ylabel('Feature Values')
axes[1].legend()

plt.tight_layout()
plt.show()

# Separate data into two groups based on EventualSepsisLabel
sepsis_data = data_cleaned[data_cleaned['EventualSepsisLabel'] == 1]
no_sepsis_data = data_cleaned[data_cleaned['EventualSepsisLabel'] == 0]

# Filter data to only include up to hour 15
sepsis_data = sepsis_data[sepsis_data['Hour'] <= 15]
no_sepsis_data = no_sepsis_data[no_sepsis_data['Hour'] <= 15]

# List of features to plot
features_to_plot = data_cleaned.columns[:-1]

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))

# Plot data for patients with sepsis
for feature in features_to_plot:
    axes[0].plot(sepsis_data['Hour'], sepsis_data[feature], label=feature, alpha=0.6)
axes[0].set_title('Patients with Sepsis')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Feature Values')
axes[0].legend()

# Plot data for patients without sepsis
for feature in features_to_plot:
    axes[1].plot(no_sepsis_data['Hour'], no_sepsis_data[feature], label=feature, alpha=0.6)
axes[1].set_title('Patients without Sepsis')
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Feature Values')
axes[1].legend()

plt.tight_layout()
plt.show()

# Load the preprocessed dataset
beforeData = pd.read_csv('data/Dataset.csv')
data = pd.read_csv('data/preprocessed_data.csv')

data_cleaned = data.drop(columns=['TimeElapsed_max', 'Hour_', 'Patient_ID'])

# Calculate 10% of the unique patient IDs
unique_patient_ids = data['Patient_ID'].unique()
sample_size = int(len(unique_patient_ids) * 1)
sampled_patient_ids = np.random.choice(unique_patient_ids, sample_size, replace=False)

# Filter the data to only include the sampled patient IDs
sampled_data = data[data['Patient_ID'].isin(sampled_patient_ids)]

# Drop the 'TimeElapsed_max' and 'Hour' columns for initial cleaning
sampled_data_cleaned = sampled_data.drop(columns=['TimeElapsed_max', 'Hour', 'Hour_'])

# Normalize the features
scaler = StandardScaler()
features = sampled_data_cleaned.columns[:-1]
sampled_data_cleaned[features] = scaler.fit_transform(sampled_data_cleaned[features])

# Separate data into two groups based on EventualSepsisLabel
sepsis_data_sampled = sampled_data_cleaned[sampled_data_cleaned['EventualSepsisLabel'] == 1]
no_sepsis_data_sampled = sampled_data_cleaned[sampled_data_cleaned['EventualSepsisLabel'] == 0]

# Filter data to only include up to hour 15
sepsis_data_sampled = sepsis_data_sampled[sepsis_data_sampled['Hour'] <= 60]
no_sepsis_data_sampled = no_sepsis_data_sampled[no_sepsis_data_sampled['Hour'] <= 60]

# Calculate the average of each feature over each hour for patients with sepsis and without sepsis
sepsis_data_avg = sepsis_data_sampled.groupby('Hour').mean()
no_sepsis_data_avg = no_sepsis_data_sampled.groupby('Hour').mean()

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))

# Plot data for patients with sepsis
for feature in features:
    axes[0].plot(sepsis_data_avg.index, sepsis_data_avg[feature], label=feature, alpha=0.6)
axes[0].set_title('Normalized Average Feature Values over Time for Patients with Sepsis')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Normalized Average Feature Values')
axes[0].legend()

# Plot data for patients without sepsis
for feature in features:
    axes[1].plot(no_sepsis_data_avg.index, no_sepsis_data_avg[feature], label=feature, alpha=0.6)
axes[1].set_title('Normalized Average Feature Values over Time for Patients without Sepsis')
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('Normalized Average Feature Values')
axes[1].legend()

plt.tight_layout()
plt.show()