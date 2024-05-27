import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the datasets
data = pd.read_csv('data/preprocessed_data.csv')
before_data = pd.read_csv('data/dataset2.csv')

# Identify shared features between the two datasets
shared_features = list(set(data.columns) & set(before_data.columns))

# Ensure there are no duplicate columns or rows
data = data.loc[:, ~data.columns.duplicated()]
data = data.drop_duplicates()
before_data = before_data.loc[:, ~before_data.columns.duplicated()]
before_data = before_data.drop_duplicates()

# Calculate 100% of the unique patient IDs from the original dataset
unique_patient_ids = data['Patient_ID'].unique()
sample_size = int(len(unique_patient_ids) * 1)
sampled_patient_ids = np.random.choice(unique_patient_ids, sample_size, replace=False)

# Filter the data to only include the sampled patient IDs and shared features
shared_features = [feature for feature in shared_features if feature != 'Patient_ID']  # Remove Patient_ID from shared features
sampled_data = data[data['Patient_ID'].isin(sampled_patient_ids)][shared_features + ['Hour', 'EventualSepsisLabel']]

# Ensure there are no duplicate columns in sampled_data
sampled_data = sampled_data.loc[:, ~sampled_data.columns.duplicated()]

# Normalize the features
scaler = StandardScaler()
features = [col for col in sampled_data.columns if col not in ['Hour', 'EventualSepsisLabel']]
sampled_data[features] = scaler.fit_transform(sampled_data[features])

# Separate data into two groups based on EventualSepsisLabel
sepsis_data_sampled = sampled_data[sampled_data['EventualSepsisLabel'] == 1]
no_sepsis_data_sampled = sampled_data[sampled_data['EventualSepsisLabel'] == 0]

# Filter data to only include up to hour 45
sepsis_data_sampled = sepsis_data_sampled[sepsis_data_sampled['Hour'] <= 45]
no_sepsis_data_sampled = no_sepsis_data_sampled[no_sepsis_data_sampled['Hour'] <= 45]

# Calculate the average of each feature over each hour for patients with sepsis and without sepsis
sepsis_data_avg = sepsis_data_sampled.groupby('Hour').mean()
no_sepsis_data_avg = no_sepsis_data_sampled.groupby('Hour').mean()

# Process before_data
before_data = before_data[shared_features + ['Hour', 'EventualSepsisLabel']]
before_data = before_data.loc[:, ~before_data.columns.duplicated()]
before_data[features] = scaler.transform(before_data[features])

# Separate before_data into two groups based on EventualSepsisLabel
sepsis_data_before = before_data[before_data['EventualSepsisLabel'] == 1]
no_sepsis_data_before = before_data[before_data['EventualSepsisLabel'] == 0]

# Filter before_data to only include up to hour 45
sepsis_data_before = sepsis_data_before[sepsis_data_before['Hour'] <= 45]
no_sepsis_data_before = no_sepsis_data_before[no_sepsis_data_before['Hour'] <= 45]

# Calculate the average of each feature over each hour for patients with sepsis and without sepsis in before_data
sepsis_data_avg_before = sepsis_data_before.groupby('Hour').mean()
no_sepsis_data_avg_before = no_sepsis_data_before.groupby('Hour').mean()

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))

# Plot data for patients with sepsis in before_data (Sample)
for feature in features:
    axes[0, 0].plot(sepsis_data_avg_before.index, sepsis_data_avg_before[feature], label=feature, alpha=0.6)
axes[0, 0].set_title('Normalized Average Feature Values over Time for Patients with Sepsis (Sample Data)')
axes[0, 0].set_xlabel('Hour')
axes[0, 0].set_ylabel('Normalized Average Feature Values')
axes[0, 0].set_ylim([-0.5, 0.5])
axes[0, 0].legend()

# Plot data for patients without sepsis in before_data (Sample)
for feature in features:
    axes[1, 0].plot(no_sepsis_data_avg_before.index, no_sepsis_data_avg_before[feature], label=feature, alpha=0.6)
axes[1, 0].set_title('Normalized Average Feature Values over Time for Patients without Sepsis (Sample Data)')
axes[1, 0].set_xlabel('Hour')
axes[1, 0].set_ylabel('Normalized Average Feature Values')
axes[1, 0].set_ylim([-0.5, 0.5])
axes[1, 0].legend()

# Plot data for patients with sepsis in sampled_data (Preprocessed)
for feature in features:
    axes[0, 1].plot(sepsis_data_avg.index, sepsis_data_avg[feature], label=feature, alpha=0.6)
axes[0, 1].set_title('Normalized Average Feature Values over Time for Patients with Sepsis (Preprocessed Data)')
axes[0, 1].set_xlabel('Hour')
axes[0, 1].set_ylabel('Normalized Average Feature Values')
axes[0, 1].set_ylim([-0.5, 0.5])
axes[0, 1].legend()

# Plot data for patients without sepsis in sampled_data (Preprocessed)
for feature in features:
    axes[1, 1].plot(no_sepsis_data_avg.index, no_sepsis_data_avg[feature], label=feature, alpha=0.6)
axes[1, 1].set_title('Normalized Average Feature Values over Time for Patients without Sepsis (Preprocessed Data)')
axes[1, 1].set_xlabel('Hour')
axes[1, 1].set_ylabel('Normalized Average Feature Values')
axes[1, 1].set_ylim([-0.5, 0.5])
axes[1, 1].legend()

plt.tight_layout()
plt.show()