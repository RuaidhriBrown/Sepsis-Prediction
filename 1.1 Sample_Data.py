import os
import pandas as pd
import numpy as np

# Load the dataset
file_path = 'data/Dataset.csv'
data = pd.read_csv(file_path)

# Drop columns with more than 40% missing values
threshold = 0.40
missing_percentage = data.isnull().mean()
columns_to_drop = missing_percentage[missing_percentage > threshold].index
data = data.drop(columns=columns_to_drop)

# Create Time-Based Features
data['TimeElapsed'] = data.groupby('Patient_ID')['Hour'].transform(lambda x: x - x.min())

# Create Eventual Sepsis Label
def create_eventual_sepsis_label(df):
    df['EventualSepsisLabel'] = df.groupby('Patient_ID')['SepsisLabel'].transform('max')
    return df

data = create_eventual_sepsis_label(data)

# Ensure the data is sorted by Patient_ID and Hour
data = data.sort_values(by=['Patient_ID', 'Hour'])

# Separate positive and negative EventualSepsisLabel patients
positive_patients = data[data['EventualSepsisLabel'] == 1]['Patient_ID'].unique()
negative_patients = data[data['EventualSepsisLabel'] == 0]['Patient_ID'].unique()

# Calculate the sample size as 10% of the original number of patients
sample_size = int(len(data['Patient_ID'].unique()) * 0.05)

# Sample an equal number of positive and negative patients
sampled_positive_ids = np.random.choice(positive_patients, size=sample_size // 2, replace=False)
sampled_negative_ids = np.random.choice(negative_patients, size=sample_size // 2, replace=False)

# Combine the sampled patient IDs
sampled_patient_ids = np.concatenate([sampled_positive_ids, sampled_negative_ids])

# Create the sampled dataset
sampled_data = data[data['Patient_ID'].isin(sampled_patient_ids)]

# Save the sampled data to CSV
sampled_data.to_csv('data/sampled_data.csv', index=False)

print("Sampled data saved to 'data/sampled_data.csv'")


