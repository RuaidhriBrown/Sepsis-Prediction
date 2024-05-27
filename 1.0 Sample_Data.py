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

# Sample a subset of patients
patient_ids = data['Patient_ID'].unique()
sampled_patient_ids = np.random.choice(patient_ids, size=int(len(patient_ids) * 0.05), replace=False)
sampled_data = data[data['Patient_ID'].isin(sampled_patient_ids)]

# Save the sampled data to CSV
sampled_data.to_csv('data/sampled_data.csv', index=False)

print("Sampled data saved to 'data/sampled_data.csv'")
