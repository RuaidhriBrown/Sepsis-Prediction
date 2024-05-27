import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from joblib import Parallel, delayed

# Load the sampled dataset
print("Loading dataset...")
sampled_data = pd.read_csv('data/sampled_data.csv')
print("Dataset loaded.")

# Function to write data to Excel in chunks
print("Saving data before preprocessing...")
with pd.ExcelWriter('data/preprocessing_steps.xlsx') as writer:
    sampled_data.to_excel(writer, sheet_name='Before_Preprocessing', index=False)
print("Data saved.")

# Function to treat outliers in the data, excluding certain variables
def treat_outliers(df, target_column, exclude_columns=[]):
    print("Treating outliers...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != target_column and col not in exclude_columns]
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    print("Outliers treated.")
    return df

# Apply outlier treatment to sampled data
sampled_data = treat_outliers(sampled_data, 'EventualSepsisLabel', exclude_columns=['SepsisLabel', 'Age', 'Hour'])

# Handle missing values according to recommendations
def handle_missing_values(df):
    print("Handling missing values...")
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')  # Drop unnecessary columns
    df = df.drop(columns=['HospAdmTime'], errors='ignore')  # Drop unnecessary columns
    
    def fill_group(group):
        group = group.sort_values(by='Hour')
        group.ffill(inplace=True)
        group.interpolate(method='linear', inplace=True)
        group.fillna(group.expanding().median(), inplace=True)
        return group

    df = df.groupby('Patient_ID').apply(fill_group).reset_index(drop=True)
    print("Missing values handled.")
    return df

# Apply the handle_missing_values function to sampled data
sampled_data = handle_missing_values(sampled_data)

# Temporal KNN imputation to ensure only past data is used for each time point
def temporal_knn_imputer(df, n_neighbors=5, n_jobs=-1):
    print("Starting temporal KNN imputation...")
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    def impute_group(group):
        group = group.sort_values(by='Hour')
        group_index = group.index
        
        if group.isnull().sum().sum() == 0:
            return group

        for i in range(len(group)):
            sub_group = group.iloc[:i+1]
            if sub_group.isnull().sum().sum() == 0:
                continue

            if len(sub_group) >= n_neighbors:
                sub_group_imputed = pd.DataFrame(imputer.fit_transform(sub_group), columns=sub_group.columns, index=sub_group.index)
                group.loc[sub_group.index] = sub_group_imputed
            else:
                # Find similar patients' data up to the current hour and within the last 10 hours
                max_hour = sub_group['Hour'].max()
                min_hour = max_hour - 10
                similar_data = df[(df['Patient_ID'] != group['Patient_ID'].iloc[0]) & 
                                  (df['Hour'] <= max_hour) & (df['Hour'] >= min_hour)]
                combined_data = pd.concat([sub_group, similar_data])
                combined_data_index = combined_data.index
                combined_data_imputed = pd.DataFrame(imputer.fit_transform(combined_data), columns=combined_data.columns, index=combined_data_index)
                group.loc[sub_group.index] = combined_data_imputed.loc[sub_group.index]
            
            if i % 100 == 0:
                print(f"Imputed up to hour {i+1} for patient {group['Patient_ID'].iloc[0]}")

        return group

    patient_groups = [group for _, group in df.groupby('Patient_ID')]
    results = Parallel(n_jobs=n_jobs)(delayed(impute_group)(group) for group in patient_groups)
    
    df_imputed = pd.concat(results).reset_index(drop=True)
    print("Temporal KNN imputation completed.")
    return df_imputed

# Apply temporal KNN imputation
sampled_data_imputed = temporal_knn_imputer(sampled_data)

print("Saving data before preprocessing...")
with pd.ExcelWriter('data/preprocessing_steps.xlsx') as writer:
    sampled_data_imputed.to_excel(writer, sheet_name='After_Preprocessing', index=False)
print("Data saved.")

# Cumulative Aggregation Function
def aggregate_patient_data(df):
    print("Aggregating patient data...")
    agg_df_list = []
    aggregation_dict = {
        'HR': ['mean', 'max', 'min', 'std'],
        'O2Sat': ['mean', 'max', 'min', 'std'],
        'Temp': ['mean', 'max', 'min', 'std'],
        'SBP': ['mean', 'max', 'min', 'std'],
        'MAP': ['mean', 'max', 'min', 'std'],
        'DBP': ['mean', 'max', 'min', 'std'],
        'Resp': ['mean', 'max', 'min', 'std'],
        'EtCO2': ['mean', 'max', 'min', 'std'],
        'BaseExcess': ['mean', 'max', 'min', 'std'],
        'HCO3': ['mean', 'max', 'min', 'std'],
        'FiO2': ['mean', 'max', 'min', 'std'],
        'pH': ['mean', 'max', 'min', 'std'],
        'PaCO2': ['mean', 'max', 'min', 'std'],
        'SaO2': ['mean', 'max', 'min', 'std'],
        'AST': ['mean', 'max', 'min', 'std'],
        'BUN': ['mean', 'max', 'min', 'std'],
        'Alkalinephos': ['mean', 'max', 'min', 'std'],
        'Calcium': ['mean', 'max', 'min', 'std'],
        'Chloride': ['mean', 'max', 'min', 'std'],
        'Creatinine': ['mean', 'max', 'min', 'std'],
        'Bilirubin_direct': ['mean', 'max', 'min', 'std'],
        'Glucose': ['mean', 'max', 'min', 'std'],
        'Lactate': ['mean', 'max', 'min', 'std'],
        'Magnesium': ['mean', 'max', 'min', 'std'],
        'Phosphate': ['mean', 'max', 'min', 'std'],
        'Potassium': ['mean', 'max', 'min', 'std'],
        'Bilirubin_total': ['mean', 'max', 'min', 'std'],
        'TroponinI': ['mean', 'max', 'min', 'std'],
        'Hct': ['mean', 'max', 'min', 'std'],
        'Hgb': ['mean', 'max', 'min', 'std'],
        'PTT': ['mean', 'max', 'min', 'std'],
        'WBC': ['mean', 'max', 'min', 'std'],
        'Fibrinogen': ['mean', 'max', 'min', 'std'],
        'Platelets': ['mean', 'max', 'min', 'std'],
        'TimeElapsed': ['max'],
    }
    
    # Filter the aggregation_dict to include only existing columns
    valid_aggregation_dict = {col: agg for col, agg in aggregation_dict.items() if col in df.columns}
    
    def aggregate_group(group):
        group = group.set_index('Hour')
        cumulative_group = group.expanding(min_periods=1).agg(valid_aggregation_dict).reset_index()
        cumulative_group.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in cumulative_group.columns]
        cumulative_group['Patient_ID'] = group['Patient_ID'].iloc[0]
        return cumulative_group
    
    agg_data = df.groupby('Patient_ID').apply(aggregate_group).reset_index(drop=True)
    print("Patient data aggregated.")
    return agg_data

# Apply the aggregation function to the filtered data
agg_data = aggregate_patient_data(sampled_data_imputed)


# Merge the EventualSepsisLabel back into the aggregated data
print("Merging EventualSepsisLabel...")
eventual_sepsis_labels = sampled_data_imputed[['Patient_ID', 'Hour', 'EventualSepsisLabel']].drop_duplicates()
agg_data = agg_data.merge(eventual_sepsis_labels, left_on=['Patient_ID', 'Hour_'], right_on=['Patient_ID', 'Hour'], how='left')

# Filter the data to exclude rows without a label
agg_data = agg_data[agg_data['EventualSepsisLabel'].notna()]

# Debugging: Check the shape and columns of agg_data before final imputation
print(f"Shape of agg_data before final imputation: {agg_data.shape}")
print(f"Columns of agg_data before final imputation: {agg_data.columns}")

# Final imputation step for if any values were missed
imputer = KNNImputer(n_neighbors=5)
agg_data_imputed = pd.DataFrame(imputer.fit_transform(agg_data), columns=agg_data.columns)

# Save the preprocessed data to CSV
agg_data_imputed.to_csv('data/preprocessed_data.csv', index=False)
print("Preprocessed data saved to 'data/preprocessed_data.csv'")

# Drop columns with all missing values
agg_data = agg_data.drop(columns=['HR_std', 'O2Sat_std', 'Temp_std', 'SBP_std', 'MAP_std', 'DBP_std', 'Resp_std'], errors='ignore')

# Append aggregated data to Excel
with pd.ExcelWriter('data/preprocessing_steps.xlsx', mode='a') as writer:
    agg_data_imputed.to_excel(writer, sheet_name='Aggregated_Data', index=False)

print("Detailed preprocessing steps saved to 'data/preprocessing_steps.xlsx'")
