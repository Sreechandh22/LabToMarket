import pandas as pd
import numpy as np
import os

def load_data(file_paths):
    data = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df['TrainId'] = file_path.split('/')[-1]
        data.append(df)
    return pd.concat(data, ignore_index=True)

def calculate_mdil_adil(df, odd_columns, even_columns):
    df['Left_MDIL'] = df[odd_columns].max(axis=1)
    df['Right_MDIL'] = df[even_columns].max(axis=1)
    
    df['Left_ADIL'] = df[odd_columns].apply(lambda row: (row.sum() - row.max()) / 5, axis=1)
    df['Right_ADIL'] = df[even_columns].apply(lambda row: (row.sum() - row.max()) / 5, axis=1)
    
    df['Left_ILF'] = df['Left_MDIL'] / df['Left_ADIL']
    df['Right_ILF'] = df['Right_MDIL'] / df['Right_ADIL']
    
    return df

def convert_to_tonnage(df, conversion_factor):
    df['Left_MDIL'] = df['Left_MDIL'] * conversion_factor
    df['Right_MDIL'] = df['Right_MDIL'] * conversion_factor
    return df

def determine_wheel_status(df):
    conditions_left = [
        (df['Left_MDIL'] >= 35) | (df['Left_ILF'] >= 4.5),
        (20 <= df['Left_MDIL']) & (df['Left_MDIL'] < 35) | (2 <= df['Left_ILF']) & (df['Left_ILF'] < 4.5)
    ]
    choices_left = ['Critical', 'Warning']
    df['Left_Wheel_Status'] = np.select(conditions_left, choices_left, default='Good')
    
    conditions_right = [
        (df['Right_MDIL'] >= 35) | (df['Right_ILF'] >= 4.5),
        (20 <= df['Right_MDIL']) & (df['Right_MDIL'] < 35) | (2 <= df['Right_ILF']) & (df['Right_ILF'] < 4.5)
    ]
    choices_right = ['Critical', 'Warning']
    df['Right_Wheel_Status'] = np.select(conditions_right, choices_right, default='Good')
    
    return df

def save_to_excel(df, filename):
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for train_id in df['TrainId'].unique():
            train_data = df[df['TrainId'] == train_id]
            train_data.to_excel(writer, sheet_name=train_id, index=False)

# Main process
conversion_factor = 0.01  # This is just an example, adjust as necessary
odd_columns = ['S01', 'S03', 'S05', 'S07', 'S09', 'S11']
even_columns = ['S02', 'S04', 'S06', 'S08', 'S10', 'S12']

critical_files = [
    'C:/Users/sreec/OneDrive/Desktop/csv_files 2/Critical_Cleaned/T20240507124521.csv',
    'C:/Users/sreec/OneDrive/Desktop/csv_files 2/Critical_Cleaned/T20240509153421.csv',
    'C:/Users/sreec/OneDrive/Desktop/csv_files 2/Critical_Cleaned/T20240609040556.csv'
]

good_files = [
    'C:/Users/sreec/OneDrive/Desktop/csv_files 2/Good_Cleaned/T20240609080529.csv',
    'C:/Users/sreec/OneDrive/Desktop/csv_files 2/Good_Cleaned/T20240609083255.csv',
    'C:/Users/sreec/OneDrive/Desktop/csv_files 2/Good_Cleaned/T20240609084817.csv',
    'C:/Users/sreec/OneDrive/Desktop/csv_files 2/Good_Cleaned/T20240609113752.csv',
    'C:/Users/sreec/OneDrive/Desktop/csv_files 2/Good_Cleaned/T20240609132748.csv'
]

# Load and preprocess data
critical_data = load_data(critical_files)
good_data = load_data(good_files)

# Calculate MDIL, ADIL, and ILF
critical_data = calculate_mdil_adil(critical_data, odd_columns, even_columns)
good_data = calculate_mdil_adil(good_data, odd_columns, even_columns)

# Convert MDIL to tonnage
critical_data = convert_to_tonnage(critical_data, conversion_factor)
good_data = convert_to_tonnage(good_data, conversion_factor)

# Determine wheel status
critical_data = determine_wheel_status(critical_data)
good_data = determine_wheel_status(good_data)

# Combine and save results
combined_data = pd.concat([critical_data, good_data], ignore_index=True)

# Extract the required columns and rename for clarity
combined_data_left = combined_data[['TrainId', 'BufferId', 'Left_MDIL', 'Left_ILF', 'Left_Wheel_Status']].copy()
combined_data_left['Side'] = 'Left'
combined_data_left = combined_data_left.rename(columns={'Left_MDIL': 'MDIL', 'Left_ILF': 'ILF', 'Left_Wheel_Status': 'Wheel_Status'})

combined_data_right = combined_data[['TrainId', 'BufferId', 'Right_MDIL', 'Right_ILF', 'Right_Wheel_Status']].copy()
combined_data_right['Side'] = 'Right'
combined_data_right = combined_data_right.rename(columns={'Right_MDIL': 'MDIL', 'Right_ILF': 'ILF', 'Right_Wheel_Status': 'Wheel_Status'})

# Combine left and right wheel data
final_combined_data = pd.concat([combined_data_left, combined_data_right], ignore_index=True)

# Save final results to Excel
save_to_excel(final_combined_data, 'wheel_conditions_with_train_info.xlsx')
