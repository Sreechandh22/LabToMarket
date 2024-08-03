import pandas as pd
import os

# Define the column names
columns = [
    'BufferId', 'Time', 'S11', 'S01', 'S03', 'S05', 'S07', 'S09', 'S06', 'S02', 'S04', 'S08', 'S10', 'S12', 
    'T3', 'T4', 'T2', 'T1', 'L1', 'L2', 'L4', 'L3', 'V1', 'V2', 'V3', 'V4', 'Unnamed: 26'
]

# Conversion factor from nm to tonnage
conversion_factor = 0.01  # Example value, adjust as necessary

# Function to read and clean data from individual files
def prepare_data(file_paths, label):
    data = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, delimiter=',', names=columns, header=0)
        df['TrainId'] = os.path.basename(file_path)
        df['Condition'] = label
        data.append(df)
    combined_data = pd.concat(data, ignore_index=True)
    return combined_data

# File paths to cleaned files
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

# Prepare data
critical_data = prepare_data(critical_files, 'Critical')
good_data = prepare_data(good_files, 'Good')

# Calculate MDIL, ADIL, and ILF for each axle
def calculate_ilf(df):
    df['Left_MDIL'] = df[['S01', 'S03', 'S05', 'S07', 'S09', 'S11']].max(axis=1) * conversion_factor
    df['Right_MDIL'] = df[['S02', 'S04', 'S06', 'S08', 'S10', 'S12']].max(axis=1) * conversion_factor
    
    df['Left_ADIL'] = df[['S01', 'S03', 'S05', 'S07', 'S09', 'S11']].apply(lambda x: (x.sum() - x.max()) / 5 * conversion_factor, axis=1)
    df['Right_ADIL'] = df[['S02', 'S04', 'S06', 'S08', 'S10', 'S12']].apply(lambda x: (x.sum() - x.max()) / 5 * conversion_factor, axis=1)
    
    df['Left_ILF'] = df['Left_MDIL'] / df['Left_ADIL']
    df['Right_ILF'] = df['Right_MDIL'] / df['Right_ADIL']
    
    return df

critical_data = calculate_ilf(critical_data)
good_data = calculate_ilf(good_data)

# Define thresholds for ILF based on provided examples
ilf_threshold_critical = 4.5
ilf_threshold_warning = 2
mdil_threshold_critical = 35
mdil_threshold_warning = 20

# Determine wheel status based on ILF values
def determine_wheel_status(ilf, mdil):
    if mdil >= mdil_threshold_critical or ilf >= ilf_threshold_critical:
        return 'Critical'
    elif mdil_threshold_warning <= mdil < mdil_threshold_critical or ilf_threshold_warning <= ilf < ilf_threshold_critical:
        return 'Warning'
    else:
        return 'Good'

# Apply the function to determine wheel status
critical_data['Left_Wheel_Status'] = critical_data.apply(lambda x: determine_wheel_status(x['Left_ILF'], x['Left_MDIL']), axis=1)
critical_data['Right_Wheel_Status'] = critical_data.apply(lambda x: determine_wheel_status(x['Right_ILF'], x['Right_MDIL']), axis=1)
good_data['Left_Wheel_Status'] = good_data.apply(lambda x: determine_wheel_status(x['Left_ILF'], x['Left_MDIL']), axis=1)
good_data['Right_Wheel_Status'] = good_data.apply(lambda x: determine_wheel_status(x['Right_ILF'], x['Right_MDIL']), axis=1)

# Extract relevant columns
left_wheels_info_critical = critical_data[['BufferId', 'Left_MDIL', 'Left_ILF', 'Left_Wheel_Status']].copy()
right_wheels_info_critical = critical_data[['BufferId', 'Right_MDIL', 'Right_ILF', 'Right_Wheel_Status']].copy()
left_wheels_info_good = good_data[['BufferId', 'Left_MDIL', 'Left_ILF', 'Left_Wheel_Status']].copy()
right_wheels_info_good = good_data[['BufferId', 'Right_MDIL', 'Right_ILF', 'Right_Wheel_Status']].copy()

# Rename columns for right wheels to match the left wheels
right_wheels_info_critical.columns = ['BufferId', 'MDIL', 'ILF', 'Wheel_Status']
left_wheels_info_critical.columns = ['BufferId', 'MDIL', 'ILF', 'Wheel_Status']
right_wheels_info_good.columns = ['BufferId', 'MDIL', 'ILF', 'Wheel_Status']
left_wheels_info_good.columns = ['BufferId', 'MDIL', 'ILF', 'Wheel_Status']

# Add the side information
left_wheels_info_critical['Side'] = 'Left'
right_wheels_info_critical['Side'] = 'Right'
left_wheels_info_good['Side'] = 'Left'
right_wheels_info_good['Side'] = 'Right'

# Combine left and right wheel information
combined_wheels_info = pd.concat([left_wheels_info_critical, right_wheels_info_critical, left_wheels_info_good, right_wheels_info_good])

# Select only the relevant columns
combined_wheels_info = combined_wheels_info[['BufferId', 'Side', 'MDIL', 'ILF', 'Wheel_Status']]

# Save the combined results to a single CSV file
combined_wheels_info.to_csv('C:/Users/sreec/OneDrive/Desktop/csv_files 2/wheel_conditions_with_train_info.csv', index=False)

print("Detailed info of wheels saved to 'wheel_conditions_with_train_info.csv'")
