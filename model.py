import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Define the column names
columns = [
    'BufferId', 'Time', 'S11', 'S01', 'S03', 'S05', 'S07', 'S09', 'S06', 'S02', 'S04', 'S08', 'S10', 'S12', 
    'T3', 'T4', 'T2', 'T1', 'L1', 'L2', 'L4', 'L3', 'V1', 'V2', 'V3', 'V4', 'Unnamed: 26'
]

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

# Combine data for analysis
combined_data = pd.concat([critical_data, good_data], ignore_index=True)

# Drop unnecessary column if present
if 'Unnamed: 26' in combined_data.columns:
    combined_data = combined_data.drop(columns=['Unnamed: 26'])

# Handle any missing values
combined_data = combined_data.dropna()

# Ensure valid data for threshold calculation by removing zero values
good_data_valid = good_data[(good_data[sensor_columns] > 0).all(axis=1)]

# Recalculate statistical measures for valid good data
good_stats = good_data_valid[sensor_columns].describe().T

# Determine thresholds based on good data
thresholds = {}
for sensor in sensor_columns:
    mean = good_stats.loc[sensor, 'mean']
    std = good_stats.loc[sensor, 'std']
    thresholds[sensor] = mean + 3 * std  # Example threshold using mean + 3*std

print("Determined thresholds for each sensor:")
print(thresholds)

# Detect spikes in critical data based on the determined thresholds
spikes = pd.DataFrame()
for sensor in sensor_columns:
    spikes[sensor] = (critical_data[sensor] > thresholds[sensor])

# Visualize the spikes detected
for sensor in sensor_columns:
    plt.figure(figsize=(15, 5))
    plt.plot(critical_data[sensor], label='Critical Data', color='red', alpha=0.6)
    plt.plot(spikes[sensor], label='Detected Spikes', color='blue', alpha=0.6)
    plt.title(f'Sensor {sensor} - Detected Spikes')
    plt.xlabel('Sample')
    plt.ylabel('Reading')
    plt.legend()
    plt.show()

# Add spikes information to combined data
for column in sensor_columns:
    combined_data[f'{column}_Spike'] = spikes[column]

# Separate the data back into training and testing
train_data = combined_data[~combined_data['TrainId'].str.contains('T20240609040556.csv')]
test_data = combined_data[combined_data['TrainId'].str.contains('T20240609040556.csv')]

# Train-Test Split
X_train = train_data.drop(columns=['Condition', 'TrainId', 'BufferId', 'Time'])
y_train = train_data['Condition']
X_test = test_data.drop(columns=['Condition', 'TrainId', 'BufferId', 'Time'])
y_test = test_data['Condition']

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Save the trained model
joblib.dump(clf, 'stress_model.pkl')

# Feature importance
feature_importances = pd.DataFrame(clf.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances['importance'], y=feature_importances.index)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Predict the condition for each wheel
combined_data['PredictedCondition'] = clf.predict(combined_data.drop(columns=['Condition', 'TrainId', 'BufferId', 'Time']))

# Group by TrainId and get the count of each condition per train
train_wheel_summary = combined_data.groupby(['TrainId', 'PredictedCondition']).size().unstack(fill_value=0)

# Detailed information about each wheel's condition
detailed_wheel_info = combined_data[['TrainId', 'BufferId', 'Time', 'PredictedCondition']]

# Identify critical wheels
critical_wheels = combined_data[combined_data['PredictedCondition'] == 'Critical']

# Save the predictions along with TrainId and wheel info
detailed_wheel_info.to_csv('C:/Users/sreec/OneDrive/Desktop/csv_files 2/wheel_conditions_with_train_info.csv', index=False)
train_wheel_summary.to_csv('C:/Users/sreec/OneDrive/Desktop/csv_files 2/train_wheel_summary.csv')
critical_wheels.to_csv('C:/Users/sreec/OneDrive/Desktop/csv_files 2/critical_wheels_info.csv', index=False)
print("Predictions for each wheel saved to 'wheel_conditions_with_train_info.csv'")
print("Summary of wheel conditions per train saved to 'train_wheel_summary.csv'")
print("Detailed info of critical wheels saved to 'critical_wheels_info.csv'")
