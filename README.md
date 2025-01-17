# Wheel Condition Analysis

This project analyzes the condition of train wheels by calculating various metrics from sensor data and determining the status of each wheel. The project leverages Python and popular libraries like Pandas and Numpy to preprocess the data, calculate metrics, and determine wheel conditions.

## Table of Contents

- [Introduction](#introduction)
- [Project Objectives](#project-objectives)
- [Technology and Tools](#technology-and-tools)
- [Setup](#setup)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Implementation Details](#implementation-details)
- [License](#license)
- [Contact](#contact)

## Introduction

This project processes train wheel sensor data to calculate metrics such as Maximum Deflection Induced Load (MDIL), Average Deflection Induced Load (ADIL), and Induced Load Factor (ILF). Based on these metrics, the project determines the status of each wheel (Good, Warning, Critical) and saves the results to an Excel file.

## Project Objectives

1. Load and preprocess train wheel sensor data.
2. Calculate MDIL, ADIL, and ILF metrics.
3. Convert MDIL to tonnage.
4. Determine the status of each wheel.
5. Save the results to an Excel file.

## Technology and Tools

- **Programming Language**: Python
- **Libraries**: Pandas, Numpy, os
- **Data Format**: CSV, Excel

## Setup

1. **Clone the repository**:

    ```sh
    git clone https://github.com/yourusername/wheel-condition-analysis.git
    cd wheel-condition-analysis
    ```

2. **Create a virtual environment and activate it**:

    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages**:

    ```sh
    pip install pandas numpy
    ```

## Usage

1. **Prepare your data files**:
   - Place your CSV files in the specified directories (e.g., `Critical_Cleaned`, `Good_Cleaned`).

2. **Run the main script**:

    ```sh
    python main.py
    ```

3. **Check the results**:
   - The output will be saved as an Excel file named `wheel_conditions_with_train_info.xlsx`.

## File Structure

        wheel-condition-analysis/
        ├── main.py
        ├── README.md
        ├── .gitattributes
        ├── model.py
        ├── csv_files/
        │ ├── Critical_Cleaned/
        │ │ ├── T20240507124521.csv
        │ │ ├── T20240509153421.csv
        │ │ └── T20240609040556.csv
        │ ├── Good_Cleaned/
        │ │ ├── T20240609080529.csv
        │ │ ├── T20240609083255.csv
        │ │ ├── T20240609084817.csv
        │ │ ├── T20240609113752.csv
        │ │ └── T20240609132748.csv
        └── wheel_conditions_with_train_info.xlsx


## Implementation Details

### Data Loading

The `load_data` function reads CSV files and concatenates them into a single DataFrame, adding a `TrainId` column to identify the source of each record.

```python
def load_data(file_paths):
    data = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df['TrainId'] = file_path.split('/')[-1]
        data.append(df)
    return pd.concat(data, ignore_index=True)

```

## Contact

For any inquiries or collaboration opportunities, please contact sreechandh2204@gmail.com