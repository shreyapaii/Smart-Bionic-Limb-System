import pandas as pd
import glob

# Get all CSV files from dataset folder
files = glob.glob("dataset/*.csv")

# Empty list to store dataframes
dataframes = []

# Read each CSV file
for file in files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Combine all dataframes into one
combined_data = pd.concat(dataframes, ignore_index=True)

# Show first 5 rows
print(combined_data.head())

# Show columns
print("\nColumns:")
print(combined_data.columns)

# Show dataset shape
print("\nShape:")
print(combined_data.shape)