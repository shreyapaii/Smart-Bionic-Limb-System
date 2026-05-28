import pandas as pd
import glob

# Load all CSV files
files = glob.glob("data/*.csv")

dataframes = []

for file in files:
    # Skip combined file if already exists
    if "combined_emg_data.csv" in file:
        continue

    df = pd.read_csv(file)
    dataframes.append(df)

# Combine all files
combined_data = pd.concat(dataframes, ignore_index=True)

# Save combined dataset
combined_data.to_csv("data/combined_emg_data.csv", index=False)

print("Combined dataset created successfully!")
print("Shape:", combined_data.shape)