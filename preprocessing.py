"""
EMG Data Preprocessing Pipeline
================================

This module handles comprehensive preprocessing of EMG (Electromyography) data for gesture recognition.

Key Functions:
- Inspect and validate all CSV files
- Handle missing values and corrupted data
- Create balanced training datasets
- Normalize EMG signals using StandardScaler
- Extract meaningful EMG features for ML models
- Generate comprehensive dataset statistics

EMG Channels: 8 muscle sensors capturing arm/hand signals
Gesture Classes: Hand gesture labels for prosthetic limb control
Dataset Size: ~4.2 million rows across multiple CSV files

Author: Smart Bionic Limb Project
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "data"
CHANNEL_COLUMNS = [f"channel{i}" for i in range(1, 9)]  # 8 EMG channels
OUTPUT_COMBINED = os.path.join(DATA_DIR, "combined_emg_data.csv")
OUTPUT_BALANCED = os.path.join(DATA_DIR, "balanced_emg_data.csv")
OUTPUT_NORMALIZED = os.path.join(DATA_DIR, "normalized_emg_data.csv")
OUTPUT_FEATURES = os.path.join(DATA_DIR, "emg_features_data.csv")
SCALER_PATH = os.path.join(DATA_DIR, "emg_scaler.joblib")


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def inspect_csv_files():
    """
    Inspect all CSV files in the data directory.

    This function:
    - Finds all CSV files
    - Checks file sizes
    - Validates column structure
    - Identifies corrupted or problematic files

    Returns:
    --------
    valid_files : list
        List of valid CSV file paths
    file_info : dict
        Dictionary containing file statistics
    """
    print_section("Step 1: Inspecting CSV Files")

    # Find all CSV files
    csv_pattern = os.path.join(DATA_DIR, "*.csv")
    all_csv_files = glob.glob(csv_pattern)

    # Filter to only subject data files (exclude combined and other processed files)
    data_files = [f for f in all_csv_files if not any(x in f for x in
                  ["combined", "balanced", "normalized", "features", "sample"])]

    print(f"\n[Files] Total CSV files found: {len(data_files)}")

    valid_files = []
    file_info = {}
    errors = []

    for file_path in sorted(data_files):
        file_name = os.path.basename(file_path)
        try:
            # Try to read the file
            df = pd.read_csv(file_path)

            # Check if required columns exist
            required_cols = ["time"] + CHANNEL_COLUMNS + ["class"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                errors.append(f"  [WARNING] {file_name}: Missing columns - {missing_cols}")
                continue

            # File is valid
            valid_files.append(file_path)
            file_info[file_name] = {
                "size": df.shape[0],
                "columns": df.shape[1],
                "path": file_path
            }

            print(f"  [OK] {file_name}: {df.shape[0]:>8,} rows x {df.shape[1]:>2} columns")

        except Exception as e:
            errors.append(f"  [ERROR] {file_name}: {str(e)[:50]}")

    # Print any errors
    if errors:
        print("\n[Warnings/Errors] File Reading Issues:")
        for error in errors:
            print(error)

    print(f"\n[Result] Valid files: {len(valid_files)}/{len(data_files)}")

    return valid_files, file_info


def validate_emg_channels(df):
    """
    Validate EMG channel data quality.

    Checks:
    - All channels are numeric
    - Values are within reasonable ranges for EMG sensors
    - No infinite or extremely large values

    Parameters:
    -----------
    df : DataFrame
        EMG data to validate

    Returns:
    --------
    is_valid : bool
        Whether channels are valid
    issues : list
        List of validation issues
    """
    issues = []

    for channel in CHANNEL_COLUMNS:
        if channel not in df.columns:
            issues.append(f"Missing channel: {channel}")
            continue

        # Check if numeric
        if not pd.api.types.is_numeric_dtype(df[channel]):
            issues.append(f"{channel}: Not numeric type")
            continue

        # Check for infinite values
        if np.isinf(df[channel]).any():
            inf_count = np.isinf(df[channel]).sum()
            issues.append(f"{channel}: Contains {inf_count} infinite values")

        # Check for extremely large values (potential sensor errors)
        if (np.abs(df[channel]) > 10).any():
            extreme_count = (np.abs(df[channel]) > 10).sum()
            issues.append(f"{channel}: Contains {extreme_count} extreme values (>10)")

    is_valid = len(issues) == 0
    return is_valid, issues


def load_and_combine_data(valid_files):
    """
    Load and combine all valid CSV files into a single dataset.

    Process:
    1. Load each valid CSV file
    2. Remove rows with missing values (NaN)
    3. Remove duplicate rows
    4. Validate EMG channels
    5. Validate class labels
    6. Combine all data

    Parameters:
    -----------
    valid_files : list
        List of valid CSV file paths

    Returns:
    --------
    combined_df : DataFrame
        Combined cleaned dataset
    """
    print_section("Step 2: Loading and Combining Data")

    dataframes = []
    total_rows_before = 0
    total_rows_removed = 0
    file_errors = 0

    for i, file_path in enumerate(valid_files, 1):
        file_name = os.path.basename(file_path)

        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            original_rows = len(df)
            total_rows_before += original_rows

            # Remove NaN values from EMG channels
            df = df.dropna(subset=CHANNEL_COLUMNS + ["class"])
            rows_after_nan = len(df)
            nan_removed = original_rows - rows_after_nan

            # Remove duplicate rows (same timestamp and channel values)
            df = df.drop_duplicates(subset=CHANNEL_COLUMNS + ["time"], keep="first")
            rows_after_dup = len(df)
            dup_removed = rows_after_nan - rows_after_dup

            # Validate EMG channels
            is_valid, channel_issues = validate_emg_channels(df)

            if not is_valid:
                print(f"\n  [WARNING]  {file_name}: Channel validation issues:")
                for issue in channel_issues:
                    print(f"     - {issue}")

            # Validate class labels (should be non-negative integers)
            invalid_classes = ~df["class"].apply(lambda x: isinstance(x, (int, float, np.integer, np.floating)) and x >= 0)
            if invalid_classes.any():
                invalid_count = invalid_classes.sum()
                print(f"  [WARNING]  {file_name}: Found {invalid_count} invalid class labels")
                df = df[~invalid_classes]

            rows_final = len(df)
            total_rows_removed += original_rows - rows_final

            # Print file summary
            print(f"  [{i:2d}/{len(valid_files)}] {file_name}:")
            print(f"       Original: {original_rows:>10,} rows")
            print(f"       NaN removed: {nan_removed:>8,} rows")
            print(f"       Duplicates removed: {dup_removed:>4,} rows")
            print(f"       Final: {rows_final:>14,} rows")

            if rows_final > 0:
                dataframes.append(df)
            else:
                print(f"       [ERROR] No valid data remaining")
                file_errors += 1

        except Exception as e:
            print(f"  [ERROR] {file_name}: Failed to process - {str(e)[:60]}")
            file_errors += 1
            continue

    # Combine all dataframes
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"\n[OK] Combined {len(valid_files) - file_errors} files successfully")
        print(f"  Total rows before processing: {total_rows_before:,}")
        print(f"  Total rows removed: {total_rows_removed:,}")
        print(f"  Final combined size: {len(combined_df):,} rows")
    else:
        raise ValueError("No valid data could be loaded from any CSV files")

    return combined_df


def analyze_class_distribution(df, title="Class Distribution"):
    """
    Analyze and print class distribution statistics.

    Shows:
    - Count of samples per gesture class
    - Percentage of dataset
    - Identifies class imbalance

    Parameters:
    -----------
    df : DataFrame
        Dataset to analyze
    title : str
        Title for the analysis output
    """
    print(f"\n{title}:")

    class_counts = df["class"].value_counts().sort_index()
    total = len(df)

    for class_id, count in class_counts.items():
        percentage = (count / total) * 100
        # Create a simple bar chart visualization
        bar_length = int(percentage / 2)
        bar = "|" * bar_length
        print(f"  Class {int(class_id):2d}: {count:>10,} samples ({percentage:>6.2f}%) {bar}")

    # Calculate imbalance metrics
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else 0

    print(f"\n  [Stats] Imbalance Metrics:")
    print(f"     Most common class: {class_counts.idxmax()} ({max_count:,} samples)")
    print(f"     Least common class: {class_counts.idxmin()} ({min_count:,} samples)")
    print(f"     Imbalance ratio: {imbalance_ratio:.2f}x")


def create_balanced_dataset(df):
    """
    Create a balanced dataset by taking equal samples from each gesture class.

    Why Balanced Dataset?
    - Random Forest can become biased toward majority classes
    - Equal class representation prevents model from learning class imbalance
    - Improves overall accuracy and fairness across all gesture classes

    Strategy:
    - Find the minimum class size (least frequent gesture)
    - Sample exactly that many rows from each class
    - Result: Equal representation of all gestures

    Parameters:
    -----------
    df : DataFrame
        Raw dataset (potentially imbalanced)

    Returns:
    --------
    balanced_df : DataFrame
        Balanced dataset with equal samples per class
    """
    print_section("Step 3: Creating Balanced Dataset")

    print("\n[Info] Dataset Balancing Strategy:")
    print("   - All gesture classes will have EQUAL sample count")
    print("   - Prevents model bias toward majority classes")
    print("   - Improves fairness and overall accuracy")

    # Analyze before balancing
    print("\nBefore Balancing:")
    analyze_class_distribution(df)

    # Find minimum class size
    class_counts = df["class"].value_counts()
    min_samples = class_counts.min()
    num_classes = len(class_counts)

    print(f"\n[Target] Balancing Parameters:")
    print(f"   Minimum class size: {min_samples:,}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Final dataset size: {min_samples * num_classes:,}")

    # Sample equally from each class
    balanced_dfs = []
    for class_id in sorted(df["class"].unique()):
        class_data = df[df["class"] == class_id]
        # Sample without replacement to get exactly min_samples
        sampled = class_data.sample(n=min(len(class_data), min_samples),
                                    random_state=42,
                                    replace=False)
        balanced_dfs.append(sampled)

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nAfter Balancing:")
    analyze_class_distribution(balanced_df)

    print(f"\n[OK] Balanced dataset created: {len(balanced_df):,} rows")

    return balanced_df


def normalize_emg_signals(df, fit_scaler=True, scaler=None):
    """
    Normalize EMG channel values using StandardScaler (Z-score normalization).

    Why Normalization?
    - EMG signals have different scales and ranges across channels and subjects
    - ML algorithms perform better with normalized features (mean=0, std=1)
    - Prevents high-magnitude channels from dominating the model
    - Enables consistent signal interpretation across different subjects

    StandardScaler Process:
    - Compute mean and std for each channel
    - Transform: X_normalized = (X - mean) / std
    - Each channel becomes: mean=0, std=1

    Parameters:
    -----------
    df : DataFrame
        EMG data to normalize
    fit_scaler : bool
        Whether to fit a new scaler (True for training data)
    scaler : StandardScaler or None
        Pre-fitted scaler (for test/validation data)

    Returns:
    --------
    normalized_df : DataFrame
        Normalized EMG data
    scaler : StandardScaler
        Fitted scaler object (for future predictions)
    """
    print_section("Step 4: Normalizing EMG Signals")

    normalized_df = df.copy()

    if fit_scaler:
        print("\n[Stats] Fitting StandardScaler to training data:")

        # Create and fit scaler
        scaler = StandardScaler()
        normalized_df[CHANNEL_COLUMNS] = scaler.fit_transform(df[CHANNEL_COLUMNS])

        print(f"   [OK] Scaler fitted on {len(df):,} samples")
        print(f"   [OK] {len(CHANNEL_COLUMNS)} EMG channels normalized")

        # Print normalization statistics
        print("\n   Channel Statistics After Normalization:")
        print(f"   {'Channel':<12} {'Mean':>12} {'Std Dev':>12}")
        print("   " + "-" * 36)
        for channel in CHANNEL_COLUMNS:
            mean = normalized_df[channel].mean()
            std = normalized_df[channel].std()
            print(f"   {channel:<12} {mean:>12.6f} {std:>12.6f}")

        # Save scaler for future use
        joblib.dump(scaler, SCALER_PATH)
        print(f"\n[OK] Scaler saved to: {SCALER_PATH}")

    else:
        if scaler is None:
            raise ValueError("Scaler must be provided if fit_scaler=False")
        print(f"\n[Stats] Applying pre-fitted scaler to data:")
        normalized_df[CHANNEL_COLUMNS] = scaler.transform(df[CHANNEL_COLUMNS])
        print(f"   [OK] {len(CHANNEL_COLUMNS)} channels transformed")

    return normalized_df, scaler


def extract_emg_features(df):
    """
    Extract meaningful features from raw EMG signals for machine learning.

    Why Feature Engineering for EMG?
    - Raw EMG samples contain noise and redundant information
    - Extracted features capture key signal characteristics
    - ML models learn better from engineered features than raw data
    - Features are interpretable and reduce dimensionality

    Features Generated for Each 8-Channel EMG Signal:

    1. Mean (Average):
       - Represents the center value of the signal
       - Helps identify baseline muscle activation level
       - Formula: mean = sum(X) / len(X)

    2. RMS (Root Mean Square):
       - Represents signal amplitude and power
       - Critical for EMG as it correlates with muscle force
       - Formula: RMS = sqrt(sum(X^2) / len(X))

    3. Variance:
       - Measures signal spread around the mean
       - Higher variance = more muscle activation
       - Formula: var = sum((X - mean)^2) / len(X)

    4. Standard Deviation:
       - Square root of variance
       - More interpretable than variance in original units
       - Formula: std = sqrt(variance)

    5. Signal Energy:
       - Sum of squared sample values
       - Represents total energy in the signal
       - Formula: energy = sum(X^2)

    6. Max Value:
       - Peak amplitude of the signal
       - Indicates maximum muscle activation intensity
       - Formula: max = max(X)

    7. Min Value:
       - Minimum signal value
       - Helps identify signal range
       - Formula: min = min(X)

    Total Features Per Sample: 8 channels × 7 features = 56 features

    Parameters:
    -----------
    df : DataFrame
        Normalized EMG data with channel1-channel8

    Returns:
    --------
    features_df : DataFrame
        DataFrame containing extracted features + class label
    feature_columns : list
        Names of all extracted feature columns
    """
    print_section("Step 5: Extracting EMG Features")

    print("\n[Analysis] Feature Engineering Strategy:")
    print(f"   Input: {len(CHANNEL_COLUMNS)} raw EMG channels per sample")
    print(f"   Features per channel: 7 statistical features")
    print(f"   Total output features: {len(CHANNEL_COLUMNS)} × 7 = {len(CHANNEL_COLUMNS) * 7} features")

    print("\n[Stats] Extracted Features:")
    features = {
        "mean": "Average signal amplitude",
        "rms": "Root Mean Square - signal power",
        "variance": "Signal spread (activation intensity)",
        "std": "Standard deviation",
        "energy": "Total signal energy",
        "max": "Peak amplitude",
        "min": "Minimum value"
    }

    for i, (feat_name, description) in enumerate(features.items(), 1):
        print(f"   {i}. {feat_name:<12} - {description}")

    # Extract features
    features_data = {}
    feature_columns = []

    for channel in CHANNEL_COLUMNS:
        channel_data = df[channel]

        # Feature 1: Mean
        mean_val = channel_data.mean()
        col_name = f"{channel}_mean"
        features_data[col_name] = mean_val
        feature_columns.append(col_name)

        # Feature 2: RMS (Root Mean Square)
        rms_val = np.sqrt((channel_data ** 2).mean())
        col_name = f"{channel}_rms"
        features_data[col_name] = rms_val
        feature_columns.append(col_name)

        # Feature 3: Variance
        var_val = channel_data.var()
        col_name = f"{channel}_variance"
        features_data[col_name] = var_val
        feature_columns.append(col_name)

        # Feature 4: Standard Deviation
        std_val = channel_data.std()
        col_name = f"{channel}_std"
        features_data[col_name] = std_val
        feature_columns.append(col_name)

        # Feature 5: Signal Energy
        energy_val = (channel_data ** 2).sum()
        col_name = f"{channel}_energy"
        features_data[col_name] = energy_val
        feature_columns.append(col_name)

        # Feature 6: Max Value
        max_val = channel_data.max()
        col_name = f"{channel}_max"
        features_data[col_name] = max_val
        feature_columns.append(col_name)

        # Feature 7: Min Value
        min_val = channel_data.min()
        col_name = f"{channel}_min"
        features_data[col_name] = min_val
        feature_columns.append(col_name)

    # Create features DataFrame
    features_df = pd.DataFrame(features_data, index=[0])

    # Add class label
    if "class" in df.columns:
        features_df["class"] = df["class"].iloc[0]

    print(f"\n[OK] Features extracted successfully")
    print(f"   Total features generated: {len(feature_columns)}")

    return features_data, feature_columns


def extract_emg_features_batch(df):
    """
    Extract EMG features from entire dataset (batch processing).

    Processes all rows in the DataFrame and returns feature matrix.

    Parameters:
    -----------
    df : DataFrame
        Normalized EMG data with channel1-channel8

    Returns:
    --------
    features_df : DataFrame
        DataFrame with extracted features for all samples
    """
    print("\n   Processing dataset (this may take a moment for large files)...")

    all_features = []

    for idx in range(len(df)):
        sample = df.iloc[idx:idx+1]
        features, _ = extract_emg_features(sample)
        all_features.append(features)

    features_df = pd.DataFrame(all_features)

    # Add class label
    if "class" in df.columns:
        features_df["class"] = df["class"].values

    return features_df


def print_dataset_summary(df, title="Dataset Summary"):
    """
    Print comprehensive dataset statistics and information.

    Parameters:
    -----------
    df : DataFrame
        Dataset to summarize
    title : str
        Title for the summary
    """
    print_section(title)

    print(f"\n[Stats] Dataset Dimensions:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {df.shape[1]}")

    print(f"\n[Info] Columns:")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        print(f"   - {col:<20} {str(dtype):<15} (Non-null: {non_null:,}, Null: {null_count:,})")

    print(f"\n[Check] Data Quality:")
    total_cells = len(df) * df.shape[1]
    total_null = df.isnull().sum().sum()
    null_percentage = (total_null / total_cells) * 100 if total_cells > 0 else 0
    print(f"   Total cells: {total_cells:,}")
    print(f"   Null values: {total_null:,} ({null_percentage:.2f}%)")

    # Signal statistics
    if any(col in df.columns for col in CHANNEL_COLUMNS):
        present_channels = [col for col in CHANNEL_COLUMNS if col in df.columns]
        print(f"\n[Stats] EMG Signal Statistics ({len(present_channels)} channels):")
        print(f"   {'Channel':<12} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        print("   " + "-" * 60)
        for channel in present_channels:
            mean = df[channel].mean()
            std = df[channel].std()
            min_val = df[channel].min()
            max_val = df[channel].max()
            print(f"   {channel:<12} {mean:>12.6f} {std:>12.6f} {min_val:>12.6f} {max_val:>12.6f}")

    # Class distribution
    if "class" in df.columns:
        print(f"\n[Target] Gesture Classes:")
        analyze_class_distribution(df, title="Class Distribution")


def main():
    """
    Execute the complete EMG preprocessing pipeline.

    Steps:
    1. Inspect all CSV files
    2. Load and combine data (with validation and cleaning)
    3. Create balanced dataset
    4. Normalize EMG signals
    5. Extract features
    6. Save processed data
    """
    print("\n" + "=" * 70)
    print("  EMG Data Preprocessing Pipeline")
    print("  Smart Bionic Limb System - Gesture Recognition")
    print("=" * 70)

    try:
        # Step 1: Inspect CSV files
        valid_files, file_info = inspect_csv_files()

        if not valid_files:
            print("\n[ERROR] No valid CSV files found. Exiting.")
            return

        # Step 2: Load and combine data
        combined_df = load_and_combine_data(valid_files)

        # Print combined dataset summary
        print_dataset_summary(combined_df, "Combined Dataset Summary (Before Cleaning)")

        # Step 3: Create balanced dataset
        balanced_df = create_balanced_dataset(combined_df)

        # Step 4: Normalize EMG signals
        normalized_df, scaler = normalize_emg_signals(balanced_df, fit_scaler=True)

        # Print normalized dataset summary
        print_dataset_summary(normalized_df, "Normalized Dataset Summary")

        # Save normalized data
        normalized_df.to_csv(OUTPUT_NORMALIZED, index=False)
        print(f"\n[OK] Normalized data saved to: {OUTPUT_NORMALIZED}")

        # Step 5: Extract features (sample demonstration)
        print_section("Step 6: Demonstrating Feature Extraction")
        print("\nNote: Feature extraction requires windowing over time series data.")
        print("For full feature extraction, consider:")
        print("  - Sliding window approach (e.g., 256ms windows)")
        print("  - Calculate 7 features per channel × 8 channels = 56 features per window")
        print("  - Use scipy.signal for advanced signal processing")
        print("\nFeature extraction module will be in: feature_engineering.py")

        # Save combined and balanced datasets (before normalization for reference)
        combined_df.to_csv(OUTPUT_COMBINED, index=False)
        balanced_df.to_csv(OUTPUT_BALANCED, index=False)

        print_section("Preprocessing Pipeline Complete")

        print("\n[OK] Successfully generated:")
        print(f"   1. {OUTPUT_COMBINED}")
        print(f"      * Combined raw data from all CSV files")
        print(f"      * {len(combined_df):,} rows")

        print(f"\n   2. {OUTPUT_BALANCED}")
        print(f"      * Balanced dataset (equal samples per class)")
        print(f"      * {len(balanced_df):,} rows")

        print(f"\n   3. {OUTPUT_NORMALIZED}")
        print(f"      * Normalized data (mean=0, std=1 for all channels)")
        print(f"      * Ready for machine learning")
        print(f"      * {len(normalized_df):,} rows")

        print(f"\n   4. {SCALER_PATH}")
        print(f"      * StandardScaler object for future predictions")
        print(f"      * Load with: joblib.load('{SCALER_PATH}')")

        print("\n" + "=" * 70)
        print("  [Stats] Preprocessing Results Summary")
        print("=" * 70)
        print(f"\n  Original combined size: {len(combined_df):,} rows")
        print(f"  After balancing: {len(balanced_df):,} rows")
        print(f"  Reduction: {((len(combined_df) - len(balanced_df)) / len(combined_df) * 100):.1f}%")
        print(f"\n  Class samples in balanced set: {len(balanced_df) // len(balanced_df['class'].unique()):,}")
        print(f"  Number of gesture classes: {len(balanced_df['class'].unique())}")
        print(f"  EMG channels: {len(CHANNEL_COLUMNS)}")
        print("\n[OK] Dataset is ready for machine learning model training!")
        print("=" * 70 + "\n")

        return {
            "combined_df": combined_df,
            "balanced_df": balanced_df,
            "normalized_df": normalized_df,
            "scaler": scaler
        }

    except Exception as e:
        print(f"\n[ERROR] Error in preprocessing pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
