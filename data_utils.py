"""
Utility Functions for EMG Data Processing
Helper functions for data validation, preprocessing, and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')


def validate_emg_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate EMG dataset structure and content
    
    Parameters:
    -----------
    df : DataFrame
        Input EMG dataset
    
    Returns:
    --------
    is_valid : bool
        Whether the dataset is valid
    issues : list
        List of validation issues found
    """
    issues = []
    
    # Check required columns
    required_cols = ['time'] + [f'channel{i}' for i in range(1, 9)]
    
    for col in required_cols:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
    
    # Check for class column (optional)
    if 'class' not in df.columns:
        issues.append("Warning: 'class' column not found - prediction only mode")
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        issues.append(f"Found {null_counts.sum()} null values")
    
    # Check data types
    for col in [c for c in required_cols if c in df.columns]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Column {col} should be numeric")
    
    # Check for negative values in time
    if 'time' in df.columns and (df['time'] < 0).any():
        issues.append("Found negative time values")
    
    is_valid = len([i for i in issues if not i.startswith("Warning")]) == 0
    
    return is_valid, issues


def preprocess_emg_signals(df: pd.DataFrame, 
                          normalize: bool = True,
                          remove_outliers: bool = True) -> pd.DataFrame:
    """
    Preprocess EMG signals
    
    Parameters:
    -----------
    df : DataFrame
        Raw EMG data
    normalize : bool
        Whether to normalize signals
    remove_outliers : bool
        Whether to remove outliers
    
    Returns:
    --------
    df_processed : DataFrame
        Processed EMG data
    """
    df_processed = df.copy()
    
    # Get channel columns
    channel_cols = [col for col in df.columns if col.startswith('channel')]
    
    # Remove outliers (optional)
    if remove_outliers:
        for col in channel_cols:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
    
    # Normalize signals (optional)
    if normalize:
        for col in channel_cols:
            mean = df_processed[col].mean()
            std = df_processed[col].std()
            if std > 0:
                df_processed[col] = (df_processed[col] - mean) / std
    
    return df_processed


def visualize_emg_signals(df: pd.DataFrame, 
                         sample_size: int = 1000,
                         save_path: str = None):
    """
    Visualize EMG signals
    
    Parameters:
    -----------
    df : DataFrame
        EMG dataset
    sample_size : int
        Number of samples to visualize
    save_path : str
        Path to save the plot
    """
    # Sample data
    df_sample = df.head(sample_size)
    
    # Get channel columns
    channel_cols = [col for col in df.columns if col.startswith('channel')]
    
    # Create subplots
    fig, axes = plt.subplots(len(channel_cols), 1, figsize=(12, 10))
    fig.suptitle('EMG Signal Visualization', fontsize=16, fontweight='bold')
    
    for idx, col in enumerate(channel_cols):
        axes[idx].plot(df_sample['time'], df_sample[col], linewidth=0.5)
        axes[idx].set_ylabel(col, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        
        if idx == len(channel_cols) - 1:
            axes[idx].set_xlabel('Time', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Signal visualization saved: {save_path}")
    
    plt.show()


def generate_sample_data(n_samples: int = 1000,
                        gestures: List[str] = None) -> pd.DataFrame:
    """
    Generate sample EMG data for testing
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    gestures : list
        List of gesture classes
    
    Returns:
    --------
    df : DataFrame
        Generated sample data
    """
    if gestures is None:
        gestures = ['rest', 'fist', 'wrist_flexion', 'wrist_extension', 'open_palm']
    
    # Generate time series
    time = np.linspace(0, 10, n_samples)
    
    # Generate random gesture labels
    classes = np.random.choice(gestures, size=n_samples)
    
    # Generate EMG signals with some patterns based on gesture
    data = {'time': time, 'class': classes}
    
    for i in range(1, 9):
        # Base signal
        signal = np.random.randn(n_samples) * 0.1
        
        # Add gesture-specific patterns
        for gesture in gestures:
            mask = classes == gesture
            if gesture == 'fist':
                signal[mask] += 0.5 + np.random.randn(mask.sum()) * 0.1
            elif gesture == 'open_palm':
                signal[mask] += 0.3 + np.random.randn(mask.sum()) * 0.1
            elif gesture == 'wrist_flexion':
                signal[mask] += 0.4 + np.random.randn(mask.sum()) * 0.1
            elif gesture == 'wrist_extension':
                signal[mask] += 0.35 + np.random.randn(mask.sum()) * 0.1
        
        data[f'channel{i}'] = signal
    
    return pd.DataFrame(data)


def print_dataset_summary(df: pd.DataFrame):
    """
    Print comprehensive dataset summary
    
    Parameters:
    -----------
    df : DataFrame
        EMG dataset
    """
    print("=" * 60)
    print("EMG Dataset Summary")
    print("=" * 60)
    
    print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Column information
    print("\n📋 Columns:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        print(f"   - {col}: {dtype} (nulls: {null_count})")
    
    # Class distribution
    if 'class' in df.columns:
        print("\n🎯 Gesture Distribution:")
        class_counts = df['class'].value_counts()
        for gesture, count in class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   - {gesture}: {count:,} ({percentage:.2f}%)")
    
    # Signal statistics
    channel_cols = [col for col in df.columns if col.startswith('channel')]
    if channel_cols:
        print("\n📈 Signal Statistics:")
        stats = df[channel_cols].describe()
        print(stats)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage
    print("EMG Data Utilities - Example Usage\n")
    
    # Generate sample data
    print("Generating sample EMG data...")
    sample_df = generate_sample_data(n_samples=1000)
    
    # Validate data
    is_valid, issues = validate_emg_data(sample_df)
    print(f"\nValidation: {'✓ PASS' if is_valid else '✗ FAIL'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    # Print summary
    print_dataset_summary(sample_df)
    
    # Save sample data
    sample_df.to_csv('data/sample_emg_data.csv', index=False)
    print("\n✓ Sample data saved to: data/sample_emg_data.csv")