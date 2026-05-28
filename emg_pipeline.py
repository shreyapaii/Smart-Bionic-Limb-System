"""
Unified EMG Data Pipeline
Automatically detects raw vs engineered data and preprocesses accordingly.
Reuses existing preprocessing and feature engineering functions.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from pathlib import Path

CHANNEL_COLUMNS = [f"channel{i}" for i in range(1, 9)]
FEATURE_STATS = ["mean", "rms", "variance", "std", "energy", "max", "min"]
SCALER_PATH = os.path.join("data", "emg_scaler.joblib")
WINDOW_SIZE = 50
STEP_SIZE = 5

# Expected engineered feature columns
EXPECTED_FEATURE_COLUMNS = [f"{ch}_{stat}" for ch in CHANNEL_COLUMNS for stat in FEATURE_STATS]


def is_raw_emg_format(df: pd.DataFrame) -> bool:
    """
    Detect if DataFrame contains raw EMG data.

    Raw EMG has:
    - Columns: time (optional), channel1-channel8, class (optional)
    - NO engineered feature columns like channel1_mean, channel1_rms, etc.
    """
    has_raw_channels = all(ch in df.columns for ch in CHANNEL_COLUMNS)
    has_engineered_features = any(col in df.columns for col in EXPECTED_FEATURE_COLUMNS)
    return has_raw_channels and not has_engineered_features


def is_engineered_format(df: pd.DataFrame) -> bool:
    """
    Detect if DataFrame contains engineered features.
    Should have most of the feature columns.
    """
    feature_cols_present = [col for col in EXPECTED_FEATURE_COLUMNS if col in df.columns]
    return len(feature_cols_present) >= 40  # At least 40 of 56 features


def validate_raw_emg(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate raw EMG data quality.

    Returns: (is_valid, error_message)
    """
    if df.empty:
        return False, "CSV file is empty"

    if len(df) < WINDOW_SIZE:
        return False, f"Not enough rows ({len(df)}) for feature extraction. Minimum: {WINDOW_SIZE} rows"

    # Check for required channels
    missing_channels = [ch for ch in CHANNEL_COLUMNS if ch not in df.columns]
    if missing_channels:
        return False, f"Missing EMG channels: {', '.join(missing_channels)}"

    # Validate numeric data
    for ch in CHANNEL_COLUMNS:
        try:
            df[ch] = pd.to_numeric(df[ch], errors='coerce')
            if df[ch].isna().sum() > 0:
                return False, f"Channel {ch} contains invalid numeric values"
            if np.isinf(df[ch]).any():
                return False, f"Channel {ch} contains infinite values"
        except Exception as e:
            return False, f"Channel {ch} validation failed: {str(e)}"

    # Check for sufficient data per class (if class column exists)
    if "class" in df.columns:
        class_counts = df["class"].value_counts()
        min_class_rows = class_counts.min()
        if min_class_rows < WINDOW_SIZE:
            return False, f"Some gesture classes have fewer than {WINDOW_SIZE} samples"

    return True, ""


def load_scaler():
    """Load the pre-fitted StandardScaler."""
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None


def normalize_emg_data(df: pd.DataFrame, scaler: StandardScaler = None) -> pd.DataFrame:
    """
    Normalize EMG channels using pre-fitted StandardScaler.
    If scaler not provided, loads from disk.
    """
    df_normalized = df.copy()

    if scaler is None:
        scaler = load_scaler()
        if scaler is None:
            raise ValueError("Scaler not found. Run preprocessing.py first.")

    df_normalized[CHANNEL_COLUMNS] = scaler.transform(df[CHANNEL_COLUMNS])
    return df_normalized


def extract_window_features(window_data: pd.DataFrame) -> dict:
    """
    Extract 7 statistical features per EMG channel from a window.
    56 total features = 8 channels × 7 statistics.
    Reuses feature engineering logic.
    """
    feats = {}
    for ch in CHANNEL_COLUMNS:
        if ch not in window_data.columns:
            raise ValueError(f"Channel {ch} not found in window data")
        x = window_data[ch].values
        feats[f"{ch}_mean"] = x.mean()
        feats[f"{ch}_rms"] = np.sqrt(np.mean(x ** 2))
        feats[f"{ch}_variance"] = np.var(x)
        feats[f"{ch}_std"] = np.std(x)
        feats[f"{ch}_energy"] = np.sum(x ** 2)
        feats[f"{ch}_max"] = x.max()
        feats[f"{ch}_min"] = x.min()
    return feats


def extract_features_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract engineered features from raw EMG data using sliding windows.
    Preserves class labels if present.
    """
    all_features = []

    # If class column exists, process per-class (preserves temporal coherence)
    if "class" in df.columns:
        for class_id in sorted(df["class"].unique()):
            class_df = df[df["class"] == class_id].reset_index(drop=True)
            n_rows = len(class_df)

            for start in range(0, n_rows - WINDOW_SIZE + 1, STEP_SIZE):
                window = class_df.iloc[start : start + WINDOW_SIZE]
                features = extract_window_features(window)
                features["class"] = class_id
                all_features.append(features)
    else:
        # No class column: process entire dataset
        n_rows = len(df)
        for start in range(0, n_rows - WINDOW_SIZE + 1, STEP_SIZE):
            window = df.iloc[start : start + WINDOW_SIZE]
            features = extract_window_features(window)
            all_features.append(features)

    return pd.DataFrame(all_features)


def process_emg_file(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, str]:
    """
    Unified pipeline: Automatically detect format and process accordingly.

    Returns:
        (engineered_features_df, input_type_detected)
    """
    if verbose:
        print(f"[Input] Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"[Input] Columns: {list(df.columns)[:10]}...")

    # Detect format
    if is_engineered_format(df):
        if verbose:
            print("[Type] ✅ Engineered features detected")
        engineered_cols = [col for col in EXPECTED_FEATURE_COLUMNS if col in df.columns]
        return df[engineered_cols].copy(), "engineered"

    elif is_raw_emg_format(df):
        if verbose:
            print("[Type] 🔬 Raw EMG detected")

        # Validate
        is_valid, error = validate_raw_emg(df)
        if not is_valid:
            raise ValueError(f"Raw EMG validation failed: {error}")

        if verbose:
            print("[Validate] ✅ Raw EMG validation passed")

        # Normalize
        df_normalized = normalize_emg_data(df)
        if verbose:
            print("[Preprocess] ✅ EMG signals normalized")

        # Extract features
        df_engineered = extract_features_from_raw(df_normalized)
        if verbose:
            print(f"[Features] ✅ Extracted {len(df_engineered)} samples × {df_engineered.shape[1]} features")
            if len(df_engineered) > 0:
                feature_cols = [col for col in df_engineered.columns if col != "class"]
                print(f"[Features] First 5 columns: {feature_cols[:5]}")

        return df_engineered, "raw"

    else:
        raise ValueError(
            "Cannot determine data format. Expected either:\n"
            "  1. Raw EMG: columns like 'channel1', 'channel2', ..., 'channel8'\n"
            "  2. Engineered features: columns like 'channel1_mean', 'channel1_rms', etc."
        )


def get_prediction_ready_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get features ready for model prediction.
    Returns only the feature columns (no class/metadata).
    """
    feature_cols = [col for col in EXPECTED_FEATURE_COLUMNS if col in df.columns]
    return df[feature_cols].fillna(0).astype(np.float32)
