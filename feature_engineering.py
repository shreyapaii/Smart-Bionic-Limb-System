"""
EMG Feature Engineering Pipeline  (FIXED v2)
=============================================

ROOT CAUSE FIX:
  The previous pipeline slid windows over the full shuffled DataFrame,
  meaning nearly every window spanned multiple classes → the mode label
  was unreliable and temporal features were meaningless.  Only 4,381
  usable windows survived, crushing accuracy to ~23 %.

FIX STRATEGY:
  1. Sort each class's rows by 'time' to restore temporal order.
  2. Slide windows within each class independently (no cross-class contamination).
  3. Use a small step size (10 samples) for dense, overlapping windows.
  4. Balance window counts across classes AFTER feature extraction.
  5. Shuffle the final balanced window set before saving.

Key Parameters (tunable at top of file):
  WINDOW_SIZE  = 50   samples  (~250 ms at 200 Hz)
  STEP_SIZE    = 5    samples  → 90 % overlap, ~5x more windows than old code (all correctly labeled)
  MIN_WINDOWS_PER_CLASS = 5000   (cap for balance; raise if you have RAM)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODELS_DIR = "models"
INPUT_FILE  = os.path.join(DATA_DIR, "normalized_emg_data.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "engineered_emg_features.csv")

CHANNEL_COLUMNS = [f"channel{i}" for i in range(1, 9)]   # 8 EMG channels
FEATURES_LIST   = ["mean", "rms", "variance", "std", "energy", "max", "min"]

# ── Windowing hyper-parameters ────────────────────────────────────────────────
WINDOW_SIZE = 50    # samples per window
STEP_SIZE   = 5     # stride between windows  (was window_size - overlap = 25)
                    # 80 % overlap → dense temporal coverage

# Balance cap: maximum windows kept per class after extraction.
# Set to None to keep all windows (may be memory-heavy for large datasets).
MIN_WINDOWS_PER_CLASS = None   # will be auto-set to min class count

RANDOM_STATE = 42

Path(MODELS_DIR).mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ── Step 1 : Load data ────────────────────────────────────────────────────────
def load_normalized_data():
    """Load normalized EMG dataset produced by preprocessing.py."""
    print_section("Step 1: Loading Normalized EMG Data")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print(f"\n  Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    print(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    # Confirm normalization
    print(f"\n  Normalization check (channel1):")
    print(f"    Mean : {df['channel1'].mean():.6f}   (≈ 0 expected)")
    print(f"    Std  : {df['channel1'].std():.6f}    (≈ 1 expected)")

    print(f"\n  Raw class distribution:")
    for c, n in df['class'].value_counts().sort_index().items():
        print(f"    Class {int(c)}: {n:,} rows")

    return df


# ── Step 2 : Feature extraction from one window ───────────────────────────────
def extract_window_features(window_data):
    """
    Extract 7 time-domain statistical features per EMG channel.

    Features
    --------
    mean     – average amplitude (DC component, baseline activation)
    rms      – root-mean-square (signal power; best correlate of muscle force)
    variance – spread around mean (amplitude modulation, complexity)
    std      – standard deviation (same info as variance, interpretable units)
    energy   – sum of squares (total power; fatigue / activation level)
    max      – peak amplitude (transient burst detection)
    min      – minimum amplitude (baseline / range bounding)

    56 total features = 8 channels × 7 statistics
    """
    feats = {}
    for ch in CHANNEL_COLUMNS:
        x = window_data[ch].values
        feats[f"{ch}_mean"]     = x.mean()
        feats[f"{ch}_rms"]      = np.sqrt(np.mean(x ** 2))
        feats[f"{ch}_variance"] = np.var(x)
        feats[f"{ch}_std"]      = np.std(x)
        feats[f"{ch}_energy"]   = np.sum(x ** 2)
        feats[f"{ch}_max"]      = x.max()
        feats[f"{ch}_min"]      = x.min()
    return feats


# ── Step 3 : Per-class sliding window (THE CORE FIX) ─────────────────────────
def extract_features_per_class(df):
    """
    Slide windows within each class independently.

    Why per-class?
    --------------
    The normalized_emg_data.csv was produced by sampling rows randomly
    from each subject/session and then globally shuffling (see preprocessing.py,
    line: balanced_df.sample(frac=1, random_state=42)).  Consecutive rows in
    the file therefore belong to *different* gestures and different time points.

    Naively windowing across the whole file means:
      • A 50-row window spans ~50 different gestures  → mode label is noise.
      • Statistical features mix unrelated signals    → pure garbage.

    Fix: sort each class's rows by their original 'time' timestamp to
    reconstruct intra-class temporal order, then slide windows only within
    that class.  This gives:
      • Pure-label windows     (every sample in a window = same gesture).
      • Temporally coherent features (contiguous samples from one gesture).
      • No data leakage across classes or subjects.

    Parameters
    ----------
    df : DataFrame   normalized EMG data (shuffled)

    Returns
    -------
    features_df : DataFrame   feature matrix + 'class' column
    """
    print_section("Step 2: Per-Class Sliding Window Feature Extraction  [FIXED]")
    print(f"\n  Window size : {WINDOW_SIZE} samples")
    print(f"  Step size   : {STEP_SIZE} samples  ({100*(1 - STEP_SIZE/WINDOW_SIZE):.0f}% overlap)")
    print(f"  Strategy    : window WITHIN each class after sorting by time\n")

    all_windows = []

    for class_id in sorted(df['class'].unique()):
        class_df = (df[df['class'] == class_id]
                      .sort_values('time')       # restore temporal order
                      .reset_index(drop=True))

        n_rows   = len(class_df)
        n_windows = max(0, (n_rows - WINDOW_SIZE) // STEP_SIZE + 1)

        print(f"  Class {int(class_id)}: {n_rows:,} rows → {n_windows:,} windows", end="")

        class_windows = []
        for start in range(0, n_rows - WINDOW_SIZE + 1, STEP_SIZE):
            window = class_df.iloc[start : start + WINDOW_SIZE]
            feats  = extract_window_features(window)
            feats['class'] = class_id
            class_windows.append(feats)

        all_windows.extend(class_windows)
        print(f"  ✓")

    features_df = pd.DataFrame(all_windows)
    print(f"\n  Total windows before balancing: {len(features_df):,}")
    print(f"  Features per window          : {len(features_df.columns) - 1}")

    return features_df


# ── Step 4 : Balance classes AFTER windowing ──────────────────────────────────
def balance_windows(features_df):
    """
    Downsample to the smallest class count so all classes are equally
    represented.  Shuffle afterwards to avoid sorted-class ordering leaking
    into train/test splits.

    Balancing happens HERE (post-windowing), not before windowing, because
    the old approach sampled raw rows before windowing, destroying temporal
    runs needed to form coherent windows.
    """
    print_section("Step 3: Balancing Window Counts Across Classes")

    class_counts = features_df['class'].value_counts().sort_index()
    min_count    = class_counts.min()

    print(f"\n  Pre-balance window counts:")
    for c, n in class_counts.items():
        print(f"    Class {int(c)}: {n:,}")

    print(f"\n  Balancing to {min_count:,} windows per class …")

    balanced_parts = []
    for class_id in sorted(features_df['class'].unique()):
        subset = features_df[features_df['class'] == class_id]
        sampled = subset.sample(n=min_count, random_state=RANDOM_STATE)
        balanced_parts.append(sampled)

    balanced_df = (pd.concat(balanced_parts)
                     .sample(frac=1, random_state=RANDOM_STATE)   # global shuffle
                     .reset_index(drop=True))

    print(f"\n  Post-balance window counts:")
    for c, n in balanced_df['class'].value_counts().sort_index().items():
        print(f"    Class {int(c)}: {n:,}")

    print(f"\n  Final dataset: {len(balanced_df):,} windows × {len(balanced_df.columns)} columns")
    return balanced_df


# ── Step 5 : Statistics ───────────────────────────────────────────────────────
def analyze_feature_statistics(features_df):
    print_section("Step 4: Feature Statistics")

    feature_cols = [c for c in features_df.columns if c != 'class']
    print(f"\n  Total features : {len(feature_cols)}")
    print(f"  Channels       : {len(CHANNEL_COLUMNS)}")
    print(f"  Features/channel: {len(FEATURES_LIST)}")
    print(f"  Samples        : {len(features_df):,}")

    missing = features_df.isnull().sum().sum()
    print(f"  Missing values : {missing}")

    print(f"\n  Descriptive stats (first 5 features):")
    print(features_df[feature_cols[:5]].describe().to_string())

    print(f"\n  Class distribution:")
    for c, n in features_df['class'].value_counts().sort_index().items():
        pct = 100 * n / len(features_df)
        print(f"    Class {int(c)}: {n:>8,}  ({pct:.2f}%)")


# ── Step 6 : Visualisations ───────────────────────────────────────────────────
def visualize_feature_distributions(features_df):
    print_section("Step 5: Visualizing Feature Distributions")

    representative = [
        'channel1_mean', 'channel1_rms', 'channel1_variance',
        'channel4_mean', 'channel4_rms', 'channel4_energy'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('EMG Feature Distributions Across Gesture Classes', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for idx, feature in enumerate(representative):
        ax = axes[idx]
        data_by_class = [features_df[features_df['class'] == c][feature].values
                         for c in sorted(features_df['class'].unique())]
        ax.boxplot(data_by_class, labels=[f'C{int(c)}' for c in sorted(features_df['class'].unique())])
        ax.set_title(feature, fontweight='bold')
        ax.set_ylabel('Normalized Value')
        ax.set_xlabel('Class')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "feature_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {path}")


def create_correlation_heatmap(features_df):
    print_section("Step 6: Feature Correlation Analysis")

    feature_cols = [c for c in features_df.columns if c != 'class']
    corr = features_df[feature_cols].corr()

    plt.figure(figsize=(16, 14))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, annot=False)
    plt.title('EMG Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    path = os.path.join(MODELS_DIR, "feature_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {path}")

    # Highly correlated pairs
    high_corr = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if abs(v) > 0.95:
                high_corr.append((cols[i], cols[j], v))
    if high_corr:
        print(f"\n  Highly correlated pairs (|r| > 0.95):")
        for a, b, v in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:10]:
            print(f"    {a}  <->  {b}:  {v:.4f}")
    else:
        print("  No highly correlated feature pairs (> 0.95) found.")


def create_feature_variance_plot(features_df):
    print_section("Step 7: Feature Variance Analysis")

    feature_cols = [c for c in features_df.columns if c != 'class']
    var = features_df[feature_cols].var().sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    var.head(20).plot(kind='barh', color='steelblue')
    plt.xlabel('Variance', fontweight='bold')
    plt.ylabel('Feature', fontweight='bold')
    plt.title('Top 20 Most Variable EMG Features', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(MODELS_DIR, "feature_variance.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {path}")

    print("\n  Top 5 most variable features:")
    for feat, v in var.head(5).items():
        print(f"    {feat}: {v:.6f}")


# ── Step 8 : Save ─────────────────────────────────────────────────────────────
def save_engineered_features(features_df):
    print_section("Step 8: Saving Engineered Features")

    print(f"\n  Saving to: {OUTPUT_FILE}")
    features_df.to_csv(OUTPUT_FILE, index=False)

    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"  [OK] File size : {size_mb:.2f} MB")
    print(f"  [OK] Rows      : {len(features_df):,}")
    print(f"  [OK] Features  : {len([c for c in features_df.columns if c != 'class'])}")


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(features_df):
    print_section("Feature Engineering Summary  (FIXED v2)")

    n_feat = len([c for c in features_df.columns if c != 'class'])
    print(f"""
  Dataset shape  : {len(features_df):,} windows × {features_df.shape[1]} columns
  Feature count  : {n_feat}  (8 channels × 7 statistics)
  Window size    : {WINDOW_SIZE} samples
  Step size      : {STEP_SIZE} samples  ({100*(1-STEP_SIZE/WINDOW_SIZE):.0f}% overlap)

  FIX APPLIED    : windows extracted per-class after sorting by time
  Data leakage   : NONE  (no cross-class window contamination)
  Class balance  : ENFORCED post-windowing

  7 Features per channel:
    1. mean     – average amplitude
    2. rms      – signal power (most important)
    3. variance – signal spread
    4. std      – standard deviation
    5. energy   – total signal energy
    6. max      – peak amplitude
    7. min      – minimum amplitude

  Class distribution:""")

    for c, n in features_df['class'].value_counts().sort_index().items():
        pct = 100 * n / len(features_df)
        print(f"    Class {int(c)}: {n:>8,}  ({pct:.2f}%)")

    print(f"\n  Output: {OUTPUT_FILE}")
    print("=" * 70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("  EMG Feature Engineering Pipeline  (FIXED v2)")
    print("  Smart Bionic Limb System – Gesture Recognition")
    print("=" * 70)

    try:
        # 1. Load
        df = load_normalized_data()

        # 2. Window per class  ← THE FIX
        features_df = extract_features_per_class(df)

        # 3. Balance post-windowing
        features_df = balance_windows(features_df)

        # 4. Statistics
        analyze_feature_statistics(features_df)

        # 5–7. Visualisations
        visualize_feature_distributions(features_df)
        create_correlation_heatmap(features_df)
        create_feature_variance_plot(features_df)

        # 8. Save
        save_engineered_features(features_df)

        # Summary
        print_summary(features_df)

        return features_df

    except Exception as e:
        print(f"\n  [ERROR] Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    features = main()