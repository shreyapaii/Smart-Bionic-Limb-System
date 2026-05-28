"""
EMG-Based Gesture Recognition - Model Training Pipeline

Trains and optimizes ML models using 56 engineered EMG features.
Compares 5 models: Random Forest, XGBoost, Gradient Boosting, ExtraTrees, LightGBM.
Performs hyperparameter tuning and generates performance reports.

Goal: Achieve accuracy > 85% (improve from 70% baseline)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
import time
import json
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

DATA_DIR = "data"
MODELS_DIR = "models"
INPUT_FILE = os.path.join(DATA_DIR, "engineered_emg_features.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_gesture_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

Path(MODELS_DIR).mkdir(exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def load_engineered_features():
    """Load and validate engineered EMG features dataset."""
    print_section("Step 1: Loading Engineered EMG Features Dataset")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    print(f"\n  Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    print(f"  Dataset shape: {df.shape[0]:,} samples x {df.shape[1]} columns")

    if 'class' not in df.columns:
        raise ValueError("Target 'class' column not found")

    feature_cols = [col for col in df.columns if col != 'class']
    if len(feature_cols) != 56:
        raise ValueError(f"Expected 56 features, got {len(feature_cols)}")

    missing = df.isnull().sum().sum()
    print(f"  Missing values: {missing}")

    X = df[feature_cols].values
    y = df['class'].values

    print(f"\n  Class Distribution:")
    unique_classes, class_counts = np.unique(y, return_counts=True)
    for class_id, count in zip(unique_classes, class_counts):
        pct = (count / len(y)) * 100
        print(f"    Class {int(class_id)}: {count:>5,} ({pct:>6.2f}%)")

    print(f"\n  [OK] Dataset loaded successfully")

    return X, y, feature_cols


def stratified_train_test_split(X, y):
    """Split data using stratified approach."""
    print_section("Step 2: Stratified Train-Test Split")

    print(f"\n  Splitting: 80% training, 20% testing (stratified)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\n  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")

    print(f"\n  [OK] Train-test split completed")

    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest with hyperparameter tuning."""
    print("\n  [Random Forest] Training...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [15, 20, 25],
        'min_samples_split': [2, 5],
    }

    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    gs = GridSearchCV(rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'y_pred': y_pred,
    }

    print(f"    Accuracy: {metrics['accuracy']:.4f}")

    return best_model, gs.best_params_, metrics


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost."""
    print("\n  [XGBoost] Training...")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
    }

    xgb_model = xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    gs = GridSearchCV(xgb_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'y_pred': y_pred,
    }

    print(f"    Accuracy: {metrics['accuracy']:.4f}")

    return best_model, gs.best_params_, metrics


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """Train Gradient Boosting."""
    print("\n  [Gradient Boosting] Training...")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
    }

    gb = GradientBoostingClassifier(random_state=RANDOM_STATE, verbose=0)
    gs = GridSearchCV(gb, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'y_pred': y_pred,
    }

    print(f"    Accuracy: {metrics['accuracy']:.4f}")

    return best_model, gs.best_params_, metrics


def train_extra_trees(X_train, X_test, y_train, y_test):
    """Train ExtraTrees."""
    print("\n  [ExtraTrees] Training...")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [15, 20, 25],
        'min_samples_split': [2, 5],
    }

    et = ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    gs = GridSearchCV(et, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'y_pred': y_pred,
    }

    print(f"    Accuracy: {metrics['accuracy']:.4f}")

    return best_model, gs.best_params_, metrics


def train_lightgbm(X_train, X_test, y_train, y_test):
    """Train LightGBM."""
    print("\n  [LightGBM] Training...")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
    }

    lgb_model = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    gs = GridSearchCV(lgb_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'y_pred': y_pred,
    }

    print(f"    Accuracy: {metrics['accuracy']:.4f}")

    return best_model, gs.best_params_, metrics


def train_all_models(X_train, X_test, y_train, y_test):
    """Train all 5 models."""
    print_section("Step 3: Model Training with Hyperparameter Tuning")

    results = {}

    print("\n  Training 5 models (5-10 minutes)...")

    rf_model, rf_params, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    results['Random Forest'] = {'model': rf_model, 'params': rf_params, 'metrics': rf_metrics}

    xgb_model, xgb_params, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)
    results['XGBoost'] = {'model': xgb_model, 'params': xgb_params, 'metrics': xgb_metrics}

    gb_model, gb_params, gb_metrics = train_gradient_boosting(X_train, X_test, y_train, y_test)
    results['Gradient Boosting'] = {'model': gb_model, 'params': gb_params, 'metrics': gb_metrics}

    et_model, et_params, et_metrics = train_extra_trees(X_train, X_test, y_train, y_test)
    results['ExtraTrees'] = {'model': et_model, 'params': et_params, 'metrics': et_metrics}

    lgb_model, lgb_params, lgb_metrics = train_lightgbm(X_train, X_test, y_train, y_test)
    results['LightGBM'] = {'model': lgb_model, 'params': lgb_params, 'metrics': lgb_metrics}

    return results


def compare_models(results):
    """Compare model performance."""
    print_section("Step 4: Model Comparison")

    comp_data = []
    for name, result in results.items():
        m = result['metrics']
        comp_data.append({'Model': name, 'Accuracy': m['accuracy'], 'Precision': m['precision'], 'Recall': m['recall'], 'F1': m['f1']})

    comp_df = pd.DataFrame(comp_data).sort_values('F1', ascending=False)

    print("\n  Performance Comparison:")
    print("  " + "-" * 75)
    print(f"  {'Model':<20} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("  " + "-" * 75)

    for idx, row in comp_df.iterrows():
        print(f"  {row['Model']:<20} {row['Accuracy']:>12.4f} {row['Precision']:>12.4f} {row['Recall']:>12.4f} {row['F1']:>12.4f}")

    print("  " + "-" * 75)

    best_name = comp_df.iloc[0]['Model']
    best_results = results[best_name]

    print(f"\n  BEST MODEL: {best_name}")
    print(f"  Accuracy: {best_results['metrics']['accuracy']:.4f} (vs 70% baseline)")

    return best_name, best_results, comp_df


def analyze_features(best_model, feature_names):
    """Analyze feature importance."""
    print_section("Step 5: Feature Importance Analysis")

    if not hasattr(best_model, 'feature_importances_'):
        print("  [Warning] Model does not support feature importance")
        return

    imp = best_model.feature_importances_
    indices = np.argsort(imp)[::-1]

    print(f"\n  Top 20 Most Important Features:")
    print(f"  " + "-" * 70)

    total_imp = np.sum(imp)
    for i in range(min(20, len(indices))):
        idx = indices[i]
        feat = feature_names[idx]
        importance = imp[idx]
        contrib = (importance / total_imp) * 100
        print(f"  {i+1:2d}. {feat:<30} {importance:>10.6f} ({contrib:>6.2f}%)")

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 8))
    top_idx = indices[:20]
    top_feat = [feature_names[i] for i in top_idx]
    top_imp = imp[top_idx]

    colors = plt.cm.viridis(np.linspace(0, 1, len(top_feat)))
    ax.barh(top_feat, top_imp, color=colors)
    ax.set_xlabel('Importance', fontweight='bold')
    ax.set_title('Top 20 Most Important EMG Features', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "feature_importance_best_model.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\n  [OK] Saved: {path}")
    plt.close()


def create_confusion_matrix(y_test, y_pred):
    """Create confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('True', fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "confusion_matrix_best_model.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\n  Confusion matrix saved: {path}")
    plt.close()


def create_comparison_plot(comp_df):
    """Create model comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(comp_df))
    width = 0.2
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, metric in enumerate(metrics):
        offset = width * (i - 1.5)
        ax.bar(x + offset, comp_df[metric], width, label=metric, color=colors[i])

    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df['Model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()
    path = os.path.join(MODELS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Model comparison saved: {path}")
    plt.close()


def save_best_model(best_model, best_name, best_results, feature_names, y_test):
    """Save best model and metadata."""
    print_section("Step 6: Saving Best Model")

    joblib.dump(best_model, BEST_MODEL_PATH)
    size = os.path.getsize(BEST_MODEL_PATH) / (1024 * 1024)
    print(f"\n  [OK] Model saved: {BEST_MODEL_PATH} ({size:.2f} MB)")

    metadata = {
        'model_name': best_name,
        'accuracy': float(best_results['metrics']['accuracy']),
        'precision': float(best_results['metrics']['precision']),
        'recall': float(best_results['metrics']['recall']),
        'f1_score': float(best_results['metrics']['f1']),
        'best_hyperparameters': {k: str(v) for k, v in best_results['params'].items()},
        'number_of_classes': len(np.unique(y_test)),
        'number_of_features': len(feature_names),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  [OK] Metadata saved: {METADATA_PATH}")
    print(f"\n  Best Model: {best_name}")
    print(f"  Accuracy: {best_results['metrics']['accuracy']:.4f}")
    print(f"  Improvement: {(best_results['metrics']['accuracy'] - 0.70) * 100:.2f}% over baseline")


def generate_report(best_name, best_results, y_test, y_pred):
    """Generate classification report."""
    print_section("Step 7: Classification Report")

    print(f"\n  Best Model: {best_name}")
    print(f"  Test Samples: {len(y_test)}")

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"\n  Per-Class Metrics:")
    print(f"  " + "-" * 75)
    print(f"  {'Class':<8} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print(f"  " + "-" * 75)

    for class_id in sorted(np.unique(y_test)):
        cid_str = str(int(class_id))
        if cid_str in report:
            r = report[cid_str]
            print(f"  {cid_str:<8} {r['precision']:>12.4f} {r['recall']:>12.4f} {r['f1-score']:>12.4f} {int(r['support']):>12,}")

    print(f"  " + "-" * 75)
    print(f"\n  Overall Accuracy: {best_results['metrics']['accuracy']:.4f}")


def main():
    """Execute complete training pipeline."""
    print("\n" + "=" * 80)
    print("  EMG-Based Gesture Recognition - Model Training Pipeline")
    print("  Smart Bionic Limb System")
    print("=" * 80)

    try:
        start = time.time()

        X, y, feature_names = load_engineered_features()
        X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)
        results = train_all_models(X_train, X_test, y_train, y_test)
        best_name, best_results, comp_df = compare_models(results)

        print_section("Step 4.5: Creating Visualizations")
        create_comparison_plot(comp_df)
        create_confusion_matrix(y_test, best_results['metrics']['y_pred'])

        analyze_features(best_results['model'], feature_names)
        save_best_model(best_results['model'], best_name, best_results, feature_names, y_test)
        generate_report(best_name, best_results, y_test, best_results['metrics']['y_pred'])

        print_section("Training Pipeline Complete")
        elapsed = time.time() - start
        print(f"\n  Total Time: {elapsed:.2f} seconds")
        print(f"\n  [OK] Best model ready for deployment!")
        print(f"  [OK] Saved to: {BEST_MODEL_PATH}")
        print("\n" + "=" * 80 + "\n")

        return best_results

    except Exception as e:
        print(f"\n  [ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
