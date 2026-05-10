"""
SafeVision AI - Model Training Script
========================================
Trains a Random Forest (or SVM) classifier on extracted PPE features
and serialises the model + scaler to disk.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

from src import config


# ─────────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────────

MODELS = {
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "gradient_boost": lambda: GradientBoostingClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    ),
    "svm": lambda: SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
}


def load_data(csv_path: str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load features and labels from the prepared CSV.

    Returns:
        X: array of shape (n_samples, TOTAL_FEATURES)
        y: array of shape (n_samples, 2) — [helmet, vest]
    """
    csv_path = csv_path or config.FEATURES_CSV
    if not os.path.exists(csv_path):
        print(f"[Train] ERROR: Feature file not found at {csv_path}")
        print("        Run data_prep.py first to generate features.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols].values.astype(np.float32)
    y = df[["helmet", "vest"]].values.astype(int)

    print(f"[Train] Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"        Helmet distribution: {dict(zip(*np.unique(y[:, 0], return_counts=True)))}")
    print(f"        Vest   distribution: {dict(zip(*np.unique(y[:, 1], return_counts=True)))}")
    return X, y


def train_model(X: np.ndarray, y: np.ndarray,
                model_name: str = "random_forest",
                test_size: float = 0.2) -> dict:
    """
    Train and evaluate a PPE classifier.

    We train TWO separate binary classifiers:
      - Helmet classifier  (y[:, 0])
      - Vest classifier    (y[:, 1])

    Returns a dict with the scaler, models, and metrics.
    """
    # ── Feature Scaling ─────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train / Test Split ──────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    results = {"scaler": scaler, "models": {}, "metrics": {}}

    for target_idx, target_name in enumerate(["helmet", "vest"]):
        print(f"\n{'='*55}")
        print(f"  Training {target_name.upper()} Classifier  ({model_name})")
        print(f"{'='*55}")

        model = MODELS[model_name]()
        model.fit(X_train, y_train[:, target_idx])

        # ── Predictions ─────────────────────────────
        y_pred = model.predict(X_test)
        y_true = y_test[:, target_idx]

        # ── Metrics ─────────────────────────────────
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Cross-validation score
        cv_scores = cross_val_score(
            MODELS[model_name](), X_scaled, y[:, target_idx],
            cv=5, scoring="accuracy", n_jobs=-1
        )

        print(f"\n  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  CV Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=["No " + target_name,
                                                   target_name.capitalize()]))
        print(f"  Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

        results["models"][target_name] = model
        results["metrics"][target_name] = {
            "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1,
            "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
        }

    return results


def save_model(results: dict,
               model_path: str = None,
               scaler_path: str = None) -> None:
    """Serialize the trained models and scaler to disk."""
    model_path = model_path or config.MODEL_PATH
    scaler_path = scaler_path or config.SCALER_PATH

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    payload = {
        "helmet_model": results["models"]["helmet"],
        "vest_model": results["models"]["vest"],
        "metrics": results["metrics"],
    }
    joblib.dump(payload, model_path)
    joblib.dump(results["scaler"], scaler_path)

    print(f"\n[Train] Model saved   -> {model_path}")
    print(f"[Train] Scaler saved  -> {scaler_path}")


# ─────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SafeVision AI - Train PPE Classifier")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to features CSV (default: data/features.csv)")
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        default="random_forest",
                        help="ML algorithm to use.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data for testing.")
    args = parser.parse_args()

    X, y = load_data(args.data)
    results = train_model(X, y, model_name=args.model, test_size=args.test_size)
    save_model(results)

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
