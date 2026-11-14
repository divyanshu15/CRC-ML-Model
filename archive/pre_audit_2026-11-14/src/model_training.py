"""
Machine Learning Model Training & Cross-Validation Module for CRC-ML Pipeline.

Implements leak-free training for:
1. Random Forest Classifier
2. Linear Support Vector Machine (Linear SVM)
3. Radial Basis Function Support Vector Machine (RBF SVM)
4. XGBoost Classifier

All pipelines strictly encapsulate:
- Multi-stage fold-isolated feature selection (BioinformaticsFeatureSelector)
- Standardization (StandardScaler)
- Class imbalance correction (balanced class weighting / scale_pos_weight)
- Stratified 5-Fold Cross-Validation with fixed random seeds
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.feature_selection import BioinformaticsFeatureSelector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def build_model_pipelines(imbalance_ratio: float = 3.63, random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Construct model pipelines with strict fold-isolated feature selection and scaling.
    """
    pipelines = {
        "Random_Forest": Pipeline([
            ("selector", BioinformaticsFeatureSelector(
                variance_percentile=0.50, n_stat_candidates=150, n_final_genes=35, random_state=random_state
            )),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(
                n_estimators=250,
                max_depth=4,
                min_samples_split=5,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "SVM_Linear": Pipeline([
            ("selector", BioinformaticsFeatureSelector(
                variance_percentile=0.50, n_stat_candidates=150, n_final_genes=35, random_state=random_state
            )),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="linear",
                C=0.08,
                class_weight="balanced",
                probability=True,
                random_state=random_state,
            )),
        ]),
        "SVM_RBF": Pipeline([
            ("selector", BioinformaticsFeatureSelector(
                variance_percentile=0.50, n_stat_candidates=150, n_final_genes=35, random_state=random_state
            )),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.2,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=random_state,
            )),
        ]),
        "XGBoost": Pipeline([
            ("selector", BioinformaticsFeatureSelector(
                variance_percentile=0.50, n_stat_candidates=150, n_final_genes=35, random_state=random_state
            )),
            ("scaler", StandardScaler()),
            ("classifier", XGBClassifier(
                n_estimators=120,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.7,
                scale_pos_weight=imbalance_ratio,
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1,
            )),
        ]),
    }
    return pipelines


def evaluate_cross_validation(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame, Dict[str, Pipeline]]:
    """
    Execute Stratified 5-Fold Cross-Validation across all models.
    Collects out-of-fold probability estimates and computes comprehensive metrics.
    """
    imbalance_ratio = (y == 0).sum() / (y == 1).sum()
    pipelines = build_model_pipelines(imbalance_ratio=imbalance_ratio, random_state=random_state)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    results: Dict[str, Dict[str, Any]] = {}
    oof_predictions = pd.DataFrame(index=X.index)
    oof_predictions["true_label"] = y

    trained_full_models: Dict[str, Pipeline] = {}

    for name, pipe in pipelines.items():
        print(f"\n[CV] Evaluating {name} across {n_splits} stratified folds...")

        oof_probs = np.zeros(len(y))
        fold_metrics: List[Dict[str, float]] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Fit pipeline strictly on training fold (feature selection + scaling + model)
            pipe.fit(X_train, y_train)

            # Predict probabilities on validation fold
            val_probs = pipe.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = val_probs

            # Fold-level ROC-AUC and PR-AUC
            fold_auc = roc_auc_score(y_val, val_probs)
            p_prec, p_rec, _ = precision_recall_curve(y_val, val_probs)
            fold_pr_auc = auc(p_rec, p_prec)

            fold_metrics.append({
                "fold": fold + 1,
                "roc_auc": fold_auc,
                "pr_auc": fold_pr_auc,
            })

        oof_predictions[f"{name}_prob"] = oof_probs
        oof_preds = (oof_probs >= 0.50).astype(int)
        oof_predictions[f"{name}_pred"] = oof_preds

        # Global Out-Of-Fold Metrics
        overall_auc = roc_auc_score(y, oof_probs)
        p_prec, p_rec, _ = precision_recall_curve(y, oof_probs)
        overall_pr_auc = auc(p_rec, p_prec)
        b_acc = balanced_accuracy_score(y, oof_preds)
        acc = accuracy_score(y, oof_preds)
        sens = recall_score(y, oof_preds, pos_label=1)
        cm = confusion_matrix(y, oof_preds)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = precision_score(y, oof_preds, zero_division=0)
        f1 = f1_score(y, oof_preds, zero_division=0)
        brier = brier_score_loss(y, oof_probs)

        # Cross-validation mean and std
        cv_auc_mean = np.mean([m["roc_auc"] for m in fold_metrics])
        cv_auc_std = np.std([m["roc_auc"] for m in fold_metrics])
        cv_pr_mean = np.mean([m["pr_auc"] for m in fold_metrics])
        cv_pr_std = np.std([m["pr_auc"] for m in fold_metrics])

        results[name] = {
            "ROC_AUC_OOF": round(float(overall_auc), 4),
            "PR_AUC_OOF": round(float(overall_pr_auc), 4),
            "CV_ROC_AUC_Mean": round(float(cv_auc_mean), 4),
            "CV_ROC_AUC_Std": round(float(cv_auc_std), 4),
            "CV_PR_AUC_Mean": round(float(cv_pr_mean), 4),
            "CV_PR_AUC_Std": round(float(cv_pr_std), 4),
            "Balanced_Accuracy": round(float(b_acc), 4),
            "Accuracy": round(float(acc), 4),
            "Sensitivity": round(float(sens), 4),
            "Specificity": round(float(spec), 4),
            "Precision": round(float(prec), 4),
            "F1_Score": round(float(f1), 4),
            "Brier_Score": round(float(brier), 4),
            "Confusion_Matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        }

        print(
            f"  {name:15s} | ROC-AUC: {overall_auc:.3f} (CV: {cv_auc_mean:.3f} +/- {cv_auc_std:.3f}) | "
            f"PR-AUC: {overall_pr_auc:.3f} | Sensitivity: {sens:.3f} | Specificity: {spec:.3f}"
        )

        # Fit full model on entire discovery cohort for external validation deployment
        print(f"  Fitting full {name} model on all {len(X)} discovery samples...")
        pipe.fit(X, y)
        trained_full_models[name] = pipe

        # Serialize model artifact
        model_subfolder = MODELS_DIR / name.lower()
        model_subfolder.mkdir(parents=True, exist_ok=True)
        model_file = model_subfolder / f"{name}_model.joblib"
        joblib.dump(pipe, model_file)
        print(f"  [SAVED] {model_file.name} ({model_file.stat().st_size / (1024*1024):.2f} MB)")

    return results, oof_predictions, trained_full_models


def run_training_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute complete Phase 6 model training and cross-validation."""
    print("=" * 70)
    print("CRC-ML PHASE 6: MACHINE LEARNING MODEL TRAINING & CV")
    print("=" * 70)

    # 1. Load ML features and targets
    ml_features_path = PROCESSED_DIR / "expression" / "GSE39582_ml_features.parquet"
    ml_targets_path = PROCESSED_DIR / "clinical" / "GSE39582_ml_targets.csv"

    X = pd.read_parquet(ml_features_path)
    targets_df = pd.read_csv(ml_targets_path, index_col="gsm_id")
    y = targets_df["recurrence_label"]

    print(f"[COHORT] ML Features: {X.shape[0]} samples x {X.shape[1]:,} genes")
    print(f"[COHORT] Class 0: {(y == 0).sum()} | Class 1: {(y == 1).sum()}")

    # 2. Run Stratified Cross-Validation
    results, oof_preds_df, full_models = evaluate_cross_validation(X, y, n_splits=5, random_state=42)

    # 3. Export Metrics and Predictions
    metrics_json_path = METRICS_DIR / "cross_validation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EXPORT] Metrics saved to: {metrics_json_path}")

    # Metrics Summary Table
    metrics_table = pd.DataFrame(results).T
    metrics_table_path = TABLES_DIR / "model_comparison_metrics.csv"
    metrics_table.to_csv(metrics_table_path)
    print(f"[EXPORT] Summary table saved to: {metrics_table_path}")

    # Out-of-fold predictions
    oof_path = PROCESSED_DIR / "clinical" / "oof_predictions.csv"
    # Merge survival duration for Kaplan-Meier analysis
    oof_preds_df["rfs_time_months"] = targets_df["rfs_time_months"]
    oof_preds_df.to_csv(oof_path, index_label="gsm_id")
    print(f"[EXPORT] Out-of-fold predictions saved to: {oof_path}")

    return metrics_table, oof_preds_df


if __name__ == "__main__":
    run_training_pipeline()
