"""
Phase 2: Machine Learning Model Training & Cross-Validation on TCGA-COADREAD.
Trains 4 model architectures on large-scale RNA-seq cohort (n=306 Stage I-III patients)
under strict Stratified 5-Fold Cross-Validation with fold-isolated feature selection.
"""

from pathlib import Path
import json
import sys
from typing import Any, Dict, List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_selection import BioinformaticsFeatureSelector

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "tcga"
MODELS_DIR = PROJECT_ROOT / "models" / "phase2_tcga"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def build_tcga_pipelines(imbalance_ratio: float, random_state: int = 42) -> Dict[str, Pipeline]:
    """Construct fold-isolated classification pipelines for TCGA RNA-seq features."""
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
                C=0.05,
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
                C=1.0,
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


def train_and_evaluate_tcga() -> None:
    """Execute complete Phase 2 cross-validation on TCGA-COADREAD."""
    print("=" * 70)
    print("PHASE 2: TCGA-COADREAD MODEL TRAINING (n=306)")
    print("=" * 70)

    # 1. Load data
    features_path = DATA_DIR / "TCGA_COADREAD_features.parquet"
    clinical_path = DATA_DIR / "TCGA_COADREAD_clinical.csv"

    X = pd.read_parquet(features_path)
    clin = pd.read_csv(clinical_path, index_col=0)
    y = clin.loc[X.index, "recurrence_label"]

    imbalance_ratio = (y == 0).sum() / (y == 1).sum()
    print(f"[DATA] Samples: {len(X)}, Class 0 (Free): {(y==0).sum()}, Class 1 (Recurrence): {(y==1).sum()}")
    print(f"[DATA] Imbalance ratio: {imbalance_ratio:.2f}")

    # 2. Setup 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = build_tcga_pipelines(imbalance_ratio=imbalance_ratio, random_state=42)

    results: Dict[str, Dict[str, Any]] = {}
    oof_predictions = pd.DataFrame(index=X.index)
    oof_predictions["true_label"] = y

    for name, pipe in pipelines.items():
        print(f"\n[CV] Evaluating {name} across 5 stratified folds on TCGA...")
        oof_probs = np.zeros(len(y))
        fold_metrics: List[Dict[str, float]] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipe.fit(X_train, y_train)
            probs_val = pipe.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = probs_val

            fold_auc = roc_auc_score(y_val, probs_val)
            p_prec, p_rec, _ = precision_recall_curve(y_val, probs_val)
            fold_pr_auc = auc(p_rec, p_prec)
            fold_metrics.append({"roc_auc": fold_auc, "pr_auc": fold_pr_auc})

        oof_predictions[f"{name}_prob"] = oof_probs
        oof_preds = (oof_probs >= 0.50).astype(int)
        oof_predictions[f"{name}_pred"] = oof_preds

        # Metrics
        roc_auc = roc_auc_score(y, oof_probs)
        p_prec, p_rec, _ = precision_recall_curve(y, oof_probs)
        pr_auc = auc(p_rec, p_prec)
        acc = accuracy_score(y, oof_preds)
        b_acc = balanced_accuracy_score(y, oof_preds)
        sens = recall_score(y, oof_preds, zero_division=0)
        spec = recall_score(y == 0, oof_preds == 0, zero_division=0)
        prec = precision_score(y, oof_preds, zero_division=0)
        f1 = f1_score(y, oof_preds, zero_division=0)
        brier = brier_score_loss(y, oof_probs)
        tn, fp, fn, tp = confusion_matrix(y, oof_preds).ravel()

        results[name] = {
            "ROC_AUC_OOF": float(np.round(roc_auc, 4)),
            "PR_AUC_OOF": float(np.round(pr_auc, 4)),
            "CV_ROC_AUC_Mean": float(np.round(np.mean([m["roc_auc"] for m in fold_metrics]), 4)),
            "CV_ROC_AUC_Std": float(np.round(np.std([m["roc_auc"] for m in fold_metrics]), 4)),
            "CV_PR_AUC_Mean": float(np.round(np.mean([m["pr_auc"] for m in fold_metrics]), 4)),
            "CV_PR_AUC_Std": float(np.round(np.std([m["pr_auc"] for m in fold_metrics]), 4)),
            "Balanced_Accuracy": float(np.round(b_acc, 4)),
            "Accuracy": float(np.round(acc, 4)),
            "Sensitivity": float(np.round(sens, 4)),
            "Specificity": float(np.round(spec, 4)),
            "Precision": float(np.round(prec, 4)),
            "F1_Score": float(np.round(f1, 4)),
            "Brier_Score": float(np.round(brier, 4)),
            "Confusion_Matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        }

        print(f"  OOF ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Bal.Acc: {b_acc:.4f} | Spec: {spec:.4f}")

        # Retrain full model and serialize
        print(f"  Fitting full {name} model on all n={len(X)} TCGA samples...")
        pipe.fit(X, y)
        model_out = MODELS_DIR / f"{name}_tcga.joblib"
        joblib.dump(pipe, model_out)
        print(f"  [SAVED] {model_out.name} ({model_out.stat().st_size / 1024:.1f} KB)")

    # 3. Export Summary Tables & Metrics
    comp_df = pd.DataFrame(results).T
    table_csv = TABLES_DIR / "phase2_tcga_model_comparison.csv"
    comp_df.to_csv(table_csv)
    print(f"\n[TABLE] Saved comparison table to: {table_csv}")

    metrics_json = METRICS_DIR / "tcga_cross_validation_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(results, f, indent=2)

    oof_path = DATA_DIR / "tcga_oof_predictions.csv"
    oof_predictions.to_csv(oof_path)
    print(f"[PREDS] Saved TCGA out-of-fold predictions to: {oof_path}")

    # 4. Generate Diagnostic Figures
    plot_tcga_figures(oof_predictions, y, results)


def plot_tcga_figures(oof_df: pd.DataFrame, y: pd.Series, results: Dict[str, Any]) -> None:
    """Generate publication-quality ROC, PR, and Confusion Matrix figures for TCGA cohort."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = {"Random_Forest": "#1f77b4", "SVM_Linear": "#2ca02c", "SVM_RBF": "#9467bd", "XGBoost": "#d62728"}

    # 1. ROC Curves
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    for name in results.keys():
        fpr, tpr, _ = roc_curve(y, oof_df[f"{name}_prob"])
        auc_val = results[name]["ROC_AUC_OOF"]
        ax.plot(fpr, tpr, label=f"{name.replace('_', ' ')} (AUC = {auc_val:.3f})", color=colors.get(name, "black"), lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance (AUC = 0.500)", alpha=0.7)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 19: Phase 2 TCGA-COADREAD Out-of-Fold ROC Curves (n=306)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig19_tcga_roc_curves.png")
    fig.savefig(FIGURES_DIR / "fig19_tcga_roc_curves.pdf")
    plt.close(fig)
    print("[FIGURE] Saved fig19_tcga_roc_curves.png")

    # 2. PR Curves
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    baseline = (y == 1).mean()
    for name in results.keys():
        p_prec, p_rec, _ = precision_recall_curve(y, oof_df[f"{name}_prob"])
        pr_auc = results[name]["PR_AUC_OOF"]
        ax.plot(p_rec, p_prec, label=f"{name.replace('_', ' ')} (PR-AUC = {pr_auc:.3f})", color=colors.get(name, "black"), lw=2)

    ax.axhline(baseline, color="gray", linestyle="--", lw=1.5, label=f"Prevalence Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 20: Phase 2 TCGA-COADREAD Out-of-Fold Precision-Recall Curves", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig20_tcga_pr_curves.png")
    fig.savefig(FIGURES_DIR / "fig20_tcga_pr_curves.pdf")
    plt.close(fig)
    print("[FIGURE] Saved fig20_tcga_pr_curves.png")

    # 3. Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), dpi=300)
    axes = axes.flatten()
    for idx, name in enumerate(results.keys()):
        cm = confusion_matrix(y, oof_df[f"{name}_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=["Free (0)", "Recurred (1)"], yticklabels=["Free (0)", "Recurred (1)"])
        axes[idx].set_title(f"{name.replace('_', ' ')} (Bal.Acc: {results[name]['Balanced_Accuracy']:.3f})", fontweight="bold")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

    plt.suptitle("Figure 21: Phase 2 TCGA-COADREAD Confusion Matrices (Threshold = 0.50)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig21_tcga_confusion_matrices.png")
    fig.savefig(FIGURES_DIR / "fig21_tcga_confusion_matrices.pdf")
    plt.close(fig)
    print("[FIGURE] Saved fig21_tcga_confusion_matrices.png")


if __name__ == "__main__":
    train_and_evaluate_tcga()

