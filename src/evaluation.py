"""
Model Evaluation, Survival Prognostication & Explainability (SHAP) Module.

Generates:
1. ROC curves comparison across all models.
2. Precision-Recall curves with baseline prevalence.
3. Confusion matrices with sensitivity/specificity breakdown.
4. Kaplan-Meier survival stratification with two-sided Log-rank test.
5. SHAP global feature explainability (beeswarm summary plot).
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Standardized publication aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 10


def plot_roc_curves(oof_df: pd.DataFrame) -> None:
    """
    Figure 9: Receiver Operating Characteristic (ROC) curves comparison.
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))

    y_true = oof_df["true_label"].values
    models = ["Random_Forest", "SVM_Linear", "SVM_RBF", "XGBoost"]
    colors = {"Random_Forest": "#2ca02c", "SVM_Linear": "#1f77b4", "SVM_RBF": "#9467bd", "XGBoost": "#e74c3c"}

    for name in models:
        prob_col = f"{name}_prob"
        if prob_col not in oof_df.columns:
            continue
        y_prob = oof_df[prob_col].values
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors.get(name, "black"), linewidth=2.2,
                label=f"{name.replace('_', ' ')} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, label="Random Guess (AUC = 0.500)")
    ax.set_title("Cross-Validated ROC Curves: Early Recurrence Prediction (Stage I-III)", fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    png_path = FIGURES_DIR / "fig09_roc_curves.png"
    pdf_path = FIGURES_DIR / "fig09_roc_curves.pdf"
    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: {png_path.name}")


def plot_pr_curves(oof_df: pd.DataFrame) -> None:
    """
    Figure 10: Precision-Recall (PR) curves comparison with baseline prevalence.
    """
    fig, ax = plt.subplots(figsize=(8, 6.5))

    y_true = oof_df["true_label"].values
    prevalence = np.mean(y_true)
    models = ["Random_Forest", "SVM_Linear", "SVM_RBF", "XGBoost"]
    colors = {"Random_Forest": "#2ca02c", "SVM_Linear": "#1f77b4", "SVM_RBF": "#9467bd", "XGBoost": "#e74c3c"}

    for name in models:
        prob_col = f"{name}_prob"
        if prob_col not in oof_df.columns:
            continue
        y_prob = oof_df[prob_col].values
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=colors.get(name, "black"), linewidth=2.2,
                label=f"{name.replace('_', ' ')} (PR-AUC = {pr_auc:.3f})")

    ax.axhline(prevalence, color="gray", linestyle="--", linewidth=1.2,
               label=f"Baseline Prevalence ({prevalence*100:.1f}%)")
    ax.set_title("Cross-Validated Precision-Recall Curves (Class Imbalanced Cohort)", fontweight="bold")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    png_path = FIGURES_DIR / "fig10_pr_curves.png"
    pdf_path = FIGURES_DIR / "fig10_pr_curves.pdf"
    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: {png_path.name}")


def plot_confusion_matrices(oof_df: pd.DataFrame) -> None:
    """
    Figure 11: Confusion matrices grid for all evaluated classifiers.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    axes = axes.flatten()

    y_true = oof_df["true_label"].values
    models = ["Random_Forest", "SVM_Linear", "SVM_RBF", "XGBoost"]

    for i, name in enumerate(models):
        pred_col = f"{name}_pred"
        if pred_col not in oof_df.columns:
            continue
        y_pred = oof_df[pred_col].values
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        annot_text = np.array([
            [f"True Neg (TN)\n{tn}\n({tn/(tn+fp)*100:.1f}%)", f"False Pos (FP)\n{fp}\n({fp/(tn+fp)*100:.1f}%)"],
            [f"False Neg (FN)\n{fn}\n({fn/(tp+fn)*100:.1f}%)", f"True Pos (TP)\n{tp}\n({tp/(tp+fn)*100:.1f}%)"],
        ])

        sns.heatmap(cm, annot=annot_text, fmt="", cmap="Blues", cbar=False, ax=axes[i],
                    annot_kws={"fontsize": 10, "fontweight": "bold"}, linewidths=1.5, linecolor="white")
        axes[i].set_title(f"{name.replace('_', ' ')}\nSens: {sens:.1%}, Spec: {spec:.1%}", fontweight="bold")
        axes[i].set_xlabel("Predicted Recurrence Status")
        axes[i].set_ylabel("True Recurrence Status")
        axes[i].set_xticklabels(["Disease-Free (0)", "Early Recur (1)"])
        axes[i].set_yticklabels(["Disease-Free (0)", "Early Recur (1)"])

    fig.suptitle("Repeated nested-CV confusion matrices (threshold tuned inside each training fold)", fontweight="bold", y=1.01)
    plt.tight_layout()
    png_path = FIGURES_DIR / "fig11_confusion_matrices.png"
    pdf_path = FIGURES_DIR / "fig11_confusion_matrices.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: {png_path.name}")


def plot_kaplan_meier_survival(oof_df: pd.DataFrame, model_name: str = "SVM_Linear") -> None:
    """
    Figure 12: Kaplan-Meier Survival Curves & Log-rank Test on Continuous Follow-up.
    Validates whether the model's risk stratification separates real survival trajectories.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))

    time_col = "rfs_time_months"
    pred_col = f"{model_name}_pred"

    km_data = oof_df.dropna(subset=[time_col, pred_col, "true_label"]).copy()
    times = km_data[time_col].values
    events = km_data["true_label"].values
    risk_group = km_data[pred_col].values  # 1 = Predicted High Risk, 0 = Predicted Low Risk

    # Log-rank test
    high_mask = (risk_group == 1)
    low_mask = (risk_group == 0)

    lr_result = logrank_test(times[high_mask], times[low_mask],
                             event_observed_A=events[high_mask],
                             event_observed_B=events[low_mask])

    kmf_low = KaplanMeierFitter()
    kmf_low.fit(times[low_mask], events[low_mask], label=f"Predicted Low Risk (n={low_mask.sum()})")
    kmf_low.plot_survival_function(ax=ax, color="#2ca02c", linewidth=2.5, ci_show=True)

    kmf_high = KaplanMeierFitter()
    kmf_high.fit(times[high_mask], events[high_mask], label=f"Predicted High Risk (n={high_mask.sum()})")
    kmf_high.plot_survival_function(ax=ax, color="#d62728", linewidth=2.5, ci_show=True)

    ax.set_title(f"Kaplan-Meier Recurrence-Free Survival by Model Stratification ({model_name.replace('_', ' ')})\n"
                 f"Log-Rank Test p-value = {lr_result.p_value:.2e}", fontweight="bold")
    ax.set_xlabel("Time from Primary Resection (Months)")
    ax.set_ylabel("Recurrence-Free Survival Probability")
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlim([0, max(times) + 5])
    ax.legend(frameon=True, loc="lower left", fontsize=11)

    png_path = FIGURES_DIR / "fig12_kaplan_meier_validation.png"
    pdf_path = FIGURES_DIR / "fig12_kaplan_meier_validation.pdf"
    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: {png_path.name} (Log-rank p-value: {lr_result.p_value:.3e})")


def compute_shap_explainability(top_n_genes: int = 20) -> None:
    """
    Figure 13: SHAP (Shapley Additive Explanations) Beeswarm Summary Plot.
    Explains the biological contribution of top predictive biomarkers.
    """
    rf_model_path = MODELS_DIR / "random_forest" / "Random_Forest_model.joblib"
    ml_features_path = PROCESSED_DIR / "expression" / "GSE39582_ml_features.parquet"
    ml_targets_path = PROCESSED_DIR / "clinical" / "GSE39582_ml_targets.csv"

    if not rf_model_path.exists():
        print(f"[SKIP] SHAP model not found: {rf_model_path}")
        return

    print("[SHAP] Loading trained Random Forest pipeline...")
    pipe = joblib.load(rf_model_path)
    X = pd.read_parquet(ml_features_path)
    targets_df = pd.read_csv(ml_targets_path, index_col="gsm_id")
    y = targets_df["recurrence_label"]

    selector = pipe.named_steps["selector"]
    scaler = pipe.named_steps["scaler"]
    classifier = pipe.named_steps["classifier"]

    # Transform features through the fitted selector and scaler
    selected_genes = selector.selected_genes_
    X_selected = X[selected_genes]
    X_scaled = scaler.transform(X_selected)
    X_scaled_df = pd.DataFrame(X_scaled, columns=selected_genes, index=X.index)

    print(f"[SHAP] Computing TreeExplainer SHAP values on {len(selected_genes)} biomarkers...")
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_scaled)

    # For binary classification: shap_values is list of [class 0, class 1] or array
    if isinstance(shap_values, list):
        shap_class1 = shap_values[1]
    elif shap_values.ndim == 3:
        shap_class1 = shap_values[:, :, 1]
    else:
        shap_class1 = shap_values

    # Plot beeswarm
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_class1, X_scaled_df, max_display=top_n_genes, show=False)
    plt.title(f"SHAP Biomarker Attribution for Colorectal Early Recurrence\n(Top {top_n_genes} Genes)",
              fontweight="bold", fontsize=12, pad=15)

    png_path = FIGURES_DIR / "fig13_shap_summary.png"
    pdf_path = FIGURES_DIR / "fig13_shap_summary.pdf"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE] Saved: {png_path.name}")

    # Export SHAP Feature Importance Table
    mean_abs_shap = np.abs(shap_class1).mean(axis=0)
    shap_df = pd.DataFrame({
        "Gene_Symbol": selected_genes,
        "Mean_Abs_SHAP": mean_abs_shap,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    shap_csv_path = RESULTS_DIR / "biomarkers" / "shap_biomarker_importance.csv"
    shap_df.to_csv(shap_csv_path, index=False)
    print(f"[EXPORT] SHAP importance ranking saved: {shap_csv_path}")


def plot_calibration_curves(oof_df: pd.DataFrame) -> None:
    """Figure 25: calibration (reliability) curves of the repeat-averaged out-of-fold probabilities."""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    fig, ax = plt.subplots(figsize=(7, 6.5))
    y_true = oof_df["true_label"].values
    colors = {"Random_Forest": "#2ca02c", "SVM_Linear": "#1f77b4", "SVM_RBF": "#9467bd", "XGBoost": "#e74c3c"}
    for name, color in colors.items():
        col = f"{name}_prob"
        if col not in oof_df.columns:
            continue
        frac_pos, mean_pred = calibration_curve(y_true, oof_df[col].values, n_bins=6, strategy="quantile")
        brier = brier_score_loss(y_true, oof_df[col].values)
        ax.plot(mean_pred, frac_pos, "o-", color=color, linewidth=2, label=f"{name.replace('_', ' ')} (Brier {brier:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.axhline(y_true.mean(), color="black", linestyle=":", linewidth=1, label=f"Prevalence ({y_true.mean():.1%})")
    ax.set_xlabel("Mean predicted probability (quantile bins)")
    ax.set_ylabel("Observed early-recurrence rate")
    ax.set_title("Calibration of out-of-fold probabilities (GSE39582)", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig25_calibration_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig25_calibration_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[FIGURE] Saved: fig25_calibration_curves.png")


def run_evaluation_suite() -> None:
    """Execute complete Phase 6 evaluation, survival analysis, and SHAP explainability."""
    print("=" * 70)
    print("CRC-ML PHASE 6: MODEL EVALUATION & EXPLAINABILITY (SHAP)")
    print("=" * 70)

    oof_path = PROCESSED_DIR / "clinical" / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"Out-of-fold predictions not found at {oof_path}. Run src/model_training.py first.")

    oof_df = pd.read_csv(oof_path, index_col="gsm_id")

    print("[EVAL] Generating ROC Curves Comparison (Fig 9)...")
    plot_roc_curves(oof_df)

    print("[EVAL] Generating Precision-Recall Curves (Fig 10)...")
    plot_pr_curves(oof_df)

    print("[EVAL] Generating Confusion Matrices Grid (Fig 11)...")
    plot_confusion_matrices(oof_df)
    plot_calibration_curves(oof_df)

    print("[EVAL] Performing Kaplan-Meier Survival Analysis & Log-Rank Test (Fig 12)...")
    plot_kaplan_meier_survival(oof_df, model_name="SVM_Linear")

    print("[EVAL] Computing SHAP Model Explainability (Fig 13)...")
    compute_shap_explainability(top_n_genes=20)

    print("\n[SUCCESS] Phase 6 Evaluation, Survival Analysis & SHAP Complete.")


if __name__ == "__main__":
    run_evaluation_suite()
