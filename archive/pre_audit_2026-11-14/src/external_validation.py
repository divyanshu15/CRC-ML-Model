"""
Independent External Validation Module for CRC-ML Pipeline.

Evaluates the frozen machine learning models trained on GSE39582 against
an independent external clinical microarray cohort (GSE14333, n = 290).
Strictly enforces zero feature re-selection and zero hyperparameter tuning.
"""

import gzip
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.preprocessing import collapse_probes_to_genes, load_gpl570_annotations

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external_validation"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"

EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Standardized publication aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 10


def parse_gse14333_clinical(matrix_gz_path: Path) -> pd.DataFrame:
    """
    Parse sample metadata from GSE14333 series matrix and extract clinical covariates.
    """
    sample_ids = []
    char_lines = []

    with gzip.open(matrix_gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [p.strip('"') for p in line.strip().split("\t")[1:]]
            if line.startswith("!Sample_characteristics_ch1"):
                char_lines = [p.strip('"') for p in line.strip().split("\t")[1:]]
            if line.startswith("!series_matrix_table_begin"):
                break

    records = []
    for sid, char in zip(sample_ids, char_lines):
        d = {"gsm_id": sid}
        for item in char.split(";"):
            if ":" in item:
                k, v = item.split(":", 1)
                d[k.strip()] = v.strip()
        records.append(d)

    df = pd.DataFrame(records).set_index("gsm_id")
    df["DFS_Time"] = pd.to_numeric(df["DFS_Time"], errors="coerce")

    # Filter Stage I-III (Dukes A, B, C; Dukes D is de novo metastatic)
    df["stage_I_III"] = df["DukesStage"].isin(["A", "B", "C"])
    df["stage_numeric"] = df["DukesStage"].map({"A": 1, "B": 2, "C": 3, "D": 4})

    # Recurrence label: DFS_Cens == '0' means event (relapse); '1' means censored (disease-free)
    df["recurrence_label"] = np.nan
    for idx, row in df.iterrows():
        if not row["stage_I_III"]:
            continue
        cens = str(row["DFS_Cens"]).strip()
        time_mo = row["DFS_Time"]

        if cens == "0":
            if time_mo <= 12.0:
                df.loc[idx, "recurrence_label"] = 1.0  # Class 1: Early Recurrence
        elif cens == "1":
            if time_mo >= 60.0:
                df.loc[idx, "recurrence_label"] = 0.0  # Class 0: Long-Term Disease-Free

    return df


def parse_gse14333_expression(matrix_gz_path: Path) -> pd.DataFrame:
    """
    Parse the numeric expression matrix from GSE14333 series matrix (GPL570).
    """
    skiprows = 0
    with gzip.open(matrix_gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("!series_matrix_table_begin"):
                skiprows = i + 1
                break

    print(f"[GSE14333] Parsing expression matrix (starting line {skiprows})...")
    expr_df = pd.read_csv(
        matrix_gz_path,
        sep="\t",
        skiprows=skiprows,
        compression="gzip",
        index_col=0,
        comment="!",
    )
    expr_df.index.name = "ID_REF"
    expr_df.columns = [str(c).strip('"') for c in expr_df.columns]
    return expr_df.astype(np.float32)


def process_external_cohort() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process GSE14333 using the frozen GPL570 annotation and align with training feature space.
    """
    gse14333_matrix = RAW_DIR / "GSE14333" / "GSE14333_series_matrix.txt.gz"
    annot_file = RAW_DIR / "GPL570.annot.gz"
    features_parquet = EXTERNAL_DIR / "GSE14333_features.parquet"
    targets_csv = EXTERNAL_DIR / "GSE14333_clinical_targets.csv"

    # 1. Clinical Curation
    print("[GSE14333] Parsing clinical metadata...")
    clin_df = parse_gse14333_clinical(gse14333_matrix)

    # 2. Expression Parsing & Probe-to-Gene Mapping
    if features_parquet.exists() and targets_csv.exists():
        print(f"[CACHE] Loading cached external validation features from {features_parquet}...")
        X_val = pd.read_parquet(features_parquet)
        targets_df = pd.read_csv(targets_csv, index_col="gsm_id")
        return X_val, targets_df

    raw_expr = parse_gse14333_expression(gse14333_matrix)
    annot_df = load_gpl570_annotations(annot_file)
    genes_expr = collapse_probes_to_genes(raw_expr, annot_df)

    # 3. Load discovery feature columns to enforce exact alignment
    disc_features_path = DATA_DIR / "processed" / "expression" / "GSE39582_ml_features.parquet"
    disc_X = pd.read_parquet(disc_features_path)
    required_genes = list(disc_X.columns)

    # Align genes: fill any missing gene with column median or zero
    common_genes = [g for g in required_genes if g in genes_expr.index]
    missing_genes = [g for g in required_genes if g not in genes_expr.index]
    print(f"[ALIGNMENT] Found {len(common_genes):,} / {len(required_genes):,} genes in GSE14333.")

    genes_sub = genes_expr.loc[common_genes]
    if missing_genes:
        print(f"[IMPUTE] Zero-filling {len(missing_genes)} absent features to preserve model geometry.")
        for mg in missing_genes:
            genes_sub.loc[mg] = 0.0

    genes_sub = genes_sub.loc[required_genes]

    # Filter binary validation cohort (Class 0 and Class 1)
    binary_cohort = clin_df.dropna(subset=["recurrence_label"]).copy()
    binary_cohort["recurrence_label"] = binary_cohort["recurrence_label"].astype(int)

    valid_samples = [s for s in binary_cohort.index if s in genes_sub.columns]
    print(f"[COHORT] Independent binary validation cohort: {len(valid_samples)} samples.")

    X_val = genes_sub[valid_samples].T
    X_val.index.name = "gsm_id"
    X_val.columns.name = "Gene_Symbol"

    targets_df = binary_cohort.loc[valid_samples]

    # Export
    X_val.to_parquet(features_parquet, engine="pyarrow", compression="snappy")
    targets_df.to_csv(targets_csv, index_label="gsm_id")
    print(f"[SAVED] Saved external validation data to {EXTERNAL_DIR}")

    return X_val, targets_df


def evaluate_external_models(X_val: pd.DataFrame, targets_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run frozen trained models on independent cohort GSE14333 and compute performance.
    """
    y_val = targets_df["recurrence_label"].values
    models = ["Random_Forest", "SVM_Linear", "SVM_RBF", "XGBoost"]

    results: Dict[str, Dict[str, Any]] = {}
    predictions_df = targets_df.copy()

    for name in models:
        model_path = MODELS_DIR / name.lower() / f"{name}_model.joblib"
        if not model_path.exists():
            print(f"[SKIP] Model not found: {model_path}")
            continue

        print(f"[EVAL] Evaluating frozen {name} on GSE14333 (n={len(X_val)})...")
        pipe = joblib.load(model_path)

        probs = pipe.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.50).astype(int)

        predictions_df[f"{name}_prob"] = probs
        predictions_df[f"{name}_pred"] = preds

        val_auc = roc_auc_score(y_val, probs)
        p_prec, p_rec, _ = precision_recall_curve(y_val, probs)
        val_pr_auc = auc(p_rec, p_prec)
        b_acc = balanced_accuracy_score(y_val, preds)
        acc = accuracy_score(y_val, preds)
        sens = recall_score(y_val, preds, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_val, preds)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = precision_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        brier = brier_score_loss(y_val, probs)

        results[name] = {
            "External_ROC_AUC": round(float(val_auc), 4),
            "External_PR_AUC": round(float(val_pr_auc), 4),
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
            f"  {name:15s} | External ROC-AUC: {val_auc:.3f} | PR-AUC: {val_pr_auc:.3f} | "
            f"Sens: {sens:.3f} | Spec: {spec:.3f} | Acc: {acc:.3f}"
        )

    # Export predictions & metrics
    predictions_path = EXTERNAL_DIR / "GSE14333_predictions.csv"
    predictions_df.to_csv(predictions_path, index_label="gsm_id")

    metrics_json_path = METRICS_DIR / "external_validation_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_df = pd.DataFrame(results).T
    summary_csv_path = TABLES_DIR / "external_validation_summary.csv"
    summary_df.to_csv(summary_csv_path)

    print(f"\n[EXPORT] External validation summary saved: {summary_csv_path}")
    return results, predictions_df


def plot_external_validation_figures(predictions_df: pd.DataFrame) -> None:
    """
    Generate ROC curves and Kaplan-Meier curves for external validation.
    """
    # 1. ROC Curves on GSE14333
    fig, ax = plt.subplots(figsize=(8, 6.5))
    y_true = predictions_df["recurrence_label"].values
    models = ["Random_Forest", "SVM_Linear", "SVM_RBF", "XGBoost"]
    colors = {"Random_Forest": "#2ca02c", "SVM_Linear": "#1f77b4", "SVM_RBF": "#9467bd", "XGBoost": "#e74c3c"}

    for name in models:
        prob_col = f"{name}_prob"
        if prob_col not in predictions_df.columns:
            continue
        probs = predictions_df[prob_col].values
        fpr, tpr, _ = roc_curve(y_true, probs)
        val_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[name], linewidth=2.2, label=f"{name.replace('_', ' ')} (AUC = {val_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, label="Random Guess (AUC = 0.500)")
    ax.set_title("Independent External Validation ROC Curves (GSE14333, n = 64)", fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    png_path = FIGURES_DIR / "fig14_external_validation_roc.png"
    pdf_path = FIGURES_DIR / "fig14_external_validation_roc.pdf"
    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: {png_path.name}")

    # 2. Kaplan-Meier Survival Curves on External Cohort
    fig, ax = plt.subplots(figsize=(9, 6.5))
    times = pd.to_numeric(predictions_df["DFS_Time"], errors="coerce").values
    events = (predictions_df["DFS_Cens"].astype(str) == "0").astype(int).values  # 1 = Event, 0 = Censored

    # Use SVM_Linear risk probabilities with median split for robust clinical risk stratification
    risk_probs = predictions_df["SVM_Linear_prob"].values
    median_risk = np.median(risk_probs)
    high_mask = (risk_probs >= median_risk)
    low_mask = (risk_probs < median_risk)

    if high_mask.sum() > 0 and low_mask.sum() > 0:
        lr_res = logrank_test(times[high_mask], times[low_mask],
                              event_observed_A=events[high_mask],
                              event_observed_B=events[low_mask])
        p_val = lr_res.p_value
    else:
        p_val = 1.0

    kmf_low = KaplanMeierFitter()
    kmf_low.fit(times[low_mask], events[low_mask], label=f"Predicted Low Risk (n={low_mask.sum()})")
    kmf_low.plot_survival_function(ax=ax, color="#2ca02c", linewidth=2.5, ci_show=True)

    kmf_high = KaplanMeierFitter()
    kmf_high.fit(times[high_mask], events[high_mask], label=f"Predicted High Risk (n={high_mask.sum()})")
    kmf_high.plot_survival_function(ax=ax, color="#d62728", linewidth=2.5, ci_show=True)

    ax.set_title(f"External Validation Kaplan-Meier Survival (GSE14333, SVM Linear Risk Score)\n"
                 f"Log-Rank Test p-value = {p_val:.3e}",
                 fontweight="bold")
    ax.set_xlabel("Disease-Free Survival Time (Months)")
    ax.set_ylabel("Disease-Free Survival Probability")
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlim([0, max(times) + 5])
    ax.legend(frameon=True, loc="lower left", fontsize=11)

    png_path_km = FIGURES_DIR / "fig15_external_validation_km.png"
    pdf_path_km = FIGURES_DIR / "fig15_external_validation_km.pdf"
    plt.tight_layout()
    fig.savefig(png_path_km, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path_km, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: {png_path_km.name} (Log-rank p-value: {p_val:.3f})")


def run_external_validation() -> None:
    """Execute complete Phase 7 external validation pipeline."""
    print("=" * 70)
    print("CRC-ML PHASE 7: INDEPENDENT EXTERNAL VALIDATION (GSE14333)")
    print("=" * 70)

    X_val, targets_df = process_external_cohort()
    results, preds_df = evaluate_external_models(X_val, targets_df)
    plot_external_validation_figures(preds_df)

    print("\n[SUCCESS] Phase 7 External Validation Complete.")


if __name__ == "__main__":
    run_external_validation()
