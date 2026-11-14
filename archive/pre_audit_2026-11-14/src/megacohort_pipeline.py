"""
Mega-Cohort Machine Learning Pipeline for Colorectal Cancer Recurrence.

Assembles, harmonizes, and benchmarks a unified multi-center cohort:
- GSE39582 (French CIT Consortium, n = 585)
- GSE14333 (Melbourne & MD Anderson, n = 290)
- GSE17536 (Moffitt / Vanderbilt, n = 177)
- GSE17537 (Vanderbilt Medical Center, n = 55)

Total: N = 1,107 patient profiles on the identical GPL570 Affymetrix platform (21,755 genes).
Applies ComBat empirical Bayes batch harmonization, curates Stage I–III recurrence endpoints,
and trains 4 ML classifiers under strict fold-isolated Stratified 5-Fold Cross-Validation.
"""

import gzip
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from combat.pycombat import pycombat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.feature_selection import BioinformaticsFeatureSelector
from src.preprocessing import collapse_probes_to_genes, load_gpl570_annotations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MEGA_DIR = PROCESSED_DIR / "megacohort"
MODELS_DIR = PROJECT_ROOT / "models" / "megacohort"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
METRICS_DIR = RESULTS_DIR / "metrics"

MEGA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Publication plotting aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 10


# ----------------------------------------------------------------------
# 1. Dataset Parsers
# ----------------------------------------------------------------------
def parse_raw_series_matrix(gz_path: Path) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Parse raw expression matrix and clinical characteristics from a GEO series matrix."""
    sample_ids = []
    char_lines = []
    skip = 0

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [p.strip('"\n') for p in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                parts = [p.strip('"\n') for p in line.strip().split("\t")[1:]]
                char_lines.append(parts)
            elif line.startswith("!series_matrix_table_begin"):
                skip = i + 1
                break

    expr_df = pd.read_csv(gz_path, sep="\t", skiprows=skip, index_col=0, comment="!")
    # Filter to only the sample columns in case extra header columns appear
    expr_df = expr_df[[c for c in expr_df.columns if c in sample_ids]]
    return expr_df, sample_ids, char_lines


def load_all_four_cohorts() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, collapse probes to genes, and assemble all 1,107 samples across the 4 cohorts.
    Returns (expression_df [samples x genes], clinical_df).
    """
    mega_expr_path = MEGA_DIR / "megacohort_raw_expression.parquet"
    mega_clin_path = MEGA_DIR / "megacohort_clinical.csv"

    if mega_expr_path.exists() and mega_clin_path.exists():
        logger.info("[CACHE] Loading cached Mega-Cohort raw expression and clinical tables...")
        expr_all = pd.read_parquet(mega_expr_path)
        clin_all = pd.read_csv(mega_clin_path, index_col=0)
        return expr_all, clin_all

    annot_df = load_gpl570_annotations()

    # --- Cohort 1: GSE39582 (n = 585) ---
    logger.info("[1/4] Processing GSE39582 (French CIT Consortium, n=585)...")
    gse39582_expr_path = PROCESSED_DIR / "expression" / "GSE39582_genes_expression.parquet"
    if gse39582_expr_path.exists():
        expr_39582 = pd.read_parquet(gse39582_expr_path).T  # Transpose to (samples x genes)
    else:
        raw_df, _, _ = parse_raw_series_matrix(RAW_DIR / "GSE39582" / "GSE39582_series_matrix.txt.gz")
        expr_39582 = collapse_probes_to_genes(raw_df, annot_df).T

    clin_39582_raw = pd.read_csv(PROCESSED_DIR / "clinical" / "GSE39582_clinical_raw.csv", index_col=0)
    clin_records = []
    for gsm in expr_39582.index:
        row = clin_39582_raw.loc[gsm] if gsm in clin_39582_raw.index else {}
        stage_raw = row.get("tnm.stage", np.nan)
        try:
            stage_clean = int(float(stage_raw)) if pd.notna(stage_raw) else np.nan
        except (ValueError, TypeError):
            stage_clean = np.nan

        relapse_val = row.get("rfs.event", np.nan)
        relapse_clean = int(float(relapse_val)) if pd.notna(relapse_val) else np.nan
        time_val = row.get("rfs.delay", np.nan)
        time_clean = float(time_val) if pd.notna(time_val) else np.nan

        clin_records.append({
            "sample_id": gsm,
            "cohort": "GSE39582",
            "stage": stage_clean,
            "recurrence_event": relapse_clean,
            "dfs_time_months": time_clean,
            "age": row.get("age.years", np.nan),
            "gender": row.get("sex", np.nan),
        })

    # --- Cohort 2: GSE14333 (n = 290) ---
    logger.info("[2/4] Processing GSE14333 (Melbourne & MD Anderson, n=290)...")
    raw_14333, sids_14333, chars_14333 = parse_raw_series_matrix(RAW_DIR / "GSE14333" / "GSE14333_series_matrix.txt.gz")
    expr_14333 = collapse_probes_to_genes(raw_14333, annot_df).T

    # Parse semicolon-separated clinical line
    first_char_line = chars_14333[0] if chars_14333 else []
    for sid, rec in zip(sids_14333, first_char_line):
        d = {}
        for item in rec.split(";"):
            if ":" in item:
                k, v = item.split(":", 1)
                d[k.strip()] = v.strip()
        dukes = d.get("DukesStage", "")
        stage_map = {"A": 1, "B": 2, "C": 3, "D": 4}
        stage_clean = stage_map.get(dukes, np.nan)

        # In GSE14333: DFS_Cens == '0' means event occurred (recurrence); '1' means censored (disease-free)
        dfs_cens = d.get("DFS_Cens", "NA")
        if dfs_cens == "0":
            recurrence_val = 1
        elif dfs_cens == "1":
            recurrence_val = 0
        else:
            recurrence_val = np.nan

        dfs_t = d.get("DFS_Time", "NA")
        time_clean = float(dfs_t) if dfs_t != "NA" and str(dfs_t).replace(".", "").isdigit() else np.nan

        clin_records.append({
            "sample_id": sid,
            "cohort": "GSE14333",
            "stage": stage_clean,
            "recurrence_event": recurrence_val,
            "dfs_time_months": time_clean,
            "age": d.get("Age_Diag", np.nan),
            "gender": d.get("Gender", np.nan),
        })

    # --- Cohort 3: GSE17536 (n = 177) ---
    logger.info("[3/4] Processing GSE17536 (Moffitt / Vanderbilt, n=177)...")
    raw_17536, sids_17536, chars_17536 = parse_raw_series_matrix(RAW_DIR / "GSE17536" / "GSE17536_series_matrix.txt.gz")
    expr_17536 = collapse_probes_to_genes(raw_17536, annot_df).T

    char_dict_17536 = {}
    for line_parts in chars_17536:
        header = line_parts[0].split(":")[0].strip() if ":" in line_parts[0] else line_parts[0][:15]
        vals = [p.split(":", 1)[-1].strip() if ":" in p else p for p in line_parts]
        char_dict_17536[header] = vals

    for idx, sid in enumerate(sids_17536):
        st_val = char_dict_17536.get("ajcc_stage", [""])[idx]
        try:
            stage_clean = int(st_val) if st_val in ["1", "2", "3", "4"] else np.nan
        except (ValueError, TypeError):
            stage_clean = np.nan

        ev_val = char_dict_17536.get("dfs_event (disease free survival; cancer recurrence)", [""])[idx]
        if ev_val == "recurrence":
            recurrence_val = 1
        elif ev_val == "no recurrence":
            recurrence_val = 0
        else:
            recurrence_val = np.nan

        t_val = char_dict_17536.get("dfs_time", [""])[idx]
        try:
            time_clean = float(t_val)
        except (ValueError, TypeError):
            time_clean = np.nan

        clin_records.append({
            "sample_id": sid,
            "cohort": "GSE17536",
            "stage": stage_clean,
            "recurrence_event": recurrence_val,
            "dfs_time_months": time_clean,
            "age": char_dict_17536.get("age", [np.nan])[idx],
            "gender": char_dict_17536.get("gender", [np.nan])[idx],
        })

    # --- Cohort 4: GSE17537 (n = 55) ---
    logger.info("[4/4] Processing GSE17537 (Vanderbilt Medical Center, n=55)...")
    raw_17537, sids_17537, chars_17537 = parse_raw_series_matrix(RAW_DIR / "GSE17537" / "GSE17537_series_matrix.txt.gz")
    expr_17537 = collapse_probes_to_genes(raw_17537, annot_df).T

    char_dict_17537 = {}
    for line_parts in chars_17537:
        header = line_parts[0].split(":")[0].strip() if ":" in line_parts[0] else line_parts[0][:15]
        vals = [p.split(":", 1)[-1].strip() if ":" in p else p for p in line_parts]
        char_dict_17537[header] = vals

    for idx, sid in enumerate(sids_17537):
        st_val = char_dict_17537.get("ajcc_stage", [""])[idx]
        try:
            stage_clean = int(st_val) if st_val in ["1", "2", "3", "4"] else np.nan
        except (ValueError, TypeError):
            stage_clean = np.nan

        ev_val = char_dict_17537.get("dfs_event (disease free survival; cancer recurrence)", [""])[idx]
        if ev_val == "recurrence":
            recurrence_val = 1
        elif ev_val == "no recurrence":
            recurrence_val = 0
        else:
            recurrence_val = np.nan

        t_val = char_dict_17537.get("dfs_time", [""])[idx]
        try:
            time_clean = float(t_val)
        except (ValueError, TypeError):
            time_clean = np.nan

        clin_records.append({
            "sample_id": sid,
            "cohort": "GSE17537",
            "stage": stage_clean,
            "recurrence_event": recurrence_val,
            "dfs_time_months": time_clean,
            "age": char_dict_17537.get("age", [np.nan])[idx],
            "gender": char_dict_17537.get("gender", [np.nan])[idx],
        })

    # Combine expression matrices
    logger.info("Aligning feature columns across cohorts...")
    shared_genes = expr_39582.columns.intersection(expr_14333.columns).intersection(expr_17536.columns).intersection(expr_17537.columns)
    logger.info(f"Shared genes across all 4 cohorts: {len(shared_genes)}")

    expr_all = pd.concat([
        expr_39582[shared_genes],
        expr_14333[shared_genes],
        expr_17536[shared_genes],
        expr_17537[shared_genes],
    ], axis=0).astype(np.float32)

    clin_all = pd.DataFrame(clin_records).set_index("sample_id")
    clin_all = clin_all.loc[expr_all.index]

    logger.info(f"[MERGED] Total Mega-Cohort samples: {expr_all.shape[0]} samples x {expr_all.shape[1]} genes")
    logger.info(f"Cohort sample counts:\n{clin_all['cohort'].value_counts()}")

    # Save cache
    expr_all.to_parquet(mega_expr_path)
    clin_all.to_csv(mega_clin_path)
    return expr_all, clin_all


# ----------------------------------------------------------------------
# 2. Batch Harmonization (ComBat)
# ----------------------------------------------------------------------
def harmonize_batches(expr_df: pd.DataFrame, clin_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ComBat empirical Bayes batch correction across the four cohorts.
    Input expr_df: (samples x genes).
    """
    harmonized_path = MEGA_DIR / "megacohort_combat_expression.parquet"
    if harmonized_path.exists():
        logger.info("[CACHE] Loading ComBat-harmonized Mega-Cohort expression matrix...")
        return pd.read_parquet(harmonized_path)

    logger.info(f"[COMBAT] Running Empirical Bayes batch correction on {expr_df.shape[0]} samples across 4 cohorts...")
    t0 = time.time()
    # pycombat expects genes as rows, samples as columns
    data_t = expr_df.T
    batch_labels = clin_df.loc[expr_df.index, "cohort"].tolist()

    combat_data_t = pycombat(data_t, batch_labels)
    combat_df = combat_data_t.T.astype(np.float32)
    logger.info(f"[COMBAT] Harmonization completed in {time.time()-t0:.2f} seconds.")

    combat_df.to_parquet(harmonized_path)
    return combat_df


# ----------------------------------------------------------------------
# 3. Model Training & Stratified 5-Fold Cross-Validation
# ----------------------------------------------------------------------
def train_megacohort_models():
    """Execute end-to-end training and cross-validation on the Stage I-III harmonized mega-cohort."""
    expr_raw, clin_all = load_all_four_cohorts()
    expr_combat = harmonize_batches(expr_raw, clin_all)

    # Filter to Stage I-III patients with valid recurrence outcome
    stage123_mask = clin_all["stage"].isin([1, 2, 3]) & clin_all["recurrence_event"].notna()
    clin_stage = clin_all[stage123_mask].copy()
    X = expr_combat.loc[clin_stage.index]
    y = clin_stage["recurrence_event"].astype(int).values

    n_recur = (y == 1).sum()
    n_free = (y == 0).sum()
    imbalance_ratio = n_free / max(n_recur, 1)

    logger.info("=" * 70)
    logger.info("MEGA-COHORT TRAINING DATASET (Stage I-III Resected CRC)")
    logger.info(f"Total Eligible Patients: {len(X)}")
    logger.info(f"Recurrence Events (Class 1): {n_recur} ({n_recur/len(X)*100:.1f}%)")
    logger.info(f"Disease-Free (Class 0): {n_free} ({n_free/len(X)*100:.1f}%)")
    logger.info(f"Class Imbalance Ratio: {imbalance_ratio:.2f} : 1")
    logger.info(f"Cohort Breakdown:\n{clin_stage['cohort'].value_counts()}")
    logger.info("=" * 70)

    # Save target modeling cohort
    X.to_parquet(MEGA_DIR / "megacohort_stage123_features.parquet")
    clin_stage.to_csv(MEGA_DIR / "megacohort_stage123_clinical.csv")

    # Define architectures
    models = {
        "Random_Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "SVM_Linear": SVC(
            kernel="linear",
            C=0.05,
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "SVM_RBF": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.03,
            scale_pos_weight=imbalance_ratio,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    oof_predictions = {"sample_id": clin_stage.index.tolist(), "true_label": y.tolist()}

    for model_name, clf in models.items():
        logger.info(f"\nEvaluating {model_name} under Stratified 5-Fold CV...")
        oof_probs = np.zeros(len(y), dtype=np.float32)
        cv_roc_aucs = []
        cv_pr_aucs = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            # Fold-isolated feature selection pipeline
            pipeline = Pipeline([
                ("selector", BioinformaticsFeatureSelector(
                    variance_percentile=0.50,
                    n_stat_candidates=150,
                    n_final_genes=35,
                    random_state=42,
                )),
                ("scaler", StandardScaler()),
                ("classifier", clf),
            ])

            pipeline.fit(X_train, y_train)
            val_probs = pipeline.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = val_probs

            fold_roc = roc_auc_score(y_val, val_probs)
            p_prec, p_rec, _ = precision_recall_curve(y_val, val_probs)
            fold_pr = -np.sum(np.diff(p_rec) * np.array(p_prec)[:-1])
            cv_roc_aucs.append(fold_roc)
            cv_pr_aucs.append(fold_pr)

        oof_predictions[f"{model_name}_prob"] = oof_probs
        oof_preds = (oof_probs >= 0.50).astype(int)

        roc_auc_oof = roc_auc_score(y, oof_probs)
        p_prec, p_rec, _ = precision_recall_curve(y, oof_probs)
        pr_auc_oof = -np.sum(np.diff(p_rec) * np.array(p_prec)[:-1])
        bal_acc = balanced_accuracy_score(y, oof_preds)
        acc = accuracy_score(y, oof_preds)
        sens = recall_score(y, oof_preds, pos_label=1, zero_division=0)
        spec = recall_score(y, oof_preds, pos_label=0, zero_division=0)
        prec = precision_score(y, oof_preds, pos_label=1, zero_division=0)
        f1 = f1_score(y, oof_preds, zero_division=0)
        brier = brier_score_loss(y, oof_probs)
        cm = confusion_matrix(y, oof_preds)
        tn, fp, fn, tp = cm.ravel()

        results[model_name] = {
            "ROC_AUC_OOF": round(float(roc_auc_oof), 4),
            "PR_AUC_OOF": round(float(pr_auc_oof), 4),
            "CV_ROC_AUC_Mean": round(float(np.mean(cv_roc_aucs)), 4),
            "CV_ROC_AUC_Std": round(float(np.std(cv_roc_aucs)), 4),
            "CV_PR_AUC_Mean": round(float(np.mean(cv_pr_aucs)), 4),
            "CV_PR_AUC_Std": round(float(np.std(cv_pr_aucs)), 4),
            "Balanced_Accuracy": round(float(bal_acc), 4),
            "Accuracy": round(float(acc), 4),
            "Sensitivity": round(float(sens), 4),
            "Specificity": round(float(spec), 4),
            "Precision": round(float(prec), 4),
            "F1_Score": round(float(f1), 4),
            "Brier_Score": round(float(brier), 4),
            "Confusion_Matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        }

        logger.info(
            f"[{model_name}] OOF ROC-AUC: {roc_auc_oof:.4f} | PR-AUC: {pr_auc_oof:.4f} "
            f"| Bal.Acc: {bal_acc:.4f} | Specificity: {spec:.4f} | Sensitivity: {sens:.4f}"
        )

        # Train and serialize final model pipeline on full Stage I-III mega-cohort
        final_pipeline = Pipeline([
            ("selector", BioinformaticsFeatureSelector(
                variance_percentile=0.50,
                n_stat_candidates=150,
                n_final_genes=35,
                random_state=42,
            )),
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])
        final_pipeline.fit(X, y)
        model_save_path = MODELS_DIR / f"{model_name}_megacohort.joblib"
        joblib.dump(final_pipeline, model_save_path)
        logger.info(f"Serialized final pipeline to {model_save_path}")

    # Save summary table & JSON
    comp_df = pd.DataFrame.from_dict(results, orient="index")
    comp_df.to_csv(TABLES_DIR / "megacohort_model_comparison.csv")
    with open(METRICS_DIR / "megacohort_cross_validation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(oof_predictions).to_csv(MEGA_DIR / "megacohort_oof_predictions.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Generate Publication Figures (Figs 22, 23, 24)
    # ------------------------------------------------------------------
    logger.info("Generating publication-quality figures for Mega-Cohort (Figs 22-24)...")
    colors = {"Random_Forest": "#10B981", "SVM_Linear": "#0284C7", "SVM_RBF": "#8B5CF6", "XGBoost": "#F59E0B"}

    # Fig 22: ROC Curves
    plt.figure(figsize=(7, 6), dpi=300)
    for model_name in models.keys():
        probs = oof_predictions[f"{model_name}_prob"]
        fpr, tpr, _ = roc_curve(y, probs)
        auc_val = results[model_name]["ROC_AUC_OOF"]
        clean_label = model_name.replace("_", " ")
        plt.plot(fpr, tpr, label=f"{clean_label} (AUC = {auc_val:.3f})", color=colors[model_name], lw=2.2)

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Guess (AUC = 0.500)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    plt.title(f"Mega-Cohort Stratified 5-Fold CV ROC Curves (N = {len(X)} Stage I–III)", fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig22_megacohort_roc_curves.png", dpi=300)
    plt.savefig(FIGURES_DIR / "fig22_megacohort_roc_curves.pdf")
    plt.close()

    # Fig 23: Precision-Recall Curves
    baseline_prev = n_recur / len(y)
    plt.figure(figsize=(7, 6), dpi=300)
    for model_name in models.keys():
        probs = oof_predictions[f"{model_name}_prob"]
        precision, recall, _ = precision_recall_curve(y, probs)
        pr_auc = results[model_name]["PR_AUC_OOF"]
        clean_label = model_name.replace("_", " ")
        plt.plot(recall, precision, label=f"{clean_label} (PR-AUC = {pr_auc:.3f})", color=colors[model_name], lw=2.2)

    plt.axhline(baseline_prev, color="black", linestyle="--", lw=1.2, alpha=0.6, label=f"Prevalence Baseline ({baseline_prev*100:.1f}%)")
    plt.xlabel("Recall (Sensitivity)", fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    plt.title(f"Mega-Cohort Precision-Recall Curves (N = {len(X)} Stage I–III)", fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig23_megacohort_pr_curves.png", dpi=300)
    plt.savefig(FIGURES_DIR / "fig23_megacohort_pr_curves.pdf")
    plt.close()

    # Fig 24: Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    axes = axes.flatten()
    for idx, (model_name, ax) in enumerate(zip(models.keys(), axes)):
        probs = oof_predictions[f"{model_name}_prob"]
        preds = (probs >= 0.50).astype(int)
        cm = confusion_matrix(y, preds)
        clean_label = model_name.replace("_", " ")

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{clean_label}\nBal.Acc: {results[model_name]['Balanced_Accuracy']:.3f} | Spec: {results[model_name]['Specificity']:.3f}", fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Disease-Free", "Recurrence"])
        ax.set_yticklabels(["Disease-Free", "Recurrence"])
        ax.set_ylabel("True Clinical Label", fontweight="bold")
        ax.set_xlabel("Predicted Risk Label", fontweight="bold")

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(j, i, f"{val:,}", ha="center", va="center", color="white" if val > cm.max() / 2 else "black", fontweight="bold", fontsize=12)

    plt.suptitle(f"Mega-Cohort Out-of-Fold Confusion Matrices (Threshold = 0.50, N = {len(X)})", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig24_megacohort_confusion_matrices.png", dpi=300)
    plt.savefig(FIGURES_DIR / "fig24_megacohort_confusion_matrices.pdf")
    plt.close()

    logger.info("Mega-Cohort pipeline successfully executed and all artifacts saved!")


if __name__ == "__main__":
    train_megacohort_models()
