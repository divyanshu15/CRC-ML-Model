"""
Mega-Cohort Machine Learning Pipeline for Colorectal Cancer Recurrence.

Assembles, harmonizes, and benchmarks a unified multi-center cohort:
- GSE39582 (French CIT Consortium, n = 585)
- GSE14333 (Melbourne & MD Anderson, n = 290)
- GSE17536 (Moffitt / Vanderbilt, n = 177)
- GSE17537 (Vanderbilt Medical Center, n = 55)

Total: N = 1,107 patient profiles on the identical GPL570 Affymetrix platform (21,755 genes).
Design (revised 2026-11-14 after audit):
- Endpoint: the same curated definition as the discovery pipeline (CLAUDE.md rule 2):
  Class 1 = recurrence <= 12 months, Class 0 = disease-free >= 60 months, others excluded.
- Batch handling: each cohort is gene-wise standardised on its own unlabelled Stage I-III
  tumours. Unlike ComBat fitted on the pooled data, this never lets a held-out cohort
  influence the training data.
- Validation: leave-one-cohort-out (LOCO); hyperparameters tuned by inner 3-fold GridSearchCV
  on the training cohorts only (same grids as src/model_training.py). Each cohort is predicted by a model trained only
  on the other cohorts, so every reported number is an external-cohort estimate.
- GSE39582 non-tumoral mucosa samples are excluded.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.feature_selection import BioinformaticsFeatureSelector
from src.preprocessing import collapse_probes_to_genes, load_gpl570_annotations
from src.model_training import build_model_pipelines
from src.stats_utils import bootstrap_auc_ci, delong_test

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
# 2. Endpoint curation & per-cohort standardisation
# ----------------------------------------------------------------------
COHORTS = ["GSE39582", "GSE14333", "GSE17536", "GSE17537"]
MODEL_NAMES = ["Random_Forest", "SVM_Linear", "SVM_RBF", "XGBoost"]


def curate_endpoint(clin: pd.DataFrame) -> pd.Series:
    """Class 1: event within 12 months. Class 0: event-free with >= 60 months follow-up. Else NaN."""
    ev, t = clin["recurrence_event"], clin["dfs_time_months"]
    label = pd.Series(np.nan, index=clin.index)
    label[(ev == 1) & (t <= 12.0)] = 1.0
    label[(ev == 0) & (t >= 60.0)] = 0.0
    return label


def standardise_within_cohort(expr: pd.DataFrame, cohort: pd.Series) -> pd.DataFrame:
    """Gene-wise z-score inside each cohort (label-free; each cohort uses only its own samples)."""
    out = []
    for c in cohort.unique():
        sub = expr.loc[cohort[cohort == c].index]
        sd = sub.std(axis=0).replace(0, np.nan)
        out.append(((sub - sub.mean(axis=0)) / sd).fillna(0.0))
    return pd.concat(out).loc[expr.index].astype(np.float32)


def _build_models(imbalance_ratio: float) -> Dict[str, object]:
    return {
        "Random_Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "SVM_Linear": SVC(kernel="linear", C=0.05, class_weight="balanced", probability=True, random_state=42),
        "SVM_RBF": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.03, scale_pos_weight=imbalance_ratio,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=-1),
    }


def _pipeline(clf) -> Pipeline:
    return Pipeline([
        ("selector", BioinformaticsFeatureSelector(
            variance_percentile=0.50, n_stat_candidates=150, n_final_genes=35, random_state=42)),
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])


def _safe_auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def _pr_auc(y, s) -> float:
    p_prec, p_rec, _ = precision_recall_curve(y, s)
    return float(-np.sum(np.diff(p_rec) * np.array(p_prec)[:-1]))


# ----------------------------------------------------------------------
# 3. Leave-one-cohort-out training & evaluation
# ----------------------------------------------------------------------
def train_megacohort_models():
    from sklearn.base import clone

    expr_raw, clin_all = load_all_four_cohorts()

    # Exclude GSE39582 non-tumoral mucosa samples; repair GSE39582 age (wrong column name in cache build)
    raw39582 = pd.read_csv(PROCESSED_DIR / "clinical" / "GSE39582_clinical_raw.csv", index_col=0)
    non_tumoral = raw39582.index[raw39582["dataset"].astype(str).str.lower() == "non tumoral"]
    clin_all = clin_all.drop(index=[s for s in non_tumoral if s in clin_all.index])
    age_39582 = raw39582["age.at.diagnosis_(year)"]
    m39582 = clin_all["cohort"] == "GSE39582"
    clin_all.loc[m39582, "age"] = age_39582.reindex(clin_all.index[m39582]).values

    stage_mask = clin_all["stage"].isin([1, 2, 3])
    clin_stage_all = clin_all[stage_mask].copy()
    expr_z = standardise_within_cohort(expr_raw.loc[clin_stage_all.index], clin_stage_all["cohort"])

    clin_stage_all["recurrence_label"] = curate_endpoint(clin_stage_all)
    clin_stage = clin_stage_all.dropna(subset=["recurrence_label"]).copy()
    clin_stage["recurrence_label"] = clin_stage["recurrence_label"].astype(int)
    clin_stage["any_recurrence_event"] = clin_stage["recurrence_event"]
    # The Streamlit app reads `recurrence_event` as the outcome shown for each patient
    clin_stage["recurrence_event"] = clin_stage["recurrence_label"]

    X = expr_z.loc[clin_stage.index]
    y = clin_stage["recurrence_label"].values
    groups = clin_stage["cohort"].values
    stage = clin_stage["stage"].astype(float).values

    logger.info("=" * 70)
    logger.info("MEGA-COHORT (Stage I-III, curated <=12 mo vs >=60 mo endpoint)")
    for c in COHORTS:
        m = groups == c
        logger.info(f"  {c}: n = {m.sum():4d} | early recurrence = {int(y[m].sum())} | disease-free = {int((y[m] == 0).sum())}")
    logger.info(f"  Total: n = {len(y)} | prevalence = {y.mean():.3f}")
    logger.info("=" * 70)

    X.to_parquet(MEGA_DIR / "megacohort_stage123_features.parquet")
    clin_stage.to_csv(MEGA_DIR / "megacohort_stage123_clinical.csv")

    oof = {name: np.full(len(y), np.nan) for name in MODEL_NAMES}
    per_cohort_rows = []

    for held_out in COHORTS:
        test = groups == held_out
        train = ~test
        if test.sum() == 0:
            continue
        ratio = (y[train] == 0).sum() / max((y[train] == 1).sum(), 1)
        row = {
            "Held_Out_Cohort": held_out,
            "N": int(test.sum()),
            "Events": int(y[test].sum()),
            "Stage_Only_AUC": round(_safe_auc(y[test], stage[test]), 4),
        }
        for name, (base, grid) in build_model_pipelines(ratio).items():
            # Hyperparameters tuned only on the training cohorts (3-fold inner CV)
            search = GridSearchCV(clone(base), grid, scoring="roc_auc", n_jobs=-1,
                                  cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42))
            search.fit(X.iloc[np.where(train)[0]], y[train])
            probs = search.best_estimator_.predict_proba(X.iloc[np.where(test)[0]])[:, 1]
            oof[name][test] = probs
            row[f"{name}_AUC"] = round(_safe_auc(y[test], probs), 4)
        per_cohort_rows.append(row)
        logger.info(f"[LOCO] held out {held_out}: " + ", ".join(f"{k}={v}" for k, v in row.items() if k.endswith("AUC")))

    per_cohort_df = pd.DataFrame(per_cohort_rows).set_index("Held_Out_Cohort")
    per_cohort_df.to_csv(TABLES_DIR / "megacohort_loco_per_cohort_auc.csv")

    # Pooled metrics over all LOCO predictions (every prediction comes from a model that never saw that cohort)
    stage_lo, stage_hi = bootstrap_auc_ci(y, stage)
    results = {
        "Stage_Only_Baseline": {
            "ROC_AUC_OOF": round(float(roc_auc_score(y, stage)), 4),
            "ROC_AUC_CI95_Low": round(stage_lo, 4),
            "ROC_AUC_CI95_High": round(stage_hi, 4),
            "PR_AUC_OOF": round(float(average_precision_score(y, stage)), 4),  # AP: see model_training.py
        }
    }
    oof_predictions = {"sample_id": clin_stage.index.tolist(), "cohort": groups.tolist(), "true_label": y.tolist(),
                       "Stage_Only_score": stage.tolist()}

    for name in MODEL_NAMES:
        probs = oof[name]
        oof_predictions[f"{name}_prob"] = probs
        preds = (probs >= 0.50).astype(int)
        lo, hi = bootstrap_auc_ci(y, probs)
        _, _, p_vs_stage = delong_test(y, probs, stage)
        cohort_aucs = per_cohort_df[f"{name}_AUC"].dropna()
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        results[name] = {
            "ROC_AUC_OOF": round(float(roc_auc_score(y, probs)), 4),
            "ROC_AUC_CI95_Low": round(lo, 4),
            "ROC_AUC_CI95_High": round(hi, 4),
            "PR_AUC_OOF": round(_pr_auc(y, probs), 4),
            "DeLong_p_vs_Stage": round(p_vs_stage, 4),
            "CV_ROC_AUC_Mean": round(float(cohort_aucs.mean()), 4),
            "CV_ROC_AUC_Std": round(float(cohort_aucs.std(ddof=0)), 4),
            "Balanced_Accuracy": round(float(balanced_accuracy_score(y, preds)), 4),
            "Accuracy": round(float(accuracy_score(y, preds)), 4),
            "Sensitivity": round(float(recall_score(y, preds, pos_label=1, zero_division=0)), 4),
            "Specificity": round(float(recall_score(y, preds, pos_label=0, zero_division=0)), 4),
            "Precision": round(float(precision_score(y, preds, pos_label=1, zero_division=0)), 4),
            "F1_Score": round(float(f1_score(y, preds, zero_division=0)), 4),
            "Brier_Score": round(float(brier_score_loss(y, probs)), 4),
            "Confusion_Matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        }
        r = results[name]
        logger.info(f"[{name}] pooled LOCO ROC-AUC: {r['ROC_AUC_OOF']:.4f} (95% CI {lo:.3f}-{hi:.3f}) | "
                    f"DeLong p vs stage: {p_vs_stage:.3f} | Sens: {r['Sensitivity']:.3f} | Spec: {r['Specificity']:.3f}")

    logger.info(f"[Stage_Only_Baseline] pooled ROC-AUC: {results['Stage_Only_Baseline']['ROC_AUC_OOF']:.4f}")

    # Final models on all four cohorts (for the app); no performance is claimed from these fits
    ratio_all = (y == 0).sum() / max((y == 1).sum(), 1)
    for name, (base, grid) in build_model_pipelines(ratio_all).items():
        search = GridSearchCV(clone(base), grid, scoring="roc_auc", n_jobs=-1,
                              cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
        search.fit(X, y)
        joblib.dump(search.best_estimator_, MODELS_DIR / f"{name}_megacohort.joblib")
    logger.info(f"Serialized final pipelines to {MODELS_DIR}")

    comp_df = pd.DataFrame.from_dict(results, orient="index")
    comp_df.to_csv(TABLES_DIR / "megacohort_model_comparison.csv")
    with open(METRICS_DIR / "megacohort_cross_validation_metrics.json", "w") as f:
        json.dump({"pooled": results, "per_held_out_cohort": per_cohort_df.reset_index().to_dict(orient="records")},
                  f, indent=2)
    pd.DataFrame(oof_predictions).to_csv(MEGA_DIR / "megacohort_oof_predictions.csv", index=False)

    # ------------------------------------------------------------------
    # 4. Figures (Figs 22, 23, 24) from pooled LOCO predictions
    # ------------------------------------------------------------------
    colors = {"Random_Forest": "#10B981", "SVM_Linear": "#0284C7", "SVM_RBF": "#8B5CF6", "XGBoost": "#F59E0B"}

    plt.figure(figsize=(7, 6), dpi=300)
    for name in MODEL_NAMES:
        fpr, tpr, _ = roc_curve(y, oof[name])
        plt.plot(fpr, tpr, label=f"{name.replace('_', ' ')} (AUC = {results[name]['ROC_AUC_OOF']:.3f})",
                 color=colors[name], lw=2.0)
    fpr, tpr, _ = roc_curve(y, stage)
    plt.plot(fpr, tpr, color="black", lw=2.4, ls="-.",
             label=f"Stage only, no model (AUC = {results['Stage_Only_Baseline']['ROC_AUC_OOF']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Guess (AUC = 0.500)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    plt.title(f"Mega-Cohort Leave-One-Cohort-Out ROC (n = {len(y)} Stage I–III)", fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig22_megacohort_roc_curves.png", dpi=300)
    plt.savefig(FIGURES_DIR / "fig22_megacohort_roc_curves.pdf")
    plt.close()

    plt.figure(figsize=(7, 6), dpi=300)
    for name in MODEL_NAMES:
        precision, recall, _ = precision_recall_curve(y, oof[name])
        plt.plot(recall, precision, label=f"{name.replace('_', ' ')} (PR-AUC = {results[name]['PR_AUC_OOF']:.3f})",
                 color=colors[name], lw=2.2)
    plt.axhline(y.mean(), color="black", linestyle="--", lw=1.2, alpha=0.6,
                label=f"Prevalence Baseline ({y.mean()*100:.1f}%)")
    plt.xlabel("Recall (Sensitivity)", fontweight="bold")
    plt.ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    plt.title(f"Mega-Cohort LOCO Precision-Recall (n = {len(y)} Stage I–III)", fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig23_megacohort_pr_curves.png", dpi=300)
    plt.savefig(FIGURES_DIR / "fig23_megacohort_pr_curves.pdf")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
    for name, ax in zip(MODEL_NAMES, axes.flatten()):
        cm = confusion_matrix(y, (oof[name] >= 0.50).astype(int), labels=[0, 1])
        ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{name.replace('_', ' ')}\nBal.Acc: {results[name]['Balanced_Accuracy']:.3f} | "
                     f"Spec: {results[name]['Specificity']:.3f}", fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Disease-Free", "Recurrence"])
        ax.set_yticklabels(["Disease-Free", "Recurrence"])
        ax.set_ylabel("True Clinical Label", fontweight="bold")
        ax.set_xlabel("Predicted Risk Label", fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontweight="bold", fontsize=12)
    plt.suptitle(f"Mega-Cohort LOCO Confusion Matrices (Threshold = 0.50, n = {len(y)})", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig24_megacohort_confusion_matrices.png", dpi=300)
    plt.savefig(FIGURES_DIR / "fig24_megacohort_confusion_matrices.pdf")
    plt.close()

    logger.info("Mega-Cohort pipeline successfully executed and all artifacts saved!")


if __name__ == "__main__":
    train_megacohort_models()
