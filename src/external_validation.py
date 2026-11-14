"""
Independent External Validation Module for CRC-ML Pipeline.

Evaluates the frozen machine learning models trained on GSE39582 against
an independent external clinical microarray cohort (GSE14333, n = 290).
Strictly enforces zero feature re-selection and zero hyperparameter tuning.

Cohort alignment
----------------
GSE14333 was RMA-normalised separately from GSE39582, so its per-gene expression
distribution is shifted (median gene level ~0.5 log2 units lower). Frozen models applied
to unaligned data collapse to predicting the majority class. Each GSE14333 gene is
therefore location/scale-matched to the GSE39582 training cohort:

    x_aligned = (x - mean_ext) / sd_ext * sd_train + mean_train

using the external cohort's *unlabelled* Stage I-III samples only (no outcome information),
which is equivalent to reference-batch mean/variance adjustment. Unaligned results are kept
as a sensitivity analysis.

Survival validation
-------------------
Binary early-vs-late metrics only cover the 64 extreme-phenotype patients. The frozen risk
score is also evaluated on *all* Stage I-III GSE14333 patients with DFS follow-up using
Harrell's C-index and a stage-adjusted Cox model.
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
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.metrics import (
    average_precision_score,
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

from src.feature_selection import STAGE_COL
from src.preprocessing import collapse_probes_to_genes, load_gpl570_annotations
from src.stats_utils import bootstrap_auc_ci, delong_test

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


GENE_MODELS = ["Random_Forest", "SVM_Linear", "SVM_RBF", "XGBoost"]
ALL_MODELS = GENE_MODELS + [f"{m}_plus_Stage" for m in GENE_MODELS]
_THR_PATH = MODELS_DIR / "final_model_thresholds.json"
FINAL_THRESHOLDS = json.loads(_THR_PATH.read_text()) if _THR_PATH.exists() else {}
KM_MODEL = "SVM_Linear"  # pre-specified before alignment; not chosen on external results

FEATURES_PARQUET = EXTERNAL_DIR / "GSE14333_features.parquet"                 # binary cohort, aligned
FEATURES_UNALIGNED_PARQUET = EXTERNAL_DIR / "GSE14333_features_unaligned.parquet"
TARGETS_CSV = EXTERNAL_DIR / "GSE14333_clinical_targets.csv"
STAGE123_FEATURES_PARQUET = EXTERNAL_DIR / "GSE14333_stage123_features.parquet"  # all Stage I-III, aligned
STAGE123_CLINICAL_CSV = EXTERNAL_DIR / "GSE14333_stage123_clinical.csv"


def align_to_reference(ext: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Per-gene location/scale matching of an external cohort (samples x genes) to the training cohort."""
    mu_t, sd_t = ref.mean(axis=0), ref.std(axis=0)
    mu_e, sd_e = ext.mean(axis=0), ext.std(axis=0)
    sd_e = sd_e.where(sd_e > 1e-8)
    aligned = (ext - mu_e) / sd_e * sd_t + mu_t
    # Genes absent or constant in the external cohort carry no information: set to the training mean
    aligned = aligned.fillna(mu_t)
    return aligned.astype(np.float32)


def process_external_cohort() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build GSE14333 feature matrices in the frozen GSE39582 gene space.

    Returns
    -------
    X_bin (aligned), X_bin_unaligned, targets (binary cohort), X_stage123 (aligned), clin_stage123
    """
    gse14333_matrix = RAW_DIR / "GSE14333" / "GSE14333_series_matrix.txt.gz"
    cached = [FEATURES_PARQUET, FEATURES_UNALIGNED_PARQUET, TARGETS_CSV, STAGE123_FEATURES_PARQUET, STAGE123_CLINICAL_CSV]
    if all(p.exists() for p in cached):
        print(f"[CACHE] Loading cached external validation matrices from {EXTERNAL_DIR}...")
        return (
            pd.read_parquet(FEATURES_PARQUET),
            pd.read_parquet(FEATURES_UNALIGNED_PARQUET),
            pd.read_csv(TARGETS_CSV, index_col="gsm_id"),
            pd.read_parquet(STAGE123_FEATURES_PARQUET),
            pd.read_csv(STAGE123_CLINICAL_CSV, index_col="gsm_id"),
        )

    print("[GSE14333] Parsing clinical metadata...")
    clin_df = parse_gse14333_clinical(gse14333_matrix)

    raw_expr = parse_gse14333_expression(gse14333_matrix)
    annot_df = load_gpl570_annotations(RAW_DIR / "GPL570.annot.gz")
    genes_expr = collapse_probes_to_genes(raw_expr, annot_df)  # genes x samples

    disc_X = pd.read_parquet(DATA_DIR / "processed" / "expression" / "GSE39582_ml_features.parquet")
    required_genes = list(disc_X.columns)
    missing = [g for g in required_genes if g not in genes_expr.index]
    print(f"[ALIGNMENT] Found {len(required_genes) - len(missing):,} / {len(required_genes):,} genes in GSE14333.")

    ext_all = genes_expr.reindex(required_genes).T  # samples x genes, NaN for missing genes
    ext_all.index.name = "gsm_id"
    ext_all.columns.name = "Gene_Symbol"

    # All Stage I-III patients with DFS follow-up (unlabelled reference for alignment + survival analysis)
    clin_stage = clin_df[clin_df["stage_I_III"] & clin_df["DFS_Time"].notna()].copy()
    clin_stage = clin_stage.loc[[s for s in clin_stage.index if s in ext_all.index]]
    clin_stage["dfs_event"] = (clin_stage["DFS_Cens"].astype(str).str.strip() == "0").astype(int)

    ext_stage = ext_all.loc[clin_stage.index]
    X_stage_aligned = align_to_reference(ext_stage, disc_X)
    X_stage_unaligned = ext_stage.fillna(disc_X.median()).astype(np.float32)

    binary = clin_stage.dropna(subset=["recurrence_label"]).copy()
    binary["recurrence_label"] = binary["recurrence_label"].astype(int)
    print(f"[COHORT] Stage I-III with DFS: {len(clin_stage)} | binary validation cohort: {len(binary)}")

    X_bin = X_stage_aligned.loc[binary.index]
    X_bin_unaligned = X_stage_unaligned.loc[binary.index]

    X_bin.to_parquet(FEATURES_PARQUET, engine="pyarrow", compression="snappy")
    X_bin_unaligned.to_parquet(FEATURES_UNALIGNED_PARQUET, engine="pyarrow", compression="snappy")
    binary.to_csv(TARGETS_CSV, index_label="gsm_id")
    X_stage_aligned.to_parquet(STAGE123_FEATURES_PARQUET, engine="pyarrow", compression="snappy")
    clin_stage.to_csv(STAGE123_CLINICAL_CSV, index_label="gsm_id")
    print(f"[SAVED] External validation matrices written to {EXTERNAL_DIR}")
    return X_bin, X_bin_unaligned, binary, X_stage_aligned, clin_stage


def _model_input(name: str, X: pd.DataFrame, stage: pd.Series) -> pd.DataFrame:
    if name.endswith("_plus_Stage"):
        X = X.copy()
        X[STAGE_COL] = stage.loc[X.index].astype(np.float32).values
    return X


def _load_model(name: str):
    path = MODELS_DIR / name.lower() / f"{name}_model.joblib"
    return joblib.load(path) if path.exists() else None


def _pr_auc(y, s) -> float:
    p, r, _ = precision_recall_curve(y, s)
    return float(auc(r, p))


def evaluate_external_models(
    X_val: pd.DataFrame, X_val_unaligned: pd.DataFrame, targets_df: pd.DataFrame
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run frozen models on the GSE14333 binary cohort and compare each with the stage-only baseline."""
    y_val = targets_df["recurrence_label"].values.astype(int)
    stage = targets_df["stage_numeric"]
    stage_score = stage.values.astype(float)
    predictions_df = targets_df.copy()
    predictions_df["Stage_Only_score"] = stage_score

    lo, hi = bootstrap_auc_ci(y_val, stage_score)
    results: Dict[str, Dict[str, Any]] = {
        "Stage_Only_Baseline": {
            "External_ROC_AUC": round(float(roc_auc_score(y_val, stage_score)), 4),
            "ROC_AUC_CI95_Low": round(lo, 4),
            "ROC_AUC_CI95_High": round(hi, 4),
            "External_PR_AUC": round(float(average_precision_score(y_val, stage_score)), 4),  # AP: see model_training.py
        }
    }
    print(f"[BASELINE] Stage-only external ROC-AUC: {results['Stage_Only_Baseline']['External_ROC_AUC']:.3f} "
          f"(95% CI {lo:.3f}-{hi:.3f})")

    for name in ALL_MODELS:
        pipe = _load_model(name)
        if pipe is None:
            print(f"[SKIP] Model not found: {name}")
            continue

        probs = pipe.predict_proba(_model_input(name, X_val, stage))[:, 1]
        # Threshold frozen from the discovery cohort (Youden on 5-fold CV predictions; see model_training.py)
        threshold = float(FINAL_THRESHOLDS.get(name, {}).get("threshold", 0.50))
        preds = (probs >= threshold).astype(int)
        preds_050 = (probs >= 0.50).astype(int)
        probs_unaligned = pipe.predict_proba(_model_input(name, X_val_unaligned, stage))[:, 1]

        predictions_df[f"{name}_prob"] = probs
        predictions_df[f"{name}_pred"] = preds
        predictions_df[f"{name}_prob_unaligned"] = probs_unaligned

        val_auc = roc_auc_score(y_val, probs)
        lo, hi = bootstrap_auc_ci(y_val, probs)
        _, _, p_vs_stage = delong_test(y_val, probs, stage_score)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()

        results[name] = {
            "External_ROC_AUC": round(float(val_auc), 4),
            "ROC_AUC_CI95_Low": round(lo, 4),
            "ROC_AUC_CI95_High": round(hi, 4),
            "External_PR_AUC": round(_pr_auc(y_val, probs), 4),
            "DeLong_p_vs_Stage": round(p_vs_stage, 4),
            "External_ROC_AUC_Unaligned": round(float(roc_auc_score(y_val, probs_unaligned)), 4),
            "Sensitivity_Unaligned": round(float(recall_score(y_val, (probs_unaligned >= 0.5).astype(int), zero_division=0)), 4),
            "Frozen_Threshold": round(threshold, 4),
            "Sensitivity_at_050": round(float(recall_score(y_val, preds_050, zero_division=0)), 4),
            "Specificity_at_050": round(float(((preds_050 == 0) & (y_val == 0)).sum() / max((y_val == 0).sum(), 1)), 4),
            "Balanced_Accuracy": round(float(balanced_accuracy_score(y_val, preds)), 4),
            "Accuracy": round(float(accuracy_score(y_val, preds)), 4),
            "Sensitivity": round(float(recall_score(y_val, preds, pos_label=1, zero_division=0)), 4),
            "Specificity": round(float(tn / (tn + fp) if (tn + fp) > 0 else 0.0), 4),
            "Precision": round(float(precision_score(y_val, preds, zero_division=0)), 4),
            "F1_Score": round(float(f1_score(y_val, preds, zero_division=0)), 4),
            "Brier_Score": round(float(brier_score_loss(y_val, probs)), 4),
            "Confusion_Matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        }
        r = results[name]
        print(
            f"  {name:27s} | AUC: {val_auc:.3f} (95% CI {lo:.3f}-{hi:.3f}) | unaligned AUC: "
            f"{r['External_ROC_AUC_Unaligned']:.3f} | DeLong p vs stage: {p_vs_stage:.3f} | "
            f"Sens: {r['Sensitivity']:.3f} | Spec: {r['Specificity']:.3f}"
        )

    predictions_df.to_csv(EXTERNAL_DIR / "GSE14333_predictions.csv", index_label="gsm_id")
    with open(METRICS_DIR / "external_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    summary_csv_path = TABLES_DIR / "external_validation_summary.csv"
    pd.DataFrame(results).T.to_csv(summary_csv_path)
    print(f"\n[EXPORT] External validation summary saved: {summary_csv_path}")
    return results, predictions_df


def _survival_rows(cohort: str, X: pd.DataFrame, times: pd.Series, events: pd.Series, stage: pd.Series):
    """C-index and stage-adjusted Cox HR per SD for every frozen model on one cohort."""
    rows = [{
        "Cohort": cohort, "Model": "Stage_Only_Baseline", "N": len(times), "Events": int(events.sum()),
        "C_Index": round(concordance_index(times, -stage, events), 4),
    }]
    scores = pd.DataFrame(index=X.index)
    for name in ALL_MODELS:
        pipe = _load_model(name)
        if pipe is None:
            continue
        risk = pipe.predict_proba(_model_input(name, X, stage))[:, 1]
        scores[name] = risk
        risk_z = (risk - risk.mean()) / (risk.std() + 1e-12)
        cox_df = pd.DataFrame({"T": times.values, "E": events.values, "risk_z": risk_z, "stage": stage.values})
        cph = CoxPHFitter().fit(cox_df, duration_col="T", event_col="E")
        hr = float(np.exp(cph.params_["risk_z"]))
        ci = np.exp(cph.confidence_intervals_.loc["risk_z"].values)
        rows.append({
            "Cohort": cohort, "Model": name, "N": len(times), "Events": int(events.sum()),
            "C_Index": round(concordance_index(times, -risk, events), 4),
            "HR_per_SD_Stage_Adjusted": round(hr, 4),
            "HR_CI95_Low": round(float(ci[0]), 4),
            "HR_CI95_High": round(float(ci[1]), 4),
            "Cox_p_Stage_Adjusted": round(float(cph.summary.loc["risk_z", "p"]), 4),
        })
        r = rows[-1]
        print(f"  [{cohort}] {name:27s} | C-index: {r['C_Index']:.3f} | stage-adjusted HR/SD: {hr:.2f} "
              f"({ci[0]:.2f}-{ci[1]:.2f}), p = {r['Cox_p_Stage_Adjusted']:.3f}")
    return rows, scores


def load_gse17536_stage123() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    GSE17536 (Moffitt, GPL570) Stage I-III patients with DFS, from the mega-cohort cache,
    aligned to the GSE39582 training cohort with its own unlabelled samples. A second,
    fully independent survival validation cohort (never used for training or tuning).
    """
    mega = DATA_DIR / "processed" / "megacohort"
    expr = pd.read_parquet(mega / "megacohort_raw_expression.parquet")
    clin = pd.read_csv(mega / "megacohort_clinical.csv", index_col=0)
    c = clin[(clin["cohort"] == "GSE17536") & clin["stage"].isin([1, 2, 3])]
    c = c.dropna(subset=["recurrence_event", "dfs_time_months"])
    c = c[c["dfs_time_months"] > 0]
    disc_X = pd.read_parquet(DATA_DIR / "processed" / "expression" / "GSE39582_ml_features.parquet")
    X = align_to_reference(expr.loc[c.index].reindex(columns=disc_X.columns), disc_X)
    return X, c


def evaluate_external_survival(X_stage: pd.DataFrame, clin_stage: pd.DataFrame) -> pd.DataFrame:
    """
    Score every Stage I-III patient of GSE14333 and GSE17536 with each frozen model and test
    whether the risk score predicts disease-free survival, alone (C-index) and adjusted for
    stage (Cox HR per SD).
    """
    rows, scores = _survival_rows(
        "GSE14333", X_stage, clin_stage["DFS_Time"].astype(float), clin_stage["dfs_event"].astype(int),
        clin_stage["stage_numeric"].astype(float))
    scores.join(clin_stage[["DFS_Time", "dfs_event", "stage_numeric"]]).to_csv(
        EXTERNAL_DIR / "GSE14333_stage123_risk_scores.csv", index_label="gsm_id")

    X17, c17 = load_gse17536_stage123()
    rows17, scores17 = _survival_rows(
        "GSE17536", X17, c17["dfs_time_months"].astype(float), c17["recurrence_event"].astype(int),
        c17["stage"].astype(float))
    scores17.join(c17[["dfs_time_months", "recurrence_event", "stage"]]).to_csv(
        EXTERNAL_DIR / "GSE17536_stage123_risk_scores.csv", index_label="gsm_id")

    surv_df = pd.DataFrame(rows + rows17)
    surv_df.to_csv(TABLES_DIR / "external_survival_validation.csv", index=False)
    print(f"[EXPORT] Survival validation saved: {TABLES_DIR / 'external_survival_validation.csv'}")
    return scores


def plot_external_validation_figures(predictions_df: pd.DataFrame, stage123_scores: pd.DataFrame,
                                     clin_stage: pd.DataFrame) -> None:
    """Fig 14: external ROC curves (with stage baseline). Fig 15: KM on all Stage I-III patients."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    y_true = predictions_df["recurrence_label"].values
    colors = {"Random_Forest": "#2ca02c", "SVM_Linear": "#1f77b4", "SVM_RBF": "#9467bd", "XGBoost": "#e74c3c"}

    for name in GENE_MODELS:
        col = f"{name}_prob"
        if col not in predictions_df.columns:
            continue
        fpr, tpr, _ = roc_curve(y_true, predictions_df[col].values)
        ax.plot(fpr, tpr, color=colors[name], linewidth=2.0,
                label=f"{name.replace('_', ' ')} (AUC = {auc(fpr, tpr):.3f})")
    fpr, tpr, _ = roc_curve(y_true, predictions_df["Stage_Only_score"].values)
    ax.plot(fpr, tpr, color="black", linewidth=2.4, linestyle="-.",
            label=f"Stage only, no model (AUC = {auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, label="Random Guess (AUC = 0.500)")
    ax.set_title(f"Independent External Validation ROC Curves (GSE14333, n = {len(y_true)}, "
                 f"{int(y_true.sum())} early recurrences)", fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig14_external_validation_roc.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig14_external_validation_roc.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[FIGURE] Saved: fig14_external_validation_roc.png")

    # Kaplan-Meier on ALL Stage I-III patients (not just the extreme-phenotype binary subset)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    times = clin_stage.loc[stage123_scores.index, "DFS_Time"].astype(float).values
    events = clin_stage.loc[stage123_scores.index, "dfs_event"].astype(int).values
    risk = stage123_scores[KM_MODEL].values
    high = risk >= np.median(risk)
    p_val = logrank_test(times[high], times[~high], event_observed_A=events[high],
                         event_observed_B=events[~high]).p_value

    for mask, label, color in [(~high, "Predicted Low Risk", "#2ca02c"), (high, "Predicted High Risk", "#d62728")]:
        KaplanMeierFitter().fit(times[mask], events[mask], label=f"{label} (n={mask.sum()})").plot_survival_function(
            ax=ax, color=color, linewidth=2.5, ci_show=True)

    ax.set_title(f"External Kaplan-Meier, all Stage I-III GSE14333 patients (n = {len(times)}, "
                 f"{events.sum()} events)\n{KM_MODEL.replace('_', ' ')} risk score, median split | "
                 f"Log-rank p = {p_val:.3g}", fontweight="bold")
    ax.set_xlabel("Disease-Free Survival Time (Months)")
    ax.set_ylabel("Disease-Free Survival Probability")
    ax.set_ylim([-0.02, 1.05])
    ax.legend(frameon=True, loc="lower left", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig15_external_validation_km.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig15_external_validation_km.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: fig15_external_validation_km.png (Log-rank p-value: {p_val:.3g})")


def run_external_validation() -> None:
    """Execute complete Phase 7 external validation pipeline."""
    print("=" * 70)
    print("CRC-ML PHASE 7: INDEPENDENT EXTERNAL VALIDATION (GSE14333)")
    print("=" * 70)

    X_bin, X_bin_unaligned, targets_df, X_stage, clin_stage = process_external_cohort()
    _, preds_df = evaluate_external_models(X_bin, X_bin_unaligned, targets_df)
    print("\n[SURVIVAL] Frozen risk scores on all Stage I-III patients (GSE14333 and GSE17536):")
    scores = evaluate_external_survival(X_stage, clin_stage)
    plot_external_validation_figures(preds_df, scores, clin_stage)
    print("\n[SUCCESS] Phase 7 External Validation Complete.")


if __name__ == "__main__":
    run_external_validation()
