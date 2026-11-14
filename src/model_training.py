"""
Machine Learning Model Training & Nested Cross-Validation Module for CRC-ML Pipeline.

Models (CLAUDE.md scope): Random Forest, Linear SVM, RBF SVM, XGBoost; each in a gene-only
form and a "genes + stage" form, compared with TNM stage alone.

Evaluation design (revised 2026-11-14):
- Repeated nested cross-validation: 5 repeats x 5 outer stratified folds. Inside every outer
  training fold, GridSearchCV (3-fold, ROC-AUC) tunes the classifier hyperparameters and the
  number of selected genes. Feature selection, scaling and tuning never see the outer test fold.
- Decision threshold: chosen inside each outer training fold (Youden's J on inner
  cross-validated predictions), then applied to the outer test fold. Metrics at 0.50 are kept too.
- Reported AUC: mean +/- SD over repeats, plus the AUC of the repeat-averaged out-of-fold
  probability with a bootstrap 95% CI and a paired DeLong test against stage alone.
- Feature-selection stability: how often each gene is selected across the 25 outer folds.
- Final deployed model: GridSearchCV (5-fold) on the full cohort; its threshold is chosen the
  same way from 5-fold cross-validated predictions on the full cohort.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.feature_selection import STAGE_COL, BioinformaticsFeatureSelector, GenesPlusStageSelector
from src.stats_utils import bootstrap_auc_ci, delong_test

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"
BIOMARKERS_DIR = RESULTS_DIR / "biomarkers"

for d in (METRICS_DIR, TABLES_DIR, BIOMARKERS_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_REPEATS = 5
N_OUTER = 5
N_INNER = 3
RANDOM_STATE = 42


def build_model_pipelines(
    imbalance_ratio: float = 3.45, random_state: int = RANDOM_STATE, selector_cls=BioinformaticsFeatureSelector
) -> Dict[str, Tuple[Pipeline, Dict[str, List[Any]]]]:
    """Base pipelines plus their hyperparameter grids (searched inside training folds only)."""

    def pipe(clf):
        return Pipeline([
            ("selector", selector_cls(variance_percentile=0.50, n_stat_candidates=150,
                                      n_final_genes=35, random_state=random_state)),
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])

    genes = {"selector__n_final_genes": [15, 35]}
    return {
        "Random_Forest": (
            pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                        random_state=random_state, n_jobs=1)),
            {**genes, "classifier__max_depth": [3, 6], "classifier__min_samples_leaf": [3, 10]},
        ),
        "SVM_Linear": (
            pipe(SVC(kernel="linear", class_weight="balanced", probability=True, random_state=random_state)),
            {**genes, "classifier__C": [0.01, 0.1, 1.0]},
        ),
        "SVM_RBF": (
            pipe(SVC(kernel="rbf", gamma="scale", class_weight="balanced", probability=True,
                     random_state=random_state)),
            {**genes, "classifier__C": [0.3, 1.0, 3.0]},
        ),
        "XGBoost": (
            pipe(XGBClassifier(n_estimators=150, subsample=0.8, colsample_bytree=0.7,
                               scale_pos_weight=imbalance_ratio, eval_metric="logloss",
                               random_state=random_state, n_jobs=1)),
            {**genes, "classifier__max_depth": [2, 3], "classifier__learning_rate": [0.03, 0.1]},
        ),
    }


def youden_threshold(y: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, probs)
    j = tpr - fpr
    t = float(thr[int(np.argmax(j))])
    return float(min(max(t, 0.0), 1.0))


def _tune(base: Pipeline, grid: Dict[str, List[Any]], X: pd.DataFrame, y: pd.Series, n_splits: int, seed: int):
    search = GridSearchCV(
        clone(base), grid, scoring="roc_auc", n_jobs=-1, refit=True,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
    )
    search.fit(X, y)
    best = search.best_estimator_
    inner_probs = cross_val_predict(
        clone(best), X, y, method="predict_proba", n_jobs=-1,
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
    )[:, 1]
    return best, search.best_params_, youden_threshold(y.values, inner_probs)


def _threshold_metrics(y: np.ndarray, probs: np.ndarray, preds: np.ndarray, suffix: str = "") -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    return {
        f"Balanced_Accuracy{suffix}": round(float(balanced_accuracy_score(y, preds)), 4),
        f"Accuracy{suffix}": round(float(accuracy_score(y, preds)), 4),
        f"Sensitivity{suffix}": round(float(recall_score(y, preds, pos_label=1, zero_division=0)), 4),
        f"Specificity{suffix}": round(float(tn / (tn + fp) if (tn + fp) > 0 else 0.0), 4),
        f"Precision{suffix}": round(float(precision_score(y, preds, zero_division=0)), 4),
        f"F1_Score{suffix}": round(float(f1_score(y, preds, zero_division=0)), 4),
        f"Confusion_Matrix{suffix}": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def _pr_auc(y, scores) -> float:
    p_prec, p_rec, _ = precision_recall_curve(y, scores)
    return float(auc(p_rec, p_prec))


def evaluate_nested_cv(
    X: pd.DataFrame, y: pd.Series, stage: pd.Series
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    imbalance_ratio = (y == 0).sum() / (y == 1).sum()
    X_stage = X.copy()
    X_stage[STAGE_COL] = stage.loc[X.index].astype(np.float32).values

    configs: Dict[str, Tuple[Pipeline, Dict, pd.DataFrame]] = {}
    for name, (p, g) in build_model_pipelines(imbalance_ratio).items():
        configs[name] = (p, g, X)
    for name, (p, g) in build_model_pipelines(imbalance_ratio, selector_cls=GenesPlusStageSelector).items():
        configs[f"{name}_plus_Stage"] = (p, g, X_stage)

    y_arr = y.values.astype(int)
    stage_score = stage.loc[X.index].values.astype(float)
    n = len(y_arr)

    results: Dict[str, Dict[str, Any]] = {}
    oof = pd.DataFrame(index=X.index)
    oof["true_label"] = y_arr
    oof["Stage_Only_score"] = stage_score

    lo, hi = bootstrap_auc_ci(y_arr, stage_score)
    results["Stage_Only_Baseline"] = {
        "ROC_AUC_OOF": round(float(roc_auc_score(y_arr, stage_score)), 4),
        "ROC_AUC_CI95_Low": round(lo, 4),
        "ROC_AUC_CI95_High": round(hi, 4),
        "PR_AUC_OOF": round(float(average_precision_score(y_arr, stage_score)), 4),  # AP: 3-level score
        "DeLong_p_vs_Stage": None,
    }
    print(f"\n[BASELINE] Stage-only ROC-AUC: {results['Stage_Only_Baseline']['ROC_AUC_OOF']:.3f} ({lo:.3f}-{hi:.3f})")

    stability_rows = []
    for name, (base, grid, X_in) in configs.items():
        t0 = time.time()
        probs_rep = np.zeros((N_REPEATS, n))
        preds_rep = np.zeros((N_REPEATS, n), dtype=int)
        repeat_aucs, chosen = [], []
        gene_counts: Counter = Counter()

        for r in range(N_REPEATS):
            outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE + r)
            for f, (tr, te) in enumerate(outer.split(X_in, y_arr)):
                best, params, thr = _tune(base, grid, X_in.iloc[tr], y.iloc[tr], N_INNER, RANDOM_STATE + 100 * r + f)
                p = best.predict_proba(X_in.iloc[te])[:, 1]
                probs_rep[r, te] = p
                preds_rep[r, te] = (p >= thr).astype(int)
                chosen.append({**params, "threshold": round(thr, 4)})
                gene_counts.update(best.named_steps["selector"].selected_genes_)
            repeat_aucs.append(roc_auc_score(y_arr, probs_rep[r]))

        probs = probs_rep.mean(axis=0)
        # Majority vote of the per-repeat tuned-threshold calls
        preds_tuned = (preds_rep.mean(axis=0) >= 0.5).astype(int)
        lo, hi = bootstrap_auc_ci(y_arr, probs)
        _, _, p_stage = delong_test(y_arr, probs, stage_score)

        oof[f"{name}_prob"] = probs
        oof[f"{name}_pred"] = preds_tuned
        oof[f"{name}_pred_050"] = (probs >= 0.5).astype(int)

        results[name] = {
            "ROC_AUC_OOF": round(float(roc_auc_score(y_arr, probs)), 4),
            "ROC_AUC_CI95_Low": round(lo, 4),
            "ROC_AUC_CI95_High": round(hi, 4),
            "PR_AUC_OOF": round(_pr_auc(y_arr, probs), 4),
            "DeLong_p_vs_Stage": round(p_stage, 4),
            "CV_ROC_AUC_Mean": round(float(np.mean(repeat_aucs)), 4),
            "CV_ROC_AUC_Std": round(float(np.std(repeat_aucs)), 4),
            "Brier_Score": round(float(brier_score_loss(y_arr, probs)), 4),
            **_threshold_metrics(y_arr, probs, preds_tuned),
            **_threshold_metrics(y_arr, probs, (probs >= 0.5).astype(int), suffix="_at_050"),
            "Tuned_Params_Most_Common": pd.Series([json.dumps({k: v for k, v in c.items() if k != "threshold"},
                                                              sort_keys=True) for c in chosen]).mode()[0],
            "Inner_Threshold_Median": round(float(np.median([c["threshold"] for c in chosen])), 4),
        }
        n_folds = N_REPEATS * N_OUTER
        for gene, cnt in gene_counts.items():
            if gene == STAGE_COL:
                continue
            stability_rows.append({"Model": name, "Gene_Symbol": gene, "Times_Selected": cnt,
                                   "Selection_Frequency": round(cnt / n_folds, 3)})

        r_ = results[name]
        print(f"  {name:27s} | AUC {r_['ROC_AUC_OOF']:.3f} ({lo:.3f}-{hi:.3f}) | repeats {r_['CV_ROC_AUC_Mean']:.3f}"
              f"+/-{r_['CV_ROC_AUC_Std']:.3f} | DeLong p vs stage {p_stage:.3f} | tuned-thr Sens {r_['Sensitivity']:.3f}"
              f" Spec {r_['Specificity']:.3f} | {time.time() - t0:.0f}s")

    stability = pd.DataFrame(stability_rows).sort_values(["Model", "Times_Selected"], ascending=[True, False])
    return results, oof, stability


def fit_final_models(X: pd.DataFrame, y: pd.Series, stage: pd.Series) -> Dict[str, Dict[str, Any]]:
    """Tune and fit each model on the full discovery cohort; save model + threshold."""
    imbalance_ratio = (y == 0).sum() / (y == 1).sum()
    X_stage = X.copy()
    X_stage[STAGE_COL] = stage.loc[X.index].astype(np.float32).values
    final_info = {}
    for selector_cls, suffix, X_in in [(BioinformaticsFeatureSelector, "", X),
                                       (GenesPlusStageSelector, "_plus_Stage", X_stage)]:
        for name, (base, grid) in build_model_pipelines(imbalance_ratio, selector_cls=selector_cls).items():
            full = f"{name}{suffix}"
            best, params, thr = _tune(base, grid, X_in, y, 5, RANDOM_STATE)
            folder = MODELS_DIR / full.lower()
            folder.mkdir(parents=True, exist_ok=True)
            joblib.dump(best, folder / f"{full}_model.joblib")
            final_info[full] = {"best_params": params, "threshold": round(thr, 4),
                                "selected_genes": list(best.named_steps["selector"].selected_genes_)}
            print(f"  [FINAL] {full:27s} params={params} threshold={thr:.3f}")
    with open(MODELS_DIR / "final_model_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(final_info, f, indent=2)
    return final_info


def run_training_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 70)
    print("CRC-ML PHASE 6: REPEATED NESTED CROSS-VALIDATION")
    print("=" * 70)

    X = pd.read_parquet(PROCESSED_DIR / "expression" / "GSE39582_ml_features.parquet")
    targets_df = pd.read_csv(PROCESSED_DIR / "clinical" / "GSE39582_ml_targets.csv", index_col="gsm_id")
    y = targets_df["recurrence_label"].astype(int)
    stage = targets_df["stage_numeric"]
    print(f"[COHORT] {X.shape[0]} samples x {X.shape[1]:,} genes | Class 0: {(y == 0).sum()} | Class 1: {(y == 1).sum()}")
    print(f"[DESIGN] {N_REPEATS} repeats x {N_OUTER} outer folds, {N_INNER}-fold inner GridSearchCV")

    results, oof, stability = evaluate_nested_cv(X, y, stage)

    with open(METRICS_DIR / "cross_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(results).T.to_csv(TABLES_DIR / "model_comparison_metrics.csv")
    stability.to_csv(BIOMARKERS_DIR / "feature_selection_stability.csv", index=False)
    oof["rfs_time_months"] = targets_df["rfs_time_months"]
    oof.to_csv(PROCESSED_DIR / "clinical" / "oof_predictions.csv", index_label="gsm_id")
    print("[EXPORT] Metrics, OOF predictions and feature-selection stability saved.")

    print("\n[FINAL] Tuning and fitting deployment models on the full cohort...")
    fit_final_models(X, y, stage)
    return pd.DataFrame(results).T, oof


if __name__ == "__main__":
    run_training_pipeline()
