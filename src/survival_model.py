"""
Survival Model for Recurrence-Free Survival in Stage I-III CRC (XGBoost Cox).

Why: binary early-vs-late classification uses only the 249 extreme-outcome GSE39582 patients
and did not beat stage, but the frozen classifier's risk score was prognostic for DFS after
adjusting for stage in GSE14333. A survival model can use every Stage I-III tumour with
follow-up (censored patients included) instead of discarding the intermediate ones.

Model (within CLAUDE.md scope): XGBoost with objective "survival:cox", in a gene-only form
and a "genes + stage" form.

Leakage control:
- Gene screening (variance filter, then |correlation| with null-model martingale residuals)
  and scaling are fitted inside training folds only.
- Repeated nested CV: 5 repeats x 5 outer folds (stratified by event); inner 3-fold grid
  search on Harrell's C-index.
- External cohorts (GSE14333, GSE17536) are gene-wise location/scale-matched to the
  discovery cohort using their own unlabelled Stage I-III samples, then scored by the frozen
  final model. The high/low risk cut-off is the discovery-cohort median, frozen.

Outputs: results/tables/survival_*.csv, results/metrics/survival_metrics.json,
results/figures/fig26, models/survival_xgb_cox/.
"""

import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.external_validation import align_to_reference
from src.feature_selection import STAGE_COL

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
BIOMARKERS_DIR = RESULTS_DIR / "biomarkers"
MODEL_DIR = PROJECT_ROOT / "models" / "survival_xgb_cox"
for d in (TABLES_DIR, METRICS_DIR, FIGURES_DIR, BIOMARKERS_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_REPEATS, N_OUTER, N_INNER = 5, 5, 3
GRID = {
    "screen__n_genes": [50, 150],
    "model__max_depth": [2, 3],
    "model__learning_rate": [0.02, 0.05],
}


# ----------------------------------------------------------------------
# Survival target encoding (XGBoost Cox: +time = event, -time = censored)
# ----------------------------------------------------------------------
def encode_target(time_: np.ndarray, event: np.ndarray) -> np.ndarray:
    t = np.maximum(np.asarray(time_, dtype=float), 0.01)
    return np.where(np.asarray(event) == 1, t, -t)


def decode_target(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    return np.abs(y), (y > 0).astype(int)


def martingale_residuals(time_: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Null-model martingale residuals: event - Nelson-Aalen cumulative hazard at own time."""
    order = np.argsort(time_)
    t_sorted, e_sorted = time_[order], event[order]
    n = len(t_sorted)
    at_risk = n - np.arange(n)
    increments = e_sorted / at_risk
    # Tied times share the cumulative hazard at the last tied position
    cumhaz_sorted = np.cumsum(increments)
    uniq, last_idx = np.unique(t_sorted[::-1], return_index=True)
    last_pos = n - 1 - last_idx
    H_at = dict(zip(uniq, cumhaz_sorted[last_pos]))
    H = np.array([H_at[t] for t in time_])
    return event - H


class SurvivalGeneScreen(BaseEstimator, TransformerMixin):
    """
    Fold-isolated gene screening for survival:
    1. keep the most variable `variance_fraction` of genes;
    2. rank by |Pearson correlation| with null-model martingale residuals; keep `n_genes`.
    If `use_stage` is True, STAGE_COL is excluded from screening and passed through.
    """

    def __init__(self, n_genes: int = 150, variance_fraction: float = 0.5, use_stage: bool = False):
        self.n_genes = n_genes
        self.variance_fraction = variance_fraction
        self.use_stage = use_stage

    def fit(self, X: pd.DataFrame, y):
        time_, event = decode_target(y)
        genes = X.drop(columns=[STAGE_COL]) if self.use_stage else X
        var = genes.var(axis=0)
        keep = var.nlargest(int(len(var) * self.variance_fraction)).index
        G = genes[keep].values.astype(np.float64)
        r = martingale_residuals(time_, event)
        Gc = G - G.mean(axis=0)
        rc = r - r.mean()
        corr = (Gc * rc[:, None]).sum(axis=0) / (np.sqrt((Gc ** 2).sum(axis=0)) * np.sqrt((rc ** 2).sum()) + 1e-12)
        top = np.argsort(-np.abs(corr))[: self.n_genes]
        self.selected_genes_ = list(keep[top])
        self.selected_corr_ = corr[top]
        return self

    def transform(self, X: pd.DataFrame):
        out = X[self.selected_genes_].values
        if self.use_stage:
            out = np.hstack([out, X[[STAGE_COL]].values])
        return out


def build_pipeline(use_stage: bool) -> Pipeline:
    return Pipeline([
        ("screen", SurvivalGeneScreen(use_stage=use_stage)),
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            objective="survival:cox", eval_metric="cox-nloglik", n_estimators=300,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=5, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=1,
        )),
    ])


def c_index(time_, event, risk) -> float:
    return float(concordance_index(time_, -np.asarray(risk, dtype=float), event))


def _predict_margin(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    Z = model[:-1].transform(X)
    return model.named_steps["model"].predict(Z, output_margin=True)


def tune(base: Pipeline, X: pd.DataFrame, y: np.ndarray, n_splits: int, seed: int):
    """Grid search on C-index with an event-stratified inner CV; returns the refitted best model."""
    _, event = decode_target(y)
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, event))
    candidates = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]

    def run(params, tr, te):
        m = clone(base).set_params(**params)
        m.fit(X.iloc[tr], y[tr])
        t, e = decode_target(y[te])
        return c_index(t, e, _predict_margin(m, X.iloc[te]))

    scores = Parallel(n_jobs=-1)(delayed(run)(p, tr, te) for p in candidates for tr, te in folds)
    scores = np.array(scores).reshape(len(candidates), len(folds)).mean(axis=1)
    best = candidates[int(np.argmax(scores))]
    model = clone(base).set_params(**best)
    model.fit(X, y)
    return model, best, float(scores.max())


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_discovery() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """All GSE39582 Stage I-III tumours with RFS follow-up (non-tumoral samples excluded)."""
    raw = pd.read_csv(PROCESSED_DIR / "clinical" / "GSE39582_clinical_raw.csv", index_col=0)
    lab = pd.read_csv(PROCESSED_DIR / "clinical" / "GSE39582_cohort_labeled.csv", index_col=0)
    tumour = raw["dataset"].astype(str).str.lower() != "non tumoral"
    clin = lab[tumour.reindex(lab.index).fillna(False) & lab["stage_numeric"].isin([1, 2, 3])].copy()
    clin = clin.dropna(subset=["relapse_event", "rfs_time_months"])
    clin = clin[clin["rfs_time_months"] > 0]
    expr = pd.read_parquet(PROCESSED_DIR / "expression" / "GSE39582_genes_expression.parquet")
    X = expr[clin.index].T.astype(np.float32)
    out = pd.DataFrame({
        "time": clin["rfs_time_months"].astype(float),
        "event": clin["relapse_event"].astype(int),
        "stage": clin["stage_numeric"].astype(float),
    }, index=clin.index)
    return X, out


def load_external(cohort: str, ref: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stage I-III samples of a GPL570 cohort from the mega-cohort cache, aligned to `ref`."""
    expr = pd.read_parquet(PROCESSED_DIR / "megacohort" / "megacohort_raw_expression.parquet")
    clin = pd.read_csv(PROCESSED_DIR / "megacohort" / "megacohort_clinical.csv", index_col=0)
    c = clin[(clin["cohort"] == cohort) & clin["stage"].isin([1, 2, 3])]
    c = c.dropna(subset=["recurrence_event", "dfs_time_months"])
    c = c[c["dfs_time_months"] > 0]
    X = expr.loc[c.index].reindex(columns=ref.columns)
    X_aligned = align_to_reference(X, ref)
    out = pd.DataFrame({
        "time": c["dfs_time_months"].astype(float),
        "event": c["recurrence_event"].astype(int),
        "stage": c["stage"].astype(float),
    }, index=c.index)
    return X_aligned, out


def with_stage(X: pd.DataFrame, stage: pd.Series) -> pd.DataFrame:
    X = X.copy()
    X[STAGE_COL] = stage.loc[X.index].astype(np.float32).values
    return X


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def bootstrap_cindex(time_, event, risk, n_boot=1000, seed=RANDOM_STATE) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(time_)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if event[i].sum() == 0:
            continue
        vals.append(c_index(time_[i], event[i], risk[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def stage_adjusted_cox(time_, event, risk, stage) -> Dict[str, float]:
    """HR per SD of the risk score adjusted for stage, plus a likelihood-ratio test vs stage alone."""
    z = (risk - risk.mean()) / (risk.std() + 1e-12)
    df = pd.DataFrame({"T": time_, "E": event, "risk_z": z, "stage": stage})
    full = CoxPHFitter().fit(df, "T", "E")
    base = CoxPHFitter().fit(df[["T", "E", "stage"]], "T", "E")
    lr = 2 * (full.log_likelihood_ - base.log_likelihood_)
    ci = np.exp(full.confidence_intervals_.loc["risk_z"].values)
    return {
        "HR_per_SD_Stage_Adjusted": round(float(np.exp(full.params_["risk_z"])), 4),
        "HR_CI95_Low": round(float(ci[0]), 4),
        "HR_CI95_High": round(float(ci[1]), 4),
        "Cox_p_Stage_Adjusted": float(full.summary.loc["risk_z", "p"]),
        "LR_test_p_vs_Stage": float(stats.chi2.sf(lr, 1)),
        "C_Index_Cox_Stage_plus_Risk": round(float(full.concordance_index_), 4),
        "C_Index_Cox_Stage_Only": round(float(base.concordance_index_), 4),
    }


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
def nested_cv(X: pd.DataFrame, clin: pd.DataFrame, use_stage: bool):
    X_in = with_stage(X, clin["stage"]) if use_stage else X
    y = encode_target(clin["time"].values, clin["event"].values)
    event = clin["event"].values
    n = len(y)
    risk_rep = np.zeros((N_REPEATS, n))
    rep_c, gene_counts, chosen = [], {}, []
    for r in range(N_REPEATS):
        outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE + r)
        for f, (tr, te) in enumerate(outer.split(X_in, event)):
            model, params, _ = tune(build_pipeline(use_stage), X_in.iloc[tr], y[tr], N_INNER,
                                    RANDOM_STATE + 100 * r + f)
            m = _predict_margin(model, X_in.iloc[te])
            risk_rep[r, te] = (m - m.mean()) / (m.std() + 1e-12)  # label-free per-fold scaling
            chosen.append(params)
            for g in model.named_steps["screen"].selected_genes_:
                gene_counts[g] = gene_counts.get(g, 0) + 1
        rep_c.append(c_index(clin["time"].values, event, risk_rep[r]))
    return risk_rep.mean(axis=0), rep_c, gene_counts, chosen


def run_survival_pipeline() -> None:
    print("=" * 70)
    print("SURVIVAL MODEL: XGBoost-Cox on all Stage I-III tumours (GSE39582)")
    print("=" * 70)
    X, clin = load_discovery()
    t, e, st = clin["time"].values, clin["event"].values, clin["stage"].values
    print(f"[COHORT] n = {len(clin)} | RFS events = {int(e.sum())} | median follow-up {np.median(t):.1f} mo")

    results = {"discovery": {}, "external": {}}
    lo, hi = bootstrap_cindex(t, e, st)
    results["discovery"]["Stage_Only"] = {"C_Index": round(c_index(t, e, st), 4), "C_CI95_Low": round(lo, 4),
                                          "C_CI95_High": round(hi, 4)}
    print(f"[BASELINE] Stage-only C-index: {c_index(t, e, st):.3f} ({lo:.3f}-{hi:.3f})")

    oof = pd.DataFrame(clin)
    stability = []
    for name, use_stage in [("XGB_Cox_Genes", False), ("XGB_Cox_Genes_plus_Stage", True)]:
        t0 = time.time()
        risk, rep_c, counts, chosen = nested_cv(X, clin, use_stage)
        oof[f"{name}_risk"] = risk
        lo, hi = bootstrap_cindex(t, e, risk)
        res = {
            "C_Index": round(c_index(t, e, risk), 4), "C_CI95_Low": round(lo, 4), "C_CI95_High": round(hi, 4),
            "C_Index_Repeat_Mean": round(float(np.mean(rep_c)), 4),
            "C_Index_Repeat_SD": round(float(np.std(rep_c)), 4),
            **stage_adjusted_cox(t, e, risk, st),
            "Tuned_Params_Most_Common": pd.Series([json.dumps(c, sort_keys=True) for c in chosen]).mode()[0],
        }
        results["discovery"][name] = res
        for g, cnt in counts.items():
            if g != STAGE_COL:
                stability.append({"Model": name, "Gene_Symbol": g, "Times_Selected": cnt,
                                  "Selection_Frequency": round(cnt / (N_REPEATS * N_OUTER), 3)})
        print(f"  {name:26s} | OOF C-index {res['C_Index']:.3f} ({lo:.3f}-{hi:.3f}) | repeats "
              f"{res['C_Index_Repeat_Mean']:.3f}+/-{res['C_Index_Repeat_SD']:.3f} | stage-adj HR/SD "
              f"{res['HR_per_SD_Stage_Adjusted']:.2f} p={res['Cox_p_Stage_Adjusted']:.2g} | "
              f"LR p vs stage {res['LR_test_p_vs_Stage']:.2g} | {time.time() - t0:.0f}s")

    oof.to_csv(PROCESSED_DIR / "clinical" / "survival_oof_risk.csv", index_label="gsm_id")
    pd.DataFrame(stability).sort_values(["Model", "Times_Selected"], ascending=[True, False]).to_csv(
        BIOMARKERS_DIR / "survival_gene_selection_stability.csv", index=False)

    # ---------------- Final models + external validation ----------------
    print("\n[FINAL] Tuning on the full discovery cohort (5-fold inner CV)...")
    y = encode_target(t, e)
    finals, cutoffs = {}, {}
    for name, use_stage in [("XGB_Cox_Genes", False), ("XGB_Cox_Genes_plus_Stage", True)]:
        X_in = with_stage(X, clin["stage"]) if use_stage else X
        model, params, cv_c = tune(build_pipeline(use_stage), X_in, y, 5, RANDOM_STATE)
        finals[name] = model
        train_margin = _predict_margin(model, X_in)
        cutoffs[name] = {"median_margin": float(np.median(train_margin)),
                         "mean": float(train_margin.mean()), "sd": float(train_margin.std())}
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        print(f"  {name}: params {params} | inner-CV C-index {cv_c:.3f}")
    with open(MODEL_DIR / "risk_cutoffs.json", "w") as f:
        json.dump(cutoffs, f, indent=2)
    pd.DataFrame({"Gene_Symbol": finals["XGB_Cox_Genes"].named_steps["screen"].selected_genes_,
                  "Corr_with_Martingale_Residual": finals["XGB_Cox_Genes"].named_steps["screen"].selected_corr_}
                 ).to_csv(BIOMARKERS_DIR / "survival_model_genes.csv", index=False)

    ext_rows, ext_scores = [], {}
    for cohort in ["GSE14333", "GSE17536"]:
        Xe, ce = load_external(cohort, X)
        te_, ee_, se_ = ce["time"].values, ce["event"].values, ce["stage"].values
        lo, hi = bootstrap_cindex(te_, ee_, se_)
        ext_rows.append({"Cohort": cohort, "Model": "Stage_Only", "N": len(ce), "Events": int(ee_.sum()),
                         "C_Index": round(c_index(te_, ee_, se_), 4), "C_CI95_Low": round(lo, 4),
                         "C_CI95_High": round(hi, 4)})
        scores = pd.DataFrame(ce)
        for name, model in finals.items():
            X_in = with_stage(Xe, ce["stage"]) if name.endswith("plus_Stage") else Xe
            risk = _predict_margin(model, X_in)
            scores[f"{name}_risk"] = risk
            high = risk >= cutoffs[name]["median_margin"]
            lr_p = logrank_test(te_[high], te_[~high], ee_[high], ee_[~high]).p_value if 0 < high.sum() < len(high) else np.nan
            lo, hi = bootstrap_cindex(te_, ee_, risk)
            row = {"Cohort": cohort, "Model": name, "N": len(ce), "Events": int(ee_.sum()),
                   "C_Index": round(c_index(te_, ee_, risk), 4), "C_CI95_Low": round(lo, 4), "C_CI95_High": round(hi, 4),
                   **stage_adjusted_cox(te_, ee_, risk, se_),
                   "N_High_Risk_Frozen_Cutoff": int(high.sum()), "LogRank_p_Frozen_Cutoff": lr_p}
            ext_rows.append(row)
            print(f"  [{cohort}] {name:26s} | C-index {row['C_Index']:.3f} ({lo:.3f}-{hi:.3f}) vs stage "
                  f"{c_index(te_, ee_, se_):.3f} | "
                  f"stage-adj HR/SD {row['HR_per_SD_Stage_Adjusted']:.2f} p={row['Cox_p_Stage_Adjusted']:.2g} | "
                  f"LR p {row['LR_test_p_vs_Stage']:.2g} | KM log-rank p {lr_p:.2g} (high {high.sum()}/{len(high)})")
        ext_scores[cohort] = scores
        scores.to_csv(DATA_DIR / "external_validation" / f"{cohort}_survival_risk_scores.csv", index_label="sample_id")

    disc_rows = [{"Cohort": "GSE39582 (nested CV)", "Model": k, "N": len(clin), "Events": int(e.sum()), **v}
                 for k, v in results["discovery"].items()]
    table = pd.DataFrame(disc_rows + ext_rows)
    table.to_csv(TABLES_DIR / "survival_model_results.csv", index=False)
    with open(METRICS_DIR / "survival_metrics.json", "w") as f:
        json.dump(json.loads(table.to_json(orient="records")), f, indent=2)
    print(f"\n[EXPORT] {TABLES_DIR / 'survival_model_results.csv'}")

    plot_km(oof, ext_scores, cutoffs)


def plot_km(oof: pd.DataFrame, ext_scores: Dict[str, pd.DataFrame], cutoffs: Dict) -> None:
    name = "XGB_Cox_Genes"
    panels = [("GSE39582 discovery\n(nested-CV out-of-fold risk, median split)", oof, None)]
    panels += [(f"{c} external\n(frozen model, discovery median cut-off)", df, cutoffs[name]["median_margin"])
               for c, df in ext_scores.items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5.2))
    for ax, (title, df, cut) in zip(axes, panels):
        risk = df[f"{name}_risk"].values
        thr = np.median(risk) if cut is None else cut
        high = risk >= thr
        t, ev = df["time"].values, df["event"].values
        p = logrank_test(t[high], t[~high], ev[high], ev[~high]).p_value
        for mask, lab, col in [(~high, "Low risk", "#2ca02c"), (high, "High risk", "#d62728")]:
            KaplanMeierFitter().fit(t[mask], ev[mask], label=f"{lab} (n={mask.sum()})").plot_survival_function(
                ax=ax, color=col, linewidth=2.2, ci_show=True)
        ax.set_title(f"{title}\nlog-rank p = {p:.2g}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Months")
        ax.set_ylabel("Recurrence/disease-free survival")
        ax.set_ylim(0, 1.05)
    fig.suptitle("XGBoost-Cox gene risk score (gene-only model)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig26_survival_model_km.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig26_survival_model_km.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[FIGURE] Saved: fig26_survival_model_km.png")


if __name__ == "__main__":
    run_survival_pipeline()
