"""
Statistical Analysis & Leakage-Free Feature Selection Module for CRC-ML Pipeline.

Implements multi-stage biomarker selection:
1. Unsupervised variance filtering.
2. Two-sample Welch's t-test with Benjamini-Hochberg FDR correction.
3. Embedded L1 Lasso / Logistic Regression sparsity selection.
4. Scikit-learn Transformer encapsulation for strict fold-isolated cross-validation.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
BIOMARKERS_DIR = RESULTS_DIR / "biomarkers"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

BIOMARKERS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def compute_differential_expression(
    X: pd.DataFrame, y: pd.Series, alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform genome-wide differential expression analysis comparing Class 1 vs Class 0.

    Parameters
    ----------
    X : pd.DataFrame
        Samples x Genes expression matrix (RMA log2 scale).
    y : pd.Series
        Binary outcome labels (1 = Early Recurrence, 0 = Disease-Free).
    alpha : float
        FDR significance threshold.

    Returns
    -------
    pd.DataFrame
        Table with Log2 Fold Change, Welch t-statistic, raw p-value, and BH adjusted FDR.
    """
    idx_1 = y[y == 1].index
    idx_0 = y[y == 0].index

    X1 = X.loc[idx_1].values
    X0 = X.loc[idx_0].values

    # Welch t-test (unequal variance)
    t_stat, p_val = stats.ttest_ind(X1, X0, axis=0, equal_var=False)

    # Log2 Fold Change: mean(Class 1) - mean(Class 0)
    mean_1 = np.mean(X1, axis=0)
    mean_0 = np.mean(X0, axis=0)
    log2_fc = mean_1 - mean_0

    # Benjamini-Hochberg FDR
    reject, fdr_q, _, _ = multipletests(p_val, alpha=alpha, method="fdr_bh")

    de_df = pd.DataFrame(
        {
            "Gene_Symbol": X.columns,
            "Mean_Class0": mean_0,
            "Mean_Class1": mean_1,
            "Log2_FC": log2_fc,
            "Welch_t_stat": t_stat,
            "p_value": p_val,
            "FDR_q_value": fdr_q,
            "Significant_FDR": reject,
        }
    )

    de_df["abs_Log2_FC"] = de_df["Log2_FC"].abs()
    de_df["neg_log10_p"] = -np.log10(de_df["p_value"].clip(lower=1e-30))
    de_df = de_df.sort_values(by="p_value", ascending=True).reset_index(drop=True)

    return de_df


def plot_volcano(de_df: pd.DataFrame, top_label_n: int = 12) -> None:
    """
    Generate publication-quality Volcano Plot of differential expression.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))

    p_cutoff = 0.01
    fc_cutoff = 0.30

    # Classify genes for coloring
    is_up = (de_df["p_value"] < p_cutoff) & (de_df["Log2_FC"] >= fc_cutoff)
    is_down = (de_df["p_value"] < p_cutoff) & (de_df["Log2_FC"] <= -fc_cutoff)
    is_ns = ~is_up & ~is_down

    ax.scatter(de_df.loc[is_ns, "Log2_FC"], de_df.loc[is_ns, "neg_log10_p"],
               color="#bdc3c7", alpha=0.5, s=15, label="Non-significant")
    ax.scatter(de_df.loc[is_up, "Log2_FC"], de_df.loc[is_up, "neg_log10_p"],
               color="#e74c3c", alpha=0.85, s=28, label=f"Upregulated in Early Recurrence (n={is_up.sum()})")
    ax.scatter(de_df.loc[is_down, "Log2_FC"], de_df.loc[is_down, "neg_log10_p"],
               color="#2980b9", alpha=0.85, s=28, label=f"Downregulated in Early Recurrence (n={is_down.sum()})")

    # Threshold guidelines
    ax.axhline(-np.log10(p_cutoff), color="black", linestyle="--", linewidth=1.0)
    ax.axvline(fc_cutoff, color="gray", linestyle=":", linewidth=1.0)
    ax.axvline(-fc_cutoff, color="gray", linestyle=":", linewidth=1.0)

    # Label top genes
    top_candidates = pd.concat([
        de_df[is_up].head(top_label_n // 2),
        de_df[is_down].head(top_label_n // 2)
    ])

    for _, row in top_candidates.iterrows():
        ax.annotate(
            row["Gene_Symbol"],
            (row["Log2_FC"], row["neg_log10_p"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=9,
            fontweight="bold",
            color="#2c3e50"
        )

    ax.set_title("Differential Gene Expression: Early Recurrence vs Disease-Free CRC", fontweight="bold")
    ax.set_xlabel("Log2 Fold Change (Early Recurrence - Disease Free)")
    ax.set_ylabel("-Log10 (Welch t-test p-value)")
    ax.legend(frameon=True, loc="upper right")

    png_path = FIGURES_DIR / "fig06_volcano_plot.png"
    pdf_path = FIGURES_DIR / "fig06_volcano_plot.pdf"
    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: {png_path.name}")


def plot_differential_heatmap(X: pd.DataFrame, y: pd.Series, top_genes: List[str]) -> None:
    """
    Generate Clustermap/Heatmap of top candidate biomarkers across patient samples.
    """
    subset_X = X[top_genes].T  # Genes x Samples

    # Standardize genes across samples (Z-score)
    z_expr = subset_X.sub(subset_X.mean(axis=1), axis=0).div(subset_X.std(axis=1), axis=0)

    # Outcome column colors
    lut = {0: "#2ca02c", 1: "#d62728"}
    col_colors = y.map(lut)
    col_colors.name = "Recurrence"

    g = sns.clustermap(
        z_expr,
        col_colors=col_colors,
        cmap="vlag",
        center=0,
        vmin=-2.5,
        vmax=2.5,
        figsize=(12, 10),
        yticklabels=True,
        xticklabels=False,
        dendrogram_ratio=(0.15, 0.15),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        cbar_kws={"label": "Expression Z-Score"}
    )
    g.fig.suptitle("Expression Heatmap of Top Recurrence Candidate Biomarkers", y=0.98, fontweight="bold")

    png_path = FIGURES_DIR / "fig07_de_genes_heatmap.png"
    pdf_path = FIGURES_DIR / "fig07_de_genes_heatmap.pdf"
    g.savefig(png_path, dpi=300, bbox_inches="tight")
    g.savefig(pdf_path, bbox_inches="tight")
    plt.close(g.fig)
    print(f"[FIGURE] Saved: {png_path.name}")


def plot_pca_visualization(X: pd.DataFrame, y: pd.Series, stage_series: pd.Series) -> None:
    """
    Module 7: Principal Component Analysis (PCA) visualization colored by
    Recurrence Outcome and Clinical TNM Stage.
    """
    from sklearn.decomposition import PCA

    # Scale before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=10, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    pca_df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "Recurrence": y.map({0: "Disease-Free", 1: "Early Recurrence"}),
        "Stage": "Stage " + stage_series.astype(int).astype(str),
    }, index=X.index)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Color by Recurrence
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2", hue="Recurrence",
        palette={"Disease-Free": "#2ca02c", "Early Recurrence": "#d62728"},
        alpha=0.85, s=60, edgecolor="black", ax=axes[0]
    )
    axes[0].set_title(f"Transcriptome PCA Stratified by Recurrence Outcome\n(PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}%)",
                      fontweight="bold")
    axes[0].set_xlabel(f"Principal Component 1 ({var_exp[0]:.1f}% Variance)")
    axes[0].set_ylabel(f"Principal Component 2 ({var_exp[1]:.1f}% Variance)")
    axes[0].legend(frameon=True, loc="upper right")

    # 2. Color by Clinical Stage
    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2", hue="Stage",
        palette={"Stage 1": "#3498db", "Stage 2": "#f39c12", "Stage 3": "#9b59b6"},
        alpha=0.85, s=60, edgecolor="black", ax=axes[1]
    )
    axes[1].set_title("Transcriptome PCA Stratified by Clinical TNM Stage", fontweight="bold")
    axes[1].set_xlabel(f"Principal Component 1 ({var_exp[0]:.1f}% Variance)")
    axes[1].set_ylabel(f"Principal Component 2 ({var_exp[1]:.1f}% Variance)")
    axes[1].legend(frameon=True, loc="upper right")

    plt.tight_layout()
    png_path = FIGURES_DIR / "fig08_pca_visualization.png"
    pdf_path = FIGURES_DIR / "fig08_pca_visualization.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: {png_path.name}")


class BioinformaticsFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Leakage-Free Multi-Stage Feature Selection Transformer.
    Integrates seamlessly into scikit-learn Pipeline so that variance filtering,
    Welch hypothesis screening, and L1 Lasso selection run strictly on training folds.
    """

    def __init__(
        self,
        variance_percentile: float = 0.50,
        n_stat_candidates: int = 150,
        n_final_genes: int = 35,
        lasso_c: float = 0.15,
        random_state: int = 42,
    ):
        self.variance_percentile = variance_percentile
        self.n_stat_candidates = n_stat_candidates
        self.n_final_genes = n_final_genes
        self.lasso_c = lasso_c
        self.random_state = random_state
        self.selected_genes_: List[str] = []

    def fit(self, X, y):
        # Convert to DataFrame if ndarray
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        y_ser = pd.Series(y, index=X_df.index)

        # Stage 1: Unsupervised Variance Filter on Training Fold
        variances = X_df.var(axis=0)
        n_keep_var = int(len(variances) * (1.0 - self.variance_percentile))
        var_genes = variances.nlargest(n_keep_var).index
        X_var = X_df[var_genes]

        # Stage 2: Supervised Welch's t-test on Training Fold
        idx_1 = y_ser[y_ser == 1].index
        idx_0 = y_ser[y_ser == 0].index

        X1 = X_var.loc[idx_1].values
        X0 = X_var.loc[idx_0].values

        t_stat, p_val = stats.ttest_ind(X1, X0, axis=0, equal_var=False)
        p_val_ser = pd.Series(p_val, index=var_genes).fillna(1.0)

        # Select top statistical candidate genes
        stat_genes = p_val_ser.nsmallest(self.n_stat_candidates).index
        X_stat = X_var[stat_genes]

        # Stage 3: Embedded Regularized Selection (L1 Logistic Regression)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_stat)

        # L1 Lasso penalization to enforce sparsity
        l1_clf = LogisticRegression(
            penalty="l1",
            C=self.lasso_c,
            solver="liblinear",
            class_weight="balanced",
            random_state=self.random_state,
            max_iter=500,
        )
        l1_clf.fit(X_scaled, y_ser)

        coefs = np.abs(l1_clf.coef_[0])
        non_zero_idx = np.where(coefs > 0)[0]

        if len(non_zero_idx) >= 10:
            # Rank non-zero genes by coefficient magnitude
            sorted_non_zero = non_zero_idx[np.argsort(-coefs[non_zero_idx])]
            selected_idx = sorted_non_zero[: self.n_final_genes]
            self.selected_genes_ = list(stat_genes[selected_idx])
        else:
            # Fallback to top statistical genes if Lasso is overly sparse
            self.selected_genes_ = list(stat_genes[: self.n_final_genes])

        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_genes_].values
        else:
            return X[:, [X.columns.get_loc(g) for g in self.selected_genes_]]


def run_feature_selection_discovery() -> pd.DataFrame:
    """Execute complete Phase 5 discovery analysis and biomarker extraction."""
    print("=" * 70)
    print("CRC-ML PHASE 5: STATISTICAL ANALYSIS & BIOMARKER DISCOVERY")
    print("=" * 70)

    # 1. Load ML features and targets
    ml_features_path = PROCESSED_DIR / "expression" / "GSE39582_ml_features.parquet"
    ml_targets_path = PROCESSED_DIR / "clinical" / "GSE39582_ml_targets.csv"

    print(f"[LOADING] Features: {ml_features_path}...")
    X = pd.read_parquet(ml_features_path)
    targets_df = pd.read_csv(ml_targets_path, index_col="gsm_id")
    y = targets_df["recurrence_label"]
    stage = targets_df["stage_numeric"]

    # 2. Genome-Wide Differential Expression Screening
    print(f"[STATISTICS] Computing Welch t-test and BH-FDR across {X.shape[1]:,} genes...")
    de_df = compute_differential_expression(X, y)

    # Save full differential expression table
    de_table_path = TABLES_DIR / "differential_expression_all_genes.csv"
    de_df.to_csv(de_table_path, index=False)
    print(f"[SAVED] Complete DE table: {de_table_path} ({len(de_df):,} genes)")

    sig_p05 = (de_df["p_value"] < 0.05).sum()
    sig_p01 = (de_df["p_value"] < 0.01).sum()
    sig_p001 = (de_df["p_value"] < 0.001).sum()
    sig_fdr = (de_df["FDR_q_value"] < 0.10).sum()

    print("\n--- Differential Expression Summary ---")
    print(f"  Genes with raw p < 0.05        : {sig_p05:,}")
    print(f"  Genes with raw p < 0.01        : {sig_p01:,}")
    print(f"  Genes with raw p < 0.001       : {sig_p001:,}")
    print(f"  Genes with FDR q < 0.10        : {sig_fdr:,}")

    # 3. Save Candidate Biomarkers Table (Top 50 candidate genes)
    top_biomarkers = de_df.head(50).copy()
    candidate_path = BIOMARKERS_DIR / "candidate_biomarkers.csv"
    top_biomarkers.to_csv(candidate_path, index=False)
    print(f"[SAVED] Top 50 candidate biomarkers table: {candidate_path}")

    # 4. Generate Volcano Plot
    print("[PLOTTING] Generating Volcano Plot (Fig 6)...")
    plot_volcano(de_df)

    # 5. Generate Heatmap of Top 30 Genes
    print("[PLOTTING] Generating Expression Heatmap of Top Biomarkers (Fig 7)...")
    top_30_genes = top_biomarkers["Gene_Symbol"].head(30).tolist()
    plot_differential_heatmap(X, y, top_30_genes)

    # 6. Generate PCA Visualization (Module 7)
    print("[PLOTTING] Generating PCA Visualization (Fig 8)...")
    plot_pca_visualization(X, y, stage)

    print("\n[SUCCESS] Phase 5 Statistical Analysis & Visualizations Complete.")
    return de_df


if __name__ == "__main__":
    run_feature_selection_discovery()

