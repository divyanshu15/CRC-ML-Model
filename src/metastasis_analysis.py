"""
Colorectal Cancer Metastasis Analysis Module for CRC-ML Pipeline.

Analyzes paired RNA-seq data from GSE50760 (18 patients x 3 tissue types = 54 samples):
- Normal Colon Mucosa
- Primary Colorectal Adenocarcinoma
- Colorectal Liver Metastasis (CRLM)

Performs:
1. Paired statistical testing (repeated-measures accounting for patient background).
2. Metastasis biomarker identification & cross-referencing with primary recurrence genes.
3. Transcriptome trajectory PCA (Normal -> Primary -> Liver Metastasis).
4. Expression heatmap across the 3 pathological states.
5. Biological pathway enrichment analysis (GSEA / KEGG).
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
BIOMARKERS_DIR = RESULTS_DIR / "biomarkers"
TABLES_DIR = RESULTS_DIR / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
BIOMARKERS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Standardized publication aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 10


def assemble_gse50760_matrix() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and concatenate all 54 sample FPKM files into a unified matrix,
    convert to log2(FPKM + 1), and create a paired sample metadata table.
    """
    samples_dir = RAW_DIR / "GSE50760" / "samples"
    matrix_parquet = PROCESSED_DIR / "expression" / "GSE50760_metastasis_expression.parquet"
    metadata_csv = PROCESSED_DIR / "clinical" / "GSE50760_sample_metadata.csv"

    if matrix_parquet.exists() and metadata_csv.exists():
        print(f"[CACHE] Loading cached metastasis expression from: {matrix_parquet}")
        log_mat = pd.read_parquet(matrix_parquet)
        meta_df = pd.read_csv(metadata_csv, index_col="sample_col")
        return log_mat, meta_df

    files = sorted(list(samples_dir.glob("*.txt.gz")))
    print(f"[GSE50760] Assembling {len(files)} sample FPKM files...")

    dfs = []
    metadata_records = []

    for f in files:
        # File name format: GSM1228184_AMC_2.1_FPKM.txt.gz
        parts = f.name.split("_")
        gsm_id = parts[0]
        patient_tag = parts[1] + "_" + parts[2].split(".")[0]  # AMC_2
        tissue_code = parts[2].split(".")[1]  # 1 = Primary, 2 = Normal, 3 = Metastasis

        tissue_map = {"1": "Primary_CRC", "2": "Normal_Colon", "3": "Liver_Metastasis"}
        tissue_name = tissue_map.get(tissue_code, "Unknown")
        col_name = f"{patient_tag}_{tissue_name}"

        sample_df = pd.read_csv(f, sep="\t", index_col=0)
        sample_df.columns = [col_name]
        dfs.append(sample_df)

        metadata_records.append({
            "sample_col": col_name,
            "gsm_id": gsm_id,
            "patient_id": patient_tag,
            "tissue_type": tissue_name,
            "tissue_code": int(tissue_code),
        })

    raw_mat = pd.concat(dfs, axis=1)
    meta_df = pd.DataFrame(metadata_records).set_index("sample_col")

    # Transform to log2(FPKM + 1)
    print("[GSE50760] Transforming to log2(FPKM + 1)...")
    log_mat = np.log2(raw_mat + 1.0).astype(np.float32)

    # Save to parquet and csv
    matrix_parquet.parent.mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    log_mat.to_parquet(matrix_parquet, engine="pyarrow", compression="snappy")
    meta_df.to_csv(metadata_csv)
    print(f"[SAVED] Metastasis matrix: {matrix_parquet} ({log_mat.shape[0]:,} genes x {log_mat.shape[1]} samples)")
    return log_mat, meta_df


# Classic hepatocyte-specific genes (plasma proteins, coagulation factors, liver CYPs). Their
# expression in a metastasis sample measures how much normal liver tissue the biopsy contains.
LIVER_MARKERS = [
    "ALB", "APOA1", "APOA2", "APOC3", "APOH", "AHSG", "FGA", "FGB", "FGG", "TF", "HP", "HPX",
    "SERPINA1", "TTR", "RBP4", "CYP2E1", "CYP3A4", "CYP2C8", "CYP2C9", "F2", "ITIH1", "ITIH2",
    "HRG", "AMBP", "ORM1", "APOB", "ALDOB", "ARG1", "HAMP", "CRP",
]


def liver_content_scores(expr_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient mean log2 expression of the hepatocyte markers in each tissue."""
    markers = [g for g in LIVER_MARKERS if g in expr_df.index]
    patients = sorted(meta_df["patient_id"].unique())
    scores = pd.DataFrame(
        {t: [float(expr_df.loc[markers, f"{p}_{t}"].mean()) for p in patients]
         for t in ["Normal_Colon", "Primary_CRC", "Liver_Metastasis"]},
        index=pd.Index(patients, name="patient_id"),
    )
    scores["Excess_Liver_Signal_Met_minus_Primary"] = scores["Liver_Metastasis"] - scores["Primary_CRC"]
    scores.to_csv(TABLES_DIR / "metastasis_liver_content_scores.csv")
    print(f"[PURITY] Hepatocyte marker score (mean log2 FPKM of {len(markers)} genes): "
          f"primary median {scores['Primary_CRC'].median():.2f}, metastasis median {scores['Liver_Metastasis'].median():.2f}")
    return scores


def compute_paired_metastasis_de(
    expr_df: pd.DataFrame, meta_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Paired Liver Metastasis vs Primary CRC comparison across the 18 patients.

    Two analyses are reported:
    - Unadjusted paired t-test (confounded by normal liver tissue in the metastasis biopsies).
    - Liver-adjusted (primary analysis): for each gene, the within-patient difference
      (metastasis - primary) is regressed on the excess hepatocyte-marker signal of that pair;
      the intercept estimates the metastasis effect at zero liver admixture. Three metastases
      have almost no liver signal, so the intercept is anchored by data rather than extrapolated.
    """
    patients = sorted(meta_df["patient_id"].unique())
    print(f"[PAIRED DE] Running paired comparison for {len(patients)} matched patient pairs...")

    prim_vals = expr_df[[f"{p}_Primary_CRC" for p in patients]].values
    meta_vals = expr_df[[f"{p}_Liver_Metastasis" for p in patients]].values
    norm_vals = expr_df[[f"{p}_Normal_Colon" for p in patients]].values

    # --- Unadjusted paired t-test ---
    t_stat, p_val = stats.ttest_rel(meta_vals, prim_vals, axis=1)
    p_val_clean = np.nan_to_num(p_val, nan=1.0)
    _, fdr_q, _, _ = multipletests(p_val_clean, method="fdr_bh")

    # --- Liver-adjusted paired regression ---
    liver = liver_content_scores(expr_df, meta_df)
    h = liver.loc[patients, "Excess_Liver_Signal_Met_minus_Primary"].values
    D = (meta_vals - prim_vals).T                      # pairs x genes
    Xd = np.column_stack([np.ones(len(h)), h])         # pairs x 2
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    beta = XtX_inv @ Xd.T @ D                          # 2 x genes
    resid = D - Xd @ beta
    dof = len(h) - 2
    sigma2 = (resid ** 2).sum(axis=0) / dof
    se_int = np.sqrt(sigma2 * XtX_inv[0, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_adj = beta[0] / se_int
    p_adj = np.nan_to_num(2 * stats.t.sf(np.abs(t_adj), dof), nan=1.0)
    _, fdr_adj, _, _ = multipletests(p_adj, method="fdr_bh")

    de_df = pd.DataFrame({
        "Gene_Symbol": expr_df.index,
        "Is_Liver_Marker": expr_df.index.isin(LIVER_MARKERS),
        "Mean_Normal": norm_vals.mean(axis=1),
        "Mean_Primary": prim_vals.mean(axis=1),
        "Mean_Metastasis": meta_vals.mean(axis=1),
        "Log2FC_Primary_vs_Normal": prim_vals.mean(axis=1) - norm_vals.mean(axis=1),
        "Log2FC_Meta_vs_Primary": meta_vals.mean(axis=1) - prim_vals.mean(axis=1),
        "Paired_t_stat": t_stat,
        "p_value": p_val_clean,
        "FDR_q_value": fdr_q,
        "Log2FC_Meta_vs_Primary_LiverAdjusted": beta[0],
        "Liver_Signal_Slope": beta[1],
        "t_LiverAdjusted": t_adj,
        "p_value_LiverAdjusted": p_adj,
        "FDR_LiverAdjusted": fdr_adj,
    })
    de_df["Significant_FDR_LiverAdjusted"] = de_df["FDR_LiverAdjusted"] < 0.05
    de_df = de_df.sort_values(by="p_value_LiverAdjusted", ascending=True).reset_index(drop=True)

    de_df.to_csv(TABLES_DIR / "metastasis_paired_de_results.csv", index=False)
    print(f"[EXPORT] Paired DE table (unadjusted + liver-adjusted) saved.")
    print(f"[PAIRED DE] FDR < 0.05: unadjusted {int((de_df['FDR_q_value'] < 0.05).sum()):,} genes | "
          f"liver-adjusted {int(de_df['Significant_FDR_LiverAdjusted'].sum()):,} genes")

    de_df[~de_df["Is_Liver_Marker"]].head(50).to_csv(BIOMARKERS_DIR / "metastasis_associated_genes.csv", index=False)
    return de_df


def plot_metastasis_pca(expr_df: pd.DataFrame, meta_df: pd.DataFrame) -> None:
    """
    Figure 16: PCA of 54 samples across the 3 pathological states
    (Normal Colon -> Primary Colorectal Cancer -> Liver Metastasis).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(expr_df.T)

    pca = PCA(n_components=5, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100

    pca_df = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "Tissue": meta_df["tissue_type"].values,
        "Patient": meta_df["patient_id"].values,
    }, index=meta_df.index)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    palette = {
        "Normal_Colon": "#2ca02c",
        "Primary_CRC": "#1f77b4",
        "Liver_Metastasis": "#d62728"
    }

    sns.scatterplot(
        data=pca_df, x="PC1", y="PC2", hue="Tissue",
        palette=palette, s=90, alpha=0.9, edgecolor="black", ax=ax
    )

    # Draw trajectories for a few representative paired patients
    for pt in ["AMC_2", "AMC_7", "AMC_18"]:
        pt_df = pca_df[pca_df["Patient"] == pt]
        if len(pt_df) == 3:
            pt_sorted = pt_df.sort_values(by="Tissue", key=lambda col: col.map({"Normal_Colon": 0, "Primary_CRC": 1, "Liver_Metastasis": 2}))
            ax.plot(pt_sorted["PC1"], pt_sorted["PC2"], color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
            ax.text(pt_sorted.iloc[2]["PC1"] + 1, pt_sorted.iloc[2]["PC2"] + 1, pt, fontsize=8, color="#555")

    ax.set_title(f"Transcriptome PCA: Colorectal Metastatic Progression (GSE50760)\n"
                 f"(PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}% Variance)", fontweight="bold")
    ax.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1f}%)")
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1f}%)")
    ax.legend(frameon=True, loc="upper right")

    png_path = FIGURES_DIR / "fig16_metastasis_pca.png"
    pdf_path = FIGURES_DIR / "fig16_metastasis_pca.pdf"
    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] Saved: {png_path.name}")


def plot_metastasis_heatmap(expr_df: pd.DataFrame, de_df: pd.DataFrame, meta_df: pd.DataFrame) -> None:
    """
    Figure 17: Clustermap showing top 35 metastasis-associated genes across all 54 samples.
    """
    top_genes = de_df[~de_df["Is_Liver_Marker"]].head(35)["Gene_Symbol"].tolist()
    sub_expr = expr_df.loc[top_genes]

    # Z-score standardize by row
    z_expr = sub_expr.sub(sub_expr.mean(axis=1), axis=0).div(sub_expr.std(axis=1), axis=0)

    # Color annotation bar for tissue type
    tissue_colors = meta_df["tissue_type"].map({
        "Normal_Colon": "#2ca02c",
        "Primary_CRC": "#1f77b4",
        "Liver_Metastasis": "#d62728"
    })
    tissue_colors.name = "Tissue State"

    g = sns.clustermap(
        z_expr,
        col_colors=tissue_colors,
        cmap="vlag",
        center=0,
        vmin=-2.5,
        vmax=2.5,
        figsize=(13, 10),
        yticklabels=True,
        xticklabels=False,
        dendrogram_ratio=(0.15, 0.15),
        cbar_pos=(0.02, 0.8, 0.03, 0.15),
        cbar_kws={"label": "Expression Z-Score"}
    )
    g.fig.suptitle("Top Liver-Adjusted Metastasis Genes (hepatocyte markers excluded)", y=0.98, fontweight="bold")

    png_path = FIGURES_DIR / "fig17_metastasis_expression_heatmap.png"
    pdf_path = FIGURES_DIR / "fig17_metastasis_expression_heatmap.pdf"
    g.savefig(png_path, dpi=300, bbox_inches="tight")
    g.savefig(pdf_path, bbox_inches="tight")
    plt.close(g.fig)
    print(f"[FIGURE] Saved: {png_path.name}")


def run_pathway_enrichment(de_df: pd.DataFrame) -> pd.DataFrame:
    """
    Figure 18: Enrichr (KEGG 2021, GO BP 2023) on genes up in metastasis after liver adjustment
    (liver-adjusted log2FC > 1, adjusted FDR < 0.05, hepatocyte markers excluded).
    If the Enrichr service is unreachable, no enrichment results are produced.
    """
    sel = (
        (de_df["Log2FC_Meta_vs_Primary_LiverAdjusted"] > 1.0)
        & (de_df["FDR_LiverAdjusted"] < 0.05)
        & (~de_df["Is_Liver_Marker"])
    )
    top_up_genes = de_df.loc[sel].sort_values("p_value_LiverAdjusted")["Gene_Symbol"].tolist()
    print(f"[ENRICHR] {len(top_up_genes)} liver-adjusted upregulated metastasis genes (submitting up to 250)...")
    enr_csv = TABLES_DIR / "metastasis_pathway_enrichment.csv"

    if len(top_up_genes) < 5:
        print("[ENRICHR] Too few genes for enrichment; skipping.")
        pd.DataFrame().to_csv(enr_csv, index=False)
        return pd.DataFrame()
    try:
        enr = gp.enrichr(
            gene_list=top_up_genes[:250],
            gene_sets=["KEGG_2021_Human", "GO_Biological_Process_2023"],
            outdir=None,
        )
        enr_df = enr.results.sort_values(by="Adjusted P-value", ascending=True)
    except Exception as e:  # never substitute invented results
        print(f"[ENRICHR WARNING] Enrichr query failed ({e}). No enrichment results written; re-run when online.")
        pd.DataFrame().to_csv(enr_csv, index=False)
        return pd.DataFrame()

    enr_df.to_csv(enr_csv, index=False)
    print(f"[EXPORT] Pathway enrichment table: {enr_csv}")

    top10 = enr_df.head(10).copy()
    top10["neg_log_p"] = -np.log10(top10["Adjusted P-value"].clip(lower=1e-15))
    top10 = top10.sort_values(by="neg_log_p", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top10["Term"], top10["neg_log_p"], color="#e74c3c", edgecolor="black", height=0.65)
    ax.set_title(f"Pathways Enriched in Liver Metastasis After Liver-Tissue Adjustment\n"
                 f"({len(top_up_genes)} genes; hepatocyte markers excluded)", fontweight="bold")
    ax.set_xlabel("-Log10 (Adjusted P-value)")
    ax.axvline(-np.log10(0.05), color="gray", linestyle="--", linewidth=1.2, label="q = 0.05")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig18_metastasis_gsea_pathways.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig18_metastasis_gsea_pathways.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[FIGURE] Saved: fig18_metastasis_gsea_pathways.png")
    return enr_df


def cross_reference_biomarkers(de_meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recurrence candidate genes (GSE39582 top 50 by nominal p) that are ALSO significantly
    changed in metastasis after liver adjustment (FDR < 0.05, hepatocyte markers excluded).
    """
    recur_biomarkers_path = BIOMARKERS_DIR / "candidate_biomarkers.csv"
    if not recur_biomarkers_path.exists():
        return pd.DataFrame()
    recur_genes = set(pd.read_csv(recur_biomarkers_path)["Gene_Symbol"])
    in_both = de_meta_df["Gene_Symbol"].isin(recur_genes)
    overlap = de_meta_df[in_both & de_meta_df["Significant_FDR_LiverAdjusted"] & ~de_meta_df["Is_Liver_Marker"]].copy()
    print(f"\n[OVERLAP] {int(in_both.sum())} of {len(recur_genes)} recurrence candidates are measured in GSE50760; "
          f"{len(overlap)} are significant in metastasis after liver adjustment (FDR < 0.05).")
    overlap.to_csv(BIOMARKERS_DIR / "recurrence_and_metastasis_shared_biomarkers.csv", index=False)
    return overlap


def generate_metastasis_report(de_df: pd.DataFrame, overlap_df: pd.DataFrame, enr_df: pd.DataFrame) -> None:
    """Write reports/metastasis_analysis_report.md entirely from the computed results."""
    report_path = REPORTS_DIR / "metastasis_analysis_report.md"
    liver = pd.read_csv(TABLES_DIR / "metastasis_liver_content_scores.csv", index_col=0)
    n_raw = int((de_df["FDR_q_value"] < 0.05).sum())
    adj = de_df["Significant_FDR_LiverAdjusted"]
    n_adj = int(adj.sum())
    n_up = int((adj & (de_df["Log2FC_Meta_vs_Primary_LiverAdjusted"] > 0)).sum())
    n_down = int((adj & (de_df["Log2FC_Meta_vs_Primary_LiverAdjusted"] < 0)).sum())
    low_liver = liver.index[liver["Liver_Metastasis"] < 3].tolist()

    content = f"""# Colorectal Cancer Liver Metastasis Analysis Report (GSE50760)

**Dataset:** GSE50760 (Illumina HiSeq 2000 RNA-seq, log2(FPKM + 1)); 18 patients x (normal colon, primary CRC, liver metastasis) = 54 samples.

*This report is generated from the pipeline outputs by `src/metastasis_analysis.py`.*

---

## 1. Normal-liver admixture in the metastasis samples

The mean log2 expression of {len([g for g in LIVER_MARKERS if g in de_df['Gene_Symbol'].values])} hepatocyte marker genes (ALB, APOA1, FGA/FGB/FGG, SERPINA1, CYP2E1, ...) is
**{liver['Primary_CRC'].median():.2f}** in primary tumours and **{liver['Liver_Metastasis'].median():.2f}** in liver metastases (medians).
{len(liver) - len(low_liver)} of {len(liver)} metastasis samples carry a strong liver signal; {len(low_liver)} ({', '.join(low_liver)}) carry almost none.
Without adjustment, "metastasis vs primary" differences are dominated by normal liver tissue
(the unadjusted top pathways were complement/coagulation, bile secretion and cytochrome P450 metabolism).

## 2. Paired differential expression

- Genes tested: {len(de_df):,}
- FDR < 0.05, unadjusted paired t-test: {n_raw:,}
- **FDR < 0.05, liver-adjusted (primary analysis): {n_adj:,}** ({n_up:,} up, {n_down:,} down in metastasis)

Liver adjustment: per gene, the within-patient difference (metastasis - primary) is regressed on the
pair's excess hepatocyte-marker signal; the intercept is the estimated metastasis effect at zero liver admixture.

### Top liver-adjusted genes (hepatocyte markers excluded)

| Rank | Gene | Adjusted log2FC | Unadjusted log2FC | Adjusted p | Adjusted FDR |
| :---: | :--- | :---: | :---: | :---: | :---: |
"""
    top = de_df[~de_df["Is_Liver_Marker"]].head(15).reset_index(drop=True)
    for i, r in top.iterrows():
        content += (f"| {i + 1} | `{r['Gene_Symbol']}` | {r['Log2FC_Meta_vs_Primary_LiverAdjusted']:+.2f} | "
                    f"{r['Log2FC_Meta_vs_Primary']:+.2f} | {r['p_value_LiverAdjusted']:.2e} | {r['FDR_LiverAdjusted']:.2e} |\n")

    content += "\n## 3. Pathway enrichment (liver-adjusted upregulated genes)\n\n"
    if enr_df is None or enr_df.empty:
        content += "No enrichment results (too few genes, or the Enrichr service was unreachable).\n"
    else:
        content += "| Library | Term | Overlap | Adjusted p |\n| :--- | :--- | :---: | :---: |\n"
        for _, r in enr_df.head(10).iterrows():
            content += f"| {r['Gene_set']} | {r['Term']} | {r['Overlap']} | {r['Adjusted P-value']:.2e} |\n"

    content += f"""
## 4. Overlap with recurrence candidates

{len(overlap_df)} of the top-50 GSE39582 recurrence candidates (ranked by nominal p; none FDR-significant for recurrence)
are significantly changed in metastasis after liver adjustment{': ' + ', '.join(f'`{g}`' for g in overlap_df['Gene_Symbol'].head(15)) if len(overlap_df) else ''}.
This overlap is hypothesis-generating only.

## 5. Caveats

- 18 pairs; the liver adjustment uses a linear model with a marker-based score, which cannot remove all admixture effects (e.g. stromal or immune differences between colon and liver).
- FPKM values from GEO; count-based methods (e.g. limma-voom/DESeq2 with a patient term) would be preferable if raw counts were available.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[REPORT] Saved research report to: {report_path}")


def run_metastasis_pipeline() -> None:
    """Execute complete Phase 8 metastasis analysis pipeline."""
    print("=" * 70)
    print("CRC-ML PHASE 8: COLORECTAL LIVER METASTASIS ANALYSIS (GSE50760)")
    print("=" * 70)

    expr_df, meta_df = assemble_gse50760_matrix()
    de_df = compute_paired_metastasis_de(expr_df, meta_df)
    plot_metastasis_pca(expr_df, meta_df)
    plot_metastasis_heatmap(expr_df, de_df, meta_df)
    enr_df = run_pathway_enrichment(de_df)
    overlap_df = cross_reference_biomarkers(de_df)
    generate_metastasis_report(de_df, overlap_df, enr_df)

    print("\n[SUCCESS] Phase 8 Metastasis Analysis Complete.")


if __name__ == "__main__":
    run_metastasis_pipeline()

