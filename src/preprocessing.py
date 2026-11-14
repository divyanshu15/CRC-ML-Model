"""
Data Preprocessing & Probe-to-Gene Mapping Module for CRC-ML Pipeline.

Implements bioinformatics platform annotation and probe aggregation:
- Maps Affymetrix GPL570 probesets (54,675) to official HGNC Gene Symbols.
- Removes control (AFFX-) and unannotated probes.
- Resolves multi-probe redundancy using maximum inter-sample variance.
- Aligns gene-level expression features with the curated Stage I-III recurrence cohort (n = 264).
"""

import gzip
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXPR_RAW_PARQUET = PROCESSED_DIR / "expression" / "GSE39582_expression.parquet"
GPL570_ANNOT_PATH = RAW_DIR / "GPL570.annot.gz"
COHORT_LABELED_PATH = PROCESSED_DIR / "clinical" / "GSE39582_cohort_labeled.csv"

# Target exports
GENES_EXPR_PARQUET = PROCESSED_DIR / "expression" / "GSE39582_genes_expression.parquet"
ML_FEATURES_PARQUET = PROCESSED_DIR / "expression" / "GSE39582_ml_features.parquet"
ML_TARGETS_CSV = PROCESSED_DIR / "clinical" / "GSE39582_ml_targets.csv"


def load_gpl570_annotations(annot_path: Path = GPL570_ANNOT_PATH) -> pd.DataFrame:
    """
    Parse GPL570 annotation table to extract probe-to-gene symbol mappings.

    Parameters
    ----------
    annot_path : Path
        Path to GPL570.annot.gz.

    Returns
    -------
    pd.DataFrame
        DataFrame with probe 'ID' as index and cleaned 'Gene_Symbol' column.
    """
    if not annot_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annot_path}")

    # Find table start
    skiprows = 0
    with gzip.open(annot_path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("!platform_table_begin"):
                skiprows = i + 1
                break

    print(f"[ANNOTATION] Loading GPL570 platform table (skiprows={skiprows})...")
    annot_df = pd.read_csv(
        annot_path,
        sep="\t",
        skiprows=skiprows,
        usecols=["ID", "Gene symbol"],
        comment="!",
        dtype=str,
    )

    # Clean probe annotations
    annot_df = annot_df.dropna(subset=["Gene symbol"]).copy()
    # If multiple gene symbols (e.g. MIR4640///DDR1), take primary protein-coding symbol
    annot_df["Gene_Symbol"] = annot_df["Gene symbol"].str.split("///").str[0].str.strip()

    # Filter invalid symbols
    annot_df = annot_df[annot_df["Gene_Symbol"] != ""]
    annot_df = annot_df[~annot_df["Gene_Symbol"].str.startswith("---")]
    annot_df = annot_df[~annot_df["ID"].str.startswith("AFFX-")]  # Drop Affymetrix hybridization controls

    annot_df = annot_df[["ID", "Gene_Symbol"]].drop_duplicates()
    annot_df.set_index("ID", inplace=True)

    print(
        f"[ANNOTATION] Successfully mapped {len(annot_df):,} probes to "
        f"{annot_df['Gene_Symbol'].nunique():,} unique HGNC Gene Symbols."
    )
    return annot_df


def collapse_probes_to_genes(
    expr_df: pd.DataFrame, annot_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Collapse multiple probesets representing the same gene symbol by selecting
    the probe exhibiting the maximum inter-sample variance.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Probes x Samples numeric matrix (float32).
    annot_df : pd.DataFrame
        Probe ID index with Gene_Symbol column.

    Returns
    -------
    pd.DataFrame
        Genes x Samples matrix with unique Gene Symbols as index.
    """
    t0 = time.time()
    # Intersect probes
    common_probes = expr_df.index.intersection(annot_df.index)
    print(f"[PREPROCESS] Retaining {len(common_probes):,} probes matching valid human genes...")

    expr_sub = expr_df.loc[common_probes].copy()
    expr_sub["Gene_Symbol"] = annot_df.loc[common_probes, "Gene_Symbol"]

    # Sample columns
    sample_cols = [c for c in expr_sub.columns if c != "Gene_Symbol"]

    # Compute variance across all samples for each probe
    print("[PREPROCESS] Calculating probe-level sample variances...")
    expr_sub["probe_variance"] = expr_sub[sample_cols].var(axis=1)

    # For duplicate gene symbols, keep the probe with maximum variance
    print("[PREPROCESS] Selecting max-variance probe for each unique gene symbol...")
    expr_sub = expr_sub.sort_values(by="probe_variance", ascending=False)
    expr_collapsed = expr_sub.drop_duplicates(subset=["Gene_Symbol"])

    # Set Gene_Symbol as index and keep only sample columns
    expr_collapsed = expr_collapsed.set_index("Gene_Symbol")[sample_cols]
    expr_collapsed = expr_collapsed.astype(np.float32)

    elapsed = time.time() - t0
    mem_mb = expr_collapsed.memory_usage(deep=True).sum() / (1024 * 1024)
    print(
        f"[COLLAPSED] Gene-level expression matrix: {expr_collapsed.shape[0]:,} genes x "
        f"{expr_collapsed.shape[1]:,} samples in {elapsed:.2f}s (RAM: {mem_mb:.2f} MB)"
    )
    return expr_collapsed


def align_ml_cohort(
    expr_genes_df: pd.DataFrame, labeled_cohort_path: Path = COHORT_LABELED_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align gene-level expression with the curated Stage I-III recurrence cohort (n = 264),
    transposing features so samples are rows and genes are columns (standard ML format).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        X (Features: n_samples x n_genes) and y (Targets: n_samples with recurrence_label)
    """
    labeled_df = pd.read_csv(labeled_cohort_path, index_col="gsm_id")

    # Filter to eligible binary classification cohort (Class 0 and Class 1)
    binary_cohort = labeled_df.dropna(subset=["recurrence_label"]).copy()
    binary_cohort["recurrence_label"] = binary_cohort["recurrence_label"].astype(int)

    # Intersect sample IDs
    valid_samples = [s for s in binary_cohort.index if s in expr_genes_df.columns]
    print(f"[ALIGNMENT] Found {len(valid_samples)} matching samples in gene expression matrix.")

    binary_cohort = binary_cohort.loc[valid_samples]

    # Transpose expression so samples are rows and genes are columns
    X = expr_genes_df[valid_samples].T
    X.index.name = "gsm_id"
    X.columns.name = "Gene_Symbol"

    # Verify zero missing values in features
    n_nans = X.isna().sum().sum()
    if n_nans > 0:
        print(f"[WARNING] Found {n_nans} missing values in expression matrix. Imputing with column median.")
        X = X.fillna(X.median())
    else:
        print("[VERIFIED] Zero missing values detected in gene expression feature space.")

    # Clinical targets table
    targets_df = binary_cohort[[
        "recurrence_label",
        "rfs_time_months",
        "stage_numeric",
        "chemotherapy"
    ]].copy()

    # Integrity verification
    assert list(X.index) == list(targets_df.index), "Feature and Target sample alignment mismatch!"

    return X, targets_df


def run_preprocessing_pipeline() -> None:
    """Execute complete Phase 4 preprocessing pipeline."""
    print("=" * 70)
    print("CRC-ML PHASE 4: PROBE-TO-GENE MAPPING & PREPROCESSING")
    print("=" * 70)

    # 1. Load raw expression parquet
    print(f"[LOADING] Raw expression matrix: {EXPR_RAW_PARQUET}...")
    expr_df = pd.read_parquet(EXPR_RAW_PARQUET)

    # 2. Load platform annotation
    annot_df = load_gpl570_annotations(GPL570_ANNOT_PATH)

    # 3. Collapse probes to genes
    expr_genes_df = collapse_probes_to_genes(expr_df, annot_df)

    # 4. Save full gene expression parquet
    print(f"[EXPORT] Saving full gene-level matrix to: {GENES_EXPR_PARQUET}...")
    expr_genes_df.to_parquet(GENES_EXPR_PARQUET, engine="pyarrow", compression="snappy")
    print(f"[SAVED] {GENES_EXPR_PARQUET.name} ({GENES_EXPR_PARQUET.stat().st_size / (1024*1024):.2f} MB)")

    # 5. Align with binary classification cohort (n = 264)
    X, targets_df = align_ml_cohort(expr_genes_df)

    # 6. Export ML features and targets
    print(f"[EXPORT] Saving ML feature matrix (samples as rows, genes as columns): {ML_FEATURES_PARQUET}...")
    X.to_parquet(ML_FEATURES_PARQUET, engine="pyarrow", compression="snappy")

    print(f"[EXPORT] Saving ML targets: {ML_TARGETS_CSV}...")
    targets_df.to_csv(ML_TARGETS_CSV, index_label="gsm_id")

    class_0_n = (targets_df["recurrence_label"] == 0).sum()
    class_1_n = (targets_df["recurrence_label"] == 1).sum()

    print("\n" + "=" * 70)
    print("PREPROCESSING & COHORT ALIGNMENT SUMMARY")
    print("=" * 70)
    print(f"  Input Raw Probesets            : {expr_df.shape[0]:,}")
    print(f"  Mapped Unique Gene Symbols     : {expr_genes_df.shape[0]:,}")
    print(f"  Total Processed Samples        : {expr_genes_df.shape[1]:,}")
    print(f"  Final ML Cohort Samples        : {len(X)}")
    print(f"  ML Feature Space               : {X.shape[0]} samples x {X.shape[1]:,} genes")
    print(f"  Class 0 (Disease-Free >= 5 yr) : {class_0_n} ({class_0_n/len(X)*100:.1f}%)")
    print(f"  Class 1 (Early Recurrence <= 1 yr): {class_1_n} ({class_1_n/len(X)*100:.1f}%)")
    print(f"  Class Imbalance Ratio          : {class_0_n / class_1_n:.2f} : 1")
    print("=" * 70)


if __name__ == "__main__":
    run_preprocessing_pipeline()

