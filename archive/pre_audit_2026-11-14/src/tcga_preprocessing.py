"""
TCGA-COADREAD Data Preprocessing & Cohort Curation Pipeline (Phase 2).
Processes RNA-seq gene expression matrices, curated clinical phenotypes,
and progression-free survival records for large-scale recurrence modeling.
"""

from pathlib import Path
import gzip
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "TCGA_COADREAD"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "tcga"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_stage_category(val: str) -> str:
    """Classify pathologic stage into standard Roman numeral categories."""
    if pd.isna(val):
        return "Unknown"
    s = str(val).upper().strip()
    if "STAGE IV" in s:
        return "Stage IV"
    elif "STAGE III" in s:
        return "Stage III"
    elif "STAGE II" in s:
        return "Stage II"
    elif "STAGE I" in s:
        return "Stage I"
    return "Other"


def process_tcga_dataset():
    """
    Load, harmonize, filter, and export TCGA-COAD and TCGA-READ datasets.
    """
    print("=" * 70)
    print("PHASE 2: TCGA-COADREAD LARGE-SCALE DATASET PREPROCESSING")
    print("=" * 70)

    # 1. Load Expression Data
    print("\n[EXPR] Loading TCGA-COAD RNA-seq expression matrix...")
    coad_expr = pd.read_csv(RAW_DIR / "COAD_HiSeqV2.gz", sep="\t", index_col=0)
    print(f"  COAD shape: {coad_expr.shape[0]} genes x {coad_expr.shape[1]} samples")

    print("[EXPR] Loading TCGA-READ RNA-seq expression matrix...")
    read_expr = pd.read_csv(RAW_DIR / "READ_HiSeqV2.gz", sep="\t", index_col=0)
    print(f"  READ shape: {read_expr.shape[0]} genes x {read_expr.shape[1]} samples")

    # Combine expression across samples
    full_expr = pd.concat([coad_expr, read_expr], axis=1)
    full_expr = full_expr.loc[:, ~full_expr.columns.duplicated()]
    print(f"[EXPR] Combined expression: {full_expr.shape[0]} genes x {full_expr.shape[1]} samples")

    # 2. Load Clinical & Survival Metadata
    print("\n[CLIN] Loading TCGA survival records (PFI, DFI, OS)...")
    surv_coad = pd.read_csv(RAW_DIR / "COAD_survival.txt", sep="\t")
    surv_read = pd.read_csv(RAW_DIR / "READ_survival.txt", sep="\t")
    surv = pd.concat([surv_coad, surv_read], ignore_index=True).drop_duplicates(subset=["sample"]).set_index("sample")

    print("[CLIN] Loading TCGA clinical phenotypes...")
    clin_coad = pd.read_csv(RAW_DIR / "COAD_clinicalMatrix", sep="\t")
    clin_read = pd.read_csv(RAW_DIR / "READ_clinicalMatrix", sep="\t")
    clin = pd.concat([clin_coad, clin_read], ignore_index=True).drop_duplicates(subset=["sampleID"]).set_index("sampleID")

    # Merge clinical and survival
    meta = surv.join(clin, how="inner", lsuffix="_surv", rsuffix="_clin")
    print(f"[CLIN] Merged metadata: {len(meta)} records")

    # 3. Filter Primary Solid Tumors (-01) with available RNA-seq
    meta_primary = meta[meta.index.str.contains(r"-01[A-Z]?$", regex=True)].copy()
    available_samples = [s for s in meta_primary.index if s in full_expr.columns]
    print(f"[FILTER] Primary tumors with RNA-seq expression: {len(available_samples)}")

    meta_sub = meta_primary.loc[available_samples].copy()

    # 4. Filter Stage I-III (Exclude de novo metastatic Stage IV)
    meta_sub["stage_clean"] = meta_sub["pathologic_stage"].apply(parse_stage_category)
    stage_1_3_mask = meta_sub["stage_clean"].isin(["Stage I", "Stage II", "Stage III"])
    cohort_clin = meta_sub[stage_1_3_mask].copy()

    print(f"[FILTER] Eligible Stage I–III CRC cohort: {len(cohort_clin)} patients")
    print("  Stage distribution:\n", cohort_clin["stage_clean"].value_counts().to_string())

    # 5. Define Clinical Endpoints
    # PFI (Progression-Free Interval) is the standard recurrence/progression endpoint
    cohort_clin["recurrence_label"] = cohort_clin["PFI"].astype(int)
    cohort_clin["pfi_time_days"] = pd.to_numeric(cohort_clin["PFI.time"], errors="coerce")
    cohort_clin["pfi_time_months"] = cohort_clin["pfi_time_days"] / 30.4375

    n_events = (cohort_clin["recurrence_label"] == 1).sum()
    n_free = (cohort_clin["recurrence_label"] == 0).sum()
    print(f"\n[ENDPOINT] Progression-Free / Recurrence Events: {n_events} ({n_events/len(cohort_clin)*100:.1f}%)")
    print(f"[ENDPOINT] Recurrence-Free Follow-up: {n_free} ({n_free/len(cohort_clin)*100:.1f}%)")

    # Align expression matrix
    cohort_samples = list(cohort_clin.index)
    X = full_expr[cohort_samples].T
    X.index.name = "sample_id"
    X.columns.name = "Gene_Symbol"

    # Convert to float32
    X = X.astype(np.float32)
    # Impute any missing gene values with column mean
    if X.isna().sum().sum() > 0:
        print("[IMPUTE] Imputing missing values with column mean...")
        X = X.fillna(X.mean())

    print(f"\n[FEATURES] Final ML Matrix: {X.shape[0]} patients x {X.shape[1]} genes (float32)")

    # 6. Export Parquet and CSV
    parquet_path = PROCESSED_DIR / "TCGA_COADREAD_features.parquet"
    clinical_path = PROCESSED_DIR / "TCGA_COADREAD_clinical.csv"

    X.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
    cohort_clin.to_csv(clinical_path, index_label="sample_id")

    print(f"[SAVED] Features: {parquet_path} ({parquet_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"[SAVED] Clinical: {clinical_path} ({clinical_path.stat().st_size / 1024:.2f} KB)")
    print("[SUCCESS] Phase 2 Preprocessing Complete.")


if __name__ == "__main__":
    process_tcga_dataset()

