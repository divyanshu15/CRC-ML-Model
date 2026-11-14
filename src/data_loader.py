"""
Dataset Acquisition and Inspection Module for CRC-ML Pipeline.

Handles automated, reproducible downloading of GEO Series Matrix files,
sample metadata extraction, expression matrix parsing, and clinical data dictionary generation.
Optimized for Apple Silicon (macOS, 16 GB unified RAM).
"""

import gzip
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Standard NCBI GEO URLs
GEO_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series"
GPL_BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/platforms"


def get_series_matrix_url(accession: str) -> str:
    """Generate NCBI HTTPS URL for a GEO Series Matrix file."""
    # GEO series directories are grouped by prefix (e.g. GSE39582 -> GSE39nnn)
    prefix = accession[:-3] + "nnn" if len(accession) > 4 else "GSEnnn"
    return f"{GEO_BASE_URL}/{prefix}/{accession}/matrix/{accession}_series_matrix.txt.gz"


def get_platform_annotation_url(platform: str) -> str:
    """Generate NCBI HTTPS URL for a GPL Platform Annotation file."""
    prefix = platform[:-3] + "nnn" if len(platform) > 4 else "GPLnnn"
    return f"{GPL_BASE_URL}/{prefix}/{platform}/annot/{platform}.annot.gz"


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> Path:
    """
    Download a remote file via HTTP/HTTPS with progress reporting and caching.

    Parameters
    ----------
    url : str
        Source URL.
    dest_path : Path
        Local file path destination.
    chunk_size : int
        Chunk size in bytes (default 1 MB).

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 0:
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"[CACHE] {dest_path.name} already exists ({size_mb:.2f} MB). Skipping download.")
        return dest_path

    print(f"[DOWNLOAD] Fetching from: {url}")
    print(f"[TARGET] Saving to: {dest_path}")

    headers = {"User-Agent": "Mozilla/5.0 (CRC-ML Research Pipeline; Bioinformatics Bot)"}
    response = requests.get(url, stream=True, headers=headers, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Downloaded {dest_path.name} ({size_mb:.2f} MB).")
    return dest_path


def parse_series_matrix_metadata(matrix_gz_path: Path) -> pd.DataFrame:
    """
    Extract sample metadata from a GEO series matrix file without parsing the expression table.

    Parameters
    ----------
    matrix_gz_path : Path
        Path to the series matrix file (.txt.gz).

    Returns
    -------
    pd.DataFrame
        DataFrame where rows are sample accession IDs (GSM...) and columns are metadata attributes.
    """
    metadata_dict: Dict[str, List[str]] = {}
    sample_ids: List[str] = []

    with gzip.open(matrix_gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            if line.startswith("!Sample_"):
                parts = line.strip().split("\t")
                header = parts[0].lstrip("!")
                values = [p.strip('"') for p in parts[1:]]

                if header == "Sample_geo_accession":
                    sample_ids = values

                if header not in metadata_dict:
                    metadata_dict[header] = values
                else:
                    # Multi-line characteristics: append with index
                    idx = 1
                    while f"{header}_{idx}" in metadata_dict:
                        idx += 1
                    metadata_dict[f"{header}_{idx}"] = values

    if not sample_ids:
        raise ValueError("Could not find Sample_geo_accession in series matrix header.")

    meta_df = pd.DataFrame(metadata_dict)
    if "Sample_geo_accession" in meta_df.columns:
        meta_df.set_index("Sample_geo_accession", inplace=True)

    return meta_df


def parse_series_matrix_expression(
    matrix_gz_path: Path, max_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Parse the numeric expression matrix from a GEO series matrix file.

    Parameters
    ----------
    matrix_gz_path : Path
        Path to the series matrix file (.txt.gz).
    max_rows : Optional[int]
        Limit number of rows read (useful for rapid debugging).

    Returns
    -------
    pd.DataFrame
        DataFrame with probe IDs as index, sample GSM IDs as columns, dtype float32.
    """
    # First, locate the row index where expression data begins
    skiprows = 0
    with gzip.open(matrix_gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if line.startswith("!series_matrix_table_begin"):
                skiprows = i + 1
                break

    print(f"[PARSING] Reading expression table starting at line {skiprows}...")
    t0 = time.time()

    # Read expression table. The header is at line skiprows, data follows.
    expr_df = pd.read_csv(
        matrix_gz_path,
        sep="\t",
        skiprows=skiprows,
        compression="gzip",
        index_col=0,
        nrows=max_rows,
        comment="!",
    )

    # Clean index and column names
    expr_df.index.name = "ID_REF"
    expr_df.columns = [str(c).strip('"') for c in expr_df.columns]

    # Convert numeric values to float32 to reduce memory footprint by 50%
    expr_df = expr_df.astype(np.float32)

    elapsed = time.time() - t0
    mem_mb = expr_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(
        f"[PARSED] Expression matrix: {expr_df.shape[0]:,} probes x {expr_df.shape[1]:,} samples "
        f"in {elapsed:.2f}s (RAM: {mem_mb:.2f} MB)"
    )

    return expr_df


def inspect_clinical_variables(meta_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Inspect clinical metadata columns and identify recurrence/survival-related fields.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Raw sample metadata DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Cleaned clinical features DataFrame and list of recurrence-related variable names.
    """
    # Flatten characteristic fields: GEO often puts key:value in characteristics_ch1
    char_cols = [c for c in meta_df.columns if "characteristics" in c.lower()]

    clinical_dict: Dict[str, Dict[str, str]] = {}
    for sample_id, row in meta_df.iterrows():
        clinical_dict[sample_id] = {}
        for col in char_cols:
            val = str(row[col])
            if ":" in val:
                k, v = val.split(":", 1)
                k = k.strip().lower().replace(" ", "_")
                v = v.strip()
                clinical_dict[sample_id][k] = v
            else:
                clinical_dict[sample_id][col] = val

    clin_df = pd.DataFrame.from_dict(clinical_dict, orient="index")

    # Detect recurrence/survival candidate keywords
    recurrence_keywords = ["relapse", "recur", "rfs", "dfs", "event", "survival", "time", "stage", "chemo"]
    recurrence_vars = [
        col for col in clin_df.columns if any(kw in col for kw in recurrence_keywords)
    ]

    return clin_df, recurrence_vars


def generate_data_dictionary(
    meta_df: pd.DataFrame, clin_df: pd.DataFrame, output_md_path: Path
) -> None:
    """
    Generate reports/data_dictionary.md with detailed documentation of all clinical variables.
    """
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Data Dictionary: GSE39582 Discovery Cohort",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Cohort Samples:** {len(clin_df)}",
        "",
        "## 1. Overview",
        "This dictionary documents the clinical and pathological variables extracted from the GSE39582 Series Matrix.",
        "Variables are parsed from NCBI GEO `!Sample_characteristics_ch1` fields.",
        "",
        "## 2. Clinical and Pathological Variables",
        "",
        "| Variable Name | Extracted Key | Non-Null Count | Null Count | Data Type | Sample Values / Distinct Categories | Recurrence Relevance |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for col in sorted(clin_df.columns):
        series = clin_df[col].dropna()
        non_null = len(series)
        null_count = len(clin_df) - non_null
        unique_vals = series.unique()
        n_unique = len(unique_vals)

        if n_unique <= 6:
            val_summary = ", ".join(f"`{v}`" for v in unique_vals[:6])
        else:
            val_summary = f"{n_unique} unique values (e.g. `{unique_vals[0]}`, `{unique_vals[1]}`)"

        # Determine recurrence relevance
        is_recur = any(
            k in col.lower()
            for k in ["relapse", "recur", "rfs", "dfs", "event", "time", "stage", "chemo"]
        )
        recur_tag = "CRITICAL (Outcome / Stratification)" if is_recur else "Clinical Covariate"

        lines.append(
            f"| `{col}` | {col} | {non_null} | {null_count} | object | {val_summary} | {recur_tag} |"
        )

    lines.extend(
        [
            "",
            "## 3. Key Recurrence Variables Identified",
            "",
            "The following variables are critical for constructing the primary binary classification endpoint:",
            "- `relapse` / `recurrence`: Binary indicator of cancer return.",
            "- `rfs` / `relapse_free_survival` (or follow-up time): Observation time elapsed from primary resection.",
            "- `stage` / `tumor_stage`: Clinical stage (Stage I, II, III, IV). Stage IV (de novo metastatic) must be excluded from the recurrence prediction cohort as per PRD Section 2.1.",
            "- `chemotherapy`: Adjuvant chemotherapy status, potential confounder.",
            "",
            "## 4. Cohort Inclusion & Exclusion Strategy",
            "- **Inclusion:** Histologically confirmed Stage I, II, or III colorectal adenocarcinoma.",
            "- **Class 1 (High Early Recurrence):** Documented relapse within <= 12 months.",
            "- **Class 0 (Long-Term Disease-Free):** Documented no relapse with follow-up >= 60 months.",
            "- **Exclusion:** Missing relapse status, de novo stage IV, or intermediate follow-up without relapse.",
        ]
    )

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[REPORT] Data dictionary saved to: {output_md_path}")


def download_validation_cohort() -> Path:
    """Download independent external validation cohort (GSE14333)."""
    gse_accession = "GSE14333"
    gse_raw_dir = RAW_DIR / gse_accession
    gse_matrix_url = get_series_matrix_url(gse_accession)
    matrix_file = gse_raw_dir / f"{gse_accession}_series_matrix.txt.gz"
    return download_file(gse_matrix_url, matrix_file)


def download_metastasis_cohort() -> Tuple[Path, Path]:
    """Download paired metastasis RNA-seq cohort (GSE50760)."""
    gse_accession = "GSE50760"
    gse_raw_dir = RAW_DIR / gse_accession
    gse_matrix_url = get_series_matrix_url(gse_accession)
    matrix_file = gse_raw_dir / f"{gse_accession}_series_matrix.txt.gz"
    download_file(gse_matrix_url, matrix_file)

    # Supplementary expression raw tar (gene-level count/RPKM files, ~7.4 MB)
    suppl_url = f"{GEO_BASE_URL}/GSE50nnn/GSE50760/suppl/GSE50760_RAW.tar"
    suppl_file = gse_raw_dir / "GSE50760_RAW.tar"
    download_file(suppl_url, suppl_file)
    return matrix_file, suppl_file


def run_discovery_acquisition() -> Dict[str, str]:
    """
    Execute Phase 2 Initial Acquisition and Inspection of GSE39582.

    Returns
    -------
    Dict[str, str]
        Summary report dictionary.
    """
    print("=" * 70)
    print("CRC-ML PHASE 2: GSE39582 DATASET ACQUISITION & INSPECTION")
    print("=" * 70)

    # 1. Download Series Matrix
    gse_accession = "GSE39582"
    gse_raw_dir = RAW_DIR / gse_accession
    gse_matrix_url = get_series_matrix_url(gse_accession)
    matrix_file = gse_raw_dir / f"{gse_accession}_series_matrix.txt.gz"

    download_file(gse_matrix_url, matrix_file)

    # 2. Download GPL570 Annotation file
    gpl_platform = "GPL570"
    gpl_annot_url = get_platform_annotation_url(gpl_platform)
    annot_file = RAW_DIR / f"{gpl_platform}.annot.gz"
    download_file(gpl_annot_url, annot_file)

    # 3. Parse Metadata & Extract Clinical Variables
    print(f"\n[INSPECT] Parsing sample metadata from {matrix_file.name}...")
    meta_df = parse_series_matrix_metadata(matrix_file)
    clin_df, recurrence_vars = inspect_clinical_variables(meta_df)

    # Save raw clinical metadata
    clin_out = PROCESSED_DIR / "clinical" / f"{gse_accession}_clinical_raw.csv"
    clin_out.parent.mkdir(parents=True, exist_ok=True)
    clin_df.to_csv(clin_out, index_label="gsm_id")
    print(f"[SAVED] Clinical metadata table: {clin_out} ({len(clin_df)} samples)")

    # 4. Generate Data Dictionary
    dict_out = REPORTS_DIR / "data_dictionary.md"
    generate_data_dictionary(meta_df, clin_df, dict_out)

    # 5. Parse Expression Matrix (Full) and serialize to Parquet for high-speed downstream access
    parquet_out = PROCESSED_DIR / "expression" / f"{gse_accession}_expression.parquet"
    parquet_out.parent.mkdir(parents=True, exist_ok=True)

    if parquet_out.exists() and parquet_out.stat().st_size > 0:
        print(f"[CACHE] Parsed expression parquet exists: {parquet_out}")
        expr_df = pd.read_parquet(parquet_out)
    else:
        expr_df = parse_series_matrix_expression(matrix_file)
        print(f"[SAVING] Serializing expression matrix to Parquet: {parquet_out}...")
        expr_df.to_parquet(parquet_out, engine="pyarrow", compression="snappy")
        print(f"[SAVED] Parquet created ({parquet_out.stat().st_size / (1024*1024):.2f} MB)")

    # 6. Verify Normalization / Log2 Range
    sample_probe_vals = expr_df.iloc[:1000, :50].values.flatten()
    val_min = float(np.nanmin(sample_probe_vals))
    val_max = float(np.nanmax(sample_probe_vals))
    val_median = float(np.nanmedian(sample_probe_vals))
    is_log2 = (val_min >= 0.0) and (val_max <= 20.0)

    # 7. Compute Storage and Dimensions
    raw_size_mb = matrix_file.stat().st_size / (1024 * 1024)
    annot_size_mb = annot_file.stat().st_size / (1024 * 1024)
    parquet_size_mb = parquet_out.stat().st_size / (1024 * 1024)
    clinical_size_kb = clin_out.stat().st_size / 1024
    total_mb = raw_size_mb + annot_size_mb + parquet_size_mb + (clinical_size_kb / 1024)

    summary = {
        "accession": gse_accession,
        "platform": gpl_platform,
        "sample_count": str(expr_df.shape[1]),
        "probe_count": str(expr_df.shape[0]),
        "clinical_sample_count": str(len(clin_df)),
        "clinical_variables_count": str(len(clin_df.columns)),
        "recurrence_variables": ", ".join(recurrence_vars),
        "raw_series_matrix_size_mb": f"{raw_size_mb:.2f} MB",
        "gpl_annotation_size_mb": f"{annot_size_mb:.2f} MB",
        "parquet_expression_size_mb": f"{parquet_size_mb:.2f} MB",
        "total_storage_mb": f"{total_mb:.2f} MB",
        "log2_min": f"{val_min:.3f}",
        "log2_max": f"{val_max:.3f}",
        "log2_median": f"{val_median:.3f}",
        "is_log2_verified": str(is_log2),
    }

    print("\n" + "=" * 70)
    print("DATASET INSPECTION SUMMARY REPORT")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    run_discovery_acquisition()
