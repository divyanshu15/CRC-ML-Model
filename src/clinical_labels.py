"""
Clinical Recurrence Label Engine for CRC-ML Pipeline.

Implements strict, scientifically defensible operational definitions for early colorectal cancer recurrence:
- Class 1 (High Early Recurrence Risk): Documented recurrence within <= 12 months (<= 365 days).
- Class 0 (Long-Term Disease-Free): Confirmed disease-free follow-up for >= 60 months (>= 5 years).
- Ambiguous / Censored Cases: Excluded to prevent label noise and artificial bias.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CLINICAL_RAW_PATH = DATA_DIR / "processed" / "clinical" / "GSE39582_clinical_raw.csv"
CLINICAL_PROCESSED_PATH = DATA_DIR / "processed" / "clinical" / "GSE39582_cohort_labeled.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def standardize_clinical_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize variable names and data types in the extracted clinical metadata.
    Specifically parses:
    - tnm.stage -> stage_clean (1, 2, 3, 4)
    - rfs.event -> relapse_event (0 = No, 1 = Yes)
    - rfs.delay -> rfs_time_months (continuous months)
    - chemotherapy.adjuvant -> chemotherapy (yes/no)
    """
    df = raw_df.copy()

    # Stage parsing
    if "tnm.stage" in df.columns:
        df["stage_numeric"] = pd.to_numeric(df["tnm.stage"], errors="coerce")
    elif "stage" in df.columns:
        df["stage_numeric"] = pd.to_numeric(df["stage"], errors="coerce")
    else:
        df["stage_numeric"] = np.nan

    # Relapse event parsing
    if "rfs.event" in df.columns:
        df["relapse_event"] = pd.to_numeric(df["rfs.event"], errors="coerce")
    elif "relapse" in df.columns:
        df["relapse_event"] = pd.to_numeric(df["relapse"], errors="coerce")
    else:
        df["relapse_event"] = np.nan

    # RFS delay in months
    if "rfs.delay" in df.columns:
        df["rfs_time_months"] = pd.to_numeric(df["rfs.delay"], errors="coerce")
    elif "rfs" in df.columns:
        df["rfs_time_months"] = pd.to_numeric(df["rfs"], errors="coerce")
    else:
        df["rfs_time_months"] = np.nan

    # Chemotherapy adjuvant status
    if "chemotherapy.adjuvant" in df.columns:
        df["chemotherapy"] = df["chemotherapy.adjuvant"].astype(str).str.strip().str.lower()
    else:
        df["chemotherapy"] = "unknown"

    return df


def build_recurrence_cohort(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply PRD Section 2.1 and Module 4 inclusion/exclusion logic:
    - Histologically confirmed Stage I, II, or III colorectal cancer.
    - Class 1: relapse_event == 1 and rfs_time_months <= 12.0
    - Class 0: relapse_event == 0 and rfs_time_months >= 60.0
    - Ambiguous cases excluded with precise clinical documentation.
    """
    df = df.copy()
    df["cohort_eligibility"] = "Eligible"
    df["exclusion_reason"] = "None"
    df["recurrence_label"] = np.nan

    exclusion_counts = {
        "Total Cohort Samples": len(df),
        "Non-Tumoral (Normal Mucosa) Sample": 0,
        "Stage IV (De Novo Metastatic)": 0,
        "Unknown or Non-I-III Stage": 0,
        "Stage I-III Subcohort": 0,
        "Missing Relapse Status": 0,
        "Missing Follow-up Duration": 0,
        "Late Recurrence (> 12 months)": 0,
        "Inadequate Disease-Free Follow-up (< 60 months)": 0,
        "Eligible Class 0 (Disease-Free >= 5 yrs)": 0,
        "Eligible Class 1 (Early Recurrence <= 1 yr)": 0,
        "Total Final Binary Classification Cohort": 0,
    }

    for idx, row in df.iterrows():
        # GSE39582 includes 19 non-tumoral mucosa samples taken from patients whose tumours are
        # also in the series (linked via `dependancy_sample`). They are not independent patients
        # and are not tumours, so they must never enter the recurrence cohort.
        if str(row.get("dataset", "")).strip().lower() == "non tumoral":
            df.loc[idx, "cohort_eligibility"] = "Excluded"
            df.loc[idx, "exclusion_reason"] = "Non-Tumoral (Normal Mucosa) Sample"
            exclusion_counts["Non-Tumoral (Normal Mucosa) Sample"] += 1
            continue

        stage = row.get("stage_numeric")

        # Stage Check
        if stage == 4.0:
            df.loc[idx, "cohort_eligibility"] = "Excluded"
            df.loc[idx, "exclusion_reason"] = "Stage IV (De Novo Metastatic)"
            exclusion_counts["Stage IV (De Novo Metastatic)"] += 1
            continue
        elif stage not in [1.0, 2.0, 3.0]:
            df.loc[idx, "cohort_eligibility"] = "Excluded"
            df.loc[idx, "exclusion_reason"] = f"Unknown or Non-I-III Stage ({stage})"
            exclusion_counts["Unknown or Non-I-III Stage"] += 1
            continue

        exclusion_counts["Stage I-III Subcohort"] += 1

        relapse = row.get("relapse_event")
        time_mo = row.get("rfs_time_months")

        if pd.isna(relapse):
            df.loc[idx, "cohort_eligibility"] = "Excluded"
            df.loc[idx, "exclusion_reason"] = "Missing Relapse Status"
            exclusion_counts["Missing Relapse Status"] += 1
            continue

        if pd.isna(time_mo):
            df.loc[idx, "cohort_eligibility"] = "Excluded"
            df.loc[idx, "exclusion_reason"] = "Missing Follow-up Duration"
            exclusion_counts["Missing Follow-up Duration"] += 1
            continue

        # Outcome Assignment
        if relapse == 1.0:
            if time_mo <= 12.0:
                df.loc[idx, "recurrence_label"] = 1.0
                exclusion_counts["Eligible Class 1 (Early Recurrence <= 1 yr)"] += 1
            else:
                df.loc[idx, "cohort_eligibility"] = "Excluded"
                df.loc[idx, "exclusion_reason"] = f"Late Recurrence at {time_mo:.1f} mo (> 12 mo)"
                exclusion_counts["Late Recurrence (> 12 months)"] += 1
        elif relapse == 0.0:
            if time_mo >= 60.0:
                df.loc[idx, "recurrence_label"] = 0.0
                exclusion_counts["Eligible Class 0 (Disease-Free >= 5 yrs)"] += 1
            else:
                df.loc[idx, "cohort_eligibility"] = "Excluded"
                df.loc[idx, "exclusion_reason"] = f"Inadequate Follow-up at {time_mo:.1f} mo (< 60 mo)"
                exclusion_counts["Inadequate Disease-Free Follow-up (< 60 months)"] += 1

    exclusion_counts["Total Final Binary Classification Cohort"] = (
        exclusion_counts["Eligible Class 0 (Disease-Free >= 5 yrs)"]
        + exclusion_counts["Eligible Class 1 (Early Recurrence <= 1 yr)"]
    )

    return df, exclusion_counts


def process_clinical_labels() -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Execute clinical standardization, cohort filtering, and export."""
    if not CLINICAL_RAW_PATH.exists():
        raise FileNotFoundError(f"Raw clinical data not found at {CLINICAL_RAW_PATH}. Run src/data_loader.py first.")

    raw_df = pd.read_csv(CLINICAL_RAW_PATH, index_col="gsm_id")
    standard_df = standardize_clinical_dataframe(raw_df)
    labeled_df, counts = build_recurrence_cohort(standard_df)

    CLINICAL_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(CLINICAL_PROCESSED_PATH, index_label="gsm_id")
    print(f"[EXPORT] Cohort labels saved to: {CLINICAL_PROCESSED_PATH}")

    return labeled_df, counts


if __name__ == "__main__":
    df, stats = process_clinical_labels()
    print("\n--- Cohort Filtering Summary ---")
    for k, v in stats.items():
        print(f"  {k:50s}: {v}")
