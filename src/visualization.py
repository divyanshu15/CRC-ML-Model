"""
Visualization Module for CRC-ML Pipeline.

Generates publication-quality figures (300 DPI PNG and vector PDF)
for exploratory data analysis, clinical cohort characterization,
and genomic expression quality control.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set standardized publication aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 13


def save_figure(fig: plt.Figure, name: str, dpi: int = 300) -> None:
    """Save figure in both high-res PNG and vector PDF formats."""
    png_path = FIGURES_DIR / f"{name}.png"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[FIGURE] Saved: {png_path.name} & {pdf_path.name}")


def plot_cohort_distribution(labeled_df: pd.DataFrame) -> None:
    """
    Figure 1: Complete patient cohort breakdown showing inclusion/exclusion accounting
    and class balance of the final binary machine learning cohort.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Total Sample Status Breakdown
    eligibility_counts = labeled_df["exclusion_reason"].value_counts()
    # Combine clean class labels for display
    plot_series = pd.Series({
        "Class 0 (Disease-Free >= 5 yr)": (labeled_df["recurrence_label"] == 0).sum(),
        "Class 1 (Early Recurrence <= 1 yr)": (labeled_df["recurrence_label"] == 1).sum(),
        "Inadequate Follow-up (< 5 yr)": labeled_df["exclusion_reason"].str.contains("Inadequate").sum(),
        "Late Recurrence (> 1 yr)": labeled_df["exclusion_reason"].str.contains("Late Recurrence").sum(),
        "Stage IV (De Novo Metastatic)": (labeled_df["exclusion_reason"] == "Stage IV (De Novo Metastatic)").sum(),
        "Missing Relapse / Other Stage": labeled_df["exclusion_reason"].isin([
            "Missing Relapse Status", "Missing Follow-up Duration"
        ]).sum() + labeled_df["exclusion_reason"].str.contains("Non-I-III").sum(),
    })

    colors_bar = ["#2ca02c", "#d62728", "#aec7e8", "#ffbb78", "#9467bd", "#c7c7c7"]
    bars = axes[0].barh(plot_series.index, plot_series.values, color=colors_bar, edgecolor="black", height=0.65)
    axes[0].set_title("GSE39582 Cohort Curation & Eligibility (N = 585)", fontweight="bold")
    axes[0].set_xlabel("Number of Samples")
    axes[0].invert_yaxis()

    for bar in bars:
        w = bar.get_width()
        axes[0].text(w + 3, bar.get_y() + bar.get_height() / 2, f"{int(w)} ({w/len(labeled_df)*100:.1f}%)",
                     va="center", fontsize=9, fontweight="bold")
    axes[0].set_xlim(0, max(plot_series.values) + 35)

    # 2. Binary Class Proportions
    binary_df = labeled_df.dropna(subset=["recurrence_label"])
    class_counts = binary_df["recurrence_label"].value_counts().sort_index()
    class_labels = [f"Class 0: Disease-Free\n(n = {int(class_counts[0])}, {class_counts[0]/len(binary_df)*100:.1f}%)",
                    f"Class 1: Early Recurrence\n(n = {int(class_counts[1])}, {class_counts[1]/len(binary_df)*100:.1f}%)"]
    axes[1].pie(class_counts, labels=class_labels, autopct="%1.1f%%", startangle=140,
                colors=["#2ca02c", "#d62728"], explode=(0, 0.08),
                wedgeprops={"edgecolor": "black", "linewidth": 1.2})
    axes[1].set_title(f"Primary Binary Classification Cohort\n(N = {len(binary_df)} Resected Stage I-III Patients)",
                      fontweight="bold")

    plt.tight_layout()
    save_figure(fig, "fig01_cohort_distribution")
    plt.close(fig)


def plot_stage_and_demographics(clinical_df: pd.DataFrame, labeled_df: pd.DataFrame) -> None:
    """
    Figure 2: Clinical stage distribution and demographics across the cohort.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. TNM Stage in Stage I-III vs Outcome
    binary_df = labeled_df.dropna(subset=["recurrence_label"]).copy()
    binary_df["Stage"] = "Stage " + binary_df["stage_numeric"].astype(int).astype(str)
    binary_df["Outcome"] = binary_df["recurrence_label"].map({0.0: "Disease-Free", 1.0: "Early Recurrence"})

    stage_order = ["Stage 1", "Stage 2", "Stage 3"]
    sns.countplot(data=binary_df, x="Stage", hue="Outcome", order=stage_order,
                  palette={"Disease-Free": "#2ca02c", "Early Recurrence": "#d62728"},
                  edgecolor="black", ax=axes[0])
    axes[0].set_title("Recurrence by Clinical Stage (Stage I-III)", fontweight="bold")
    axes[0].set_ylabel("Patient Count")
    axes[0].set_xlabel("")
    axes[0].legend(title="Clinical Outcome", frameon=True)

    # 2. Age distribution
    age_col = next((c for c in clinical_df.columns if "age" in c.lower()), None)
    if age_col:
        ages = pd.to_numeric(clinical_df[age_col], errors="coerce").dropna()
        sns.histplot(ages, kde=True, bins=25, color="#1f77b4", edgecolor="black", ax=axes[1])
        axes[1].axvline(ages.median(), color="red", linestyle="--", linewidth=1.5,
                        label=f"Median Age: {ages.median():.1f} yrs")
        axes[1].set_title(f"Patient Age Distribution (n = {len(ages)})", fontweight="bold")
        axes[1].set_xlabel("Age at Diagnosis (Years)")
        axes[1].set_ylabel("Patient Count")
        axes[1].legend(frameon=True)

    # 3. Patient Sex Distribution
    sex_col = next((c for c in clinical_df.columns if "sex" in c.lower()), None)
    if sex_col:
        sex_counts = clinical_df[sex_col].value_counts()
        axes[2].pie(sex_counts, labels=[f"{k} (n={v})" for k, v in sex_counts.items()],
                    autopct="%1.1f%%", colors=["#4C72B0", "#DD8452"], startangle=90,
                    wedgeprops={"edgecolor": "black", "linewidth": 1.2})
        axes[2].set_title("Patient Sex Distribution", fontweight="bold")

    plt.tight_layout()
    save_figure(fig, "fig02_stage_and_demographics")
    plt.close(fig)


def plot_survival_and_recurrence_timelines(labeled_df: pd.DataFrame) -> None:
    """
    Figure 3: Relapse-free survival (RFS) time distributions illustrating the operational
    distinction between early recurrence (<= 1 yr) and long-term follow-up (>= 5 yr).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    # 1. Histogram of All RFS Delays in Months
    times = labeled_df["rfs_time_months"].dropna()
    sns.histplot(times, bins=40, kde=True, color="#34495e", edgecolor="black", ax=axes[0])
    axes[0].axvline(12, color="#d62728", linestyle="--", linewidth=2, label="1-Year Boundary (12 Months)")
    axes[0].axvline(60, color="#2ca02c", linestyle="--", linewidth=2, label="5-Year Disease-Free Boundary (60 Months)")
    axes[0].set_title("Distribution of Relapse-Free Survival Follow-up Duration", fontweight="bold")
    axes[0].set_xlabel("Follow-up / Event Time (Months)")
    axes[0].set_ylabel("Patient Count")
    axes[0].legend(frameon=True, loc="upper right")

    # 2. Cumulative Recurrence Events Timeline
    relapse_cases = labeled_df[labeled_df["relapse_event"] == 1.0].sort_values(by="rfs_time_months")
    relapse_times = relapse_cases["rfs_time_months"].dropna().values
    cumulative_pct = np.arange(1, len(relapse_times) + 1) / len(relapse_times) * 100

    axes[1].plot(relapse_times, cumulative_pct, color="#d62728", linewidth=2.5, label="Cumulative Recurrences (%)")
    axes[1].axvline(12, color="black", linestyle=":", linewidth=1.5)
    pct_within_1yr = (relapse_times <= 12).sum() / len(relapse_times) * 100
    axes[1].plot(12, pct_within_1yr, "ro", markersize=8)
    axes[1].text(14, pct_within_1yr - 5, f"{pct_within_1yr:.1f}% Recurrences\nwithin 1 Year",
                 fontsize=10, fontweight="bold", color="#d62728")

    axes[1].set_title("Cumulative Recurrence Timing Post-Resection", fontweight="bold")
    axes[1].set_xlabel("Time from Primary Resection (Months)")
    axes[1].set_ylabel("Cumulative Recurrences (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_xlim(0, max(relapse_times) + 5)
    axes[1].legend(frameon=True, loc="lower right")

    plt.tight_layout()
    save_figure(fig, "fig03_survival_and_recurrence_timelines")
    plt.close(fig)


def plot_clinical_missingness(clinical_df: pd.DataFrame) -> None:
    """
    Figure 4: Audit of missing clinical attributes across all extracted fields.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    missing_pct = (clinical_df.isnull().sum() / len(clinical_df) * 100).sort_values(ascending=False)
    # Filter to columns that have either some missingness or are primary covariates
    missing_pct = missing_pct.iloc[:22]

    colors = ["#e74c3c" if p > 30 else "#f39c12" if p > 0 else "#27ae60" for p in missing_pct.values]
    bars = ax.barh(missing_pct.index, missing_pct.values, color=colors, edgecolor="black", height=0.65)
    ax.set_title("Missing Data Audit Across GSE39582 Clinical Variables (N = 585)", fontweight="bold")
    ax.set_xlabel("Missingness Percentage (%)")
    ax.invert_yaxis()

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, max(missing_pct.values) + 12)

    plt.tight_layout()
    save_figure(fig, "fig04_clinical_missingness")
    plt.close(fig)


def plot_expression_distributions(expr_df: pd.DataFrame) -> None:
    """
    Figure 5: Gene expression intensity density distribution and sample boxplots,
    confirming RMA log2 scale stability without sample outliers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Overall Density Distribution (Sample of 5,000 probes x 30 samples)
    subset_vals = expr_df.iloc[:5000, :30].values.flatten()
    sns.kdeplot(subset_vals, color="#2980b9", fill=True, alpha=0.3, linewidth=2, ax=axes[0])
    axes[0].set_title("Gene Expression Intensity Distribution (RMA Log2 Verified)", fontweight="bold")
    axes[0].set_xlabel("Log2 Expression Intensity")
    axes[0].set_ylabel("Kernel Density")
    axes[0].axvline(np.median(subset_vals), color="red", linestyle="--", linewidth=1.5,
                    label=f"Median Intensity: {np.median(subset_vals):.2f}")
    axes[0].legend(frameon=True)

    # 2. Boxplot across 20 Representative Patient Samples
    sample_sub = expr_df.iloc[::20, :20]  # Every 20th probe across first 20 samples
    sample_sub.plot(kind="box", ax=axes[1], flierprops={"marker": ".", "markersize": 1, "alpha": 0.2},
                    patch_artist=True, boxprops={"facecolor": "#d4e6f1", "edgecolor": "#1b4f72"})
    axes[1].set_title("Intensity Boxplots Across Representative Samples", fontweight="bold")
    axes[1].set_xlabel("Sample GSM IDs")
    axes[1].set_ylabel("Log2 Expression Intensity")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90, fontsize=8)

    plt.tight_layout()
    save_figure(fig, "fig05_expression_qc_distribution")
    plt.close(fig)


def run_eda_visualizations() -> None:
    """Execute all Phase 3 exploratory visualizations."""
    print("=" * 70)
    print("CRC-ML PHASE 3: GENERATING PUBLICATION EDA FIGURES")
    print("=" * 70)

    clinical_raw_path = PROJECT_ROOT / "data" / "processed" / "clinical" / "GSE39582_clinical_raw.csv"
    cohort_labeled_path = PROJECT_ROOT / "data" / "processed" / "clinical" / "GSE39582_cohort_labeled.csv"
    expr_parquet_path = PROJECT_ROOT / "data" / "processed" / "expression" / "GSE39582_expression.parquet"

    clinical_df = pd.read_csv(clinical_raw_path, index_col="gsm_id")
    labeled_df = pd.read_csv(cohort_labeled_path, index_col="gsm_id")

    print("[EDA] Plotting Figure 1: Cohort Distribution & Class Proportions...")
    plot_cohort_distribution(labeled_df)

    print("[EDA] Plotting Figure 2: Stage & Demographic Breakdown...")
    plot_stage_and_demographics(clinical_df, labeled_df)

    print("[EDA] Plotting Figure 3: Survival & Recurrence Event Timelines...")
    plot_survival_and_recurrence_timelines(labeled_df)

    print("[EDA] Plotting Figure 4: Clinical Missingness Audit...")
    plot_clinical_missingness(clinical_df)

    print("[EDA] Loading expression parquet for Quality Control plots...")
    expr_df = pd.read_parquet(expr_parquet_path)
    print("[EDA] Plotting Figure 5: Expression QC Density & Boxplots...")
    plot_expression_distributions(expr_df)

    print("\n[SUCCESS] All Phase 3 exploratory figures generated in results/figures/.")


if __name__ == "__main__":
    run_eda_visualizations()

