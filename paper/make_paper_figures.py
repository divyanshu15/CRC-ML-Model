"""
Build the paper-specific figures and copy the pipeline figures the paper uses into paper/figures/.

Run from the project root after the pipeline:  python paper/make_paper_figures.py
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
TABLES = ROOT / "results" / "tables"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

PIPELINE_FIGS = {
    "fig01_cohort_distribution.png": "fig_cohort.png",
    "fig06_volcano_plot.png": "fig_volcano.png",
    "fig09_roc_curves.png": "fig_internal_roc.png",
    "fig25_calibration_curves.png": "fig_calibration.png",
    "fig14_external_validation_roc.png": "fig_external_roc.png",
    "fig15_external_validation_km.png": "fig_external_km.png",
    "fig22_megacohort_roc_curves.png": "fig_loco_roc.png",
    "fig26_survival_model_km.png": "fig_survival_model_km.png",
    "fig16_metastasis_pca.png": "fig_metastasis_pca.png",
}


def copy_pipeline_figures() -> None:
    for src, dst in PIPELINE_FIGS.items():
        shutil.copy2(ROOT / "results" / "figures" / src, OUT / dst)


def study_design() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#334155", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#334155", lw=1.3))

    blue, green, amber, grey = "#DBEAFE", "#DCFCE7", "#FEF3C7", "#F1F5F9"
    box(0.2, 2.9, 3.2, 1.3, "GSE39582 (GPL570, n = 585)\nexclude 19 normal mucosa,\nStage IV / unknown stage", blue)
    box(0.2, 4.7, 3.2, 1.3, "Binary cohort: n = 249\n56 relapse <= 12 mo\n193 disease-free >= 60 mo", blue)
    box(0.2, 1.1, 3.2, 1.3, "Survival cohort: n = 485\nall Stage I-III tumours\n137 RFS events", blue)
    arrow(1.8, 4.2, 1.8, 4.7)
    arrow(1.8, 2.9, 1.8, 2.4)

    box(3.9, 2.9, 3.2, 3.1, "Repeated nested CV (5 x 5)\n\nin-fold: variance filter,\nWelch t-test / martingale screen,\nL1 selection, scaling,\nGridSearchCV, Youden threshold\n\nRF | SVM-lin | SVM-RBF | XGBoost\n(+ stage variants) ; XGBoost-Cox\n\nbaseline: TNM stage alone", green)
    arrow(3.4, 5.35, 3.9, 4.6)
    arrow(3.4, 1.75, 3.9, 3.2)

    box(7.6, 4.7, 3.2, 1.3, "GSE14333 (n = 226 Stage I-III)\nbinary n = 64; DFS survival", amber)
    box(7.6, 2.9, 3.2, 1.3, "GSE17536 (n = 135 Stage I-III)\nDFS survival only", amber)
    box(7.6, 1.1, 3.2, 1.3, "Leave-one-cohort-out (n = 356)\nGSE39582 + 14333 + 17536 + 17537", grey)
    arrow(7.1, 4.4, 7.6, 5.2)
    arrow(7.1, 3.9, 7.6, 3.5)
    arrow(7.1, 3.1, 7.6, 1.9)
    ax.text(9.2, 0.55, "Frozen models; label-free per-gene alignment\nto the discovery cohort", ha="center", fontsize=8, style="italic")
    ax.text(5.5, 0.55, "Secondary: TCGA-COADREAD (n = 57 curated);\nGSE50760 paired liver metastases (18 x 3)", ha="center", fontsize=8, style="italic")
    ax.set_title("Study design", fontweight="bold", fontsize=12)
    fig.savefig(OUT / "fig_study_design.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def forest_plot() -> None:
    df = pd.read_csv(TABLES / "external_survival_validation.csv")
    df = df[df["Model"] != "Stage_Only_Baseline"].copy()
    surv = pd.read_csv(TABLES / "survival_model_results.csv")
    surv = surv[surv["Model"].str.startswith("XGB_Cox") & surv["Cohort"].isin(["GSE14333", "GSE17536"])]
    surv = surv.rename(columns={"Model": "Model"})[["Cohort", "Model", "HR_per_SD_Stage_Adjusted",
                                                     "HR_CI95_Low", "HR_CI95_High", "Cox_p_Stage_Adjusted"]]
    df = pd.concat([df[surv.columns], surv], ignore_index=True)
    label = {
        "Random_Forest": "Random Forest", "SVM_Linear": "SVM (linear)", "SVM_RBF": "SVM (RBF)", "XGBoost": "XGBoost",
        "Random_Forest_plus_Stage": "Random Forest + stage", "SVM_Linear_plus_Stage": "SVM (linear) + stage",
        "SVM_RBF_plus_Stage": "SVM (RBF) + stage", "XGBoost_plus_Stage": "XGBoost + stage",
        "XGB_Cox_Genes": "XGBoost-Cox (survival)", "XGB_Cox_Genes_plus_Stage": "XGBoost-Cox + stage (survival)",
    }
    order = list(label)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    for ax, cohort, color in zip(axes, ["GSE14333", "GSE17536"], ["#1f77b4", "#d62728"]):
        sub = df[df["Cohort"] == cohort].set_index("Model").reindex(order)
        y = np.arange(len(order))[::-1]
        for yi, (m, r) in zip(y, sub.iterrows()):
            c = color if not m.startswith("XGB_Cox") else "#6b7280"
            ax.errorbar(r["HR_per_SD_Stage_Adjusted"], yi,
                        xerr=[[r["HR_per_SD_Stage_Adjusted"] - r["HR_CI95_Low"]], [r["HR_CI95_High"] - r["HR_per_SD_Stage_Adjusted"]]],
                        fmt="s", color=c, ecolor=c, capsize=3, ms=6)
            ax.text(2.55, yi, f"p={r['Cox_p_Stage_Adjusted']:.3f}", va="center", fontsize=8)
        ax.axvline(1, color="black", ls="--", lw=1)
        ax.axhline(1.5, color="#9ca3af", lw=0.8)
        ax.set_xlim(0.6, 2.85)
        ax.set_yticks(y)
        ax.set_yticklabels([label[m] for m in order])
        n = {"GSE14333": "n = 226, 50 events", "GSE17536": "n = 135, 32 events"}[cohort]
        ax.set_title(f"{cohort} ({n})", fontweight="bold")
        ax.set_xlabel("Stage-adjusted hazard ratio per SD of risk score (95% CI)")
    fig.suptitle("External disease-free survival: risk score added to TNM stage", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "fig_forest_survival.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    copy_pipeline_figures()
    study_design()
    forest_plot()
    print(f"Figures written to {OUT}")
