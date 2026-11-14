"""
Colorectal Cancer (CRC) Recurrence & Metastasis Biomarker Explorer
Interactive Research Interface for Genomic Risk Stratification.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Ensure project root is in sys.path so custom pipeline transformers unpickle correctly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_selection import BioinformaticsFeatureSelector

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="CRC Recurrence & Metastasis Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Medical Clinical Styling (Off-White Palette)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global off-white medical palette - enforce across all modes */
    :root, [data-theme="dark"], [data-theme="light"], .stApp {
        --background-color: #F8FAFC !important;
        --secondary-background-color: #F1F5F9 !important;
        --text-color: #0F172A !important;
        --primary-color: #0284C7 !important;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    /* Sidebar off-white clinical background */
    [data-testid="stSidebar"] {
        background-color: #F1F5F9 !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #0F172A !important;
    }

    /* Remove floating translucent top header bar and decoration strip completely */
    [data-testid="stHeader"], header[data-testid="stHeader"], [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Clean layout starting right at the top without any obstruction */
    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }

    /* Prominent Dropdown / Selectbox Border & Highlight */
    div[data-testid="stSelectbox"] > div > div,
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 2px solid #64748B !important; /* Prominent dark-slate border */
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08) !important;
        min-height: 42px !important;
        transition: all 0.2s ease-in-out !important;
    }

    /* Hover state: highlight border in clinical blue */
    div[data-testid="stSelectbox"] > div > div:hover,
    div[data-baseweb="select"] > div:hover {
        border-color: #0284C7 !important;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.2) !important;
    }

    /* Focus-within state: prominent active ring */
    div[data-testid="stSelectbox"] > div > div:focus-within,
    div[data-baseweb="select"]:focus-within > div {
        border-color: #0284C7 !important;
        box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.25) !important;
    }

    /* Dropdown text */
    div[data-testid="stSelectbox"] [data-baseweb="select"] * {
        color: #0F172A !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Dropdown labels */
    div[data-testid="stSelectbox"] label p {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.01em !important;
        margin-bottom: 6px !important;
    }

    /* Dropdown menu popover options */
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #FFFFFF !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15) !important;
    }
    li[role="option"] {
        color: #0F172A !important;
        padding: 10px 14px !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background-color: #E0F2FE !important;
        color: #0369A1 !important;
        font-weight: 600 !important;
    }

    /* Text search inputs */
    div[data-testid="stTextInput"] input {
        background-color: #FFFFFF !important;
        border: 2px solid #64748B !important;
        border-radius: 8px !important;
        color: #0F172A !important;
        font-weight: 500 !important;
        padding: 10px 14px !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #0284C7 !important;
        box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.25) !important;
    }

    /* Clinical card container for metrics */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05) !important;
        min-height: 110px !important;
    }

    /* Metric values: prevent ellipsis truncation and wrap cleanly */
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] * {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
        color: #0F172A !important;
    }

    /* Metric labels: prevent ellipsis and ensure readability */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] * {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        color: #475569 !important;
        font-size: 0.825rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.025em !important;
        line-height: 1.3 !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* Clean clinical tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px !important;
        padding: 8px 16px !important;
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        color: #475569 !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border-color: #0284C7 !important;
    }

    /* Informational banners */
    div[data-testid="stAlert"] {
        border-radius: 8px !important;
        border: 1px solid #BAE6FD !important;
        background-color: #F0F9FF !important;
        color: #0369A1 !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #0284C7 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
BIOMARKERS_DIR = RESULTS_DIR / "biomarkers"


def show_figure(filename: str, caption: str):
    """Safely render an image with graceful fallback if file is missing."""
    p = FIGURES_DIR / filename
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"Figure '{filename}' is not available.")


# ---------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    """Load Phase 1 serialized classification pipelines."""
    models = {}
    model_configs = {
        "Linear SVM": MODELS_DIR / "svm_linear" / "SVM_Linear_model.joblib",
        "XGBoost": MODELS_DIR / "xgboost" / "XGBoost_model.joblib",
        "Random Forest": MODELS_DIR / "random_forest" / "Random_Forest_model.joblib",
        "RBF SVM": MODELS_DIR / "svm_rbf" / "SVM_RBF_model.joblib",
    }
    for name, path in model_configs.items():
        if path.exists():
            models[name] = joblib.load(path)
    return models


@st.cache_resource
def load_tcga_models():
    """Load Phase 2 TCGA-COADREAD classification pipelines."""
    tcga_dir = PROJECT_ROOT / "models" / "phase2_tcga"
    models = {}
    model_configs = {
        "Linear SVM": tcga_dir / "SVM_Linear_tcga.joblib",
        "XGBoost": tcga_dir / "XGBoost_tcga.joblib",
        "Random Forest": tcga_dir / "Random_Forest_tcga.joblib",
        "RBF SVM": tcga_dir / "SVM_RBF_tcga.joblib",
    }
    for name, path in model_configs.items():
        if path.exists():
            models[name] = joblib.load(path)
    return models


@st.cache_resource
def load_megacohort_models():
    """Load Phase 3 Combined GPL570 Mega-Cohort classification pipelines."""
    mega_dir = PROJECT_ROOT / "models" / "megacohort"
    models = {}
    model_configs = {
        "Random Forest": mega_dir / "Random_Forest_megacohort.joblib",
        "Linear SVM": mega_dir / "SVM_Linear_megacohort.joblib",
        "RBF SVM": mega_dir / "SVM_RBF_megacohort.joblib",
        "XGBoost": mega_dir / "XGBoost_megacohort.joblib",
    }
    for name, path in model_configs.items():
        if path.exists():
            models[name] = joblib.load(path)
    return models


@st.cache_data
def load_discovery_data():
    """Load GSE39582 discovery expression matrix and curated clinical outcomes."""
    features_path = DATA_DIR / "processed" / "expression" / "GSE39582_ml_features.parquet"
    clinical_path = DATA_DIR / "processed" / "clinical" / "GSE39582_cohort_labeled.csv"
    oof_path = DATA_DIR / "processed" / "clinical" / "oof_predictions.csv"

    X = pd.read_parquet(features_path) if features_path.exists() else None
    clin = pd.read_csv(clinical_path, index_col=0) if clinical_path.exists() else None
    oof = pd.read_csv(oof_path, index_col=0) if oof_path.exists() else None
    return X, clin, oof


@st.cache_data
def load_external_data():
    """Load GSE14333 independent validation features and clinical metadata."""
    features_path = DATA_DIR / "external_validation" / "GSE14333_features.parquet"
    targets_path = DATA_DIR / "external_validation" / "GSE14333_clinical_targets.csv"
    preds_path = DATA_DIR / "external_validation" / "GSE14333_predictions.csv"

    X = pd.read_parquet(features_path) if features_path.exists() else None
    targets = pd.read_csv(targets_path, index_col=0) if targets_path.exists() else None
    preds = pd.read_csv(preds_path, index_col=0) if preds_path.exists() else None
    return X, targets, preds


@st.cache_data
def load_tcga_data():
    """Load Phase 2 TCGA-COADREAD features and clinical targets."""
    features_path = DATA_DIR / "processed" / "tcga" / "TCGA_COADREAD_features.parquet"
    clinical_path = DATA_DIR / "processed" / "tcga" / "TCGA_COADREAD_clinical.csv"
    oof_path = DATA_DIR / "processed" / "tcga" / "tcga_oof_predictions.csv"

    X = pd.read_parquet(features_path) if features_path.exists() else None
    clin = pd.read_csv(clinical_path, index_col=0) if clinical_path.exists() else None
    oof = pd.read_csv(oof_path, index_col=0) if oof_path.exists() else None
    return X, clin, oof


@st.cache_data
def load_megacohort_data():
    """Load Phase 3 Combined GPL570 Mega-Cohort features and clinical targets."""
    features_path = DATA_DIR / "processed" / "megacohort" / "megacohort_stage123_features.parquet"
    clinical_path = DATA_DIR / "processed" / "megacohort" / "megacohort_stage123_clinical.csv"
    oof_path = DATA_DIR / "processed" / "megacohort" / "megacohort_oof_predictions.csv"

    X = pd.read_parquet(features_path) if features_path.exists() else None
    clin = pd.read_csv(clinical_path, index_col=0) if clinical_path.exists() else None
    oof = pd.read_csv(oof_path) if oof_path.exists() else None
    return X, clin, oof


@st.cache_data
def load_biomarkers_data():
    """Load candidate gene tables, SHAP attributions, and pathway enrichment."""
    candidates_path = BIOMARKERS_DIR / "candidate_biomarkers.csv"
    shap_path = BIOMARKERS_DIR / "shap_biomarker_importance.csv"
    shared_path = BIOMARKERS_DIR / "recurrence_and_metastasis_shared_biomarkers.csv"
    pathway_path = TABLES_DIR / "metastasis_pathway_enrichment.csv"

    cand_df = pd.read_csv(candidates_path) if candidates_path.exists() else None
    shap_df = pd.read_csv(shap_path) if shap_path.exists() else None
    shared_df = pd.read_csv(shared_path) if shared_path.exists() else None
    pathway_df = pd.read_csv(pathway_path) if pathway_path.exists() else None
    return cand_df, shap_df, shared_df, pathway_df


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("CRC Recurrence Analysis")
st.sidebar.caption("Transcriptomic Predictors of 12-Month Recurrence in Stage I–III Disease")
st.sidebar.divider()

navigation = st.sidebar.radio(
    "Navigation",
    [
        "Patient Risk Scoring",
        "Cross-Validation Benchmarks",
        "Differential Expression",
        "External Validation (GSE14333)",
        "Liver Metastasis Triplets (GSE50760)",
        "Biological Mechanisms",
    ],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown("**Cohort & System Summary**")
st.sidebar.text(
    "• Phase 1 Discovery: GSE39582 (n=264)\n"
    "• Phase 1 External: GSE14333 (n=64)\n"
    "• Phase 1 Paired: GSE50760 (54 samples)\n"
    "• Phase 2 RNA-seq: TCGA (n=306, 20,530g)\n"
    "• Phase 3 Mega-Cohort: N=1,107 (n=885)\n"
    "• Platforms: GPL570 & HiSeq V2\n"
    "• Mega RF AUC: 0.659 | PR: 0.396\n"
    "• Hardware: Apple Silicon M4 (16GB)"
)

# Load data into session
models = load_models()
models_tcga = load_tcga_models()
models_mega = load_megacohort_models()
X_disc, clin_disc, oof_disc = load_discovery_data()
X_ext, clin_ext, preds_ext = load_external_data()
X_tcga, clin_tcga, oof_tcga = load_tcga_data()
X_mega, clin_mega, oof_mega = load_megacohort_data()
cand_df, shap_df, shared_df, pathway_df = load_biomarkers_data()

# ---------------------------------------------------------
# Top Clinical Header Banner (fills top empty space)
# ---------------------------------------------------------
st.markdown(
    """
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1.5px solid #CBD5E1;
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    ">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.5rem;">🧬</span>
            <div>
                <span style="font-weight: 800; font-size: 1.15rem; color: #0F172A; letter-spacing: -0.01em;">
                    Colorectal Cancer (CRC) Genomic Recurrence & Metastasis Analyzer
                </span>
                <span style="display: block; font-size: 0.82rem; color: #64748B; font-weight: 500;">
                    Stage I–III Multi-Cohort Transcriptomic Risk Stratification Pipeline
                </span>
            </div>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
            <span style="background-color: #E0F2FE; color: #0369A1; font-weight: 700; font-size: 0.75rem; padding: 5px 12px; border-radius: 9999px; border: 1px solid #BAE6FD;">
                Active ML Pipeline
            </span>
            <span style="background-color: #F1F5F9; color: #334155; font-weight: 600; font-size: 0.75rem; padding: 5px 12px; border-radius: 9999px; border: 1px solid #CBD5E1;">
                External AUC: 0.676
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 1. PATIENT RISK SCORING
# =========================================================
if navigation == "Patient Risk Scoring":
    st.title("Patient Recurrence Risk Scoring")
    st.markdown(
        "Evaluate individual patient expression profiles using frozen machine learning pipelines. "
        "Predictions estimate the probability of disease recurrence within 12 months post-resection."
    )

    col1, col2, col3 = st.columns([1.5, 2, 1.5])

    with col1:
        cohort_choice = st.selectbox(
            "Select Cohort",
            [
                "Phase 1: Discovery Cohort (GSE39582, n=264)",
                "Phase 1: External Cohort (GSE14333, n=64)",
                "Phase 2: Large Cohort (TCGA-COADREAD, n=306)",
                "Phase 3: Combined GPL570 Mega-Cohort (n=885 / N=1,107)",
            ],
        )

    if "Mega" in cohort_choice:
        active_cohort = "megacohort"
        active_X = X_mega
        active_clin = clin_mega
        active_models = models_mega
    elif "TCGA" in cohort_choice:
        active_cohort = "tcga"
        active_X = X_tcga
        active_clin = clin_tcga
        active_models = models_tcga
    elif "GSE39582" in cohort_choice:
        active_cohort = "discovery"
        active_X = X_disc
        active_clin = clin_disc
        active_models = models
    else:
        active_cohort = "external"
        active_X = X_ext
        active_clin = clin_ext
        active_models = models

    if active_X is not None and active_clin is not None:
        with col2:
            patient_options = []
            for samp in active_X.index:
                if samp in active_clin.index:
                    row = active_clin.loc[samp]
                    if active_cohort == "megacohort":
                        outcome = "Recurrence" if row.get("recurrence_event") == 1 else "Disease-Free"
                        raw_st = row.get("stage", "Unknown")
                        stage_str = f"Stage {int(raw_st)}" if str(raw_st).replace(".0", "").isdigit() else f"Stage {raw_st}"
                        c_src = str(row.get("cohort", ""))
                        patient_options.append(f"{samp} ({outcome}, {stage_str}, {c_src})")
                    elif active_cohort == "tcga":
                        outcome = "Recurrence" if row.get("recurrence_label") == 1 else "Disease-Free"
                        raw_st = row.get("pathologic_stage", row.get("stage_clean", "Stage Unknown"))
                        patient_options.append(f"{samp} ({outcome}, {raw_st})")
                    else:
                        outcome = "Early Recurrence" if row.get("recurrence_label") == 1 else "Disease-Free >=5yr"
                        raw_st = row.get("stage_numeric")
                        if pd.isna(raw_st):
                            raw_st = row.get("tnm.stage", row.get("DukesStage", "N/A"))
                        stage_str = f"Stage {int(raw_st)}" if str(raw_st).replace(".0", "").isdigit() else f"Stage {raw_st}"
                        patient_options.append(f"{samp} ({outcome}, {stage_str})")
                else:
                    patient_options.append(samp)

            selected_str = st.selectbox("Select Patient Sample", patient_options)
            selected_gsm = selected_str.split(" ")[0]

        with col3:
            selected_model_name = st.selectbox(
                "Classification Model",
                list(active_models.keys()),
                index=0,
            )

        st.divider()

        chosen_pipeline = active_models.get(selected_model_name)
        if chosen_pipeline is not None:
            patient_row = active_X.loc[[selected_gsm]]
            pred_prob = chosen_pipeline.predict_proba(patient_row)[0, 1]
            pred_binary = int(pred_prob >= 0.50)

            clin_row = active_clin.loc[selected_gsm]

            if active_cohort == "megacohort":
                true_label = int(clin_row.get("recurrence_event", 0))
                raw_dfs = clin_row.get("dfs_time_months")
                stage_val = clin_row.get("stage")
                stage_str = f"Stage {int(stage_val)}" if str(stage_val).replace(".0", "").isdigit() else f"Stage {stage_val}"
                cohort_src = str(clin_row.get("cohort", ""))
                stage_display = f"{stage_str} ({cohort_src})"
                truth_str = "Recurrence Event" if true_label == 1 else "Disease-Free"
                follow_up_label = "Disease-Free Follow-up"
                stage_label = "Stage (Cohort Origin)"
            elif active_cohort == "tcga":
                true_label = int(clin_row.get("recurrence_label", 0))
                raw_dfs = clin_row.get("pfi_time_months")
                stage_display = str(clin_row.get("pathologic_stage", clin_row.get("stage_clean", "Stage Unknown")))
                truth_str = "PFI Event (Recurrence)" if true_label == 1 else "Disease-Free (PFI Censored)"
                follow_up_label = "PFI Follow-up"
                stage_label = "Pathologic Stage"
            else:
                true_label = int(clin_row.get("recurrence_label", 0))
                raw_dfs = clin_row.get("rfs_time_months")
                if pd.isna(raw_dfs):
                    raw_dfs = clin_row.get("DFS_Time")
                if pd.isna(raw_dfs):
                    raw_dfs = clin_row.get("rfs.delay", "N/A")

                raw_stage = clin_row.get("stage_numeric")
                if pd.isna(raw_stage):
                    raw_stage = clin_row.get("tnm.stage", clin_row.get("DukesStage", "N/A"))

                mmr_val = clin_row.get("mmr.status")
                if pd.isna(mmr_val):
                    mmr_val = clin_row.get("mmr_status", "")

                stage_display = f"Stage {int(raw_stage)}" if str(raw_stage).replace(".0", "").isdigit() else f"Stage {raw_stage}"
                if mmr_val and str(mmr_val).lower() != "nan":
                    stage_display += f" ({mmr_val})"

                truth_str = "Recurrence <= 12 mo" if true_label == 1 else "Disease-Free >= 5 yr"
                follow_up_label = "Disease-Free Follow-up"
                stage_label = "Stage / MMR Status"

            # Column 2 widened to 1.45x to comfortably accommodate full outcome text without truncation
            res1, res2, res3, res4 = st.columns([1.05, 1.45, 1.0, 1.0])

            with res1:
                st.metric(
                    label=f"Predicted Risk ({selected_model_name})",
                    value=f"{pred_prob * 100:.1f}%",
                    delta="High Risk (>=50%)" if pred_binary == 1 else "Low Risk (<50%)",
                    delta_color="inverse" if pred_binary == 1 else "normal",
                )

            with res2:
                st.metric(
                    label="Documented Clinical Outcome",
                    value=truth_str,
                    delta="Concordant" if pred_binary == true_label else "Discordant",
                )

            with res3:
                time_display = f"{float(raw_dfs):.1f} months" if str(raw_dfs).replace(".", "").isdigit() else str(raw_dfs)
                st.metric(label=follow_up_label, value=time_display)

            with res4:
                st.metric(label=stage_label, value=stage_display)

            st.progress(float(pred_prob), text=f"Estimated Recurrence Probability: {pred_prob:.3f}")

            st.subheader("Key Biomarker Expression (vs. Cohort Average)")
            key_genes = ["KLK10", "UBD", "MSLN", "DMRT2", "COCH", "FLT1", "MMP3", "SPACA3"]
            available_genes = [g for g in key_genes if g in active_X.columns]

            if available_genes:
                patient_expr = active_X.loc[selected_gsm, available_genes]
                cohort_mean = active_X[available_genes].mean()
                fold_diff = patient_expr - cohort_mean

                comp_df = pd.DataFrame({
                    "Gene Symbol": available_genes,
                    "Patient Value (log2)": patient_expr.values,
                    "Cohort Mean (log2)": cohort_mean.values,
                    "Deviation (log2 diff)": fold_diff.values,
                })

                st.dataframe(
                    comp_df.style.format({
                        "Patient Value (log2)": "{:.2f}",
                        "Cohort Mean (log2)": "{:.2f}",
                        "Deviation (log2 diff)": "{:+.2f}",
                    }),
                    width="stretch",
                )

                norm_desc = "log2(norm_count + 1) RNA-seq" if active_cohort == "tcga" else "log2 Robust Multi-array Average (RMA)"
                st.caption(
                    f"Expression values are {norm_desc} normalized. "
                    "A positive deviation indicates higher transcript abundance in this patient relative to the cohort mean."
                )
    else:
        st.warning("Processed feature matrices were not found in data/processed/. Run preprocessing scripts first.")


# =========================================================
# 2. CROSS-VALIDATION BENCHMARKS
# =========================================================
elif navigation == "Cross-Validation Benchmarks":
    st.title("Cross-Validation Model Benchmarks")
    st.markdown(
        "Benchmarking four classification architectures evaluated under strict Stratified 5-Fold Cross-Validation. "
        "All variance thresholding, Welch's t-test ranking, and scaling transformations were fitted strictly inside each training fold."
    )

    phase_tab1, phase_tab2, phase_tab3 = st.tabs([
        "Phase 1: Microarray Discovery (GSE39582, n=264)",
        "Phase 2: RNA-seq Large Cohort (TCGA-COADREAD, n=306)",
        "Phase 3: Combined GPL570 Mega-Cohort (n=885 / N=1,107)",
    ])

    with phase_tab1:
        st.subheader("Phase 1: GSE39582 Discovery Cohort")
        st.markdown(
            "Evaluated on 264 Stage I–III patients (57 early recurrence within 12 months, 207 disease-free >= 5 years) "
            "profiled across 21,755 genes on Affymetrix HG-U133 Plus 2.0 microarrays."
        )

        metrics_csv_path = TABLES_DIR / "model_comparison_metrics.csv"
        if metrics_csv_path.exists():
            comp_df = pd.read_csv(metrics_csv_path, index_col=0)
            st.dataframe(
                comp_df[["ROC_AUC_OOF", "PR_AUC_OOF", "Balanced_Accuracy", "Accuracy", "Sensitivity", "Specificity", "F1_Score", "Brier_Score"]].style.format("{:.4f}"),
                width="stretch",
            )

        tab_roc, tab_pr, tab_cm, tab_km, tab_shap = st.tabs([
            "ROC Curves", "Precision-Recall", "Confusion Matrices", "Survival Stratification", "SHAP Feature Attributions"
        ])

        with tab_roc:
            show_figure("fig09_roc_curves.png", "Figure 9: Out-of-fold ROC curves across 5 cross-validation folds (GSE39582).")

        with tab_pr:
            show_figure("fig10_pr_curves.png", "Figure 10: Out-of-fold Precision-Recall curves. Dashed line marks the 21.6% class prevalence baseline.")

        with tab_cm:
            show_figure("fig11_confusion_matrices.png", "Figure 11: Out-of-fold confusion matrices at standard 0.50 decision threshold.")

        with tab_km:
            show_figure("fig12_kaplan_meier_validation.png", "Figure 12: Kaplan-Meier disease-free survival curves stratified by predicted risk category.")

        with tab_shap:
            show_figure("fig13_shap_summary.png", "Figure 13: SHAP summary plot for the top 20 genes (XGBoost model).")
            if shap_df is not None:
                st.subheader("Top SHAP Biomarker Rankings")
                st.dataframe(shap_df.head(15).style.format({"Mean_Abs_SHAP": "{:.5f}"}), width="stretch")

    with phase_tab2:
        st.subheader("Phase 2: TCGA-COADREAD High-Throughput RNA-seq Cohort")
        st.markdown(
            "Evaluated on 306 Stage I–III primary colorectal adenocarcinoma patients (66 recurrence/progression events, 240 disease-free) "
            "profiled across 20,530 genes by Illumina HiSeq RNA sequencing. Evaluates progression-free interval (PFI) endpoints."
        )

        tcga_metrics_path = TABLES_DIR / "phase2_tcga_model_comparison.csv"
        if tcga_metrics_path.exists():
            tcga_comp_df = pd.read_csv(tcga_metrics_path, index_col=0)
            st.dataframe(
                tcga_comp_df[["ROC_AUC_OOF", "PR_AUC_OOF", "Balanced_Accuracy", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1_Score", "Brier_Score"]].style.format("{:.4f}"),
                width="stretch",
            )

        tab_tcga_roc, tab_tcga_pr, tab_tcga_cm, tab_tcga_analysis = st.tabs([
            "TCGA ROC Curves", "TCGA Precision-Recall", "TCGA Confusion Matrices", "Platform Comparison & Discussion"
        ])

        with tab_tcga_roc:
            show_figure("fig19_tcga_roc_curves.png", "Figure 19: Out-of-fold ROC curves across 5 cross-validation folds on TCGA-COADREAD (n=306).")

        with tab_tcga_pr:
            show_figure("fig20_tcga_pr_curves.png", "Figure 20: Out-of-fold Precision-Recall curves on TCGA-COADREAD. Dashed line marks the 21.6% baseline prevalence.")

        with tab_tcga_cm:
            show_figure("fig21_tcga_confusion_matrices.png", "Figure 21: Out-of-fold confusion matrices for TCGA-COADREAD models at 0.50 threshold.")

        with tab_tcga_analysis:
            st.markdown("""
            **Methodological & Biological Insights Across Platforms:**
            - **Platform Consistency:** 16,262 genes overlap between Phase 1 (GPL570) and Phase 2 (TCGA RNA-seq). Candidate biomarkers (*KLK10*, *KLK6*, *UBD*, *MSLN*, *FLT1*) are expressed and detectable in both technologies.
            - **High Specificity Protection:** Random Forest achieved **95.0% specificity** ($TN = 228, FP = 12$) and Linear SVM achieved **88.3% specificity** on TCGA. In clinical adjuvant chemotherapy contexts, minimizing false positives is critical to avoid unnecessary cytotoxic exposure.
            - **RNA-seq Dynamic Range vs. Microarray Hybridization:** Bulk RNA-seq captures dynamic transcript ranges, but tumor purity variations and stromal infiltration in surgical resections introduce inter-sample noise. Cross-fold feature selection successfully isolated reproducible prognostic transcripts without information leakage.
            """)

    with phase_tab3:
        st.subheader("Phase 3: Multi-Center Combined GPL570 Mega-Cohort (N = 1,107)")
        st.markdown(
            "Unified meta-analysis across four clinical cohorts: GSE39582 (n=510), GSE14333 (n=226), "
            "GSE17536 (n=135), and GSE17537 (n=14) on the identical Affymetrix GPL570 platform (21,755 genes). "
            "Applies ComBat empirical Bayes batch harmonization and evaluates 885 Stage I–III patients (225 recurrence events, 660 disease-free) "
            "under strict fold-isolated Stratified 5-Fold Cross-Validation."
        )

        mega_metrics_path = TABLES_DIR / "megacohort_model_comparison.csv"
        if mega_metrics_path.exists():
            mega_comp_df = pd.read_csv(mega_metrics_path, index_col=0)
            st.dataframe(
                mega_comp_df[["ROC_AUC_OOF", "PR_AUC_OOF", "Balanced_Accuracy", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1_Score", "Brier_Score"]].style.format("{:.4f}"),
                width="stretch",
            )

        tab_mega_roc, tab_mega_pr, tab_mega_cm, tab_mega_analysis = st.tabs([
            "Mega-Cohort ROC Curves", "Mega-Cohort Precision-Recall", "Mega-Cohort Confusion Matrices", "ComBat Harmonization & Scaling Insights"
        ])

        with tab_mega_roc:
            show_figure("fig22_megacohort_roc_curves.png", "Figure 22: Out-of-fold ROC curves across 5 cross-validation folds on the Combined GPL570 Mega-Cohort (N = 885 Stage I–III).")

        with tab_mega_pr:
            show_figure("fig23_megacohort_pr_curves.png", "Figure 23: Out-of-fold Precision-Recall curves. Dashed line marks the 25.4% baseline prevalence.")

        with tab_mega_cm:
            show_figure("fig24_megacohort_confusion_matrices.png", "Figure 24: Out-of-fold confusion matrices for Mega-Cohort models at 0.50 threshold.")

        with tab_mega_analysis:
            st.markdown("""
            **Key Insights from the Combined Mega-Cohort Benchmark:**
            - **Sample Size Scaling:** Expanding the training cohort from single-center discovery ($n=264$) to multi-center harmonization ($n=885$ Stage I–III, $N=1,107$ total) yielded marked increases in out-of-fold discrimination: Random Forest OOF ROC-AUC rose to **0.659** and PR-AUC reached **0.396** (vs 25.4% prevalence baseline).
            - **ComBat Empirical Bayes Harmonization:** Removing inter-hospital batch variations using parametric empirical Bayes standardizing eliminated center-specific scanner and RNA handling artifacts while strictly preserving genuine tumor recurrence biology.
            - **Robust Specificity:** SVM architectures maintained high specificities (**92.4%** on RBF SVM, **90.3%** on Linear SVM), confirming that multi-center cross-validation continues to prioritize protection against adjuvant chemotherapy overtreatment.
            """)


# =========================================================
# 3. DIFFERENTIAL EXPRESSION
# =========================================================
elif navigation == "Differential Expression":
    st.title("Differential Gene Expression Analysis")
    st.markdown(
        "Genome-wide screening between early recurrence (<=12 months, n=57) and long-term recurrence-free (>=5 years, n=207) "
        "tumors in GSE39582 using Welch's t-test and Benjamini-Hochberg false discovery rate control."
    )

    tab_volcano, tab_heatmap, tab_pca, tab_table = st.tabs([
        "Volcano Plot", "Clustering Heatmap", "Principal Components", "Candidate Biomarkers"
    ])

    with tab_volcano:
        show_figure("fig06_volcano_plot.png", "Figure 6: Differential expression volcano plot. Red points indicate FDR < 0.05; blue points indicate nominal p < 0.05.")

    with tab_heatmap:
        show_figure("fig07_de_genes_heatmap.png", "Figure 7: Hierarchical clustering of discovery cohort samples over top differentially expressed transcripts.")

    with tab_pca:
        show_figure("fig08_pca_visualization.png", "Figure 8: Principal component analysis of global expression profiles (PC1 vs PC2).")

    with tab_table:
        if cand_df is not None:
            st.subheader("Differentially Expressed Candidate Genes")
            search_query = st.text_input("Filter by gene symbol (e.g. KLK10, UBD, MSLN):", "")
            display_df = cand_df.copy()
            if search_query:
                display_df = display_df[display_df["Gene_Symbol"].str.contains(search_query.strip(), case=False, na=False)]
            st.dataframe(display_df.head(50), width="stretch")


# =========================================================
# 4. EXTERNAL VALIDATION (GSE14333)
# =========================================================
elif navigation == "External Validation (GSE14333)":
    st.title("Independent External Validation (GSE14333)")
    st.markdown(
        "Evaluation of frozen models on an independent external cohort of 64 Stage I–III patients from "
        "Peter MacCallum Cancer Centre. Gene alignment was mapped to discovery feature columns, and models were evaluated "
        "without re-estimating coefficients or tuning hyperparameters."
    )

    ext_csv_path = TABLES_DIR / "external_validation_summary.csv"
    if ext_csv_path.exists():
        ext_df = pd.read_csv(ext_csv_path, index_col=0)
        st.subheader("External Cohort Performance")
        st.dataframe(
            ext_df[["External_ROC_AUC", "External_PR_AUC", "Balanced_Accuracy", "Accuracy", "Sensitivity", "Specificity", "Brier_Score"]].style.format("{:.4f}"),
            width="stretch",
        )

    st.info(
        "**Observation:** The linear SVM yielded the highest discrimination on external data (ROC-AUC 0.676, PR-AUC 0.354 versus the 0.234 prevalence baseline), "
        "outperforming tree ensembles. Regularized linear margins proved less sensitive to inter-laboratory microenvironment "
        "and batch shifts than deeper non-linear models."
    )

    col1, col2 = st.columns(2)
    with col1:
        show_figure("fig14_external_validation_roc.png", "Figure 14: Independent validation ROC curves on GSE14333.")
    with col2:
        show_figure("fig15_external_validation_km.png", "Figure 15: Kaplan-Meier disease-free survival by predicted risk group on GSE14333.")


# =========================================================
# 5. LIVER METASTASIS TRIPLETS (GSE50760)
# =========================================================
elif navigation == "Liver Metastasis Triplets (GSE50760)":
    st.title("Liver Metastasis Progression Dynamics (GSE50760)")
    st.markdown(
        "RNA-sequencing analysis of 18 matched patient triplets (Normal Mucosa -> Primary Colorectal Tumor -> Liver Metastasis; 54 samples) "
        "to track transcriptomic changes that emerge in primary disease and persist during hepatic colonization."
    )

    col1, col2 = st.columns(2)
    with col1:
        show_figure("fig16_metastasis_pca.png", "Figure 16: Paired progression trajectories across normal mucosa, primary tumor, and liver metastasis in PCA space.")
    with col2:
        show_figure("fig17_metastasis_expression_heatmap.png", "Figure 17: Expression heatmap of paired triplet biomarkers across the three disease stages.")

    show_figure("fig18_metastasis_gsea_pathways.png", "Figure 18: Pathway enrichment across progression-associated transcripts (KEGG and GO Biological Processes).")

    if shared_df is not None:
        st.subheader(f"Shared Recurrence and Metastasis Transcripts ({len(shared_df)} Genes)")
        st.markdown(
            "Transcripts that showed significant differentiation both in the primary recurrence discovery cohort "
            "and in the paired primary-to-metastasis progression series."
        )
        st.dataframe(shared_df, width="stretch")

    if pathway_df is not None:
        st.subheader("Functional Pathway Enrichment")
        st.dataframe(
            pathway_df.head(20)[["Gene_set", "Term", "Overlap", "P-value", "Adjusted P-value", "Genes"]],
            width="stretch",
        )


# =========================================================
# 6. BIOLOGICAL MECHANISMS
# =========================================================
elif navigation == "Biological Mechanisms":
    st.title("Biological Interpretation of Discovered Biomarkers")
    st.markdown(
        "Candidate genes selected through differential expression, cross-validated feature ranking, "
        "and SHAP attribution converge on several known hallmarks of colorectal cancer dissemination."
    )

    st.subheader("Key Discovered Genes")

    st.markdown("""
    **KLK10 and KLK6 (Kallikrein-Related Peptidases)**  
    Both serine proteases showed marked upregulation in primary tumors from patients experiencing early recurrence (KLK10 log2 fold change = +1.11, nominal p = 0.007). Kallikreins degrade extracellular matrix components, facilitate basement membrane degradation, and cleave pro-collagen, which facilitates vascular invasion and early micrometastatic seeding.

    **UBD / FAT10 (Ubiquitin D)**  
    UBD ranked as the top predictive feature in SHAP attribution across tree-based models. Unlike canonical ubiquitin, FAT10 targets proteins for proteasomal degradation via an E1-E2-E3 cascade without polyubiquitin chain formation. Its expression in colorectal carcinoma has been linked to bypass of the spindle assembly checkpoint and accelerated chromosomal instability.

    **MSLN (Mesothelin)**  
    Mesothelin expression was elevated in recurring tumors (log2 fold change = +0.93). Primarily recognized as a cell-surface glycoprotein in mesothelial tissues, ectopic expression in colorectal adenocarcinoma enhances peritoneal and hematogenous seeding through interaction with mucin 16 (MUC16/CA125).

    **FLT1 (VEGFR1 - Vascular Endothelial Growth Factor Receptor 1)**  
    FLT1 emerged consistently across multivariate pipelines. As a high-affinity tyrosine kinase receptor for VEGF-A and VEGF-B, FLT1 promotes vascular remodeling and endothelial cell survival, supporting tumor neovascularization and hepatic pre-metastatic niche conditioning.
    """)

    st.divider()

    st.subheader("Methodological Safeguards")
    st.markdown("""
    To avoid common pitfalls in high-dimensional genomic modeling ($p \\gg n$):
    - **Fold-isolated feature selection:** Gene filtering, statistical ranking, and scaling were executed exclusively within training folds. No discovery-wide differential expression informed the training splits.
    - **No artificial synthetic oversampling:** Class balancing relied on cost-sensitive loss weighting rather than synthetic interpolation (SMOTE), preserving true biological expression distributions.
    - **Independent cohort validation:** Final performance was benchmarked on GSE14333 without tuning hyperparameters on the external data.
    """)
