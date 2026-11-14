# CLAUDE.md - CRC Machine Learning Research Project Guidelines

## Project Overview
This project is an academic research pipeline for predicting early disease recurrence in Stage I–III Colorectal Cancer (CRC) patients using gene expression profiles (microarrays: GSE39582, GSE14333) and investigating metastasis-associated transcriptomics (RNA-seq: GSE50760).

- **Domain:** Bioinformatics, Cancer Genomics, Machine Learning
- **Target Platform:** Apple Silicon (M4, 16 GB unified RAM, macOS)
- **Primary Environment:** Python 3.10 virtualenv (`.venv`)

---

## Environment & Common Commands

### Virtual Environment Setup & Activation
```bash
# Activate existing virtual environment
source .venv/bin/activate

# Install / update dependencies
pip install -r requirements.txt
```

### Running Pipeline Scripts
```bash
# Data download and acquisition inspection
python src/data_loader.py

# Clinical label generation & curation
python src/clinical_labels.py

# Data preprocessing & probe-to-gene mapping
python src/preprocessing.py

# Feature selection
python src/feature_selection.py

# Model training & internal evaluation
python src/model_training.py
python src/evaluation.py

# Multi-cohort cache + leave-one-cohort-out benchmark (needed before external validation)
python src/megacohort_pipeline.py

# Independent external validation (GSE14333; survival also on GSE17536)
python src/external_validation.py

# Survival model (XGBoost-Cox) with two-cohort external validation
python src/survival_model.py

# Paired liver metastasis analysis (GSE50760)
python src/metastasis_analysis.py

# Launch interactive Streamlit showcase app
streamlit run app.py
```

### Running Jupyter Notebooks
```bash
# Register kernel into Jupyter
python -m ipykernel install --user --name=crc-ml --display-name "Python (CRC-ML)"

# Launch JupyterLab / Notebook
jupyter lab
```

---

## Critical Bioinformatics & Machine Learning Principles

1. **Strict Prevention of Data Leakage (Mandatory):**
   - Feature selection, scaling, normalization, and hyperparameter tuning MUST only be fitted on training folds.
   - Never perform differential expression or variance filtering on the entire dataset prior to train/test splitting.
   - Use `sklearn.pipeline.Pipeline` or fold-isolated transforms for cross-validation.

2. **Curation of Clinical Recurrence Labels:**
   - **Class 1 (High Early Recurrence Risk):** Documented locoregional or distant recurrence within 1 year (<= 365 days / 12 months) of primary resection.
   - **Class 0 (Low Recurrence / Disease-Free):** Documented disease-free status with at least 5 years (>= 60 months / 1825 days) of confirmed clinical follow-up.
   - **Exclusion:** Patients with unknown recurrence status, non-cancer death, or intermediate follow-up without recurrence (< 5 years) must be excluded from binary classification to prevent label noise and artificial bias.
   - Never impute or synthesize clinical recurrence outcomes from cancer stage or other features.

3. **Multi-Cohort & Platform Integrity:**
   - **GSE39582 (Discovery Cohort):** GPL570 Affymetrix HG-U133 Plus 2.0 (~566 samples). Used for EDA, preprocessing, label creation, feature selection, ML training, and internal CV.
   - **GSE14333 (Independent Validation Cohort):** GPL570 Affymetrix. Strictly held out. Preprocessing and feature set are frozen and applied without retraining or re-selecting features.
   - **GSE50760 (Metastasis Cohort):** GPL11154 Illumina RNA-seq (18 patients: 18 normal, 18 primary, 18 liver metastasis). Must be analyzed as a paired experimental study. NEVER merge directly with microarray data.

4. **Model Architecture Scope:**
   - Implement only:
     - Random Forest (`RandomForestClassifier`)
     - Support Vector Machine (`SVC` with Linear and RBF kernels)
     - XGBoost (`XGBClassifier`)
   - Do NOT implement deep neural networks, CNNs, or GPU-dependent architectures.
   - Support class imbalance via `class_weight='balanced'` or `scale_pos_weight`.

5. **Hardware & Memory Optimization (16 GB RAM Constraint):**
   - Convert gene expression floats to `float32` (halves memory usage vs `float64`).
   - Store intermediate matrices using Apache Parquet (`pyarrow`) or compressed Feather/CSV format.
   - Do not download raw FASTQ or unneeded raw CEL files. Use GEO Series Matrix / processed matrices.

6. **Reproducibility & Scientific Honesty:**
   - Always pin random states (`random_state=42`).
   - Save models with `joblib`.
   - Never fabricate or embellish AUC, accuracy, or biomarker claims. A scientifically rigorous AUC of 0.68 without data leakage is vastly superior to a flawed 0.95 with leakage.

---

## Directory Structure
```
CRC-ML-Model/
├── PRD.md                       # Product Requirements Document
├── DESIGN.md                    # Detailed System Architecture & Bioinformatics Design
├── CLAUDE.md                    # Agent & Developer Operating Instructions
├── requirements.txt             # Pinned Python package dependencies
├── .gitignore                   # Git ignore rules for data, models, and virtualenv
├── data/
│   ├── raw/                     # Original GEO downloads (Series Matrix, metadata)
│   ├── processed/               # Cleaned expression matrices, clinical tables
│   └── external_validation/     # GSE14333 processed validation inputs
├── notebooks/                   # Milestone exploratory and analysis notebooks
├── src/                         # Modular Python implementation modules
├── models/                      # Serialized trained model artifacts (.joblib)
├── results/                     # Figures (PNG/PDF), metrics (CSV/JSON), biomarkers
└── reports/                     # Data dictionary, methodology, logs, limitations
```

