# PRD.md

# Machine Learning-Based Early Prediction Model for Colorectal Cancer Using Gene Expression Profiles

**Project Type:** M.Tech Research Project  
**Domain:** Bioinformatics, Machine Learning, Cancer Genomics  
**Primary Language:** Python  
**Development Environment:** Jupyter Notebook / VS Code  
**Target Hardware:** Apple MacBook Air M4, 16 GB RAM, 512 GB SSD  
**Development Schedule:** 8–10 hours per week  
**Difficulty:** Medium–High

---

# 1. Project Overview

## 1.1 Project Description

Develop a machine learning-based research pipeline capable of identifying colorectal cancer patients at increased risk of early disease recurrence using gene expression profiles and associated clinical information.

The system will analyze publicly available colorectal cancer transcriptomics datasets, identify important gene-expression patterns, select predictive biomarkers and train machine learning models to distinguish patients with different recurrence outcomes.

The project will also investigate gene-expression differences associated with colorectal cancer metastasis using an independent RNA-sequencing dataset.

The final system must be scientifically reproducible, computationally lightweight and suitable for an M.Tech dissertation and potential research publication.

## 1.2 Research Problem

Colorectal cancer patients with similar clinical stages can experience substantially different treatment outcomes.

Some patients experience recurrence after surgery and treatment, while others remain disease-free for extended periods.

Traditional clinical indicators do not always capture the molecular differences responsible for these outcomes.

This project investigates whether gene-expression profiles can identify molecular patterns associated with recurrence risk and aggressive colorectal cancer behaviour.

## 1.3 Main Research Objective

Develop and evaluate machine learning models that predict early recurrence risk in stage I–III colorectal cancer patients using gene-expression profiles.

## 1.4 Secondary Objectives

1. Identify genes associated with colorectal cancer recurrence.
2. Identify potential molecular biomarkers associated with aggressive disease.
3. Investigate gene-expression differences between primary colorectal tumors and liver metastases.
4. Compare Random Forest, Support Vector Machine and XGBoost.
5. Evaluate predictive performance using appropriate classification and survival-analysis metrics.
6. Investigate whether identified genes have biological relevance in colorectal cancer progression.
7. Validate the predictive model using an independent dataset.

---

# 2. Research Scope

## 2.1 Primary Research Task

Binary classification:

- Class 0: Lower recurrence risk.
- Class 1: Higher early recurrence risk.

The primary endpoint should be based on documented recurrence or relapse information.

Use the following operational definition where the required follow-up information is available:

- Early recurrence: Locoregional recurrence or distant metastasis within one year after primary tumor resection.
- Long-term disease-free comparison group: No documented recurrence with at least five years of follow-up.

Patients with insufficient follow-up, unknown recurrence status or ambiguous clinical outcomes must not be assigned an artificial class.

**Important:** These are research labels, not a clinically validated prediction of treatment response.

## 2.2 Secondary Research Task

Metastasis-associated gene-expression analysis:

Compare:

- Primary colorectal cancer.
- Liver metastatic colorectal cancer.

Use GSE50760 for this investigation.

The secondary task must remain separate from the primary recurrence-prediction training pipeline.

## 2.3 Research Limitations

The project must not claim that:

- Every metastatic tumor is treatment-resistant.
- Every early-stage tumor has low recurrence risk.
- Gene-expression profiles alone determine treatment difficulty.
- A high-risk ML prediction proves that a patient will develop recurrence.

Treatment resistance, invasiveness, metastasis and recurrence are related but distinct biological and clinical concepts.

---

# 3. Dataset Selection and Download Requirements

All datasets must come from publicly available research repositories.

## 3.1 Primary Dataset: GSE39582

**Repository:** NCBI Gene Expression Omnibus (GEO)

**Accession:** GSE39582

**Platform:** GPL570 – Affymetrix Human Genome U133 Plus 2.0 Array

**Organism:** Homo sapiens

**Data Type:** Gene expression microarray

**Expected Cohort:** Approximately 566 tumor samples in commonly used processed versions.

**Dataset URL:**

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39582

### Download Requirements

Download:

1. GEO Series Matrix File.
2. Associated sample metadata.
3. Clinical information.
4. Recurrence or relapse-free survival information.
5. Platform annotation file (GPL570).

Prefer the processed expression matrix initially.

Do not download all individual raw CEL files during the initial development phase.

### Purpose

This dataset will be used for:

- Exploratory data analysis.
- Gene-expression preprocessing.
- Clinical metadata analysis.
- Recurrence label construction.
- Feature selection.
- ML model development.
- Internal validation.

The final eligible sample count must be established after inspecting the clinical metadata.

## 3.2 Independent Validation Dataset: GSE14333

**Repository:** NCBI GEO

**Accession:** GSE14333

**Platform:** GPL570 – Affymetrix Human Genome U133 Plus 2.0 Array

**Data Type:** Gene expression microarray.

**Dataset URL:**

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14333

### Download Requirements

Download:

1. Processed gene-expression matrix.
2. Sample annotations.
3. Clinical metadata.
4. Recurrence and follow-up information.
5. Platform annotation.

### Purpose

Use this dataset for independent validation of the recurrence-prediction model.

Do not use its samples for training or feature selection.

Check sample identifiers and published cohort overlap before considering it independent. Remove any duplicated or overlapping patients.

## 3.3 Secondary Dataset: GSE50760

**Repository:** NCBI GEO

**Accession:** GSE50760

**Platform:** GPL11154 – Illumina HiSeq 2000

**Data Type:** RNA sequencing.

**Sample Count:** 54 samples from 18 patients.

Sample distribution:

- 18 normal colon samples.
- 18 primary colorectal cancer samples.
- 18 liver metastatic samples.

**Dataset URL:**

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE50760

### Download Requirements

Download:

1. GSE50760_RAW.tar.
2. Processed gene-expression supplementary files.
3. Sample metadata.
4. Sample-to-patient mapping.
5. Gene annotation information.

Prefer processed data for initial analysis.

### Purpose

Investigate:

- Metastasis-associated genes.
- Differential expression between primary and metastatic tumors.
- Potential invasion-related biomarkers.
- Biological pathways associated with disease progression.

Because samples are derived from the same 18 patients, statistical analysis must account for the paired experimental design.

Do not treat all 54 samples as 54 independent patients.

Do not combine this dataset directly with GSE39582 for model training.

---

# 4. Dataset Download Priority

Download in this order:

| Priority | Dataset  | Required files                         |
| -------- | -------- | -------------------------------------- |
| 1        | GSE39582 | Processed matrix + clinical metadata   |
| 2        | GSE14333 | Processed matrix + clinical metadata   |
| 3        | GSE50760 | Processed expression + sample metadata |

Start implementation with GSE39582 only.

Add the other datasets after the primary pipeline works.

## 4.1 Future Optional Dataset: TCGA-COAD

Repository: Genomic Data Commons.

https://portal.gdc.cancer.gov/

Project: TCGA-COAD.

Data type: Transcriptome Profiling.

Data subtype: Gene Expression Quantification.

Workflow: STAR - Counts.

Download open-access RNA-seq data and associated clinical information.

TCGA-COAD should be treated as a separate RNA-seq validation cohort.

Do not directly concatenate its expression values with Affymetrix microarray values.

---

# 5. Technology Stack

## 5.1 Programming Language

Python 3.11 or a compatible stable Python version.

## 5.2 Development Tools

- Jupyter Notebook.
- VS Code.
- Git.
- GitHub.
- Terminal.

## 5.3 Python Libraries

### Data Processing

- pandas
- numpy
- scipy

### Visualization

- matplotlib
- seaborn
- plotly (optional)

### Bioinformatics

- GEOparse
- gseapy
- mygene
- statsmodels

### Machine Learning

- scikit-learn
- xgboost

### Statistical Analysis

- scipy
- statsmodels
- lifelines

### Model Interpretation

- SHAP

### File Management

- pathlib
- gzip
- tarfile
- pickle or joblib

## 5.4 Optional R/Bioconductor Tools

Use R only where it provides a clear bioinformatics advantage.

Recommended:

- GEOquery
- limma
- sva
- Biobase
- hgu133plus2.db

R and Bioconductor may be used for microarray preprocessing and ComBat batch correction.

Python remains the primary development language.

---

# 6. Hardware and Storage Constraints

The project must be optimized for the following machine:

- Apple M4 processor.
- 16 GB unified memory.
- 512 GB SSD.
- macOS.

## 6.1 Storage Budget

Target project storage:

| Component                        |    Estimated storage |
| -------------------------------- | -------------------: |
| GSE39582 processed data          |          100 MB–1 GB |
| GSE14333 processed data          |            50–500 MB |
| GSE50760 processed data          |            10–500 MB |
| Optional raw GSE39582 CEL files  | Approximately 4.4 GB |
| Optional raw GSE50760 files      |  Approximately 10 MB |
| Python environment               |               2–5 GB |
| Intermediate datasets            |               1–3 GB |
| Model outputs and visualizations |          500 MB–1 GB |

These are planning estimates, not guaranteed download sizes. Actual sizes must be recorded during downloading.

### Recommended Storage Allocation

Reserve approximately 10–15 GB of free storage for the project.

Do not download raw FASTQ sequencing files.

Do not perform genome alignment or read-level sequencing analysis.

## 6.2 Memory Optimization

The system must:

- Load expression matrices using efficient pandas data types.
- Avoid unnecessary copies of large DataFrames.
- Save intermediate processed datasets.
- Use NumPy arrays for model training.
- Limit parallel processing.
- Avoid GPU dependencies.
- Use reproducible random seeds.
- Allow individual pipeline stages to execute independently.

The complete project must run on a 16 GB RAM laptop.

---

# 7. Proposed Project Architecture

```text
colorectal-cancer-ml/
│
├── PRD.md
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── GSE39582/
│   │   ├── GSE14333/
│   │   └── GSE50760/
│   │
│   ├── processed/
│   │   ├── expression/
│   │   ├── clinical/
│   │   └── metadata/
│   │
│   └── external_validation/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_clinical_label_creation.ipynb
│   ├── 04_differential_gene_analysis.ipynb
│   ├── 05_feature_selection.ipynb
│   ├── 06_pca_visualization.ipynb
│   ├── 07_model_training.ipynb
│   ├── 08_model_evaluation.ipynb
│   ├── 09_external_validation.ipynb
│   └── 10_metastasis_analysis.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── clinical_labels.py
│   ├── feature_selection.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── visualization.py
│
├── models/
│   ├── random_forest/
│   ├── svm/
│   └── xgboost/
│
├── results/
│   ├── figures/
│   ├── tables/
│   ├── metrics/
│   └── biomarkers/
│
├── reports/
│   ├── data_dictionary.md
│   ├── methodology.md
│   ├── experiment_log.md
│   └── limitations.md
│
└── logs/
```

Do not create unnecessary files or folders before their corresponding implementation stage.

---

# 8. Functional Requirements

## Module 1: Dataset Acquisition

Implement a reproducible data acquisition process.

Requirements:

- Download GEO datasets.
- Save original downloaded files.
- Record dataset accession numbers.
- Record file sizes.
- Record sample counts.
- Record download dates.
- Preserve original sample identifiers.
- Create a data dictionary.

The program must detect existing downloaded files and avoid downloading them repeatedly.

## Module 2: Exploratory Data Analysis

Perform exploratory analysis on GSE39582.

Generate:

1. Sample distribution plot.
2. Clinical stage distribution.
3. Recurrence distribution.
4. Gene-expression distribution.
5. Missing-value analysis.
6. Sample correlation heatmap.
7. Gene-expression boxplots.
8. Sample metadata summary.

Report:

- Total number of samples.
- Number of tumor samples.
- Number of normal samples, if present.
- Number of recurrence cases.
- Number of non-recurrence cases.
- Number of excluded samples.
- Missing clinical variables.

## Module 3: Data Preprocessing

Perform:

1. Missing-value inspection.
2. Removal of invalid or duplicated samples.
3. Probe-to-gene mapping.
4. Duplicate probe handling.
5. Gene filtering.
6. Expression normalization verification.
7. Appropriate log transformation if necessary.

Important:

The GSE39582 processed matrix may already contain RMA-normalized, log2-transformed expression values.

Do not normalize or log-transform the data again without checking its original processing.

Preserve both the original and processed expression matrices.

## Module 4: Clinical Label Creation

Create recurrence labels using actual clinical outcomes.

Rules:

- Positive class: documented early recurrence within one year.
- Negative class: documented absence of recurrence with at least five years of follow-up.
- Exclude uncertain cases from this primary binary analysis.

Create a clinical metadata table containing:

- Sample ID.
- Patient ID.
- Age.
- Sex.
- Tumor stage.
- T stage.
- N stage.
- M stage.
- Recurrence status.
- Follow-up duration.
- Recurrence duration.
- Adjuvant chemotherapy status, if available.

Do not infer missing recurrence status from tumor stage.

Do not use future information as an input feature when claiming early prediction.

## Module 5: Differential Gene Expression Analysis

Identify genes associated with recurrence.

Use:

- limma, preferably for Affymetrix microarray analysis.
- Appropriate statistical testing.
- Multiple-testing correction using Benjamini-Hochberg FDR.

Generate:

- Differentially expressed gene table.
- Volcano plot.
- Heatmap of selected genes.
- Gene-expression comparison plots.

Report adjusted p-values and fold changes.

## Module 6: Feature Selection

Reduce the gene-expression feature space.

Implement:

1. Variance-based gene filtering.
2. Statistical feature selection.
3. L1-regularized logistic regression.
4. Random Forest feature importance.

Feature selection must happen exclusively within the training data.

For cross-validation, repeat feature selection inside each training fold.

Never use the complete dataset to select genes before splitting into training and testing data.

Store the selected gene names and selection parameters.

## Module 7: PCA Visualization

Perform Principal Component Analysis.

Generate:

- PCA plot using the first two principal components.
- Explained variance plot.
- Sample clustering visualization.

Color samples by:

- Recurrence status.
- Clinical stage.

PCA is for exploratory visualization, not a substitute for model evaluation.

## Module 8: Machine Learning Model Development

Develop three primary models.

### Model A: Random Forest

Implement:

- RandomForestClassifier.
- Hyperparameter tuning.
- Feature importance.
- Cross-validation.

### Model B: Support Vector Machine

Implement:

- SVC.
- StandardScaler.
- Linear kernel initially.
- RBF kernel as an additional experiment.
- Hyperparameter tuning.

### Model C: XGBoost

Implement:

- XGBClassifier.
- Controlled tree depth.
- Learning rate tuning.
- Regularization.
- Early stopping where appropriate.

Use class weighting or other suitable imbalance-handling approaches.

Do not apply SMOTE before train/test splitting.

Do not use deep learning, CNNs or neural networks.

## Module 9: Model Validation

Use:

- Stratified train/test splitting.
- Stratified cross-validation.
- Fixed random seeds.
- Independent external validation.

The final external validation must use the same predefined gene features and preprocessing logic as the trained model.

Do not select features again using external validation labels.

Do not tune model hyperparameters on the external validation cohort.

## Module 10: Model Evaluation

Calculate:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- PR-AUC.
- Confusion matrix.
- Sensitivity.
- Specificity.

For recurrence outcomes with follow-up times and censoring, also investigate:

- Kaplan-Meier curves.
- Log-rank test.
- Cox proportional hazards analysis.
- Time-dependent ROC-AUC, if appropriate.

Do not present ordinary classification accuracy as a survival-prediction metric.

Generate comparative visualizations for all three models.

## Module 11: Model Explainability

Use SHAP where computationally practical.

Identify:

- Top predictive genes.
- Gene importance rankings.
- Positive and negative contributions.
- Common genes selected across different models.

Generate:

- SHAP summary plot.
- Feature importance bar chart.
- Gene-expression comparison plots.

Do not describe model importance as proof of biological causation.

## Module 12: Metastasis Analysis

Analyze GSE50760 separately.

Compare:

- Primary CRC vs liver metastatic CRC.

Use patient-matched statistical analysis wherever sample pairing is confirmed.

Generate:

- Differential gene expression table.
- Volcano plot.
- Heatmap.
- PCA visualization.
- Metastasis-associated gene list.

Perform gene-set enrichment analysis using gseapy and appropriate pathway databases.

Compare identified metastasis-associated genes with the primary model's selected genes.

Treat overlapping genes as research candidates requiring independent validation.

---

# 9. Scientific and Statistical Requirements

These requirements are mandatory.

## 9.1 Avoid Data Leakage

The following must only be fitted using training data:

- Feature selection.
- Standardization.
- Imputation.
- Hyperparameter tuning.
- Any supervised preprocessing.

Use scikit-learn Pipeline wherever appropriate.

## 9.2 Batch Effects

GSE39582 and GSE14333 use the same microarray platform, but they may still have different laboratory and processing effects.

Inspect batch effects before combining data.

Prefer independent external validation over blindly merging datasets.

If batch correction is necessary, ensure the procedure does not use validation outcome labels.

## 9.3 Class Imbalance

Check class distribution before model training.

Use:

- class_weight where supported.
- Stratified splitting.
- Precision-recall analysis.

## 9.4 Reproducibility

Every experiment must record:

- Dataset version.
- Sample identifiers.
- Selected genes.
- Model parameters.
- Random seed.
- Training/test split.
- Performance metrics.

Save trained models using joblib.

---

# 10. Expected Visualizations

The project must generate publication-quality figures.

Required figures:

1. Dataset sample distribution.
2. Clinical stage distribution.
3. Recurrence distribution.
4. Missing-value visualization.
5. Gene-expression distribution.
6. Sample correlation heatmap.
7. PCA visualization.
8. Differential expression volcano plot.
9. Differential expression heatmap.
10. Random Forest feature importance.
11. SVM evaluation.
12. XGBoost feature importance.
13. ROC curve comparison.
14. Precision-recall curves.
15. Confusion matrices.
16. SHAP summary plot.
17. Kaplan-Meier recurrence-free survival curves, where supported.
18. Metastasis-associated gene-expression heatmap.

All figures must have:

- Clear titles.
- Axis labels.
- Legends.
- Readable font sizes.
- Appropriate figure dimensions.
- High-resolution PNG output.
- Optional PDF output.

---

# 11. Expected Research Results

The final project should provide:

1. A cleaned and documented colorectal cancer gene-expression dataset.
2. A clinically defined recurrence classification dataset.
3. A list of recurrence-associated genes.
4. Three trained ML models.
5. A comparative model-performance report.
6. Independent validation results.
7. A list of potential molecular biomarkers.
8. Metastasis-associated gene-expression results.
9. Biological pathway analysis.
10. A reproducible research methodology.

Do not target a predetermined accuracy.

A scientifically valid result with modest performance is preferable to an artificially inflated accuracy caused by data leakage.

---

# 12. Development Milestones

## Phase 1: Environment Setup

- Install Python.
- Configure Jupyter Notebook.
- Create project directories.
- Install required libraries.
- Verify system compatibility.

Deliverable: Working Python environment.

## Phase 2: Dataset Acquisition

- Download GSE39582.
- Inspect expression matrix.
- Extract clinical metadata.
- Verify sample identifiers.
- Create data dictionary.

Deliverable: Documented primary dataset.

## Phase 3: Exploratory Analysis

- Analyze clinical variables.
- Investigate recurrence distribution.
- Generate exploratory plots.
- Identify missing data.

Deliverable: EDA report.

## Phase 4: Data Preprocessing

- Process gene-expression matrix.
- Map probes to genes.
- Verify normalization.
- Create clinical outcome labels.
- Document exclusions.

Deliverable: Clean expression and clinical datasets.

## Phase 5: Statistical Analysis

- Differential expression.
- Feature filtering.
- Multiple-testing correction.
- Biomarker investigation.

Deliverable: Candidate gene list.

## Phase 6: Machine Learning

- Train Random Forest.
- Train SVM.
- Train XGBoost.
- Tune hyperparameters.
- Compare models.

Deliverable: Trained models and internal evaluation.

## Phase 7: Independent Validation

- Acquire GSE14333.
- Verify independent samples.
- Apply the frozen preprocessing and gene-selection procedure.
- Evaluate the models.

Deliverable: External validation report.

## Phase 8: Metastasis Analysis

- Acquire GSE50760.
- Analyze paired primary and metastatic samples.
- Identify metastasis-associated genes.
- Perform pathway enrichment.

Deliverable: Metastasis analysis report.

## Phase 9: Final Research Documentation

- Consolidate findings.
- Generate figures.
- Document limitations.
- Prepare methodology.
- Prepare M.Tech dissertation.
- Draft a research manuscript.

Deliverable: Complete research project.

---

# 13. Non-Functional Requirements

The application must:

- Run locally on macOS.
- Work with 16 GB RAM.
- Require no dedicated GPU.
- Avoid unnecessary cloud computing costs.
- Preserve original research data.
- Maintain reproducibility.
- Support incremental execution.
- Avoid downloading duplicate datasets.
- Use modular Python code.
- Provide understandable error messages.
- Record processing time and storage usage.

---

# 14. What Must NOT Be Implemented

Do not implement:

- Deep learning.
- CNNs.
- Transformer models.
- GPU-based training.
- Raw FASTQ alignment pipelines.
- Genome assembly.
- A web application.
- A mobile application.
- A commercial clinical diagnostic system.
- Unnecessary dashboards.
- Automated treatment recommendations.

The focus is transcriptomics, statistical analysis, machine learning and biological interpretation.

---

# 15. Final Project Deliverables

The completed project must contain:

| Deliverable             | Expected output   |
| ----------------------- | ----------------- |
| Dataset documentation   | Markdown          |
| Cleaned expression data | CSV/Parquet       |
| Clinical metadata       | CSV               |
| EDA                     | Jupyter Notebook  |
| Preprocessing           | Python/Notebook   |
| Differential expression | CSV + figures     |
| Feature selection       | Gene list         |
| ML models               | Joblib files      |
| Model evaluation        | Metrics + figures |
| External validation     | Report            |
| Metastasis analysis     | Report + figures  |
| Final documentation     | Markdown/PDF      |
| Dissertation material   | Research chapters |

---

# 16. Instructions for the AI Coding Agent

Act as a bioinformatics engineer and machine learning research assistant.

The developer of this project is a beginner in bioinformatics and machine learning.

Follow these instructions strictly:

1. Explain the scientific purpose of every implementation stage.
2. Explain unfamiliar biological terminology.
3. Use clean, modular and well-documented Python code.
4. Do not generate the entire project in one step.
5. Implement one milestone at a time.
6. Verify the output of each stage before continuing.
7. Never invent dataset sample counts or clinical metadata.
8. Inspect actual GEO files before making assumptions about their structure.
9. Never invent missing recurrence labels.
10. Prevent data leakage at every stage.
11. Keep all downloaded datasets in the data directory.
12. Record the original and processed dataset sizes.
13. Optimize all operations for a 16 GB RAM MacBook.
14. Prefer processed GEO expression matrices.
15. Avoid GPU-dependent libraries.
16. Create visualizations that explain the biological and statistical findings.
17. Save intermediate outputs to avoid repeating expensive operations.
18. Maintain an experiment log.
19. Clearly distinguish established scientific findings from model-generated hypotheses.
20. Never fabricate accuracy, AUC, biological significance or publication claims.

## Initial Execution Instructions

Do not begin by implementing machine learning models.

First perform these tasks:

1. Inspect the development environment.
2. Create the project directory structure.
3. Create requirements.txt.
4. Install and verify dependencies.
5. Download GSE39582 processed expression data.
6. Download associated clinical metadata.
7. Inspect the actual files and their dimensions.
8. Identify recurrence-related clinical variables.
9. Generate a data dictionary.
10. Report the actual dataset storage requirement.

Stop after completing the initial dataset inspection and report.

Wait for the developer to review the dataset structure before proceeding to preprocessing.

---

# 17. Research Integrity

This project is an academic research prototype.

It is not intended for clinical diagnosis or treatment selection.

Any predictive biomarkers must be treated as candidates until independently validated.

All research papers, datasets, software packages and external resources must be properly cited.

The final manuscript must clearly state:

- Dataset limitations.
- Cohort selection criteria.
- Sample exclusions.
- Class imbalance.
- Batch effects.
- Model limitations.
- External validation limitations.
- Potential clinical implications without overstating them.

**Final Research Goal:**

Develop a reproducible, interpretable and computationally efficient machine learning pipeline that investigates whether colorectal cancer gene-expression profiles can identify patients at elevated risk of early recurrence and uncover molecular signatures associated with metastatic disease.
