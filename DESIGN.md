# System & Scientific Design Document: Machine Learning-Based Early Prediction of Colorectal Cancer Recurrence

**Project Title:** Machine Learning-Based Early Prediction Model for Colorectal Cancer Using Gene Expression Profiles  
**Degree Track:** M.Tech Research Project  
**Author / Team:** Bioinformatics & Machine Learning Engineering  
**Version:** 1.0.0  
**Target Architecture:** Apple Silicon (macOS, M4, 16 GB Unified RAM, 512 GB SSD)  

---

## 1. Executive Summary & Scientific Motivation

### 1.1 The Clinical Challenge
Colorectal cancer (CRC) remains the second leading cause of cancer-related mortality globally. Although surgical resection with curative intent is standard for Stage I–III CRC, 20% to 35% of patients experience disease recurrence—predominantly within the first 1 to 3 years following surgery. Standard clinical staging (TNM staging: Tumor penetration, Node involvement, Metastasis) stratifies anatomical disease spread but frequently fails to capture underlying molecular heterogeneity. Consequently, patients with identical TNM stages often exhibit divergent clinical trajectories: some remain long-term disease-free, while others experience aggressive early relapse.

### 1.2 The Research Solution
This project establishes a reproducible, scientifically defensible bioinformatics and machine learning pipeline that leverages genome-wide transcriptomic profiles (Affymetrix GeneChip arrays and Illumina RNA-seq) to:
1. Identify high-confidence transcriptomic biomarkers associated with early recurrence (< 1 year).
2. Train classical machine learning classifiers (Random Forest, Linear/RBF Support Vector Machines, and XGBoost) to stratify recurrence risk.
3. Validate predictive fidelity on a fully independent external clinical microarray cohort (GSE14333) with frozen parameters.
4. Investigate transcriptomic alterations mediating colorectal liver metastasis using paired primary-metastatic RNA-sequencing data (GSE50760).

```
   ┌────────────────────────────────────────────────────────────────────────┐
   │                       PRIMARY RESEARCH PIPELINE                         │
   │                                                                        │
   │  GSE39582 (n≈566) Microarray                                            │
   │        │                                                               │
   │        ▼                                                               │
   │  [Curate Clinical Endpoint] ──► Class 1 (Recur <= 1yr) vs Class 0 (DFI >= 5yr)
   │        │                                                               │
   │        ▼                                                               │
   │  [Strict Stratified Split] ──► Train Fold (80%) vs Test Fold (20%)     │
   │        │                                                               │
   │        ├─► [Feature Selection: Variance -> limma/Welch t-test BH-FDR]   │
   │        ├─► [L1 / Lasso / RF Feature Importance]                        │
   │        └─► [Train RF, SVM, XGBoost with Stratified 5-Fold CV]          │
   │                                                                        │
   │        ▼                                                               │
   │  [Independent Frozen Validation on GSE14333 (n≈290)]                   │
   │                                                                        │
   │  ═══════════════════════════════════════════════════════════════════   │
   │                      SECONDARY RESEARCH TASK                           │
   │                                                                        │
   │  GSE50760 RNA-seq (18 patients: Normal, Primary CRC, Liver Metastasis) │
   │        │                                                               │
   │        ▼                                                               │
   │  [Paired Differential Expression + GSEA Biological Pathway Enrichment] │
   └────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Biological Foundations & Terminology

To ensure conceptual clarity for bioinformatics researchers and computer scientists alike, key biological concepts governing this project are defined below:

| Terminology | Biological Definition & Context in CRC |
| :--- | :--- |
| **Transcriptomics** | The complete set of RNA transcripts produced by the genome under specific physiological or pathological conditions. Quantifies gene activity (expression levels). |
| **Microarray (Affymetrix GPL570)** | High-throughput hybridization platform using short oligonucleotide probe pairs (probesets) to quantify the abundance of known mRNA sequences. |
| **RNA-Sequencing (RNA-seq)** | High-throughput sequencing platform measuring cDNA read counts; provides broad dynamic range and avoids probe saturation issues. |
| **TNM Staging** | Clinical staging system: **T** (depth of primary tumor invasion through colon wall layers: T1–T4), **N** (regional lymph node metastasis: N0–N2), and **M** (distant metastasis: M0–M1). Stages I–III represent localized to regional disease eligible for curative resection. |
| **Locoregional vs Distant Recurrence** | Locoregional recurrence is cancer regrowth at the initial surgical bed or regional mesenteric lymph nodes. Distant recurrence is metastatic spread to distant organs, predominantly the liver (via mesenteric/portal venous circulation) and lungs. |
| **Disease-Free Interval (DFI / RFS)** | The time elapsed from primary curative surgery until documented disease relapse (recurrence or metastasis) or death from any cause. |
| **Probeset-to-Gene Mapping** | Translating Affymetrix probe IDs (e.g., `204224_s_at`) to standard HUGO Gene Nomenclature Committee (HGNC) gene symbols (e.g., `CDX2`). Multiple probesets often target one gene and must be summarized (e.g., maximum variance or mean). |
| **Batch Effect** | Systematic technical non-biological variation introduced by laboratory handling, chip lot, reagent batches, or scanning dates across different experiment batches. |

---

## 3. Dataset Architecture & Cohort Selection

### 3.1 Primary Discovery Cohort: GSE39582 (CIT Consortium)
- **Repository:** NCBI Gene Expression Omnibus (GEO).
- **Accession:** GSE39582 (Marisa et al., *PLoS Medicine* 2013).
- **Platform:** GPL570 (Affymetrix Human Genome U133 Plus 2.0 Array).
- **Total Samples:** 566 colorectal tumor resections (+ 19 non-tumoral mucosa).
- **Target Sub-cohort:** Stage I–III patients with documented recurrence status and follow-up duration.
- **Microarray Normalization:** RMA (Robust Multichip Average), log2-transformed.
- **Clinical Variables of Interest:**
  - `characteristics_ch1.7.age` / Age
  - `characteristics_ch1.8.sex` / Sex
  - `characteristics_ch1.9.stage` / Stage (Stage 1, 2, 3, 4)
  - `characteristics_ch1.10.t` / T stage
  - `characteristics_ch1.11.n` / N stage
  - `characteristics_ch1.12.m` / M stage
  - `characteristics_ch1.13.relapse` / Relapse status (0 = No relapse, 1 = Relapse)
  - `characteristics_ch1.14.rfs` / Relapse-free survival time (months)
  - `characteristics_ch1.15.chemotherapy` / Adjuvant chemotherapy received (Yes/No)

### 3.2 External Validation Cohort: GSE14333
- **Repository:** NCBI GEO (Jorissen et al., *Clin Cancer Res* 2009).
- **Accession:** GSE14333.
- **Platform:** GPL570 (Identical Affymetrix HG-U133 Plus 2.0 Array).
- **Cohort Size:** 290 colorectal cancer patients.
- **Independence Gate:** Sample patient IDs and clinical registries must be checked for zero patient overlap with GSE39582. Features, scaling parameters, and decision boundaries derived from GSE39582 are applied in a completely frozen state to GSE14333.

### 3.3 Metastasis Cohort: GSE50760 (Paired RNA-Seq)
- **Repository:** NCBI GEO (Kim et al., *Genome Med* 2014).
- **Accession:** GSE50760.
- **Platform:** GPL11154 (Illumina HiSeq 2000).
- **Experimental Design:** Paired cohort of 18 patients, each contributing 3 matched tissue specimens:
  - 18 Normal colon mucosa
  - 18 Primary colorectal adenocarcinoma
  - 18 Colorectal liver metastases (CRLM)
- **Total Samples:** 54 samples across 18 unique biological subjects.
- **Statistical Model:** Must use paired analysis (e.g. repeated-measures ANOVA or paired Wilcoxon / paired limma-voom) to control for inter-individual patient background.

---

## 4. Operational Classification Definitions & Clinical Label Engine

To prevent subjective outcome assignment and preserve clinical research integrity, recurrence labels are constructed strictly as follows:

```
               PATIENT SELECTION CRITERIA (Stage I–III Resected CRC)
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
      Recurrence Reported                        Recurrence Reported
            YES (1)                                     NO (0)
              │                                               │
    ┌─────────┴─────────┐                           ┌─────────┴─────────┐
    ▼                   ▼                           ▼                   ▼
Time <= 12 mo     Time > 12 mo               Follow-up >= 60 mo   Follow-up < 60 mo
    │                   │                           │                   │
    ▼                   ▼                           ▼                   ▼
[ CLASS 1 ]         [ EXCLUDE ]                 [ CLASS 0 ]         [ EXCLUDE ]
High Early         Late Recurrence             Long-term           Inadequate
Recurrence Risk    (Confounding)               Disease-Free        Observation Period
```

1. **Class 1 (High Early Recurrence Risk):**
   - Documented recurrence (`relapse == 1` or `rfs_event == 1`)
   - Time to recurrence $\le 12.0$ months (or $\le 365$ days).
2. **Class 0 (Long-Term Disease-Free / Low Risk):**
   - No documented recurrence (`relapse == 0` or `rfs_event == 0`)
   - Total follow-up duration $\ge 60.0$ months (5 years) without relapse.
3. **Excluded Cohort (Censored / Ambiguous):**
   - Patients with no recurrence but follow-up $< 60$ months (unknown whether recurrence occurred in years 2–5).
   - Patients with late recurrence (recurrence between month 13 and 60), to avoid blurring molecular signatures distinguishing acute, aggressive early recurrence from indolent recurrence.
   - Non-cancer related deaths or unverified recurrence timelines.

---

## 5. Statistical Methodology & Machine Learning Architecture

### 5.1 Strict Data Leakage Prevention Framework
Data leakage is the most prevalent flaw in bioinformatics machine learning. To eliminate leakage:
1. **Splitting Before Any Supervised Processing:**
   $$\mathcal{D}_{\text{train}}, \mathcal{D}_{\text{test}} = \text{StratifiedSplit}(\mathcal{D}, \text{test\_size}=0.20, \text{seed}=42)$$
2. **Feature Selection within Folds:**
   Feature selection (variance thresholding, hypothesis testing, lasso shrinkage) is computed **solely** on $\mathcal{D}_{\text{train}}$ (or training folds $\mathcal{D}_{\text{train}}^{(k)}$ inside K-fold cross-validation). $\mathcal{D}_{\text{test}}$ is never observed during biomarker selection.
3. **Pipeline Encapsulation:**
   Standardization, transformation, and imputation steps are bound directly to the estimator pipeline.

### 5.2 Multistage Feature Selection Engine
Affymetrix HG-U133 Plus 2.0 interrogates 54,675 probesets representing $\sim 20,000$ unique genes. Training on 20,000 features with $N \approx 200\text{--}400$ samples induces the "curse of dimensionality" ($p \gg N$) and severe overfitting. A 3-stage filtration pipeline is employed:

```
54,675 Probesets
       │
       ▼
[ Probe-to-Gene Collapse ] ──► Max-variance probe per gene symbol (~20,000 Genes)
       │
       ▼
[ Stage 1: Unsupervised Variance Filter ] ──► Remove lowest 50% invariant genes (~10,000 Genes)
       │
       ▼
[ Stage 2: Hypothesis Testing / Differential Expression ] ──► Welch's t-test / limma with BH-FDR < 0.05
       │                                                      Rank top ~200–500 Candidate Genes
       ▼
[ Stage 3: Embedded Regularization (L1 Lasso / ElasticNet / RF Importance) ] ──► Top 20–50 Biomarkers
```

### 5.3 Machine Learning Algorithms

#### 1. Random Forest Classifier
- **Model:** Ensemble of decorrelated decision trees using bagging and random feature subspaces.
- **Strengths:** Robust to non-linear interactions; resistant to outliers; provides Gini impurity feature importance.
- **Hyperparameter Grid:**
  - `n_estimators`: `[100, 200, 500]`
  - `max_depth`: `[3, 5, 8, None]`
  - `min_samples_split`: `[2, 5, 10]`
  - `class_weight`: `['balanced', 'balanced_subsample']`

#### 2. Support Vector Machine (SVM)
- **Model:** Maximum-margin hyperplane separation.
- **Kernels:**
  - Linear Kernel: $K(x, x') = \langle x, x' \rangle$ (Interpretable feature weights $w$).
  - Radial Basis Function (RBF) Kernel: $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ (Captures non-linear decision boundaries).
- **Regularization:** $C \in [0.01, 0.1, 1.0, 10.0]$ with `class_weight='balanced'`.

#### 3. Extreme Gradient Boosting (XGBoost)
- **Model:** Additive boosted tree model minimizing regularized objective:
  $$\mathcal{L}^{(t)} = \sum_{i=1}^n l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$
- **Handling Imbalance:** `scale_pos_weight = (count(negative) / count(positive))`.
- **Hyperparameter Grid:**
  - `learning_rate`: `[0.01, 0.05, 0.1]`
  - `max_depth`: `[2, 3, 4]` (Shallow trees to prevent overfitting high-dimensional genomic features)
  - `subsample`: `[0.7, 0.9]`
  - `colsample_bytree`: `[0.6, 0.8]`

### 5.4 Model Evaluation Suite
1. **Classification Metrics:**
   - ROC-AUC (Area Under Receiver Operating Characteristic)
   - PR-AUC (Precision-Recall Area Under Curve; essential for imbalanced cohorts)
   - Balanced Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score
   - Brier Score (Probability calibration error)
2. **Survival Analysis Integration:**
   - Stratify validation cohort into Predicted High-Risk vs Predicted Low-Risk groups using model probability thresholds ($\ge 0.5$).
   - Compute Kaplan-Meier survival curves on continuous relapse-free survival time.
   - Perform two-sided Log-rank test ($p < 0.05$) to establish whether ML risk stratification yields statistically significant separation in true survival time.
3. **Model Explainability (SHAP):**
   - Compute Shapley Additive Explanations (`TreeExplainer` for RF/XGBoost, `LinearExplainer`/`KernelExplainer` for SVM).
   - Visualize global feature attribution (Beeswarm plot) and individual patient risk drivers (Waterfall plot).

---

## 6. Detailed Module Specifications (`src/`)

```text
src/
├── __init__.py               # Package initialization
├── data_loader.py            # Automated GEO downloading, caching, series matrix parsing
├── clinical_labels.py        # Clinical outcome curation, exclusion logic, label generation
├── preprocessing.py          # GPL570 probe-to-gene mapping, log2 check, variance filtering
├── feature_selection.py      # Statistical testing (t-test/FDR), L1 Lasso, embedded ranking
├── model_training.py         # Stratified CV, Grid/RandomizedSearchCV, model serialization
├── evaluation.py             # Classification metrics, ROC/PR curves, Kaplan-Meier & Log-rank
└── visualization.py          # Publication-ready figure generation (PNG 300 DPI & PDF)
```

### Module Responsibilities:
- `data_loader.py`:
  - `download_geo_series(geo_accession, target_dir)`: Downloads matrix via GEOparse/HTTP with MD5/checksum and local caching.
  - `extract_expression_matrix(geo_object)`: Returns clean sample-by-probe DataFrame in `float32`.
  - `extract_clinical_metadata(geo_object)`: Returns parsed clinical attributes DataFrame.
- `clinical_labels.py`:
  - `build_recurrence_cohort(clinical_df)`: Applies the 1-year vs 5-year decision rules, documents exclusion reasons, returns tagged cohort table.
- `preprocessing.py`:
  - `map_probes_to_genes(expr_df, gpl_platform_df)`: Aggregates multiple probesets to unique HGNC genes.
  - `verify_normalization(expr_df)`: Checks dynamic range ($\approx 2\text{--}16$) to verify log2 transformation status.
- `feature_selection.py`:
  - `differential_expression_screen(X_train, y_train, top_n)`: Computes two-sample Welch t-tests and Benjamini-Hochberg FDR adjustments on training fold only.
  - `lasso_feature_selector(X_train, y_train, alpha)`: Identifies sparse non-zero coefficients.
- `model_training.py`:
  - `train_random_forest(X_train, y_train, cv)`
  - `train_svm(X_train, y_train, cv)`
  - `train_xgboost(X_train, y_train, cv)`
- `evaluation.py`:
  - `evaluate_classifier(model, X_test, y_test)`: Calculates full classification matrix.
  - `kaplan_meier_analysis(time, event, risk_scores)`: Computes log-rank statistic and survival curves.
- `visualization.py`:
  - Standardized publication formatting: `figure_dpi=300`, Seaborn `whitegrid`, Nature/Cancer Research color palette.

---

## 7. Computational & Memory Budget (MacBook M4, 16 GB RAM)

| Asset / Operation | Memory / Disk Impact | Optimization Technique |
| :--- | :--- | :--- |
| **GSE39582 Matrix (566 x 54,675)** | $\approx 240\text{ MB}$ raw gzip, $1.2\text{ GB}$ uncompressed CSV | Converted to `np.float32` in pandas: memory footprint reduces from $240\text{ MB}$ to $\approx 120\text{ MB}$ RAM. Saved as `.parquet` ($85\text{ MB}$). |
| **Probe-to-Gene Mapping** | Temporary peak memory $\approx 500\text{ MB}$ | Run in chunked dictionary mappings; garbage-collect raw probe matrix immediately after collapse. |
| **K-Fold Model Training** | CPU computation across 8 cores | Set `n_jobs=4` to preserve responsiveness and prevent thermal throttling on MacBook Air. |
| **Storage Reservation** | Total project footprint: $\approx 3.5\text{ GB}$ | Raw matrices retained in `data/raw/`; processed parquets in `data/processed/`. Intermediate duplications eliminated. |

---

## 8. Milestone Execution Plan

In accordance with PRD Section 16, the implementation executes in sequential, verified milestones:

```
[ Milestone 1: Environment & Architecture Verification ]  <-- IN PROGRESS
  - Create directory structure, requirements.txt, virtualenv, CLAUDE.md, DESIGN.md.
  - Verify Apple Silicon dependencies (numpy, scipy, scikit-learn, statsmodels, GEOparse).

[ Milestone 2: Dataset Acquisition & Data Dictionary (GSE39582) ]
  - Download GSE39582 series matrix and clinical metadata.
  - Inspect sample identifiers, probe counts, clinical fields.
  - Generate reports/data_dictionary.md.
  - STOP & PRESENT FOR REVIEW.

[ Milestone 3: Exploratory Data Analysis & Quality Control ]
  - Characterize clinical stage distributions, missing variables, follow-up times.
  - Generate exploratory figures in results/figures/.

[ Milestone 4: Clinical Label Engine & Cohort Curation ]
  - Filter Stage I–III patients.
  - Generate Class 1 (<= 1yr) and Class 0 (>= 5yr) recurrence cohort.
  - Log all sample exclusions in reports/experiment_log.md.

[ Milestone 5: Preprocessing & Probe Annotation ]
  - Collapse Affymetrix probesets to HGNC symbols.
  - Verify log2 scaling and normalize if required.

[ Milestone 6: Differential Expression & Feature Selection (Leakage-Free) ]
  - Isolate training set. Perform Welch t-test + BH-FDR.
  - Identify top candidate recurrence biomarkers.

[ Milestone 7: Model Training & Hyperparameter Tuning ]
  - Fit Random Forest, Linear SVM, RBF SVM, XGBoost with Stratified 5-Fold CV.
  - Save best models to models/.

[ Milestone 8: Model Evaluation & SHAP Explainability ]
  - Comparative ROC, PR curves, Confusion Matrices, Brier score.
  - Compute SHAP values for biological attribution.

[ Milestone 9: Independent External Validation (GSE14333) ]
  - Evaluate frozen model on independent external cohort GSE14333.
  - Kaplan-Meier survival separation with Log-rank test.

[ Milestone 10: Metastasis Sub-Pipeline (GSE50760) & Synthesis ]
  - Paired RNA-seq analysis of primary vs liver metastases.
  - Pathway enrichment (GSEA / KEGG / Reactome).
  - Compile final dissertation reports and limitations.
```

