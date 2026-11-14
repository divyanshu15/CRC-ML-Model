# Research Methodology: CRC Early Recurrence Prediction Pipeline

## 1. Study Design & Cohort Definition
- **Target Disease:** Colorectal Adenocarcinoma (Stage I, II, III).
- **Primary Endpoint:** Early Disease Recurrence vs Long-Term Disease-Free Survival.
  - **Class 1 (High Early Recurrence):** Documented locoregional or distant recurrence within 12 months post-resection.
  - **Class 0 (Low Risk / Disease-Free):** Documented absence of recurrence with >= 60 months of confirmed follow-up.
  - **Excluded Cohort:** Intermediate follow-up (< 60 months without recurrence), late recurrence (> 12 months), non-cancer deaths, or missing recurrence status.

## 2. Genomic Data Sources
- **Discovery Cohort (GSE39582):** 566 colorectal tumors hybridized onto Affymetrix HG-U133 Plus 2.0 (GPL570) arrays. RMA normalized and log2 transformed.
- **Independent Validation Cohort (GSE14333):** 290 colorectal tumors profiled on the same GPL570 platform.
- **Metastasis Exploration Cohort (GSE50760):** 18 matched triplets (normal colon, primary CRC, liver metastasis; 54 samples total) sequenced on Illumina HiSeq 2000 (GPL11154).

## 3. Preprocessing & Probe Annotation
- Probesets on the GPL570 platform (54,675 features) are mapped to official HGNC gene symbols using the GPL570 annotation table.
- Multi-probeset resolution: When multiple probesets map to a single gene symbol, the probeset exhibiting the maximum interquartile range (IQR) / variance across samples is retained.
- Log2 dynamic range verification: Expression values are checked for $E \in [1, 16]$. No duplicate log-transformations are applied.

## 4. Leakage-Free Statistical Feature Selection
Feature selection is isolated strictly inside cross-validation training folds:
1. **Unsupervised Variance Filtering:** Remove lowest 50% non-variable genes across training samples.
2. **Differential Expression Screening:** Two-sample Welch's t-test comparing Class 1 vs Class 0, followed by Benjamini-Hochberg False Discovery Rate (FDR) adjustment ($q < 0.05$).
3. **Regularized Selection:** L1-penalized Logistic Regression (Lasso) and Random Forest mean decrease in Gini impurity to select a parsimonious biomarker panel (20–50 genes).

## 5. Machine Learning Models & Cross-Validation
- **Models:**
  - Random Forest Classifier (balanced class weighting, 200 trees)
  - Linear & RBF Support Vector Machines (standardized features, balanced margin penalties)
  - XGBoost Classifier (`scale_pos_weight` tuned to class imbalance, shallow max depth $\le 3$)
- **Validation Scheme:** Stratified 5-Fold Cross-Validation, repeated across 5 random seeds.

## 6. Evaluation Framework
- **Discrimination:** ROC-AUC, PR-AUC, Sensitivity, Specificity, Balanced Accuracy.
- **Survival Prognostication:** Kaplan-Meier survival curves and two-sided Log-rank test on predicted risk strata.
- **Explainability:** SHAP (Shapley Additive Explanations) summary plots for top predictive features.
- **Independent Validation:** Frozen model inference on GSE14333.

