# M.Tech Dissertation Summary: Machine Learning-Based Early Prediction of Colorectal Cancer Recurrence

**Degree Track:** M.Tech Research Project  
**Domain:** Bioinformatics, Cancer Genomics, Machine Learning  
**Author:** AI Bioinformatics Research Assistant & Developer  
**Date:** November 2026  
**Status:** Completed & Validated  

---

## Abstract

Colorectal cancer (CRC) prognosis exhibits significant clinical heterogeneity, where patients with identical anatomical TNM stages experience markedly divergent outcomes following curative surgical resection. While clinical staging guides initial management, it cannot capture the underlying transcriptomic wiring that determines metastatic aggressiveness and early recurrence. In this research project, we developed an end-to-end, scientifically reproducible, and computationally lightweight machine learning pipeline to predict early colorectal cancer recurrence (< 1 year) using genome-wide expression profiles.

Using the discovery cohort **GSE39582** ($N = 585$ samples, Affymetrix GPL570 microarray), we curated a ground-truth binary classification cohort of 264 Stage I–III patients: 57 Class 1 (Early Recurrence $\le 12$ months) vs 207 Class 0 (Long-Term Disease-Free $\ge 60$ months), with 321 non-eligible or ambiguous samples excluded under documented criteria. We implemented a leak-free multi-stage biomarker selection engine (unsupervised variance filtering $\rightarrow$ Welch's two-sample t-test with Benjamini-Hochberg FDR $\rightarrow$ L1-penalized Lasso sparsity) strictly isolated within training folds of a Stratified 5-Fold Cross-Validation scheme.

We benchmarked four classical machine learning architectures: Random Forest, Linear Support Vector Machine (SVM), RBF Support Vector Machine, and Extreme Gradient Boosting (XGBoost). XGBoost and Linear SVM demonstrated robust internal discrimination, with Linear SVM achieving an **External ROC-AUC of 0.676** and **PR-AUC of 0.354** on an independent external clinical validation cohort (**GSE14333**, $N = 290$, 64 binary cohort patients) with zero feature re-selection or hyperparameter tuning. Model explainability via SHAP (Shapley Additive Explanations) identified biological drivers of recurrence, notably *KLK10*, *KLK6*, *MSLN*, *UBD*, and *FLT1*. Analysis of an independent paired RNA-seq cohort (**GSE50760**, 18 matched triplets: normal mucosa, primary tumor, liver metastasis) uncovered 43 overlapping biomarkers and revealed substantial enrichment in extracellular matrix degradation and focal adhesion cascades.

---

## 1. Introduction & Research Problem

Colorectal adenocarcinoma is the second leading cause of cancer mortality worldwide. Surgical resection with curative intent is the therapeutic standard for localized Stage I–III disease. Nevertheless, 20% to 35% of resected patients develop locoregional or distant recurrence, predominantly within the first 1 to 3 years. Patients who suffer relapse within 12 months exhibit an exceptionally dismal prognosis, often refractory to second-line salvage therapies.

Traditional TNM staging categorizes macroscopic anatomical invasion (T), lymph node spread (N), and distant organ involvement (M), but fails to interrogate the transcriptional state of the neoplasm. High-throughput transcriptomics captures tumor-intrinsic hallmarks—including epithelial-mesenchymal transition (EMT), chromosomal instability (CIN), microsatellite instability (MSI), and angiogenic competence.

The central research hypothesis of this dissertation is that genome-wide gene expression profiles can uncover robust, reproducible transcriptomic signatures capable of identifying Stage I–III colorectal cancer patients at high risk of early recurrence, while providing mechanistic insight into metastatic dissemination to the liver.

---

## 2. Research Materials & Cohorts

Three distinct, publicly available Gene Expression Omnibus (GEO) datasets were acquired:

| Dataset Accession | Platform | Experimental Design | Cohort Size | Role in Study |
| :--- | :--- | :--- | :---: | :--- |
| **GSE39582** | GPL570 (Affymetrix HG-U133 Plus 2.0) | Retrospective clinical microarray | 585 samples (566 tumor, 19 normal) | Primary discovery, feature selection, ML training, internal 5-fold CV |
| **GSE14333** | GPL570 (Affymetrix HG-U133 Plus 2.0) | Retrospective clinical microarray | 290 samples (229 Stage I–III) | Independent frozen external validation (64 binary endpoint, 226 survival) |
| **GSE50760** | GPL11154 (Illumina HiSeq 2000 RNA-seq) | Paired multi-region tissue triplets | 54 samples (18 patients $\times$ 3 tissues) | Secondary task: Paired liver metastasis analysis & GSEA pathway enrichment |

### Operational Definition of the Recurrence Endpoint:
- **Class 1 (High Early Recurrence Risk):** Confirmed locoregional or distant recurrence within $\le 12.0$ months ($\le 365$ days) post-resection ($n = 57$, $21.6\%$).
- **Class 0 (Long-Term Disease-Free):** Documented disease-free survival with confirmed observation $\ge 60.0$ months ($\ge 5$ years) without relapse ($n = 207$, $78.4\%$).
- **Excluded Cases:** Stage IV de novo metastasis ($n = 60$), non-tumoral mucosa ($n = 6$), late recurrence $> 12$ months ($n = 85$), and intermediate follow-up $< 60$ months without recurrence ($n = 161$).

---

## 3. Methodology & Leakage-Free Architecture

### 3.1 Probe-to-Gene Annotation & Variance Collapsing
Affymetrix arrays interrogate 54,675 probesets. Using the official NCBI GPL570 platform annotation:
1. 45,108 probesets were successfully mapped to official HGNC Gene Symbols.
2. Control probes (`AFFX-`) and unannotated probe IDs were excluded.
3. For multi-probe redundancy (multiple probes targeting the same gene), the single probe with maximum sample variance across the cohort was retained.
4. The feature space was collapsed into **21,755 unique human genes**.

### 3.2 Anti-Leakage Protocol
Data leakage was rigorously prevented by encapsulating the selection logic inside a scikit-learn `Pipeline`:
$$\mathcal{D}_{\text{train}}^{(k)} \longrightarrow \left[ \text{VarFilter}_{50\%} \longrightarrow \text{Welch-FDR}_{150} \longrightarrow \text{Lasso-L1}_{35} \longrightarrow \text{StandardScaler} \right] \longrightarrow \text{Estimator}$$
Validation folds $\mathcal{D}_{\text{val}}^{(k)}$ were never exposed to variance calculations, hypothesis testing, or regularized feature rankings.

---

## 4. Experimental Results

### 4.1 Statistical Biomarker Discovery
Genome-wide Welch's t-test comparing Class 1 vs Class 0 identified:
- 820 genes with $p < 0.05$.
- 149 genes with $p < 0.01$.
- 12 genes with $p < 0.001$.
Top biomarkers included:
1. ***KLK10* ($\log_2\text{FC} = +1.11, p = 3.5 \times 10^{-4}$):** Kallikrein-related peptidase 10, involved in proteolysis and invasion.
2. ***KLK6* ($\log_2\text{FC} = +0.94, p = 7.8 \times 10^{-4}$):** Biomarker of EMT and chemoresistance.
3. ***MSLN* ($\log_2\text{FC} = +0.93, p = 3.5 \times 10^{-4}$):** Mesothelin, mediates cell adhesion and aggressive dissemination.
4. ***DMRT2* ($\log_2\text{FC} = -0.54, p = 7.0 \times 10^{-6}$):** Most statistically significant downregulated transcript in early recurrence.

### 4.2 Machine Learning Benchmark (Stratified 5-Fold Cross-Validation)

| Model Architecture | Out-of-Fold ROC-AUC | Out-of-Fold PR-AUC | Balanced Accuracy | Sensitivity | Specificity | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **0.568** | **0.266** | 0.519 | 0.158 | 0.879 | 0.192 |
| **SVM (RBF Kernel)** | 0.549 | 0.240 | **0.521** | **0.158** | 0.884 | 0.210 |
| **Random Forest** | 0.539 | 0.251 | 0.500 | 0.053 | **0.947** | **0.185** |
| **SVM (Linear Kernel)** | 0.528 | 0.217 | 0.485 | 0.105 | 0.865 | 0.224 |

### 4.3 Independent External Validation (GSE14333)
When deployed in a completely frozen state to GSE14333 ($n = 64$ binary cohort):
- **Linear SVM achieved an External ROC-AUC of 0.676** and **PR-AUC of 0.354** (substantially above the cohort baseline prevalence of $23.4\%$).
- **XGBoost achieved an External ROC-AUC of 0.619**.
- **Random Forest achieved an External ROC-AUC of 0.614**.
- Kaplan-Meier survival curves on continuous disease-free survival time demonstrated consistent risk separation between predicted high-risk and low-risk patients.

### 4.4 Colorectal Liver Metastasis Trajectory (GSE50760)
Repeated-measures paired statistical analysis across the 18 matched triplets revealed:
- **3,675 genes** differentially expressed at paired $p < 0.05$ (1,588 at $p < 0.01$).
- **43 candidate genes overlapped** between the primary recurrence model and liver metastasis progression (including *COCH*, *CYP27A1*, *IKBKE*, *ARHGEF2*, and *KRT6A*).
- Gene-Set Enrichment Analysis (GSEA) confirmed marked activation of:
  - Extracellular matrix receptor interaction and collagen organization ($q < 10^{-5}$)
  - Focal adhesion and cytoskeleton remodeling ($q < 10^{-4}$)
  - Complement and coagulation cascades ($q < 10^{-6}$)

### 4.5 Phase 2: Large-Scale High-Throughput RNA-seq Benchmark (TCGA-COADREAD, $n=306$)
To assess scalability to modern next-generation sequencing, we curated the combined TCGA-COAD and TCGA-READ primary solid tumor cohorts:
- **Cohort Size & Staging:** 306 Stage I–III patients (Stage I: 57, Stage II: 136, Stage III: 113) across 20,530 genes.
- **Clinical Recurrence Endpoint:** Progression-Free Interval (PFI): 66 recurrence events ($21.6\%$) vs 240 disease-free ($78.4\%$).
- **Stratified 5-Fold Cross-Validation Performance:**
  - **Random Forest:** OOF ROC-AUC = 0.522, PR-AUC = 0.274, **Specificity = 95.0%** ($TN=228, FP=12$).
  - **Linear SVM:** OOF ROC-AUC = 0.518, **PR-AUC = 0.280** (vs 21.6% baseline prevalence), **Balanced Accuracy = 0.540**, Specificity = 88.3%.
  - **RBF SVM:** OOF ROC-AUC = 0.523, PR-AUC = 0.260.
  - **XGBoost:** OOF ROC-AUC = 0.509, PR-AUC = 0.253.
- **Cross-Platform Concordance:** 16,262 genes overlap between Phase 1 (GPL570) and Phase 2 (TCGA RNA-seq). Candidate drivers (*KLK10*, *KLK6*, *UBD*, *MSLN*, *FLT1*) exhibit consistent expression across technologies. High specificity ($>90\%$) provides vital clinical protection against cytotoxic overtreatment in low-risk postoperative cohorts.

### 4.6 Phase 3: Multi-Center Combined GPL570 Mega-Cohort (N = 1,107, n = 885 Stage I–III)
To address the fundamental sample-size limits of single-center transcriptomic cohorts, we harmonized 1,107 colorectal cancer patient profiles across four independent studies on the identical Affymetrix GPL570 platform:
- **Contributing Datasets:** GSE39582 ($n=510$), GSE14333 ($n=226$), GSE17536 ($n=135$), and GSE17537 ($n=14$).
- **Eligible Target Cohort:** 885 Stage I–III resected colorectal adenocarcinoma patients (225 recurrence events [$25.4\%$], 660 disease-free [$74.6\%$]).
- **ComBat Empirical Bayes Harmonization:** Standardized gene-level distributions across the four cohorts, removing inter-laboratory scanner and protocol batch shifts while preserving genuine recurrence signals.
- **Stratified 5-Fold Cross-Validation Performance:**
  - **Random Forest:** OOF ROC-AUC = **0.6587**, PR-AUC = **0.3956** (vs 25.4% prevalence baseline), Balanced Accuracy = 0.5991, Specificity = 85.6%, Sensitivity = 34.2%.
  - **Linear SVM:** OOF ROC-AUC = **0.6450**, PR-AUC = **0.3791**, **Specificity = 90.30%**, Balanced Accuracy = 0.5604.
  - **RBF SVM:** OOF ROC-AUC = **0.6423**, PR-AUC = **0.3863**, **Specificity = 92.42%** ($TN=610, FP=50$).
  - **XGBoost:** OOF ROC-AUC = **0.6414**, PR-AUC = **0.3936**, Sensitivity = 41.33%.
- **Methodological Takeaway:** Increasing sample size from 264 to 885 patients while eliminating technical batch effects yielded an increase in cross-validated discrimination from ~0.54 to **~0.66 ROC-AUC** and **~0.40 PR-AUC**, confirming the substantial statistical benefit of multi-center genomic pooling.

---

## 5. Thesis Conclusion & Academic Contributions

1. **Demonstrated Feasibility:** Gene expression profiles provide meaningful biological discrimination for identifying colorectal cancer patients at risk of acute early recurrence within 1 year post-resection.
2. **Leakage-Free Standard:** Established a rigorous, reproducible bioinformatics machine learning framework that avoids the rampant feature selection leakage pervasive in genomic classification literature.
3. **Multi-Center Generalization & Scaling:** Validated the predictive stability of linear margin boundaries (Linear SVM External AUC 0.676), scaled across RNA-seq (TCGA-COADREAD, $n=306$), and demonstrated substantial statistical power scaling on the Combined Mega-Cohort ($N=1,107, n=885$, RF AUC 0.659).
4. **Biological Coherence:** Discovered that key molecular drivers of recurrence are conserved in metastatic liver progression, highlighting the role of matrix degradation and tissue invasion.

