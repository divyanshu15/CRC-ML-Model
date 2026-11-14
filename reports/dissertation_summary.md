# M.Tech Dissertation Summary: Machine Learning-Based Early Prediction of Colorectal Cancer Recurrence

**Degree Track:** M.Tech Research Project  
**Domain:** Bioinformatics, Cancer Genomics, Machine Learning  
**Author:** AI Bioinformatics Research Assistant & Developer  
**Date:** November 2026  
**Status:** Revised after pipeline audit (14 Nov 2026)  

---

## Abstract

Colorectal cancer (CRC) prognosis exhibits significant clinical heterogeneity, where patients with identical anatomical TNM stages experience markedly divergent outcomes following curative surgical resection. While clinical staging guides initial management, it cannot capture the underlying transcriptomic wiring that determines metastatic aggressiveness and early recurrence. In this research project, we developed an end-to-end, scientifically reproducible, and computationally lightweight machine learning pipeline to predict early colorectal cancer recurrence (< 1 year) using genome-wide expression profiles.

Using the discovery cohort **GSE39582** ($N = 585$ samples, Affymetrix GPL570 microarray), we curated a binary classification cohort of 249 Stage I–III tumours: 56 Class 1 (early recurrence $\le 12$ months) vs 193 Class 0 (disease-free $\ge 60$ months), excluding 336 samples under documented criteria, including 19 non-tumoral mucosa samples. We implemented a leak-free multi-stage gene selection engine (variance filter $\rightarrow$ Welch's t-test $\rightarrow$ L1-penalised logistic regression) isolated within the training folds of Stratified 5-Fold Cross-Validation, and compared every model with TNM stage alone.

We benchmarked Random Forest, Linear SVM, RBF SVM and XGBoost, each with and without stage as an extra feature, using repeated nested cross-validation (hyperparameters, gene count and decision threshold all chosen inside training folds). **No gene-expression model discriminated early recurrence better than stage alone**: nested-CV ROC-AUC was 0.51–0.54 versus 0.626 for stage (DeLong p = 0.02–0.09), and on the independent external cohort **GSE14333** (64 binary patients, frozen models, per-gene alignment to the training distribution) 0.70–0.77 versus 0.773 for stage (all DeLong p ≥ 0.44). However, the frozen gene risk scores were **prognostic for disease-free survival independently of stage in two independent external cohorts**: all 226 Stage I–III GSE14333 patients (stage-adjusted Cox HR 1.35–1.48 per SD, p = 0.001–0.008) and all 135 Stage I–III GSE17536 patients (HR 1.30–1.63, p < 0.05 for 6 of 8 models). A dedicated XGBoost-Cox survival model trained on all 485 Stage I–III discovery tumours added to stage internally and in GSE14333 but did not replicate in GSE17536. No gene reached FDR < 0.10 in differential expression, and gene selection was unstable across folds. In the paired GSE50760 metastasis cohort most metastasis biopsies contain normal liver tissue; after adjusting for it, no gene differed significantly between metastases and primaries. A four-cohort leave-one-cohort-out analysis (n = 356) and TCGA (n = 57 under the curated endpoint) also did not beat stage.

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
Genome-wide Welch's t-test comparing Class 1 ($n=56$) vs Class 0 ($n=193$):
- 1,144 genes with nominal $p < 0.05$; 218 with $p < 0.01$; 28 with $p < 0.001$.
- **No gene passes Benjamini–Hochberg FDR $q < 0.10$** (lowest $q = 0.15$).
Leading candidates (nominal $p$): *COCH* ($\log_2\text{FC} = -0.75$, $p = 1.0 \times 10^{-5}$), *DMRT2* ($-0.55$, $1.4 \times 10^{-5}$), *SPACA3* ($-0.27$, $3.0 \times 10^{-5}$), *UBD* ($-0.82$, $5.0 \times 10^{-4}$), *KLK10* ($+1.04$, $7.6 \times 10^{-4}$), *MSLN* ($+0.86$, $1.0 \times 10^{-3}$), *KLK6* ($+0.92$, $1.1 \times 10^{-3}$). These are candidates for follow-up, not established biomarkers.

### 4.2 Machine Learning Benchmark (Repeated Nested Cross-Validation, $n = 249$)

5 repeats × 5 outer folds; inner 3-fold GridSearchCV over classifier settings and gene count (15/35); threshold by Youden's J inside each training fold.

| Model | ROC-AUC (95% CI) | Mean ± SD over repeats | DeLong $p$ vs stage | Sensitivity | Specificity | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage only (no model)** | **0.626** (0.557–0.697) | — | — | — | — | — |
| Random Forest | 0.537 (0.451–0.625) | 0.521 ± 0.026 | 0.089 | 0.429 | 0.611 | 0.193 |
| SVM (Linear) | 0.523 (0.435–0.611) | 0.515 ± 0.041 | 0.059 | 0.518 | 0.560 | 0.204 |
| SVM (RBF) | 0.507 (0.422–0.594) | 0.510 ± 0.022 | 0.023 | 0.643 | 0.472 | 0.208 |
| XGBoost | 0.526 (0.438–0.613) | 0.513 ± 0.028 | 0.065 | 0.429 | 0.534 | 0.199 |
| SVM (Linear) + stage | 0.539 (0.452–0.626) | 0.528 ± 0.031 | 0.087 | 0.607 | 0.461 | 0.201 |

Feature selection was unstable: only 2–6 genes per model were chosen in ≥ 50% of the 25 outer folds (most stable: *EPSTI1*, 84%). Calibration curves are flat around the 22.5% prevalence.

### 4.3 Independent External Validation (GSE14333)
Frozen models and thresholds, no feature re-selection or tuning. GSE14333 genes were location/scale-matched to GSE39582 using GSE14333's unlabelled Stage I–III samples.
- **Binary endpoint ($n = 64$, 15 events):** stage alone 0.773 (0.675–0.855); Random Forest 0.748, XGBoost 0.740, Linear SVM 0.739, RBF SVM 0.705; best genes + stage model (XGBoost) 0.773. No model differs significantly from stage (DeLong $p \ge 0.44$). At the frozen thresholds, sensitivity 0.53–0.87 (vs 0.07–0.27 at 0.50).
- **Survival endpoint (all 226 Stage I–III patients, 50 DFS events):** C-index 0.62–0.65 for gene models vs 0.675 for stage; the gene risk score is independently prognostic after adjusting for stage (e.g. Linear SVM HR 1.47 per SD, 95% CI 1.17–1.86, $p = 0.001$). Median split of the Linear SVM score: log-rank $p = 3.6 \times 10^{-3}$.

- **Second external survival cohort (GSE17536, $n = 135$, 32 events; never used in training):** stage C-index 0.657; Linear SVM 0.653 alone and 0.676 with stage; stage-adjusted HR 1.59 (1.20–2.12), $p = 0.002$ for Linear SVM; 6 of 8 models $p < 0.05$.

### 4.3b Survival Model (XGBoost-Cox, all Stage I–III tumours)
- Discovery: 485 GSE39582 Stage I–III tumours, 137 RFS events; repeated nested CV with fold-isolated martingale-residual gene screening.
- Nested-CV C-index 0.587 (genes) / 0.599 (genes + stage) vs 0.619 for stage; stage-adjusted HR 1.24 per SD, likelihood-ratio $p = 0.013$ vs stage alone.
- External: GSE14333 HR 1.51 ($p = 0.004$); GSE17536 HR 1.02 ($p = 0.91$), i.e. not replicated. The extreme-phenotype classifiers transferred better than the survival model.

### 4.4 Colorectal Liver Metastasis Trajectory (GSE50760)
- **Liver admixture:** the mean log2 FPKM of 30 hepatocyte genes is 1.1 in primaries and 7.6 in metastases; 15 of 18 metastases carry a strong liver signal.
- **Unadjusted paired t-test:** 700 genes at FDR < 0.05, with top pathways complement/coagulation, bile secretion and P450 metabolism (liver functions).
- **Liver-adjusted paired regression (primary analysis):** no gene at FDR < 0.05. No pathway enrichment is reported, and no recurrence candidate overlaps an adjusted-significant metastasis gene.
- **Correction:** the earlier "43 shared biomarkers" counted all recurrence candidates measured in GSE50760, and an offline fallback in the enrichment code could write invented pathway results; that fallback has been removed.

### 4.5 Phase 2: TCGA-COADREAD RNA-seq (curated endpoint, $n = 57$)
- 306 Stage I–III primary tumours have expression and PFI data, but only 57 meet the curated endpoint (28 PFI events ≤ 12 mo, 29 event-free ≥ 60 mo) because TCGA follow-up is short.
- 10 × repeated 5-fold CV: gene models 0.40–0.43 ROC-AUC (below chance, a small-sample artefact) vs 0.635 for stage. No conclusion about gene expression can be drawn from TCGA under this endpoint.
- The earlier analysis (any PFI event, n = 306, AUC ~0.52) used an endpoint not comparable with Phase 1.

### 4.6 Phase 3: Multi-Center GPL570 Mega-Cohort (Leave-One-Cohort-Out, $n = 356$)
- **Endpoint:** the same curated rule as Phase 1 (≤ 12 mo vs ≥ 60 mo). Usable patients: GSE39582 249, GSE14333 64, GSE17536 41, GSE17537 2 (78 events in total).
- **Batch handling:** gene-wise standardisation within each cohort on its own unlabelled Stage I–III tumours (replacing ComBat fitted on pooled data before CV).
- **Validation:** each cohort predicted by models trained on the others.
- **Pooled results (inner-tuned):** stage alone 0.680 (0.626–0.733); Random Forest 0.663, Linear SVM 0.647, XGBoost 0.631, RBF SVM 0.623 (all DeLong $p \ge 0.22$).
- **Per held-out cohort:** GSE39582 0.59–0.62 (stage 0.63); GSE14333 0.75–0.86 (stage 0.77); GSE17536 0.89–0.95 (stage 0.79, only 7 events).
- **Correction to the earlier version:** the previously reported AUC of 0.66 on $n = 885$ used "any recurrence" as the event (short-follow-up censored patients counted as disease-free), included GSE14333 in training, and ran ComBat before CV. The apparent gain over Phase 1 came from the endpoint change, not from sample size.

---

## 5. Thesis Conclusion & Academic Contributions

1. **Honest negative result for classification:** under strict fold isolation, gene-expression models did not discriminate early recurrence better than TNM stage alone, internally or externally.
2. **Positive signal for prognosis beyond stage, replicated:** frozen gene risk scores predicted disease-free survival independently of stage in two external cohorts (GSE14333 HR ≈ 1.35–1.48, GSE17536 HR ≈ 1.30–1.63 per SD), despite near-chance internal binary discrimination. A survival-trained XGBoost-Cox model did not transfer as well. This is the result most worth developing further, ideally with a survival-trained model and a larger external cohort.
3. **Leakage-free, reproducible framework:** fold-isolated feature selection, repeated nested CV with in-fold tuning and thresholds, a stage baseline with DeLong tests and bootstrap CIs, label-free external cohort alignment, and leave-one-cohort-out validation.
4. **Data-quality lessons:** normal-tissue samples hidden in a public tumour series, cross-cohort normalisation shifts, endpoint drift in pooled analyses, and liver-tissue contamination in metastasis biopsies each materially changed the conclusions, and are documented here.
