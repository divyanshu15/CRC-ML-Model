# Data Dictionary: GSE39582 Discovery Cohort

**Generated:** 2026-11-07 13:35:09
**Total Cohort Samples:** 585

## 1. Overview
This dictionary documents the clinical and pathological variables extracted from the GSE39582 Series Matrix.
Variables are parsed from NCBI GEO `!Sample_characteristics_ch1` fields.

## 2. Clinical and Pathological Variables

| Variable Name | Extracted Key | Non-Null Count | Null Count | Data Type | Sample Values / Distinct Categories | Recurrence Relevance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Sample_characteristics_ch1_32` | Sample_characteristics_ch1_32 | 2 | 583 | object | `` | Clinical Covariate |
| `age.at.diagnosis_(year)` | age.at.diagnosis_(year) | 585 | 0 | object | 115 unique values (e.g. `34.5`, `36.4`) | Clinical Covariate |
| `braf.mutation` | braf.mutation | 585 | 0 | object | `WT`, `M`, `N/A` | Clinical Covariate |
| `braf.mutation.dna` | braf.mutation.dna | 585 | 0 | object | `N/A`, `c.1799T>A` | Clinical Covariate |
| `braf.mutation.exon.number` | braf.mutation.exon.number | 585 | 0 | object | `N/A`, `c.1799T>A` | Clinical Covariate |
| `braf.mutation.protein` | braf.mutation.protein | 585 | 0 | object | `N/A`, `p.V600E` | Clinical Covariate |
| `chemotherapy.adjuvant` | chemotherapy.adjuvant | 585 | 0 | object | `N`, `Y`, `N/A` | CRITICAL (Outcome / Stratification) |
| `chemotherapy.adjuvant.type` | chemotherapy.adjuvant.type | 585 | 0 | object | `N/A`, `5FU`, `FOLFOX`, `other`, `FUFOL`, `FOLFIRI` | CRITICAL (Outcome / Stratification) |
| `cimp.status` | cimp.status | 585 | 0 | object | `-`, `N/A`, `+` | Clinical Covariate |
| `cin.status` | cin.status | 585 | 0 | object | `+`, `-`, `N/A` | Clinical Covariate |
| `cit.molecularsubtype` | cit.molecularsubtype | 585 | 0 | object | 7 unique values (e.g. `C5`, `C4`) | Clinical Covariate |
| `dataset` | dataset | 585 | 0 | object | `discovery`, `validation`, `Non Tumoral` | Clinical Covariate |
| `dependancy_sample` | dependancy_sample | 585 | 0 | object | 18 unique values (e.g. `N/A`, `CIT546`) | Clinical Covariate |
| `kras.mutation` | kras.mutation | 585 | 0 | object | `M`, `WT`, `N/A` | Clinical Covariate |
| `kras.mutation.dna` | kras.mutation.dna | 585 | 0 | object | 27 unique values (e.g. `c.35G>A`, `N/A`) | Clinical Covariate |
| `kras.mutation.exon.number` | kras.mutation.exon.number | 585 | 0 | object | 27 unique values (e.g. `c.35G>A`, `N/A`) | Clinical Covariate |
| `kras.mutation.protein` | kras.mutation.protein | 585 | 0 | object | 15 unique values (e.g. `p.G12D`, `N/A`) | Clinical Covariate |
| `mmr.status` | mmr.status | 585 | 0 | object | `pMMR`, `dMMR`, `N/A` | Clinical Covariate |
| `organism` | organism | 585 | 0 | object | `Homo sapiens` | Clinical Covariate |
| `os.delay_(months)` | os.delay_(months) | 585 | 0 | object | 147 unique values (e.g. `10`, `9`) | Clinical Covariate |
| `os.event` | os.event | 585 | 0 | object | `1`, `0`, `N/A` | CRITICAL (Outcome / Stratification) |
| `rfs.delay` | rfs.delay | 585 | 0 | object | 140 unique values (e.g. `0`, `4`) | CRITICAL (Outcome / Stratification) |
| `rfs.event` | rfs.event | 585 | 0 | object | `1`, `0`, `N/A` | CRITICAL (Outcome / Stratification) |
| `rna.extraction.method` | rna.extraction.method | 583 | 2 | object | 7 unique values (e.g. `Manual-Trizol`, `Manual-Cesium Chloride`) | Clinical Covariate |
| `sex` | sex | 585 | 0 | object | `Male`, `Female` | Clinical Covariate |
| `tnm.m` | tnm.m | 585 | 0 | object | `M1`, `M0`, `N/A`, `MX` | Clinical Covariate |
| `tnm.n` | tnm.n | 585 | 0 | object | `N2`, `N0`, `N1`, `N3`, `N+`, `N/A` | Clinical Covariate |
| `tnm.stage` | tnm.stage | 585 | 0 | object | `4`, `2`, `1`, `3`, `0`, `N/A` | CRITICAL (Outcome / Stratification) |
| `tnm.t` | tnm.t | 585 | 0 | object | 7 unique values (e.g. `T4`, `T3`) | Clinical Covariate |
| `tp53.mutation` | tp53.mutation | 585 | 0 | object | `M`, `WT`, `N/A` | Clinical Covariate |
| `tp53.mutation.dna` | tp53.mutation.dna | 585 | 0 | object | 49 unique values (e.g. `N/A`, `c.610G>T`) | Clinical Covariate |
| `tp53.mutation.exon.number` | tp53.mutation.exon.number | 585 | 0 | object | 49 unique values (e.g. `N/A`, `c.610G>T`) | Clinical Covariate |
| `tp53.mutation.protein` | tp53.mutation.protein | 585 | 0 | object | 48 unique values (e.g. `N/A`, `p.E204X`) | Clinical Covariate |
| `tumor.location` | tumor.location | 585 | 0 | object | `distal`, `proximal`, `N/A` | Clinical Covariate |

## 3. Key Recurrence Variables Identified

The following variables are critical for constructing the primary binary classification endpoint:
- `relapse` / `recurrence`: Binary indicator of cancer return.
- `rfs` / `relapse_free_survival` (or follow-up time): Observation time elapsed from primary resection.
- `stage` / `tumor_stage`: Clinical stage (Stage I, II, III, IV). Stage IV (de novo metastatic) must be excluded from the recurrence prediction cohort as per PRD Section 2.1.
- `chemotherapy`: Adjuvant chemotherapy status, potential confounder.

## 4. Cohort Inclusion & Exclusion Strategy & Observed Counts

- **Total Samples in Series Matrix:** 585
  - Primary Colorectal Tumors: 566
  - Non-Tumoral Mucosa / Normal: 19
- **Tumor Stage Filtering (TNM Stage I–III):**
  - Stage I: 38
  - Stage II: 271
  - Stage III: 210
  - Stage IV (Excluded - De Novo Metastatic): 60
  - Stage 0 / Non-Tumoral / Missing (Excluded): 6
  - **Stage I–III Eligible Subcohort:** 502 tumours (after excluding 19 non-tumoral samples)
- **Recurrence Classification Outcomes:**
  - **Class 1 (High Early Recurrence Risk, <= 12 months):** 56 patients (22.5%)
  - **Class 0 (Long-Term Disease-Free, >= 60 months):** 193 patients (77.5%)
  - **Total Binary Classification Dataset:** 249 tumours (the 19 non-tumoral mucosa samples, `dataset = "Non Tumoral"`, are excluded)
- **Exclusion Accounting (to prevent label noise & data leakage):**
  - Non-tumoral mucosa samples: 19
  - Stage IV or unknown stage: 64
  - Late Recurrence (> 12 months): 84 patients
  - Inadequate Follow-up (< 60 months without documented recurrence): 160 patients
  - Missing Recurrence Status: 9 patients
  - Total Excluded from Binary Endpoint: 336 samples
