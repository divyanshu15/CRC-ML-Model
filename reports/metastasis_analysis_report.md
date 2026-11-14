# Colorectal Cancer Liver Metastasis Analysis Report (GSE50760)

**Dataset:** GSE50760 (Illumina HiSeq 2000 RNA-seq, log2(FPKM + 1)); 18 patients x (normal colon, primary CRC, liver metastasis) = 54 samples.

*This report is generated from the pipeline outputs by `src/metastasis_analysis.py`.*

---

## 1. Normal-liver admixture in the metastasis samples

The mean log2 expression of 30 hepatocyte marker genes (ALB, APOA1, FGA/FGB/FGG, SERPINA1, CYP2E1, ...) is
**1.14** in primary tumours and **7.57** in liver metastases (medians).
15 of 18 metastasis samples carry a strong liver signal; 3 (AMC_13, AMC_22, AMC_3) carry almost none.
Without adjustment, "metastasis vs primary" differences are dominated by normal liver tissue
(the unadjusted top pathways were complement/coagulation, bile secretion and cytochrome P450 metabolism).

## 2. Paired differential expression

- Genes tested: 23,505
- FDR < 0.05, unadjusted paired t-test: 700
- **FDR < 0.05, liver-adjusted (primary analysis): 0** (0 up, 0 down in metastasis)

Liver adjustment: per gene, the within-patient difference (metastasis - primary) is regressed on the
pair's excess hepatocyte-marker signal; the intercept is the estimated metastasis effect at zero liver admixture.

### Top liver-adjusted genes (hepatocyte markers excluded)

| Rank | Gene | Adjusted log2FC | Unadjusted log2FC | Adjusted p | Adjusted FDR |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | `FAM138F` | -0.04 | -0.01 | 1.82e-04 | 1.00e+00 |
| 2 | `ANKRD18DP` | -0.02 | +0.00 | 2.62e-04 | 1.00e+00 |
| 3 | `SERPINI2` | -0.03 | -0.01 | 3.99e-04 | 1.00e+00 |
| 4 | `ZNF354A` | +0.48 | +0.04 | 4.84e-04 | 1.00e+00 |
| 5 | `SDF2` | -0.66 | -0.29 | 6.20e-04 | 1.00e+00 |
| 6 | `OLIG3` | -0.03 | -0.00 | 8.99e-04 | 1.00e+00 |
| 7 | `MC3R` | +0.05 | +0.01 | 9.26e-04 | 1.00e+00 |
| 8 | `KRTAP3-3` | +0.24 | +0.07 | 9.84e-04 | 1.00e+00 |
| 9 | `GBA` | -0.45 | -0.26 | 1.07e-03 | 1.00e+00 |
| 10 | `SUPT6H` | -0.77 | -0.11 | 1.18e-03 | 1.00e+00 |
| 11 | `C11orf88` | -0.12 | -0.02 | 1.26e-03 | 1.00e+00 |
| 12 | `RNU4ATAC` | -4.66 | -0.74 | 1.34e-03 | 1.00e+00 |
| 13 | `LOC650368` | -1.06 | -0.37 | 1.35e-03 | 1.00e+00 |
| 14 | `FAM90A7P` | +0.06 | +0.01 | 1.39e-03 | 1.00e+00 |
| 15 | `DUX2` | +0.02 | -0.00 | 1.54e-03 | 1.00e+00 |

## 3. Pathway enrichment (liver-adjusted upregulated genes)

No enrichment results (too few genes, or the Enrichr service was unreachable).

## 4. Overlap with recurrence candidates

0 of the top-50 GSE39582 recurrence candidates (ranked by nominal p; none FDR-significant for recurrence)
are significantly changed in metastasis after liver adjustment.
This overlap is hypothesis-generating only.

## 5. Caveats

- 18 pairs; the liver adjustment uses a linear model with a marker-based score, which cannot remove all admixture effects (e.g. stromal or immune differences between colon and liver).
- FPKM values from GEO; count-based methods (e.g. limma-voom/DESeq2 with a patient term) would be preferable if raw counts were available.
