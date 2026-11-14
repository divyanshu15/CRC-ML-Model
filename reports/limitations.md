# Scientific & Methodological Limitations

This research project operates under defined scientific and clinical limitations that must be explicitly acknowledged in any thesis or publication:

1. **Retrospective Microarray Data:**
   - Datasets GSE39582 and GSE14333 originate from retrospective observational cohorts collected over a decade ago. Treatment regimens (such as standard-of-care FOLFOX/FOLFIRI adjuvant chemotherapy) have evolved.

2. **Inherent Biological Complexity:**
   - Colorectal cancer recurrence is not determined exclusively by tumor gene expression. Tumor microenvironment (stromal infiltration, tumor-infiltrating lymphocytes), surgical margins, anatomical lymph node yield, and post-operative surveillance frequency also strongly influence clinical outcomes.

3. **Batch Effects Across Cohorts:**
   - Even though GSE39582 and GSE14333 share the GPL570 platform, differences in RNA extraction protocols, hybridization facilities, and scanner calibration introduce technical batch variance.

4. **Class Imbalance:**
   - Acute early recurrence (< 1 year) represents a minority subset of resected CRC patients (~15–25%). Imbalance mitigation techniques (class weighting, PR-AUC assessment) are necessary.

5. **Gene models do not outperform stage (audit, 14 Nov 2026):**
   - Internal nested-CV (0.51–0.54 vs 0.63), external binary (0.70–0.77 vs 0.77) and pooled leave-one-cohort-out (0.62–0.66 vs 0.68) ROC-AUCs are all at or below TNM stage alone. Only the stage-adjusted survival analysis on 226 external patients shows added value.

6. **Small external event counts:**
   - The external binary cohort has 15 events and GSE17536 has 7, so external AUC confidence intervals span roughly ±0.13.

7. **Small tuning grids:**
   - Nested CV searches small grids (6–8 settings per model) to fit the 16 GB / laptop budget; larger searches might change results slightly. TCGA (n = 57) uses fixed hyperparameters.

8. **Cross-cohort normalisation:**
   - External predictions require per-gene alignment to the training distribution; without it, frozen models default to the majority class. Predicted probabilities are not calibrated across cohorts.

9. **Liver-tissue contamination in GSE50760:**
   - 15 of 18 metastasis biopsies contain substantial normal liver tissue. After a marker-based linear adjustment no gene is significant; the adjustment cannot remove all admixture effects, and 18 pairs give little power.

10. **No FDR-significant single genes:**
   - No gene reaches FDR < 0.10 for early recurrence in GSE39582; gene selection is unstable across folds, and SHAP rankings come from a model with nested-CV AUC ≈ 0.54, so they are exploratory.

11. **TCGA endpoint:**
   - Under the curated ≤ 12 mo vs ≥ 60 mo endpoint only 57 TCGA patients qualify, too few for meaningful modelling.

12. **Multiple comparisons in survival validation:**
   - Eight classifier variants were tested in two external cohorts; the consistency across models and cohorts is the main evidence. A single pre-specified model validated in a further cohort is still needed. The XGBoost-Cox survival model did not replicate in GSE17536.

13. **Research-Only Scope:**
   - This machine learning pipeline is an academic exploration of transcriptomic patterns and is **not** a clinical diagnostic or treatment decision support system.

