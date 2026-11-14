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

5. **Research-Only Scope:**
   - This machine learning pipeline is an academic exploration of transcriptomic patterns and is **not** a clinical diagnostic or treatment decision support system.

