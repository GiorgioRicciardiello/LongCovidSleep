# Long COVID Sleep Symptom Analysis

This project provides a comprehensive analysis of sleep disturbances among Long COVID patients, using structured questionnaire data (ASQ, ISI, ESS, PSQI, rMEQ) from a specialized clinic. The work goes beyond prevalence estimates to uncover symptom domains, identify patient subgroups, and explore clinical and demographic predictors of poor sleep outcomes.

---

## Key Insights from the Data

1. **High Burden of Sleep Disturbances**

   * Nearly **9 in 10 patients** reported at least one sleep complaint.
   * Insomnia (42.5%), sleep-related breathing issues (57.5%), and excessive daytime sleepiness (28.5%) emerged as particularly prevalent.
   * These rates exceed many prior reports, underscoring the magnitude of the problem in Long COVID.

2. **Distinct but Overlapping Symptom Domains**

   * Factor analysis revealed **nine interpretable dimensions**, including insomnia/unrefreshing sleep, fatigue/post-exertional malaise, parasomnia behaviors, and respiratory complaints.
   * This multidimensional structure highlights that Long COVID sleep problems are not a single entity, but a cluster of interlinked issues.

    
<img src="results/github/umap_nice_plot.png" width="400">
<img src="results/github/radar_clusters.png" width="400">
<img src="results/github/heatmap_loadings_all.png" width="400">



3. **Patient Subgroups with Higher Symptom Load**

   * Gaussian mixture modeling identified two broad patient groups.
   * One subgroup (“**High-Symptom Cluster**”) had higher BMI, more severe insomnia and daytime sleepiness, and greater prevalence of gastrointestinal and parasomnia complaints.
   * The other subgroup reported milder, though still significant, sleep symptoms.

4. **Predictors of Insomnia**

   * **Hospitalization during acute COVID** was strongly linked to later insomnia (OR \~4.4).
   * Individuals identifying as **Multiracial** showed a trend toward higher insomnia risk (OR \~3.2).
   * No consistent associations were found with age, sex, or illness duration.

---

## Relevance

* **For clinicians**: Detailed screening beyond insomnia is critical. Symptoms such as parasomnias, restless legs, and circadian disruption are under-recognized yet impair quality of life.
* **For healthcare systems**: Incorporating structured sleep assessments into Long COVID care pathways can improve patient management and reduce downstream burden (fatigue, reduced work capacity).
* **For researchers**: The weak cluster separation highlights the need for larger, multi-center studies and integration of objective sleep measures to identify reproducible subtypes.

---

## Methodology at a Glance

* **Questionnaires**: ASQ, ISI, ESS, PSQI, rMEQ.
* **Statistical methods**: Parallel analysis, PCA with varimax rotation, Gaussian mixture modeling, logistic regression, cross-validation.
* **Software**: Python (scikit-learn, statsmodels, scipy, umap-learn).
* **Validation**: Internal cluster validity metrics and cross-validated predictive performance.

---

## 🚀 Strategic Takeaway

This project demonstrates that sleep disturbances in Long COVID are **common, heterogeneous, and clinically meaningful**. Identifying subgroups with greater symptom load enables targeted interventions and helps guide both **clinical resource allocation** and **future research priorities**.

