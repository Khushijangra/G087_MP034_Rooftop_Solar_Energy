# Faculty Viva Voce & Technical Defense Guide

**Group:** G087 | **Problem ID:** MP034 | **Faculty:** Dr. Gaurav Tripathi  
**Candidate:** Khushi Jangra (Data Lead)

This quick-reference guide prepares the candidate for the individual viva by providing rigorous mathematical, architectural, and code-level justifications for every project decision.

---

## 1. Data Cleaning & Integrity Viva Questions

### Q1: Why did you divide `panel_area_m2` by 10,000 only for New Colony?
**Answer:** During our grouping analysis (`train.groupby('area')['panel_area_m2'].mean()`), New Colony had a mean of ~332,501 while Industrial and Old City had means of ~33.2. A ratio of 10,000 indicates that New Colony values were entered in square centimetres ($\text{cm}^2$) rather than square metres ($\text{m}^2$). Since $1\text{ m}^2 = (100\text{ cm})^2 = 10,000\text{ cm}^2$, dividing by 10,000 strictly normalizes the physical units across the entire city.

### Q2: How did you identify target typos versus natural outliers?
**Answer:** Sorting `energy_kwh` revealed a distinct multimodal break: 991 records sat between 12.3 and 64.4 kWh, followed by a vast empty gap, and then exactly 9 records clustered between 249.6 and 533.6 kWh. Dividing those 9 records by 10 brought their distribution into exact alignment (24.96 to 53.36 kWh) with rooftop solar physics.

---

## 2. Drift & Retraining Viva Questions

### Q3: What is PSI and why was `temp_c` flagged?
**Answer:** Population Stability Index (PSI) measures the change in distribution between a baseline dataset $P$ and a target dataset $Q$:
$$\text{PSI} = \sum_{i=1}^k (Q_i - P_i) \times \ln\left(\frac{Q_i}{P_i}\right)$$
Using 10 decile bins, `temp_c` yielded a PSI of $3.6698$ (well above the critical threshold of 0.25), reflecting seasonal temperature drift between spring/summer (training set) and monsoon/post-monsoon (later set).

### Q4: Explain the 10% model promotion rule.
**Answer:** We evaluated baseline v1 and retrained v2 on the chronologically holdout final 30% of `later.csv`. v1 achieved an MAE of 4.6934 kWh, while v2 achieved 4.1535 kWh. The relative improvement was:
$$\frac{4.6934 - 4.1535}{4.6934} = 11.503\% \ge 10.0\%$$
Meeting the client's non-negotiable threshold, v2 was automatically promoted to `@champion` in the MLflow model registry.
