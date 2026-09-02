# patient-segmentation-kmeans
Unsupervised K-Means clustering to segment hospital patients into actionable care-priority groups based on visit frequency, stay duration, and spending behavior.
# Patient Segmentation with K-Means Clustering

Unsupervised clustering of hospital patients into behavior-based segments — built to
identify which patients need proactive care management versus which don't, without
any pre-existing labels.

## Business Problem

Hospitals treat large numbers of patients with no systematic way to tell who needs
extra attention. Treating every patient identically wastes resources on low-need
patients while high-complexity patients risk being under-monitored until a costly
readmission. Real healthcare cost research shows a small fraction of patients often
account for a disproportionate share of total spend — the goal of this project is to
surface that group (and others) directly from patient data, so care teams can act on
groups instead of reviewing every chart manually.

## Dataset

**Source:** "Hospital data" (Kaggle, by ibnarahat) — *A Data-Driven Patient Profiling Study*

**Columns:**
| Column | Description |
|---|---|
| `Patient_ID` | Unique identifier (dropped before modeling) |
| `Age` | Patient age |
| `Gender` | Male / Female |
| `Blood_Type` | Blood type |
| `Chronic_Condition` | 0–4 scale (severity/count of chronic conditions) |
| `BMI` | Body Mass Index |
| `Annual_Visits` | Number of hospital visits per year |
| `Avg_Stay_Duration` | Average length of stay, in days |
| `Total_Spending` | Total annual cost |
| `Insurance_Type` | Insurance category — significant missing values, likely representing uninsured patients rather than missing data |

## Methodology

1. **EDA** — checked for duplicates and nulls (`Insurance_Type` had meaningful missing
   values, treated as its own "None" category rather than imputed); reviewed skew and
   outliers on numeric columns.
2. **Feature engineering** — dropped `Patient_ID`; selected `Age`, `Chronic_Condition`,
   `BMI`, `Annual_Visits`, `Avg_Stay_Duration`, and `Total_Spending` as clustering
   features (behavioral/clinical signal); `Gender` and `Insurance_Type` reserved for
   post-hoc profiling rather than clustering inputs.
3. **Preprocessing** — `StandardScaler` applied to all clustering features so no single
   feature (e.g., `Total_Spending`, largest raw range) dominates the distance
   calculation.
4. **Choosing K** — Elbow Method and Silhouette Score computed for K = 2–7.
   - Elbow curve was gradual with no sharp bend (weak evidence on its own).
   - Silhouette peaked at K=2 (0.317), with K=3 second-highest.
   - **K=3 was selected over the statistically "best" K=2** because it revealed a
     third, genuinely distinct persona — frequent-visit, high-BMI outpatients — that
     K=2 was folding into a broader "low-need" group. This is a case where cluster
     interpretability and business usefulness outweighed a marginally higher
     Silhouette score.
5. **Modeling** — `KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)`.
6. **Validation** — PCA (2 components) used to visually confirm cluster separation
   across all six clustering features at once, not just two at a time.

## Results — Cluster Profiles (K=3)

| Cluster | Age | Chronic Condition | BMI | Annual Visits | Avg Stay (days) | Total Spending | Count |
|---|---|---|---|---|---|---|---|
| 0 — Low-Risk / Healthy | 37.1 | 1.1 | 33.8 | 2.2 | 2.4 | $924.5 | 38 |
| 1 — High-Complexity / Inpatient | 60.0 | 1.5 | 39.1 | 4.1 | 6.0 | $1,794.1 | 36 |
| 2 — Frequent Monitoring / High-BMI | 38.1 | 1.5 | 44.8 | 5.9 | 2.6 | $1,262.3 | 35 |

### Persona summaries

- **Low-Risk / Healthy** — younger, low visit frequency, short/no stays, lowest
  spending. Minimal intervention needed; standard preventive care is sufficient.
- **High-Complexity / Inpatient** — older, longest average stays, highest spending.
  Best candidates for proactive care coordination and readmission-prevention
  programs — the group most likely to drive disproportionate cost if left unmanaged.
- **Frequent Monitoring / High-BMI** — similar age to the healthy group but the
  *highest* visit frequency and BMI, despite short individual stays. Suggests an
  actively-managed, weight-related condition requiring frequent outpatient
  monitoring rather than inpatient care — a distinct care pathway from Cluster 1.

## Visualizations

- Elbow Method & Silhouette Score plots across K=2–7 (justifying the K decision).
- Direct scatter: `Annual_Visits` vs `Avg_Stay_Duration`, colored by persona.
- PCA (2 components) scatter across all six clustering features, confirming Clusters
  1 and 2 are cleanly separated, with Cluster 0 showing natural, expected overlap at
  its edges.

## Key Takeaways / Business Recommendations

- Route **Cluster 1** patients into proactive care coordination — this group carries
  the highest cost and longest stays and is the highest-value target for early
  intervention.
- **Cluster 2** patients need a different response than their "healthy-looking" age
  and stay-length would suggest — frequent visits and high BMI point to an ongoing
  condition needing consistent outpatient monitoring, not emergency escalation.
- **Cluster 0** patients can remain on standard care pathways — reallocating
  resources away from this group toward Clusters 1 and 2 is the efficiency gain this
  segmentation enables.
- Statistical "best K" (Silhouette) and "most useful K" aren't always the same
  choice — validate against domain interpretability, not just the metric.

## Limitations

- Cluster boundaries reflect the specific features chosen; adding lab results,
  diagnosis codes, or medication data could reveal further sub-segments.
- `Chronic_Condition`'s exact meaning (severity vs. count) was inferred, not
  confirmed from the dataset source — worth verifying before using this in a real
  clinical setting.
- K-Means assumes roughly spherical clusters; the moderate Silhouette scores
  (~0.24–0.32) suggest patient behavior here is more continuous than cleanly
  separated — a reasonable, expected finding for real-world health data, but worth
  stating explicitly rather than overstating cluster purity.

## Tools

`pandas` · `scikit-learn` (`KMeans`, `StandardScaler`, `PCA`, `silhouette_score`) ·
`matplotlib`

## Possible Next Steps

- Compare against DBSCAN to test whether density-based clustering finds the same
  three groups or reveals additional structure/outliers.
- Build a simple Streamlit interface (as done for the mall customer project) so care
  teams can classify a new patient's segment on entry.

  ## ⚠️ Critical Model Limitation: The Hard-Assignment Bottleneck of K-Means

During deployment testing, a critical failure mode was identified: **K-Means cannot say "I don't know" or handle anomalous multi-feature edge cases.**

### The Discovery
When testing acute or unusual patient profiles—such as a 37-year-old patient with low total spending ($1,000) but a 7-day stay and 3 chronic conditions—the model confidently assigned the patient to **Cluster 0 ("Low-Risk / Healthy")**.

### Why K-Means Fails Here
1. **Mathematical Force-Fitting:** K-Means uses hard clustering. It calculates Euclidean distance across all scaled features and assigns the observation to the single nearest centroid, even if that centroid is mathematically far away in absolute terms.
2. **Feature Dominance:** Low annual spending and younger age pulled the vector closer to Cluster 0, completely diluting severe acute metrics like a 7-day inpatient stay.
3. **Inability to Detect Anomaly vs. Normality:** Healthcare is non-linear and unpredictable. Pure K-Means has no mechanism to flag a novel feature combination as an outlier or "unclassifiable."

### Why Explicit `if/else` Rules Are NOT the Solution
Hardcoding clinical threshold rules (e.g., `if stay_duration > 5: high_risk`) invalidates the machine learning paradigm. Healthcare edge cases (e.g., prolonged stays due to sudden weakness, acute poisonings, or diagnostic delays) are too diverse to capture via static logic.

### Proposed Production Fixes
To resolve this without resorting to static hardcoded rules:
- **Transition to Density-Based Clustering (DBSCAN):** Unlike K-Means, DBSCAN natively incorporates a noise class (`-1`) for sparse, unclassifiable observations that do not fit defined clusters.
- **Distance-Based Outlier Gating:** Measure the distance between the input vector and the nearest centroid against training distribution variances. If $D_{\text{min}} > \mu + 2\sigma$, flag the patient for **Manual Clinical Review** rather than assigning a confidence label.