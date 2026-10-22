# Cloud Anomaly Detection using Machine Learning

Detecting anomalous cloud workloads from resource-usage telemetry (CPU, memory, disk I/O, network I/O) with a supervised **Random Forest Classifier**.

Mini-project for MBA (Artificial Intelligence & Data Science), Graphic Era (Deemed to be University), Dehradun.

## Problem

Cloud monitoring often relies on static alert thresholds that create alert fatigue and miss unusual patterns. This project explores whether a machine learning model can classify a workload snapshot as **normal** or **anomalous** from its resource metrics. That helps DevOps teams prioritise the alerts that matter.

## Dataset

`cloud_dataset.csv`: 14,400 rows × 8 columns. Per-minute readings for one day (2025-07-01) across 10 users and 5 workload types (Database_Query, Video_Streaming, Web_Service, Crypto_Mining, Backup).

| Column | Description |
|---|---|
| `Timestamp` | Reading time (1-minute intervals) |
| `CPU_Usage` | CPU utilisation (%) |
| `Memory_Usage` | Memory utilisation (%) |
| `Disk_IO` | Disk I/O activity |
| `Network_IO` | Network I/O activity |
| `Workload_Type` | Type of workload running |
| `User_ID` | User identifier |
| `Anomaly_Label` | Target: 0 = normal, 1 = anomaly |

**Class balance:** 13,143 normal vs 1,257 anomalies (~8.7%). The data is imbalanced.

Source: [Cloud Resource Usage Dataset for Anomaly Detection (Kaggle)](https://www.kaggle.com/datasets/programmer3/cloud-resource-usage-dataset-for-anomaly-detection)

## Approach

1. **EDA**: dataset shape and types, missing-value check, class distribution, feature histograms, boxplots for outliers, correlation heatmap
2. **Preprocessing**: median imputation (no missing values were actually present); features `CPU_Usage`, `Memory_Usage`, `Network_IO`, `Disk_IO`
3. **Split**: 80/20 train-test, **stratified** to keep the anomaly ratio
4. **Model**: `RandomForestClassifier(n_estimators=100, class_weight='balanced')`, where class weighting handles the imbalance
5. **Evaluation**: accuracy, confusion matrix, precision / recall / F1, feature importance
6. **Interactive prediction**: enter resource values to classify a new service as normal or anomalous

## Results

| Metric | Value |
|---|---|
| Accuracy | 1.00 |
| Precision / Recall / F1 (anomaly class) | 1.00 / 1.00 / 1.00 |
| Test set | 2,880 rows (2,629 normal, 251 anomalies) |

**Interpreting the perfect score honestly:** further analysis shows that in this dataset every anomaly has `CPU_Usage` > 80% and every normal row has `CPU_Usage` ≤ 80%. The labels follow a CPU threshold rule, and the Random Forest learns that rule exactly. CPU usage dominates the feature importance for this reason. The pipeline works as intended, but a perfect score here reflects how the dataset was labelled. It doesn't show how the model would perform on real-world anomalies, which are messier.

## Limitations & next steps

- Labels are threshold-based, so the task is easier than real anomaly detection
- `Workload_Type` and time-based features (hour, rolling averages) aren't used yet
- Next steps: test on data with real incident labels; compare with **unsupervised** methods (Isolation Forest, One-Class SVM) that don't need labels; add ROC-AUC / PR-AUC and cross-validation

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Project_Final.ipynb
```

The notebook was written in Google Colab and reads the data from `/content/cloud_dataset.csv`. To run it locally, change that path to `cloud_dataset.csv`.

## Tech stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Jupyter / Google Colab

## Author

**Abhay Rawat**, MBA (AI & DS), Graphic Era (Deemed to be University) · [LinkedIn](https://www.linkedin.com/in/abhay-rawat-4a79b52b4/)
