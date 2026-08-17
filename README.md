# Data Mining Regression Dashboard 

A data mining project analyzing product defect data (`defects_data.csv`), combining OLAP-style aggregation, dimensionality reduction, regression, clustering, and frequent pattern mining — with results presented in both a Jupyter notebook and an interactive Tkinter GUI.

## Contents

- **`DataMining.ipynb`** — Full analysis notebook: preprocessing, OLAP roll-up/drill-down/pivot, PCA, Decision Tree / KNN / Linear Regression, KMeans clustering, and Apriori/FP-Growth frequent pattern mining.
- **`GUI.py`** — Tkinter dashboard that mirrors the notebook's results across 5 tabs: Model Performance, PCA, KNN Analysis, KMeans Clustering, and OLAP.
- **`defects_data.csv`** — Dataset of product defects (type, severity, location, inspection method, repair cost).


```

> Requires a display (won't run headless over plain SSH without X forwarding).

## Key Finding

Regression models were evaluated on `repair_cost` using defect attributes (`defect_type`, `severity`, `defect_location`, `inspection_method`, `product_id`):

| Model | R² |
|---|---|
| Linear Regression | ≈ -0.21 |
| Decision Tree | ≈ -0.88 |
| KNN | negative across all tested K |

All models perform worse than simply predicting the mean, indicating `repair_cost` is not well explained by these features in this dataset. OLAP aggregation supports this: average repair cost is nearly flat (~₹500–515) across every severity level and defect type.
