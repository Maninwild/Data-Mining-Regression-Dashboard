# Data Mining Regression Dashboard

A reproducible analysis of 1,000 product-defect records. The project combines regression benchmarking, OLAP-style summaries, PCA, KMeans clustering, and frequent-itemset mining in a shared Python module used by both a notebook and a Tkinter dashboard.

## What was improved

- Loads `defects_data.csv` relative to the project instead of relying on the current working directory or Google Colab paths.
- Keeps `defect_id` out of the model and converts the raw date into month and weekday features.
- Compares every regressor with a mean baseline on the same held-out test split.
- Scales KNN inputs and selects `k` with five-fold cross-validation on training data only.
- Uses constrained tree settings to reduce overfitting.
- Runs PCA and KMeans on standardized features.
- Provides dependency-free Apriori frequent-itemset mining for this small dataset.
- Shares one analysis implementation across the notebook, GUI, and automated tests.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the project

Launch the desktop dashboard:

```bash
python GUI.py
```

Tkinter needs a graphical desktop. On some Linux systems, the OS package for Tkinter must be installed separately.

Or open the analysis notebook:

```bash
jupyter notebook "Data Mining Regression Dashboard.ipynb"
```

Jupyter itself is optional and is not included in `requirements.txt` because the GUI and test suite do not need it.

## Reproduced model results

The checked-in pipeline uses a deterministic 80/20 split (`random_state=42`). Current held-out results are:

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Mean baseline | -0.003 | ₹254.62 | ₹293.22 |
| Linear regression | -0.164 | ₹267.49 | ₹315.81 |
| Decision tree | -0.042 | ₹257.90 | ₹298.78 |
| Scaled KNN (`k=13`) | -0.098 | ₹263.94 | ₹306.68 |

Negative held-out R² is an analytical result, not a display error: these recorded attributes do not predict repair cost better than the training-set mean. The severity groups tell a similar story, with average repair costs ranging only from about ₹501.63 to ₹514.43. Richer cost drivers would be needed for a useful predictive model.

## Project structure

| Path | Purpose |
|---|---|
| `analysis.py` | Validated loading, features, models, PCA, clustering, OLAP, and itemsets |
| `GUI.py` | Five-tab Tkinter dashboard |
| `Data Mining Regression Dashboard.ipynb` | Reproducible exploratory workflow |
| `defects_data.csv` | Source dataset |
| `test_analysis.py` | Pipeline regression tests |
| `.github/workflows/ci.yml` | Automated validation on pushes and pull requests |

## Validate

```bash
python -m compileall -q analysis.py GUI.py test_analysis.py
python -m unittest -v
```

## License

MIT — see [LICENSE](LICENSE).
