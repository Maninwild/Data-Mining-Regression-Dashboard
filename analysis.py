"""Reusable analysis pipeline for the product-defects dataset."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


DATA_PATH = Path(__file__).resolve().with_name("defects_data.csv")
RANDOM_STATE = 42


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the source CSV."""
    frame = pd.read_csv(Path(path))
    required = {
        "defect_id",
        "product_id",
        "defect_type",
        "defect_date",
        "defect_location",
        "severity",
        "inspection_method",
        "repair_cost",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["defect_date"] = pd.to_datetime(frame["defect_date"], errors="coerce")
    frame["repair_cost"] = pd.to_numeric(frame["repair_cost"], errors="coerce")
    frame = frame.dropna(subset=list(required)).reset_index(drop=True)
    if frame.empty:
        raise ValueError("Dataset contains no complete valid rows")
    return frame


def prepare_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create model features without using row identifiers or raw date strings."""
    features = frame.drop(columns=["repair_cost", "defect_id", "defect_date"]).copy()
    features["product_id"] = features["product_id"].astype("string")
    features["defect_month"] = frame["defect_date"].dt.month.astype(float)
    features["defect_weekday"] = frame["defect_date"].dt.dayofweek.astype(float)
    encoded = pd.get_dummies(features, drop_first=False, dtype=float)
    return encoded, frame["repair_cost"].astype(float)


def _metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, predictions)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
    }


def evaluate_models(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Evaluate a mean baseline and three regressors on one held-out split."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    baseline = DummyRegressor(strategy="mean")
    linear = TransformedTargetRegressor(
        regressor=Pipeline(
            [("scale", StandardScaler()), ("model", LinearRegression())]
        ),
        transformer=StandardScaler(),
    )
    tree = DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )
    knn_search = GridSearchCV(
        Pipeline(
            [("scale", StandardScaler()), ("model", KNeighborsRegressor())]
        ),
        {"model__n_neighbors": list(range(1, 16))},
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    models = {
        "Mean baseline": baseline,
        "Linear regression": linear,
        "Decision tree": tree,
        "Scaled KNN": knn_search,
    }
    rows: list[dict[str, float | str]] = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        rows.append({"model": name, **_metrics(y_test, model.predict(x_test))})

    curve = pd.DataFrame(
        {
            "k": knn_search.cv_results_["param_model__n_neighbors"].astype(int),
            "mean_cv_r2": knn_search.cv_results_["mean_test_score"],
        }
    ).sort_values("k")
    return pd.DataFrame(rows), curve, int(knn_search.best_params_["model__n_neighbors"])


def cluster_projection(features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a two-dimensional PCA projection and KMeans cluster labels."""
    scaled = StandardScaler().fit_transform(features)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    projection = pca.fit_transform(scaled)
    clusters = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE).fit_predict(scaled)
    return projection, clusters, pca.explained_variance_ratio_


def olap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repair-cost count mean and total by severity."""
    return (
        frame.groupby("severity", observed=True)["repair_cost"]
        .agg(defects="count", average_cost="mean", total_cost="sum")
        .sort_values("average_cost", ascending=False)
    )


def mine_frequent_patterns(
    frame: pd.DataFrame,
    min_support: float = 0.3,
) -> pd.DataFrame:
    """Mine frequent defect-type and severity itemsets per product.

    This small Apriori implementation keeps the project reproducible without a
    separate association-rule package. Candidate sets are retained only when
    all of their immediate subsets were frequent in the previous pass.
    """
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be greater than 0 and at most 1")

    transactions = [
        frozenset(
            [
                *(f"type:{value}" for value in group["defect_type"].unique()),
                *(f"severity:{value}" for value in group["severity"].unique()),
            ]
        )
        for _, group in frame.groupby("product_id", observed=True)
    ]
    if not transactions:
        return pd.DataFrame(columns=["support", "itemsets", "size"])

    all_items = sorted(set().union(*transactions))
    previous = {frozenset([item]) for item in all_items}
    rows: list[dict[str, object]] = []
    size = 1
    while previous:
        frequent: set[frozenset[str]] = set()
        for candidate in sorted(previous, key=lambda values: sorted(values)):
            support = sum(candidate <= transaction for transaction in transactions) / len(transactions)
            if support >= min_support:
                frequent.add(candidate)
                rows.append({"support": support, "itemsets": candidate, "size": size})

        size += 1
        candidate_pool = {
            frozenset(candidate)
            for candidate in combinations(all_items, size)
            if all(frozenset(subset) in frequent for subset in combinations(candidate, size - 1))
        }
        previous = candidate_pool

    result = pd.DataFrame(rows, columns=["support", "itemsets", "size"])
    if result.empty:
        return result
    return (
        result.assign(_items=result["itemsets"].map(lambda values: ", ".join(sorted(values))))
        .sort_values(["support", "size", "_items"], ascending=[False, True, True])
        .drop(columns="_items")
        .reset_index(drop=True)
    )


def run_analysis(path: str | Path = DATA_PATH) -> dict[str, object]:
    """Run every analysis used by the GUI and notebook."""
    frame = load_dataset(path)
    features, target = prepare_features(frame)
    metrics, knn_curve, best_k = evaluate_models(features, target)
    projection, clusters, explained_variance = cluster_projection(features)
    return {
        "frame": frame,
        "features": features,
        "metrics": metrics,
        "knn_curve": knn_curve,
        "best_k": best_k,
        "projection": projection,
        "clusters": clusters,
        "explained_variance": explained_variance,
        "olap": olap_summary(frame),
    }
