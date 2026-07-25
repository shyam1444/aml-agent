"""
Anomaly Detection Tool using scikit-learn (IsolationForest + LocalOutlierFactor).
Computes ML anomaly scores per entity, normalized to [0.0, 1.0].
"""

from typing import Any
import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def detect_anomalies_ml(
    features_df: pl.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42
) -> dict[str, Any]:
    """
    Runs IsolationForest and LOF over computed feature matrix.
    Returns:
    - scores_df: pl.DataFrame (Account, ml_score, if_score, lof_score)
    - model: fitted IsolationForest instance (for SHAP)
    - feature_names: list of column names used
    """
    if features_df is None or len(features_df) == 0:
        return {
            "scores_df": pl.DataFrame(),
            "model": None,
            "feature_cols": []
        }

    # Select numerical feature columns
    ignore_cols = {"Account", "parsed_timestamp"}
    feature_cols = [c for c in features_df.columns if c not in ignore_cols and features_df[c].dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]]

    if not feature_cols:
        return {
            "scores_df": pl.DataFrame(),
            "model": None,
            "feature_cols": []
        }

    X = features_df.select(feature_cols).to_numpy()

    # Fill NaNs or Infs
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # 1. Isolation Forest
    clf_if = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state
    )
    clf_if.fit(X)
    raw_if_scores = -clf_if.score_samples(X)  # Higher = more anomalous

    # Min-Max normalize IF scores to [0, 1]
    if_min, if_max = raw_if_scores.min(), raw_if_scores.max()
    if_denom = (if_max - if_min) if if_max > if_min else 1.0
    norm_if_scores = (raw_if_scores - if_min) / if_denom

    # 2. Local Outlier Factor
    n_neighbors = min(20, max(2, len(X) - 1))
    clf_lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    _ = clf_lof.fit_predict(X)
    raw_lof_scores = -clf_lof.negative_outlier_factor_
    lof_min, lof_max = raw_lof_scores.min(), raw_lof_scores.max()
    lof_denom = (lof_max - lof_min) if lof_max > lof_min else 1.0
    norm_lof_scores = (raw_lof_scores - lof_min) / lof_denom

    # Ensembled ML score (70% IF + 30% LOF)
    ml_scores = 0.70 * norm_if_scores + 0.30 * norm_lof_scores

    scores_df = features_df.select(["Account"]).with_columns([
        pl.Series("ml_score", ml_scores),
        pl.Series("if_score", norm_if_scores),
        pl.Series("lof_score", norm_lof_scores)
    ])

    return {
        "scores_df": scores_df,
        "model": clf_if,
        "feature_cols": feature_cols,
        "X_data": X
    }
