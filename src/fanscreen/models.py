"""Model specifications.

All models use fixed, untuned settings so that the comparison with the one-predictor
Melton model is not driven by hyperparameter search on 104 catchments.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = {"LR": "Logistic regression", "RF": "Random Forest", "GB": "Gradient boosting"}


def build_models() -> dict:
    """Return fresh, unfitted copies of the three model types."""
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "RF": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, max_features="sqrt", random_state=1, n_jobs=1
        ),
        "GB": HistGradientBoostingClassifier(
            max_depth=2, learning_rate=0.05, max_iter=100, min_samples_leaf=5, random_state=1
        ),
    }
