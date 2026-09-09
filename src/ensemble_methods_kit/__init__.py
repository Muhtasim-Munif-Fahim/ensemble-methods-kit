"""ensemble-methods-kit: a from-scratch NumPy toolkit of ensemble learning methods.

The package re-implements common ensemble strategies (bagging, random forest,
gradient boosting, voting, stacking and blending) on top of a small CART
decision tree so that the whole workflow is dependency-light and easy to follow.
"""

from .utils import (
    DecisionTree,
    accuracy_score,
    balanced_sample_weights,
    bootstrap_sample,
    clone_estimator,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    train_test_split,
)
from .bagging import BaggingClassifier
from .random_forest import RandomForestClassifier
from .adaboost import AdaBoostClassifier
from .gradient_boosting import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from .voting import VotingClassifier
from .stacking import (
    BlendingClassifier,
    LogisticRegression,
    StackingClassifier,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DecisionTree",
    "BaggingClassifier",
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "VotingClassifier",
    "StackingClassifier",
    "BlendingClassifier",
    "LogisticRegression",
    "AdaBoostClassifier",
    "accuracy_score",
    "balanced_sample_weights",
    "bootstrap_sample",
    "clone_estimator",
    "confusion_matrix",
    "f1_score",
    "log_loss",
    "mean_squared_error",
    "precision_score",
    "r2_score",
    "recall_score",
    "train_test_split",
]
