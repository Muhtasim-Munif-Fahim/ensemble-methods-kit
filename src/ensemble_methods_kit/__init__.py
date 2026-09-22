"""ensemble-methods-kit: a from-scratch NumPy toolkit of ensemble learning methods.

The package re-implements common ensemble strategies (bagging, random forest,
extremely randomized trees, AdaBoost, gradient boosting, histogram gradient
boosting, voting, stacking and blending). Most of them sit on a small CART
decision tree; histogram gradient boosting grows its own binned Newton trees.
The whole workflow stays dependency-light and easy to follow.
"""

from .utils import (
    DecisionTree,
    accuracy_score,
    balanced_sample_weights,
    bootstrap_sample,
    clone_estimator,
    confusion_matrix,
    mean_decrease_impurity,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    train_test_split,
)
from .bagging import BaggingClassifier
from .random_forest import RandomForestClassifier
from .extra_trees import ExtraTreesClassifier
from .adaboost import AdaBoostClassifier
from .gradient_boosting import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from .histogram_gradient_boosting import HistogramGradientBoostingClassifier
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
    "ExtraTreesClassifier",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistogramGradientBoostingClassifier",
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
    "mean_decrease_impurity",
    "f1_score",
    "log_loss",
    "mean_squared_error",
    "precision_score",
    "r2_score",
    "recall_score",
    "roc_auc_score",
    "train_test_split",
]
