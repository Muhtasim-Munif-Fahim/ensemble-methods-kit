"""ensemble-methods-kit: a from-scratch NumPy toolkit of ensemble learning methods.

The package re-implements common ensemble strategies (bagging, random subspace (classifier and regressor), random forest,
extremely randomized trees (classifier and regressor), AdaBoost (classifier and regressor), gradient boosting, histogram gradient
boosting (classifier and regressor), voting, stacking (classifier and regressor), blending (classifier and regressor), Rotation Forest, Caruana ensemble selection, and KNORA dynamic ensemble selection). Bagging and random forest cover
classification and regression. Most estimators sit on a small CART decision
tree; histogram gradient boosting grows its own binned Newton trees.
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
from .bagging import BaggingClassifier, BaggingRegressor, RandomSubspaceClassifier, RandomSubspaceRegressor
from .rotation_forest import RotationForestClassifier
from .random_forest import RandomForestClassifier, RandomForestRegressor
from .extra_trees import ExtraTreesClassifier, ExtraTreesRegressor
from .adaboost import AdaBoostClassifier, AdaBoostRegressor
from .gradient_boosting import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from .histogram_gradient_boosting import (
    HistogramGradientBoostingClassifier,
    HistogramGradientBoostingRegressor,
)
from .voting import VotingClassifier, VotingRegressor
from .ensemble_selection import EnsembleSelectionClassifier, EnsembleSelectionRegressor
from .des import KNORAClassifier
from .stacking import (
    BlendingClassifier,
    BlendingRegressor,
    LogisticRegression,
    RidgeRegression,
    StackingClassifier,
    StackingRegressor,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DecisionTree",
    "BaggingClassifier",
    "BaggingRegressor",
    "RandomSubspaceClassifier",
    "RandomSubspaceRegressor",
    "RotationForestClassifier",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistogramGradientBoostingClassifier",
    "HistogramGradientBoostingRegressor",
    "VotingClassifier",
    "VotingRegressor",
    "EnsembleSelectionClassifier",
    "EnsembleSelectionRegressor",
    "KNORAClassifier",
    "StackingClassifier",
    "StackingRegressor",
    "BlendingClassifier",
    "BlendingRegressor",
    "LogisticRegression",
    "RidgeRegression",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
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
