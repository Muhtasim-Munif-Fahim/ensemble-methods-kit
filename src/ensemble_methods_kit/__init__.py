"""ensemble-methods-kit: a from-scratch NumPy toolkit of ensemble learning methods.

The package re-implements common ensemble strategies (bagging, random subspace (classifier and regressor), random forest,
extremely randomized trees (classifier and regressor), AdaBoost (classifier and regressor), gradient boosting, histogram gradient
boosting (classifier and regressor), voting, stacking (classifier and regressor), blending (classifier and regressor), Rotation Forest, Caruana ensemble selection, KNORA dynamic ensemble selection, OLA/LCA dynamic classifier selection, Kuncheva diversity measures with diversity-based pruning, EasyEnsemble / BalanceCascade for imbalanced classification, Meinshausen quantile regression forests for prediction intervals, and the van der Laan Super Learner). Bagging and random forest cover
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
from .quantile_forest import QuantileRegressionForest, interval_coverage, pinball_loss
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
from .super_learner import SuperLearnerClassifier, SuperLearnerRegressor, nnls, project_to_simplex
from .ensemble_selection import EnsembleSelectionClassifier, EnsembleSelectionRegressor
from .des import DCSClassifier, KNORAClassifier
from .imbalance import BalanceCascadeClassifier, EasyEnsembleClassifier
from .diversity import (
    average_pairwise_diversity,
    correlation_coefficient,
    disagreement,
    diversity_prune,
    diversity_report,
    double_fault,
    ensemble_member_predictions,
    entropy_diversity,
    interrater_agreement,
    kohavi_wolpert_variance,
    majority_vote,
    oracle_matrix,
    pairwise_diversity,
    q_statistic,
)
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
    "SuperLearnerClassifier",
    "SuperLearnerRegressor",
    "nnls",
    "project_to_simplex",
    "DecisionTree",
    "BaggingClassifier",
    "BaggingRegressor",
    "RandomSubspaceClassifier",
    "RandomSubspaceRegressor",
    "RotationForestClassifier",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "QuantileRegressionForest",
    "interval_coverage",
    "pinball_loss",
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
    "DCSClassifier",
    "EasyEnsembleClassifier",
    "BalanceCascadeClassifier",
    "average_pairwise_diversity",
    "correlation_coefficient",
    "disagreement",
    "diversity_prune",
    "diversity_report",
    "double_fault",
    "ensemble_member_predictions",
    "entropy_diversity",
    "interrater_agreement",
    "kohavi_wolpert_variance",
    "majority_vote",
    "oracle_matrix",
    "pairwise_diversity",
    "q_statistic",
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
