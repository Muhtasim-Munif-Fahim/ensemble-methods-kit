# ensemble-methods-kit

A from-scratch NumPy toolkit of **ensemble learning methods** with a complete
demo workflow that trains several ensembles, benchmarks them against each other
and against scikit-learn baselines, and writes a Markdown report.

Most estimators are implemented on top of a small CART decision tree. Histogram
gradient boosting is the exception: it bins each feature and grows Newton trees
on those bins. The package has a single hard runtime dependency (`numpy`) and
the algorithms are easy to read and extend.

## Features

- **DecisionTree** — CART with gini / entropy splits for classification and
  variance reduction for regression, plus feature sub-sampling and Mean
  Decrease Impurity (`feature_importances_`).
- **BaggingClassifier / BaggingRegressor** — bootstrap aggregating of CART
  decision trees. The regressor averages tree predictions. Pass
  `oob_score=True` to record an out-of-bag R² (`oob_score_`) and the
  left-out predictions (`oob_prediction_`).
- **RandomSubspaceClassifier / RandomSubspaceRegressor** — Ho (1998) Random
  Subspace Method: each tree sees a fixed random feature subset
  (`max_features` fraction) and, optionally, bootstrap rows. The classifier
  soft-votes like bagging; the regressor averages tree predictions. Both
  store `estimators_features_` and remapped `feature_importances_`.
- **RotationForestClassifier** — Rodriguez, Kuncheva & Alonso (2006) Rotation
  Forest: each tree is trained in a PCA-rotated feature space built from
  random feature groups (with class/instance bootstraps). Soft-votes like
  bagging; stores `rotation_matrices_` and projected `feature_importances_`.
- **RandomForestClassifier / RandomForestRegressor** — bagging with random
  feature sub-sampling (`max_features`: `"sqrt"`, `"log2"`, or an integer
  count). Both expose impurity-based `feature_importances_` (Mean Decrease
  Impurity). The regressor accepts the same optional out-of-bag score.
- **QuantileRegressionForest** — Meinshausen (2006) quantile regression
  forest.
  - The fitted trees define nearest-neighbour weights
    `w_i(x) = mean_t 1{leaf_t(X_i) = leaf_t(x)} / |leaf_t(x)|` over the
    training targets. Their weighted CDF estimates the full conditional
    distribution.
  - `predict_quantiles` and `predict_interval` give distribution-free
    prediction intervals that widen where the noise does.
  - `predict` still returns the forest mean.
  - `oob_predict_quantiles` gives honest out-of-bag intervals for checking
    calibration.
  - `leaf_weighting="inbag"` reproduces the forest mean exactly.
  - Also ships `pinball_loss`, `interval_coverage` and `DecisionTree.apply`
    (leaf indices).
- **ExtraTreesClassifier / ExtraTreesRegressor** — extremely randomized trees: random feature
  sub-sampling plus random split thresholds on the shared DecisionTree
  (`splitter="random"`, `bootstrap=False` by default). The regressor uses
  variance-reducing CART trees. Same MDI `feature_importances_` as Random Forest.
- **AdaBoostClassifier** — SAMME adaptive boosting of decision-tree weak
  learners with a weighted majority vote.
- **AdaBoostRegressor** — AdaBoost.R2 adaptive boosting of variance-reducing
  decision-tree weak learners with a weighted-median prediction.
- **LogitBoostClassifier** — LogitBoost (Friedman, Hastie & Tibshirani 2000), which takes
  Newton steps on the multinomial log-likelihood. Each round fits one weighted
  least-squares regression tree per class to the clipped working response
  `(y - p) / (p (1 - p))`, then centres the class contributions. For two classes it
  reduces exactly to their binary Algorithm 3.
- **GentleBoostClassifier** — Gentle AdaBoost. Weighted least-squares trees fit
  `y in {-1, +1}` under `w = exp(-y F)`, and the bounded steps are added to `F`.
  Multiclass problems use one-vs-rest. Probabilities come from `1 / (1 + exp(-2F))`.
  Both expose `staged_predict`, `decision_function` and the per-round `train_loss_`.
  Both are built on `WeightedRegressionTree`, an exact weighted-SSE tree.
- **GradientBoostingClassifier / Regressor** — additive trees fit on the loss
  gradient (log-loss / least-squares), with an exact scan of each feature.
- **HistogramGradientBoostingClassifier** — the histogram variant of gradient
  boosting. Features are quantile-binned, then each boosting iteration grows
  Newton trees on those bins (one tree per class when there are more than two
  classes). Supports binary and multiclass targets, `predict_proba`, and
  `staged_predict_proba` after every iteration. Gain-based
  `feature_importances_` summarise which bins the trees split on.
- **HistogramGradientBoostingRegressor** — the least-squares counterpart.
  Same quantile binning and Newton histogram trees, but each stage fits the
  residual `y - F` (unit Hessian). Exposes `predict`, `staged_predict`, and
  gain-based `feature_importances_`.
- **VotingClassifier** — hard (weighted majority) and soft (weighted
  probability) voting. `fit` clones each base estimator. Soft votes align
  every model's probability columns to the ensemble class order, and an
  entry may be `"drop"` to leave that model out.
- **VotingRegressor** — weighted average of independent regressors. Same
  clone / `"drop"` / `weights` conventions as the classifier.
- **StackingClassifier** — out-of-fold meta-features + a logistic-regression
  meta-learner.
- **StackingRegressor** — out-of-fold base predictions + a ridge / linear
  meta-learner (parallel to `StackingClassifier`).
- **BlendingClassifier** — hold-out meta-learning on a validation split.
- **BlendingRegressor** — hold-out base predictions + a ridge meta-learner (parallel to `BlendingClassifier`).
- **EnsembleSelectionClassifier / EnsembleSelectionRegressor** — Caruana et al.
  (2004) ensemble selection: fit a library of models on a training split, then
  greedily add (with replacement) whichever model most improves the averaged
  validation predictions (log-loss / Brier / error, or MSE / MAE). Sorted
  initialisation (`n_init`), best-of-path truncation (`use_best`), and refit
  on all data. Exposes `weights_`, `selection_order_`, `validation_scores_`.
- **KNORAClassifier** — Ko, Sabourin & Britto (2008) dynamic ensemble
  selection. A pool (bagged trees by default, or your own `(name, estimator)`
  list) is fitted on a training split; at prediction time the `k` nearest
  points of a held-out DSEL split form each query's region of competence.
  `method="eliminate"` (KNORA-E) keeps only the models that classify every
  neighbour correctly (shrinking `k` until an oracle exists);
  `method="union"` (KNORA-U) weights each model by its correct neighbours.
  Exposes `competence_weights`, `n_selected`, `dsel_correct_`, `pool_accuracy_`.
- **DCSClassifier** — Woods, Kegelmeyer & Bowyer (1997) dynamic classifier
  *selection*: same pool / DSEL / neighbour machinery as KNORA, but each query
  is decided by the **single** most locally competent member.
  `method="ola"` (overall local accuracy) scores members by their accuracy on
  the `k` neighbours; `method="lca"` (local class accuracy) scores each member
  by its local precision for the label it assigns to the query. Members tied
  for the best competence (within `tie_tolerance`) vote together. Exposes
  `competences`, `selected_estimator`, `competence_weights`, `n_selected`.
- **Diversity measures & pruning** (`diversity.py`) — Kuncheva & Whitaker
  (2003) oracle-output diversity: pairwise Q-statistic, correlation (rho),
  disagreement and double-fault (`pairwise_diversity`,
  `average_pairwise_diversity`), plus the non-pairwise entropy E,
  Kohavi-Wolpert variance and inter-rater agreement kappa.
  `ensemble_member_predictions(model, X)` collects member votes from any
  fitted ensemble with `estimators_` (handles subspace / rotation members),
  `diversity_report` summarises everything, and `diversity_prune` greedily
  keeps `n_select` members trading accuracy against disagreement
  (`alpha * accuracy + (1 - alpha) * diversity`).

- **SuperLearnerRegressor / SuperLearnerClassifier** implement the van der Laan, Polley & Hubbard (2007) Super Learner.
  - Every candidate learner is fit by V-fold cross-validation to get out-of-fold predictions. Ensemble weights are then chosen to minimise the cross-validated risk.
  - Weight methods: `nnls` (NNLS rescaled to sum to 1, the R `SuperLearner` default), `convex` (simplex-constrained least squares or Brier score), `log_loss` (classifier negative log-likelihood on the simplex) and `discrete` (CV model selection).
  - Learners with zero weight are not refit. `summary()` lists each learner's CV risk and weight.
  - Everything, including the Lawson-Hanson NNLS solver, is pure NumPy.

## Installation

```bash
pip install -e .
```

For the optional demo comparisons against scikit-learn:

```bash
pip install -e .[demo]
```

## Usage

Train an ensemble on the Iris dataset:

```python
from sklearn.datasets import load_iris
from ensemble_methods_kit import RandomForestClassifier, train_test_split, accuracy_score

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=25, max_depth=5, random_state=0)
clf.fit(X_tr, y_tr)
print("accuracy:", accuracy_score(y_te, clf.predict(X_te)))
print("MDI importances:", clf.feature_importances_)
```

LogitBoost / Gentle AdaBoost with depth-2 trees:

```python
from ensemble_methods_kit import GentleBoostClassifier, LogitBoostClassifier

lb = LogitBoostClassifier(n_estimators=150, learning_rate=0.5, max_depth=2).fit(X_tr, y_tr)
lb.predict_proba(X_te)          # softmax of the additive class scores
lb.train_loss_[-1]              # training log-loss after the last round
gb = GentleBoostClassifier(n_estimators=150, learning_rate=0.5).fit(X_tr, y_tr)
[accuracy_score(y_te, p) for p in gb.staged_predict(X_te)][-1]
```

Prediction intervals from a quantile regression forest (heteroscedastic noise):

```python
import numpy as np
from ensemble_methods_kit import QuantileRegressionForest, interval_coverage

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, (800, 3))
y = np.sin(X[:, 0]) + (0.1 + 0.3 * X[:, 0]) * rng.normal(size=800)
qrf = QuantileRegressionForest(n_estimators=100, random_state=0).fit(X[:600], y[:600])
lo, hi = qrf.predict_interval(X[600:], coverage=0.9)     # wider for large X[:, 0]
print(interval_coverage(y[600:], lo, hi))                 # ~0.9
print(qrf.predict_quantiles(X[600:605], [0.1, 0.5, 0.9]))
oob = qrf.oob_predict_quantiles([0.05, 0.95])             # honest in-sample check
```

Continuing the Iris split above, combine heterogeneous classifiers with a
hard or soft vote. The objects you pass in are cloned, so they stay unfitted
until you fit them yourself:

```python
from ensemble_methods_kit import DecisionTree, RandomForestClassifier, VotingClassifier

vote = VotingClassifier(
    estimators=[
        ("forest", RandomForestClassifier(n_estimators=25, max_depth=5, random_state=0)),
        ("tree", DecisionTree(max_depth=4, random_state=0)),
    ],
    voting="soft",
    weights=[2.0, 1.0],
)
vote.fit(X_tr, y_tr)
print("soft vote:", accuracy_score(y_te, vote.predict(X_te)))
print("mean class probability:", vote.predict_proba(X_te).mean(axis=0))

hard = VotingClassifier(vote.estimators, voting="hard")
hard.fit(X_tr, y_tr)
print("hard vote:", accuracy_score(y_te, hard.predict(X_te)))
```

Average heterogeneous regressors the same way:

```python
from ensemble_methods_kit import (
    DecisionTree,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)

vote = VotingRegressor(
    estimators=[
        ("rf", RandomForestRegressor(n_estimators=20, max_depth=5, random_state=1)),
        ("gbr", GradientBoostingRegressor(n_estimators=40, learning_rate=0.1, max_depth=2, random_state=1)),
        ("dt", DecisionTree(criterion="variance", max_depth=5, random_state=1)),
    ],
    weights=[2.0, 1.0, 1.0],
)
vote.fit(X_tr, y_tr)  # continuous targets
print(vote.predict(X_te[:3]))
```


Let greedy ensemble selection pick weights for a model library (continuing
the Iris split above). Models that never help get weight zero:

```python
from ensemble_methods_kit import (
    DecisionTree,
    EnsembleSelectionClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)

sel = EnsembleSelectionClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=25, max_depth=5, random_state=0)),
        ("et", ExtraTreesClassifier(n_estimators=25, max_depth=5, random_state=0)),
        ("stump", DecisionTree(max_depth=1, random_state=0)),
    ],
    n_iterations=20,
    metric="log_loss",
    random_state=0,
)
sel.fit(X_tr, y_tr)
print("selection accuracy:", accuracy_score(y_te, sel.predict(X_te)))
print("weights:", sel.named_weights_)
```

Fit a random-forest regressor on a noisy linear target. `max_features`
controls how many columns each split may use, and `oob_score=True` asks for
the out-of-bag R² (bootstrap must stay on):

```python
import numpy as np
from ensemble_methods_kit import RandomForestRegressor, mean_squared_error

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 4))
y = X @ np.array([1.5, -2.0, 0.5, 0.0]) + 0.3 * rng.normal(size=200)

reg = RandomForestRegressor(
    n_estimators=40, max_depth=6, max_features="sqrt", oob_score=True, random_state=0
)
reg.fit(X, y)
print("mse:", mean_squared_error(y, reg.predict(X)))
print("oob R²:", reg.oob_score_)
```

`feature_importances_` is the Mean Decrease Impurity (MDI) ranking: each
split contributes its weighted impurity decrease to the chosen feature.
`RandomForestClassifier`, `RandomForestRegressor`, and `ExtraTreesClassifier`
average that vector over their trees (also available as
`mean_decrease_impurity(estimators)`).
AdaBoost keeps its own SAMME-weighted split-usage importances.

Measure and prune ensemble diversity:

```python
from ensemble_methods_kit import (
    RandomForestClassifier, diversity_prune, diversity_report, ensemble_member_predictions,
)

rf = RandomForestClassifier(n_estimators=25, random_state=0).fit(X_tr, y_tr)
P = ensemble_member_predictions(rf, X_val)         # (25, n_val) member votes
print(diversity_report(P, y_val)["disagreement"])
pruned = diversity_prune(P, y_val, n_select=7, alpha=0.5)
print(pruned["selected"], pruned["pruned_accuracy"], pruned["full_accuracy"])
```

Run the command-line demo (writes `examples/output/demo_report.md`):

```bash
python -m ensemble_methods_kit
# or, after install:
ensemble-methods
```

### Super Learner

```python
from ensemble_methods_kit import (SuperLearnerRegressor, RidgeRegression,
                                  RandomForestRegressor, GradientBoostingRegressor)

sl = SuperLearnerRegressor(
    [("ridge", RidgeRegression(alpha=1.0)),
     ("rf", RandomForestRegressor(n_estimators=50, max_depth=6)),
     ("gbr", GradientBoostingRegressor(n_estimators=100))],
    method="nnls", cv=10, random_state=0,
).fit(X_train, y_train)
sl.summary()              # [{'name', 'cv_risk', 'weight'}, ...]
y_hat = sl.predict(X_test)
```

`SuperLearnerClassifier` takes the same arguments and defaults to `method="log_loss"`, which uses convex weights on the predicted probabilities. `cv_risk_` holds each learner's out-of-fold loss, and `cv_predictions_` the full out-of-fold matrix.

## Project layout

```
ensemble-methods-kit/
├── pyproject.toml
├── README.md
├── src/ensemble_methods_kit/
│   ├── __init__.py
│   ├── __main__.py        # console-script entry point
│   ├── utils.py           # metrics, split, DecisionTree
│   ├── bagging.py
│   ├── random_forest.py
│   ├── quantile_forest.py # Meinshausen quantile regression forest
│   ├── extra_trees.py
│   ├── adaboost.py
│   ├── logitboost.py      # LogitBoost, Gentle AdaBoost, weighted LS tree
│   ├── gradient_boosting.py
│   ├── histogram_gradient_boosting.py
│   ├── voting.py
│   ├── ensemble_selection.py
│   ├── des.py             # KNORA-E / KNORA-U, OLA / LCA dynamic selection
│   ├── diversity.py       # diversity measures + pruning
│   ├── stacking.py
│   └── blending.py
├── tests/
└── examples/
    ├── run_demo.py
    └── output/demo_report.md
```

## License

MIT
