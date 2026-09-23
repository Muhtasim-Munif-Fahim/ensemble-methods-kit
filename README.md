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
- **BaggingClassifier** — bootstrap aggregating of decision trees.
- **RandomForestClassifier** — bagging with random feature sub-sampling.
  Exposes impurity-based `feature_importances_` (Mean Decrease Impurity).
- **ExtraTreesClassifier** — extremely randomized trees: random feature
  sub-sampling plus random split thresholds on the shared DecisionTree.
  Same MDI `feature_importances_` as Random Forest.
- **AdaBoostClassifier** — SAMME adaptive boosting of decision-tree weak
  learners with a weighted majority vote.
- **GradientBoostingClassifier / Regressor** — additive trees fit on the loss
  gradient (log-loss / least-squares), with an exact scan of each feature.
- **HistogramGradientBoostingClassifier** — the histogram variant of gradient
  boosting. Features are quantile-binned, then each boosting iteration grows
  Newton trees on those bins (one tree per class when there are more than two
  classes). Supports binary and multiclass targets, `predict_proba`, and
  `staged_predict_proba` after every iteration. Gain-based
  `feature_importances_` summarise which bins the trees split on.
- **VotingClassifier** — hard (weighted majority) and soft (weighted
  probability) voting. `fit` clones each base estimator. Soft votes align
  every model's probability columns to the ensemble class order, and an
  entry may be `"drop"` to leave that model out.
- **StackingClassifier** — out-of-fold meta-features + a logistic-regression
  meta-learner.
- **BlendingClassifier** — hold-out meta-learning on a validation split.

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

`feature_importances_` is the Mean Decrease Impurity (MDI) ranking: each
split contributes its weighted impurity decrease to the chosen feature.
`RandomForestClassifier` and `ExtraTreesClassifier` average that vector
over their trees (also available as `mean_decrease_impurity(estimators)`).
AdaBoost keeps its own SAMME-weighted split-usage importances.

Run the command-line demo (writes `examples/output/demo_report.md`):

```bash
python -m ensemble_methods_kit
# or, after install:
ensemble-methods
```

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
│   ├── extra_trees.py
│   ├── adaboost.py
│   ├── gradient_boosting.py
│   ├── histogram_gradient_boosting.py
│   ├── voting.py
│   ├── stacking.py
│   └── blending.py
├── tests/
└── examples/
    ├── run_demo.py
    └── output/demo_report.md
```

## License

MIT
