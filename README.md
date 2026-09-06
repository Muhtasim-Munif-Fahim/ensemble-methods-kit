# ensemble-methods-kit

A from-scratch NumPy toolkit of **ensemble learning methods** with a complete
demo workflow that trains several ensembles, benchmarks them against each other
and against scikit-learn baselines, and writes a Markdown report.

The estimators are implemented on top of a small CART decision tree so the whole
package has a single hard runtime dependency (`numpy`) and the algorithms are
easy to read and extend.

## Features

- **DecisionTree** — CART with gini / entropy splits for classification and
  variance reduction for regression, plus feature sub-sampling.
- **BaggingClassifier** — bootstrap aggregating of decision trees.
- **RandomForestClassifier** — bagging with random feature sub-sampling.
- **GradientBoostingClassifier / Regressor** — additive trees fit on the loss
  gradient (log-loss / least-squares).
- **VotingClassifier** — soft and hard voting over arbitrary estimators.
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
```

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
│   ├── gradient_boosting.py
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
