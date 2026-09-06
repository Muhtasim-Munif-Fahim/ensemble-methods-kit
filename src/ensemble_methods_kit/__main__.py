"""Console-script entry point and demo workflow.

Running ``python -m ensemble_methods_kit`` (or the ``ensemble-methods``
console script) trains every estimator in the kit on a classification task,
optionally compares them to scikit-learn baselines, benchmarks the gradient
boosting regressor on a regression task, and writes a Markdown report.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

from . import (
    BaggingClassifier,
    BlendingClassifier,
    DecisionTree,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
    accuracy_score,
    log_loss,
    r2_score,
    train_test_split,
)

__all__ = ["run_demo", "main"]

_DEMO_RANDOM_STATE = 0


def _maybe_use_sklearn():
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


def _load_classification(random_state: int):
    """Return X, y for a 3-class classification task.

    Prefers ``sklearn.datasets.load_iris``; falls back to a synthetic 3-class
    blob dataset built with numpy so the demo runs with numpy alone.
    """
    if _maybe_use_sklearn():
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)
        return X, y, "Iris (sklearn.datasets.load_iris)", load_iris().target_names
    rng = np.random.default_rng(random_state)
    k, n, d = 3, 50, 4
    centers = np.array([[0, 0, 0, 0], [3, 3, 3, 3], [-3, 3, -3, 3]], dtype=float)
    Xs = [rng.normal(c, 1.0, (n, d)) for c in centers]
    X = np.vstack(Xs)
    y = np.concatenate([np.full(n, i) for i in range(k)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx], "Synthetic 3-class blobs", np.array(["class_0", "class_1", "class_2"])


def _load_regression(random_state: int):
    if _maybe_use_sklearn():
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=200, n_features=6, n_informative=6, noise=5.0,
            random_state=random_state,
        )
        return X, y, "make_regression (sklearn)"
    rng = np.random.default_rng(random_state)
    X = rng.normal(0, 1, (200, 6))
    w = rng.normal(0, 1, 6)
    y = X @ w + 5.0 * rng.normal(0, 1, 200)
    return X, y, "Synthetic linear regression"


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return float(np.mean(f1s))


def _build_classifiers() -> List[Tuple[str, object]]:
    return [
        ("DecisionTree", DecisionTree(criterion="gini", max_depth=5, random_state=_DEMO_RANDOM_STATE)),
        ("Bagging", BaggingClassifier(n_estimators=25, max_samples=1.0, random_state=_DEMO_RANDOM_STATE)),
        ("RandomForest", RandomForestClassifier(n_estimators=30, max_depth=5, random_state=_DEMO_RANDOM_STATE)),
        (
            "GradientBoosting",
            GradientBoostingClassifier(n_estimators=60, learning_rate=0.1, max_depth=2,
                                       random_state=_DEMO_RANDOM_STATE),
        ),
        (
            "Voting(soft)",
            VotingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=20, max_depth=5, random_state=1)),
                    ("gbc", GradientBoostingClassifier(n_estimators=30, learning_rate=0.1,
                                                       max_depth=2, random_state=1)),
                ],
                voting="soft",
            ),
        ),
        (
            "Stacking",
            StackingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=15, max_depth=5, random_state=1)),
                    ("gbc", GradientBoostingClassifier(n_estimators=20, learning_rate=0.1,
                                                      max_depth=2, random_state=1)),
                ],
                cv=3,
                random_state=_DEMO_RANDOM_STATE,
            ),
        ),
        (
            "Blending",
            BlendingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=15, max_depth=5, random_state=1)),
                    ("gbc", GradientBoostingClassifier(n_estimators=20, learning_rate=0.1,
                                                      max_depth=2, random_state=1)),
                ],
                validation_fraction=0.3,
                random_state=_DEMO_RANDOM_STATE,
            ),
        ),
    ]


def _build_sklearn_baselines():
    """Return scikit-learn baselines when available, else an empty list."""
    if not _maybe_use_sklearn():
        return []
    from sklearn.ensemble import GradientBoostingClassifier as SKGBC
    from sklearn.ensemble import RandomForestClassifier as SKRF

    return [
        ("sklearn-RandomForest", SKRF(n_estimators=30, max_depth=5, random_state=_DEMO_RANDOM_STATE)),
        ("sklearn-GradientBoosting", SKGBC(n_estimators=60, learning_rate=0.1, max_depth=2,
                                           random_state=_DEMO_RANDOM_STATE)),
    ]


def run_demo(output_path: str = "demo_report.md", use_sklearn: bool = True) -> str:
    """Run the demo workflow and write a Markdown report to ``output_path``."""
    sections: List[str] = []
    sections.append("# ensemble-methods-kit - Demo Report\n")
    sections.append(f"Generated by `python -m ensemble_methods_kit` on "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}.\n")

    # ---- classification
    X, y, dataset_name, target_names = _load_classification(_DEMO_RANDOM_STATE)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=_DEMO_RANDOM_STATE, stratify=y
    )
    sections.append(f"## 1. Classification — {dataset_name}\n")
    sections.append(f"Samples: {X.shape[0]}  |  Features: {X.shape[1]}  |  "
                    f"Classes: {len(np.unique(y))}  |  Train: {X_tr.shape[0]}  "
                    f"Test: {X_te.shape[0]}\n")
    if target_names is not None:
        sections.append(f"Class names: {[str(c) for c in target_names]}\n")

    rows: List[Tuple[str, float, float, float, float]] = []
    models = _build_classifiers() + ( _build_sklearn_baselines() if use_sklearn else [])

    for name, model in models:
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        elapsed = time.perf_counter() - t0
        preds = model.predict(X_te)
        proba = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
        acc = accuracy_score(y_te, preds)
        f1 = _macro_f1(y_te, preds)
        ll = log_loss(y_te, proba) if proba is not None else float("nan")
        rows.append((name, acc, f1, ll, elapsed))

    rows.sort(key=lambda r: r[1], reverse=True)
    sections.append(
        "\n| Model | Accuracy | Macro-F1 | Log-loss | Train time (s) |\n"
        "|---|---|---|---|---|\n"
        + "\n".join(
            f"| {name} | {acc:.4f} | {f1:.4f} | {ll:.4f} | {elapsed:.3f} |"
            for name, acc, f1, ll, elapsed in rows
        )
    )
    sections.append(f"\n**Best model:** {rows[0][0]} (accuracy {rows[0][1]:.4f}).\n")

    # ---- regression
    Xr, yr, reg_name = _load_regression(_DEMO_RANDOM_STATE)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.25,
                                                  random_state=_DEMO_RANDOM_STATE)
    sections.append(f"\n## 2. Regression - {reg_name}\n")
    sections.append(f"Samples: {Xr.shape[0]}  |  Features: {Xr.shape[1]}  |  "
                    f"Train: {Xr_tr.shape[0]}  |  Test: {Xr_te.shape[0]}\n")
    gbr = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=_DEMO_RANDOM_STATE
    ).fit(Xr_tr, yr_tr)
    gbr_r2 = r2_score(yr_te, gbr.predict(Xr_te))
    tree_r2 = r2_score(yr_te, DecisionTree(criterion="variance", max_depth=5,
                                           random_state=_DEMO_RANDOM_STATE).fit(Xr_tr, yr_tr).predict(Xr_te))
    sections.append(
        "\n| Model | R² |\n|---|---|\n"
        f"| GradientBoostingRegressor | {gbr_r2:.4f} |\n"
        f"| DecisionTree (depth 5)    | {tree_r2:.4f} |\n"
    )

    # ---- notes
    sections.append("\n## 3. Notes\n")
    sections.append(
        "- All estimators are implemented from scratch on a single NumPy CART "
        "decision tree; the only hard dependency is `numpy`.\n"
        "- scikit-learn is used solely for datasets and baseline comparisons; "
        "it is an optional extra (`pip install -e .[demo]`).\n"
        "- Tree depth, learning rate, number of estimators and split fractions are "
        "deliberately modest to keep the demo fast and deterministic.\n"
    )

    report = "\n".join(sections)
    path = output_path
    if path:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="ensemble-methods",
                                     description="Run the ensemble-methods-kit demo workflow.")
    parser.add_argument("-o", "--output", default="demo_report.md",
                        help="Path to write the Markdown report (default: demo_report.md).")
    parser.add_argument("--no-sklearn", action="store_true",
                        help="Do not use scikit-learn even if it is installed.")
    args = parser.parse_args(argv)
    report = run_demo(output_path=args.output, use_sklearn=not args.no_sklearn)
    print(f"Wrote {len(report)} characters of Markdown to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
