"""Run the ensemble-methods-kit demo and write the Markdown report.

Usage (from the repository root)::

    python examples/run_demo.py

This trains every estimator in the kit on the Iris dataset, compares the
results to scikit-learn baselines when scikit-learn is installed, benchmarks a
gradient-boosting regressor on a regression task, and writes
``examples/output/demo_report.md``.
"""

from __future__ import annotations

from pathlib import Path

from ensemble_methods_kit.__main__ import run_demo

OUTPUT_PATH = Path(__file__).resolve().parent / "output" / "demo_report.md"


def main() -> int:
    report = run_demo(output_path=str(OUTPUT_PATH), use_sklearn=True)
    print(f"Demo report written to {OUTPUT_PATH} ({len(report)} characters of Markdown).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
