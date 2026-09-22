"""Standalone quality and integrity checks for the phone classifier.

Run from the project root:

    python quality_check.py

The module intentionally uses the same KNN implementation as the application,
while keeping all checks independent from the Tkinter UI.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from config import K, PHONE_LABELS, REPORTS_DIR
from data_validation import dataset_summary, load_training_arrays
from knn_model import KNNClassifier


CANDIDATE_K = tuple(range(1, 20, 2))
CV_SPLITS = 5
CV_SEEDS = (42, 43, 44, 45, 46)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    labels = (0, 1)
    matrix = []
    per_class = {}

    for actual in labels:
        row = []
        for predicted in labels:
            row.append(int(np.sum((y_true == actual) & (y_pred == predicted))))
        matrix.append(row)

    f1_values = []
    recalls = []
    for label in labels:
        tp = matrix[label][label]
        fp = matrix[1 - label][label]
        fn = matrix[label][1 - label]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        f1_values.append(f1)
        recalls.append(recall)
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Для стратифицированной проверки нужны два класса.")
    if counts.min() < n_splits:
        raise ValueError("Недостаточно примеров одного из классов для выбранного числа фолдов.")

    rng = np.random.default_rng(seed)
    fold_indices: list[list[int]] = [[] for _ in range(n_splits)]
    for class_value in classes:
        indices = np.flatnonzero(y == class_value)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            fold_indices[position % n_splits].append(int(index))
    return [np.asarray(sorted(indices), dtype=int) for indices in fold_indices]


def cross_validate_k(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    seeds: Iterable[int] = CV_SEEDS,
    n_splits: int = CV_SPLITS,
) -> dict[str, float]:
    scores: list[dict[str, object]] = []
    all_indices = np.arange(len(y))

    for seed in seeds:
        for test_indices in stratified_folds(y, n_splits, seed):
            train_indices = np.setdiff1d(all_indices, test_indices)
            model = KNNClassifier(k).fit(x[train_indices], y[train_indices])
            prediction = model.predict(x[test_indices])
            scores.append(classification_metrics(y[test_indices], prediction))

    accuracy = np.asarray([float(score["accuracy"]) for score in scores])
    macro_f1 = np.asarray([float(score["macro_f1"]) for score in scores])
    balanced_accuracy = np.asarray([float(score["balanced_accuracy"]) for score in scores])
    return {
        "mean_accuracy": float(accuracy.mean()),
        "std_accuracy": float(accuracy.std()),
        "mean_macro_f1": float(macro_f1.mean()),
        "std_macro_f1": float(macro_f1.std()),
        "mean_balanced_accuracy": float(balanced_accuracy.mean()),
        "std_balanced_accuracy": float(balanced_accuracy.std()),
        "evaluations": len(scores),
    }


def majority_baseline_cv(
    y: np.ndarray,
    seeds: Iterable[int] = CV_SEEDS,
    n_splits: int = CV_SPLITS,
) -> dict[str, float]:
    scores: list[dict[str, object]] = []
    all_indices = np.arange(len(y))
    for seed in seeds:
        for test_indices in stratified_folds(y, n_splits, seed):
            train_indices = np.setdiff1d(all_indices, test_indices)
            values, counts = np.unique(y[train_indices], return_counts=True)
            majority_label = int(values[np.argmax(counts)])
            prediction = np.full(len(test_indices), majority_label, dtype=int)
            scores.append(classification_metrics(y[test_indices], prediction))

    accuracy = np.asarray([float(score["accuracy"]) for score in scores])
    macro_f1 = np.asarray([float(score["macro_f1"]) for score in scores])
    balanced_accuracy = np.asarray([float(score["balanced_accuracy"]) for score in scores])
    return {
        "mean_accuracy": float(accuracy.mean()),
        "mean_macro_f1": float(macro_f1.mean()),
        "mean_balanced_accuracy": float(balanced_accuracy.mean()),
    }


def holdout_score(x: np.ndarray, y: np.ndarray, k: int) -> dict[str, object]:
    # A single holdout is reported as a supplementary check. The repeated
    # stratified CV result remains the primary estimate for this small dataset.
    test_indices = stratified_folds(y, n_splits=4, seed=42)[0]
    train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
    model = KNNClassifier(k).fit(x[train_indices], y[train_indices])
    prediction = model.predict(x[test_indices])
    return {
        "test_rows": len(test_indices),
        "metrics": classification_metrics(y[test_indices], prediction),
    }


def choose_k(results: list[dict[str, object]]) -> int:
    best = max(
        results,
        key=lambda item: (
            float(item["mean_macro_f1"]),
            float(item["mean_accuracy"]),
            -int(item["k"]),
        ),
    )
    return int(best["k"])


def write_reports(
    output_dir: Path,
    summary: dict[str, object],
    results: list[dict[str, object]],
    chosen_k: int,
    baseline: dict[str, float],
    holdout: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "quality_by_k.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "k",
            "mean_accuracy",
            "std_accuracy",
            "mean_macro_f1",
            "std_macro_f1",
            "mean_balanced_accuracy",
            "std_balanced_accuracy",
            "evaluations",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    report = {
        "dataset": summary,
        "candidate_k": list(CANDIDATE_K),
        "cross_validation": {
            "folds": CV_SPLITS,
            "seeds": list(CV_SEEDS),
            "primary_metric": "mean_macro_f1",
            "results": results,
        },
        "majority_baseline": baseline,
        "chosen_k": chosen_k,
        "production_config_k": K,
        "holdout_check": holdout,
        "label_mapping": {str(key): value for key, value in PHONE_LABELS.items()},
        "notes": [
            "The normalized dataset is small, so cross-validation spread must be reported.",
            "The metro question is not a model feature and is omitted from the UI.",
        ],
    }
    (output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        x_values = [int(item["k"]) for item in results]
        accuracy_values = [float(item["mean_accuracy"]) for item in results]
        f1_values = [float(item["mean_macro_f1"]) for item in results]
        plt.figure(figsize=(8, 5))
        plt.plot(x_values, accuracy_values, marker="o", label="Accuracy")
        plt.plot(x_values, f1_values, marker="o", label="Macro F1")
        plt.axvline(chosen_k, linestyle="--", label=f"Выбранное K = {chosen_k}")
        plt.xlabel("K")
        plt.ylabel("Среднее значение по кросс-валидации")
        plt.title("Качество KNN для разных значений K")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "quality_by_k.png", dpi=150)
        plt.close()
    except ImportError:
        print("matplotlib не установлен: график не создан, CSV и JSON отчёты сохранены.")


def run_quality_check(dataset_path: str | Path, output_dir: str | Path) -> int:
    dataframe, x, y = load_training_arrays(dataset_path)
    summary = dataset_summary(dataframe)

    results: list[dict[str, object]] = []
    for k in CANDIDATE_K:
        cv_result = cross_validate_k(x, y, k)
        results.append({"k": k, **cv_result})

    chosen_k = choose_k(results)
    baseline = majority_baseline_cv(y)
    holdout = holdout_score(x, y, chosen_k)
    write_reports(Path(output_dir), summary, results, chosen_k, baseline, holdout)

    print(f"Строк в датасете: {summary['rows']}")
    print(f"Распределение классов: {summary['class_counts']}")
    print(f"Базовая accuracy: {baseline['mean_accuracy']:.3f}")
    print("Результаты по K:")
    for result in results:
        marker = " <- выбран" if int(result["k"]) == chosen_k else ""
        print(
            f"  K={result['k']:2d}: "
            f"accuracy={result['mean_accuracy']:.3f} +/- {result['std_accuracy']:.3f}, "
            f"macro_f1={result['mean_macro_f1']:.3f} +/- {result['std_macro_f1']:.3f}"
            f"{marker}"
        )
    print(f"Выбранное K по macro F1: {chosen_k}")
    print(f"K в config.py: {K}")
    if chosen_k != K:
        print("ВНИМАНИЕ: config.K отличается от значения, выбранного проверкой качества.")
    print(f"Отчёты сохранены в: {Path(output_dir).resolve()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка качества классификатора KNN")
    parser.add_argument("--dataset", default=None, help="Путь к нормализованному CSV")
    parser.add_argument("--output-dir", default=str(REPORTS_DIR), help="Каталог для отчётов")
    args = parser.parse_args()
    dataset_path = args.dataset if args.dataset else None
    from config import DATASET_PATH

    return run_quality_check(dataset_path or DATASET_PATH, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
