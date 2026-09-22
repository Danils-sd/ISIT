"""Small educational implementation of k nearest neighbors."""

from __future__ import annotations

import numpy as np


class KNNClassifier:
    """KNN classifier with Euclidean distance and deterministic tie handling."""

    def __init__(self, k: int) -> None:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("K должно быть положительным целым числом.")
        self.k = k
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        features = np.asarray(x, dtype=float)
        labels = np.asarray(y, dtype=int)

        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("Обучающие признаки должны быть непустой двумерной матрицей.")
        if labels.ndim != 1 or len(labels) != len(features):
            raise ValueError("Количество меток должно совпадать с количеством строк.")
        if not np.isfinite(features).all():
            raise ValueError("Обучающие признаки содержат нечисловые значения.")
        if self.k > len(features):
            raise ValueError("K не может быть больше размера обучающей выборки.")

        self._x = features
        self._y = labels
        self.classes_ = np.unique(labels)
        if len(self.classes_) < 2:
            raise ValueError("Для классификации нужны минимум два класса.")
        return self

    def _require_fitted(self) -> tuple[np.ndarray, np.ndarray]:
        if self._x is None or self._y is None:
            raise RuntimeError("Модель ещё не обучена. Сначала вызовите fit().")
        return self._x, self._y

    def neighbor_indices(self, sample: np.ndarray) -> np.ndarray:
        features, _ = self._require_fitted()
        point = np.asarray(sample, dtype=float)
        if point.ndim != 1 or point.shape[0] != features.shape[1]:
            raise ValueError("Размер вектора признаков не совпадает с обучающими данными.")
        if not np.isfinite(point).all():
            raise ValueError("Вектор признаков содержит нечисловые значения.")

        distances = np.linalg.norm(features - point, axis=1)
        return np.argsort(distances, kind="stable")[: self.k]

    def predict_one(self, sample: np.ndarray) -> int:
        _, labels = self._require_fitted()
        indices = self.neighbor_indices(sample)
        neighbor_labels = labels[indices]
        classes = np.unique(labels)
        counts = np.asarray([(neighbor_labels == cls).sum() for cls in classes])
        winners = classes[counts == counts.max()]

        if len(winners) == 1:
            return int(winners[0])

        # Deterministic tie-break: choose the label of the closest tied
        # neighbor. This matters for tests and even K values.
        for index in indices:
            if labels[index] in winners:
                return int(labels[index])
        raise RuntimeError("Не удалось разрешить ничью при голосовании.")

    def predict(self, samples: np.ndarray) -> np.ndarray:
        matrix = np.asarray(samples, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise ValueError("Предсказываемые признаки должны быть двумерной матрицей.")
        return np.asarray([self.predict_one(row) for row in matrix], dtype=int)
