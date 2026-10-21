"""Validation and loading of the normalized training dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DATASET_PATH, EYE_FEATURE_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN


class DatasetValidationError(ValueError):
    """Raised when the training CSV cannot safely be used by the model."""


def validation_errors(dataframe: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_columns = sorted(required - set(dataframe.columns))
    if missing_columns:
        errors.append(f"Отсутствуют столбцы: {', '.join(missing_columns)}")
        return errors

    if dataframe.empty:
        errors.append("Датасет пуст.")
        return errors

    required_frame = dataframe[list(FEATURE_COLUMNS) + [TARGET_COLUMN]]
    missing_values = int(required_frame.isna().sum().sum())
    if missing_values:
        errors.append(f"В обязательных столбцах найдено пропусков: {missing_values}.")

    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(dataframe[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            errors.append(f"Признак {column} содержит нечисловые или бесконечные значения.")
            continue
        if ((values < 0) | (values > 1)).any():
            errors.append(f"Признак {column} содержит значения вне диапазона [0, 1].")

    target = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
        errors.append("Целевая метка phone содержит нечисловые или бесконечные значения.")
    elif not set(target.astype(int).unique()).issubset({0, 1}):
        errors.append("Целевая метка phone должна содержать только 0 и 1.")

    eye_sum = dataframe[list(EYE_FEATURE_COLUMNS)].sum(axis=1)
    if not np.isclose(eye_sum.to_numpy(dtype=float), 1).all():
        errors.append("В каждой строке ровно один признак цвета глаз должен быть равен 1.")

    if TARGET_COLUMN in dataframe:
        class_count = dataframe[TARGET_COLUMN].nunique(dropna=True)
        if class_count < 2:
            errors.append("В датасете должен присутствовать минимум один пример каждого класса.")

    return errors


def validate_dataset(dataframe: pd.DataFrame) -> None:
    errors = validation_errors(dataframe)
    if errors:
        raise DatasetValidationError("\n".join(errors))


def load_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Датасет не найден: {dataset_path}")
    dataframe = pd.read_csv(dataset_path, encoding="utf-8-sig")
    validate_dataset(dataframe)
    return dataframe


def load_training_arrays(path: str | Path = DATASET_PATH) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dataframe = load_dataset(path)
    x = dataframe.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    y = dataframe.loc[:, TARGET_COLUMN].to_numpy(dtype=int)
    return dataframe, x, y


def dataset_summary(dataframe: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(dataframe)),
        "feature_count": len(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "class_counts": {
            str(int(label)): int(count)
            for label, count in dataframe[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "missing_values": int(dataframe.isna().sum().sum()),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "extra_columns": sorted(
            set(dataframe.columns) - (set(FEATURE_COLUMNS) | {TARGET_COLUMN})
        ),
    }
