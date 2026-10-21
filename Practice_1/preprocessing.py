"""Convert questionnaire answers to the feature vector used by KNN."""

from collections.abc import Mapping

import numpy as np

from config import EYE_FEATURE_COLUMNS, FEATURE_COLUMNS
from questionnaire import VISIBLE_QUESTIONS, validate_answer


# These bounds reproduce the normalization in the supplied dataset:
# years: (value - 2) / (8 - 2), clipped to [0, 1]
# games: value / 40, clipped to [0, 1]
PHONE_CHANGE_MIN_YEARS = 2
PHONE_CHANGE_MAX_YEARS = 8
GAMES_MAX_HOURS = 40


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def normalize_phone_change_years(years: int) -> float:
    span = PHONE_CHANGE_MAX_YEARS - PHONE_CHANGE_MIN_YEARS
    return _clip01((years - PHONE_CHANGE_MIN_YEARS) / span)


def normalize_games_hours(hours: int) -> float:
    return _clip01(hours / GAMES_MAX_HOURS)


def answers_to_vector(answers: Mapping[str, object]) -> np.ndarray:
    """Encode all visible answers in the exact training-column order."""

    expected_keys = {question.key for question in VISIBLE_QUESTIONS}
    missing = expected_keys - set(answers)
    if missing:
        raise ValueError(f"Не заполнены ответы: {', '.join(sorted(missing))}")

    values: dict[str, float] = {}
    eye_value: str | None = None

    for question in VISIBLE_QUESTIONS:
        raw_value = validate_answer(question, answers[question.key])

        if question.kind == "choice":
            values[question.key] = float(question.encoding[str(raw_value)])
        elif question.kind == "integer" and question.key == "phone_change_years":
            values[question.key] = normalize_phone_change_years(int(raw_value))
        elif question.kind == "integer" and question.key == "games_hours":
            values[question.key] = normalize_games_hours(int(raw_value))
        elif question.kind == "eye_color":
            eye_value = str(question.encoding[str(raw_value)])
        else:
            raise ValueError(f"Неизвестный вопрос: {question.key}")

    if eye_value not in EYE_FEATURE_COLUMNS:
        raise ValueError("Не удалось закодировать цвет глаз.")
    for column in EYE_FEATURE_COLUMNS:
        values[column] = 1.0 if column == eye_value else 0.0

    vector = np.asarray([values[column] for column in FEATURE_COLUMNS], dtype=float)
    if vector.shape != (len(FEATURE_COLUMNS),):
        raise ValueError("Сформирован вектор признаков неправильного размера.")
    if not np.isfinite(vector).all() or ((vector < 0) | (vector > 1)).any():
        raise ValueError("Вектор признаков содержит значения вне диапазона [0, 1].")
    return vector
