"""Question definitions and raw answer validation.

The mappings in this file reproduce the encoding observed by comparing the
source form responses with dataset_normalized.csv. The target question is
intentionally not included in VISIBLE_QUESTIONS.
"""

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class Question:
    key: str
    text: str
    kind: str
    options: tuple[str, ...] = ()
    encoding: Mapping[str, int | str] = field(default_factory=dict)


VISIBLE_QUESTIONS = (
    Question(
        key="attention_feel",
        text="Как вы себя чувствуете, когда оказались в центре внимания в малознакомой компании?",
        kind="choice",
        options=("Комфортно", "Дискомфортно"),
        encoding={"Комфортно": 1, "Дискомфортно": 0},
    ),
    Question(
        key="decision_style",
        text="Как чаще всего принимаются ваши повседневные решения о покупках?",
        kind="choice",
        options=("Рационально", "Эмоционально"),
        encoding={"Рационально": 0, "Эмоционально": 1},
    ),
    Question(
        key="complex_tech",
        text="Покупая сложную технику или мебель, какой подход вам ближе?",
        kind="choice",
        options=("Чтобы всё сразу работало из коробки", "Самому всё собрать"),
        encoding={
            "Чтобы всё сразу работало из коробки": 0,
            "Самому всё собрать": 1,
        },
    ),
    Question(
        key="everyday_choice",
        text="Что вы выберете при покупке повседневной вещи?",
        kind="choice",
        options=("Какой-то уникальный вариант", "Что-то популярное"),
        encoding={"Какой-то уникальный вариант": 1, "Что-то популярное": 0},
    ),
    Question(
        key="battery_drain",
        text="Быстро ли твой телефон разряжается?",
        kind="choice",
        options=("Да", "Нет"),
        encoding={"Да": 1, "Нет": 0},
    ),
    Question(
        key="shoots_often",
        text="Ты часто снимаешь на телефон фото/видео?",
        kind="choice",
        options=("Да", "Нет"),
        encoding={"Да": 1, "Нет": 0},
    ),
    Question(
        key="design_vs_durability",
        text="При выборе предмета интерьера или одежды, что станет решающим фактором?",
        kind="choice",
        options=("Дизайн, эстетика", "Долговечность, польза"),
        encoding={"Дизайн, эстетика": 1, "Долговечность, польза": 0},
    ),
    Question(
        key="phone_lags",
        text="Часто ли твой телефон тормозит/лагает?",
        kind="choice",
        options=("Да", "Нет"),
        encoding={"Да": 1, "Нет": 0},
    ),
    Question(
        key="apps_uniform",
        text="Тебе важно, чтобы все приложения со временем работали одинаково?",
        kind="choice",
        options=("Да", "Нет"),
        encoding={"Да": 1, "Нет": 0},
    ),
    Question(
        key="likes_travel",
        text="Любите ли вы путешествовать?",
        kind="choice",
        options=("Да", "Нет"),
        encoding={"Да": 1, "Нет": 0},
    ),
    Question(
        key="phone_change_years",
        text="Как часто ты меняешь смартфон на новый? Введите число лет.",
        kind="integer",
    ),
    Question(
        key="games_hours",
        text="Как часто вы играете в видеоигры? Введите примерное количество часов в неделю.",
        kind="integer",
    ),
    Question(
        key="posts_social",
        text="Часто ли вы публикуете фотографии или видео в социальных сетях?",
        kind="choice",
        options=("Да", "Нет"),
        encoding={"Да": 1, "Нет": 0},
    ),
    Question(
        key="watches_on_phone",
        text="Часто ли вы смотрите фильмы и сериалы на телефоне?",
        kind="choice",
        options=("Да", "Нет"),
        encoding={"Да": 1, "Нет": 0},
    ),
    Question(
        key="eye_color",
        text="Цвет глаз?",
        kind="eye_color",
        options=("Голубой", "Карий", "Серый", "Серо-голубой", "Зелёный", "Другой"),
        encoding={
            "Голубой": "eye_blue",
            "Карий": "eye_brown",
            "Серый": "eye_gray",
            "Серо-голубой": "eye_gray_blue",
            "Зелёный": "eye_green",
            "Другой": "eye_other",
        },
    ),
)

OMITTED_QUESTION_TEXT = (
    "Ближайшая станция метро? В текущем нормализованном датасете "
    "для этого вопроса нет признака."
)

TARGET_QUESTION_TEXT = "У тебя айфон или андроид?"


def validate_answer(question: Question, value: object) -> object:
    """Validate one raw UI answer and return a normalized raw value.

    Numeric normalization is deliberately kept in preprocessing.py; this
    function only validates that the user entered a non-negative integer.
    """

    if question.kind in {"choice", "eye_color"}:
        if value not in question.encoding:
            raise ValueError("Выберите один из предложенных вариантов.")
        return str(value)

    if question.kind == "integer":
        raw = str(value).strip()
        if not raw:
            raise ValueError("Введите целое число.")
        try:
            parsed = int(raw)
        except ValueError as exc:
            raise ValueError("Введите целое число без дополнительных символов.") from exc
        if parsed < 0:
            raise ValueError("Число не может быть отрицательным.")
        return parsed

    raise ValueError(f"Неизвестный тип вопроса: {question.kind}")
