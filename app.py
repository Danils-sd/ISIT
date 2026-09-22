"""Tkinter application for sequential phone type prediction."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

from config import K, PHONE_LABELS
from data_validation import load_training_arrays
from knn_model import KNNClassifier
from preprocessing import answers_to_vector
from questionnaire import VISIBLE_QUESTIONS, Question, validate_answer


class PhoneClassifierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Определение типа телефона")
        self.root.geometry("780x560")
        self.root.minsize(620, 440)

        _, x, y = load_training_arrays()
        self.model = KNNClassifier(K).fit(x, y)

        self.answers: dict[str, object] = {}
        self.current_index = 0
        self.current_answer_var = tk.StringVar()

        self._build_layout()
        self._show_question()

    def _build_layout(self) -> None:
        self.header = tk.Label(
            self.root,
            text="Определим тип телефона по ответам",
            font=("Arial", 20, "bold"),
            pady=18,
        )
        self.header.pack(fill="x")

        self.form_frame = tk.Frame(self.root, padx=36, pady=12)
        self.form_frame.pack(fill="both", expand=True)

        self.progress_label = tk.Label(self.form_frame, anchor="w", font=("Arial", 11))
        self.progress_label.pack(fill="x", pady=(0, 16))

        self.question_label = tk.Label(
            self.form_frame,
            anchor="w",
            justify="left",
            wraplength=700,
            font=("Arial", 15),
        )
        self.question_label.pack(fill="x", pady=(0, 20))

        self.answer_frame = tk.Frame(self.form_frame)
        self.answer_frame.pack(fill="both", expand=True, anchor="nw")

        self.error_label = tk.Label(
            self.form_frame,
            text="",
            fg="#b00020",
            anchor="w",
            justify="left",
        )
        self.error_label.pack(fill="x", pady=(8, 8))

        self.controls = tk.Frame(self.form_frame)
        self.controls.pack(fill="x", side="bottom", pady=(8, 0))

        self.back_button = tk.Button(self.controls, text="Назад", command=self._go_back, width=12)
        self.back_button.pack(side="left")

        self.next_button = tk.Button(
            self.controls,
            text="Далее",
            command=self._go_next,
            width=18,
            default="active",
        )
        self.next_button.pack(side="right")
        self.notice_label = tk.Label(
            self.form_frame,
            text="Вопрос о станции метро не используется: в обучающем датасете нет этого признака.",
            fg="#666666",
            anchor="w",
            justify="left",
            wraplength=700,
        )
        self.notice_label.pack(fill="x", pady=(12, 0))

        self.result_frame = tk.Frame(self.root, padx=36, pady=36)
        self.result_label = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 22, "bold"),
            pady=24,
        )
        self.result_label.pack(fill="x")
        self.result_detail = tk.Label(
            self.result_frame,
            text="",
            font=("Arial", 12),
            fg="#555555",
            pady=10,
        )
        self.result_detail.pack(fill="x")
        tk.Button(
            self.result_frame,
            text="Пройти заново",
            command=self._restart,
            width=20,
        ).pack(pady=24)

    def _show_question(self) -> None:
        question = VISIBLE_QUESTIONS[self.current_index]
        for child in self.answer_frame.winfo_children():
            child.destroy()

        self.progress_label.configure(
            text=f"Вопрос {self.current_index + 1} из {len(VISIBLE_QUESTIONS)}"
        )
        self.question_label.configure(text=question.text)
        self.error_label.configure(text="")
        self.current_answer_var.set(str(self.answers.get(question.key, "")))

        if question.kind in {"choice", "eye_color"}:
            for option in question.options:
                tk.Radiobutton(
                    self.answer_frame,
                    text=option,
                    variable=self.current_answer_var,
                    value=option,
                    tristatevalue="unselected",
                    anchor="w",
                    justify="left",
                    wraplength=680,
                    font=("Arial", 12),
                    padx=8,
                    pady=7,
                ).pack(fill="x", anchor="w")
        elif question.kind == "integer":
            tk.Label(
                self.answer_frame,
                text="Введите целое неотрицательное число:",
                anchor="w",
                font=("Arial", 12),
            ).pack(anchor="w", pady=(0, 8))
            entry = tk.Entry(
                self.answer_frame,
                textvariable=self.current_answer_var,
                width=18,
                font=("Arial", 13),
            )
            entry.pack(anchor="w")
            entry.focus_set()

        self.back_button.configure(state="normal" if self.current_index else "disabled")
        is_last = self.current_index == len(VISIBLE_QUESTIONS) - 1
        self.next_button.configure(text="Показать результат" if is_last else "Далее")

    def _read_current_answer(self, question: Question) -> object:
        return validate_answer(question, self.current_answer_var.get())

    def _go_next(self) -> None:
        question = VISIBLE_QUESTIONS[self.current_index]
        try:
            self.answers[question.key] = self._read_current_answer(question)
        except ValueError as error:
            self.error_label.configure(text=str(error))
            return

        if self.current_index == len(VISIBLE_QUESTIONS) - 1:
            self._show_result()
            return

        self.current_index += 1
        self._show_question()

    def _go_back(self) -> None:
        if self.current_index == 0:
            return
        self.current_index -= 1
        self._show_question()

    def _show_result(self) -> None:
        try:
            vector = answers_to_vector(self.answers)
            label = self.model.predict_one(vector)
        except (ValueError, RuntimeError) as error:
            self.error_label.configure(text=str(error))
            return

        self.form_frame.pack_forget()
        self.result_label.configure(text=f"Предполагаемый тип телефона: {PHONE_LABELS[label]}")
        self.result_detail.configure(
            text=f"Прогноз рассчитан методом k ближайших соседей при K = {K}."
        )
        self.result_frame.pack(fill="both", expand=True)

    def _restart(self) -> None:
        self.result_frame.pack_forget()
        self.answers.clear()
        self.current_index = 0
        self.form_frame.pack(fill="both", expand=True)
        self._show_question()


def main() -> None:
    root = tk.Tk()
    try:
        PhoneClassifierApp(root)
    except Exception as error:  # Show a readable startup error instead of a traceback-only failure.
        messagebox.showerror("Не удалось запустить приложение", str(error))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
