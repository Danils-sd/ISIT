import unittest

import numpy as np

from config import EYE_FEATURE_COLUMNS, FEATURE_COLUMNS, PHONE_LABELS
from data_validation import load_training_arrays
from knn_model import KNNClassifier
from preprocessing import (
    answers_to_vector,
    normalize_games_hours,
    normalize_phone_change_years,
)
from questionnaire import VISIBLE_QUESTIONS


class CoreTests(unittest.TestCase):
    def test_training_dataset_contract(self) -> None:
        dataframe, x, y = load_training_arrays()
        self.assertEqual(dataframe.shape[0], 44)
        self.assertEqual(x.shape, (44, len(FEATURE_COLUMNS)))
        self.assertEqual(y.shape, (44,))
        self.assertEqual(set(y.tolist()), {0, 1})

    def test_label_mapping(self) -> None:
        self.assertEqual(PHONE_LABELS[0], "iPhone")
        self.assertEqual(PHONE_LABELS[1], "Android")

    def test_questionnaire_vector_shape_and_eye_one_hot(self) -> None:
        answers = {}
        for question in VISIBLE_QUESTIONS:
            if question.kind == "integer":
                answers[question.key] = 5
            else:
                answers[question.key] = question.options[0]

        vector = answers_to_vector(answers)
        self.assertEqual(vector.shape, (len(FEATURE_COLUMNS),))
        self.assertTrue(np.isfinite(vector).all())
        self.assertTrue(((vector >= 0) & (vector <= 1)).all())
        eye_positions = [FEATURE_COLUMNS.index(column) for column in EYE_FEATURE_COLUMNS]
        self.assertEqual(float(vector[eye_positions].sum()), 1.0)

    def test_numeric_normalization_matches_source_dataset(self) -> None:
        self.assertEqual(normalize_phone_change_years(2), 0.0)
        self.assertAlmostEqual(normalize_phone_change_years(5), 0.5)
        self.assertEqual(normalize_phone_change_years(8), 1.0)
        self.assertEqual(normalize_games_hours(0), 0.0)
        self.assertEqual(normalize_games_hours(40), 1.0)
        self.assertEqual(normalize_games_hours(130), 1.0)

    def test_knn_majority_vote(self) -> None:
        x = np.asarray([[0.0], [0.1], [0.2], [1.0]])
        y = np.asarray([0, 0, 1, 1])
        model = KNNClassifier(3).fit(x, y)
        self.assertEqual(model.predict_one(np.asarray([0.12])), 0)

    def test_knn_tie_uses_closest_neighbor(self) -> None:
        x = np.asarray([[0.0], [1.0]])
        y = np.asarray([0, 1])
        model = KNNClassifier(2).fit(x, y)
        self.assertEqual(model.predict_one(np.asarray([0.1])), 0)

    def test_knn_rejects_invalid_k(self) -> None:
        with self.assertRaises(ValueError):
            KNNClassifier(0)
        with self.assertRaises(ValueError):
            KNNClassifier(3).fit(np.asarray([[0.0], [1.0]]), np.asarray([0, 1]))

    def test_app_radiobuttons_tristate_prevented(self) -> None:
        import tkinter as tk
        from app import PhoneClassifierApp

        root = tk.Tk()
        try:
            app = PhoneClassifierApp(root)
            root.update()

            rbs = [w for w in app.answer_frame.winfo_children() if isinstance(w, tk.Radiobutton)]
            self.assertGreater(len(rbs), 0)
            for rb in rbs:
                self.assertNotEqual(rb.cget("tristatevalue"), "")

            # Answer and move to next question
            app.current_answer_var.set(rbs[0].cget("value"))
            app._go_next()
            root.update()

            next_rbs = [w for w in app.answer_frame.winfo_children() if isinstance(w, tk.Radiobutton)]
            self.assertGreater(len(next_rbs), 0)
            self.assertEqual(app.current_answer_var.get(), "")
            for rb in next_rbs:
                self.assertNotEqual(rb.cget("tristatevalue"), "")
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
