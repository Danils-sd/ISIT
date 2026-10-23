import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from config import POSITIVE_LABEL, LABELS


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model_name, group in df.groupby("model"):
        y_true = group["true_label"].tolist()
        y_pred = group["predicted_label"].tolist()

        n_total = len(group)
        n_unknown = sum(1 for p in y_pred if p == "unknown")

        y_pred_fixed = [p if p in LABELS else POSITIVE_LABEL for p in y_pred]

        acc = accuracy_score(y_true, y_pred_fixed)
        prec = precision_score(y_true, y_pred_fixed,
                               pos_label=POSITIVE_LABEL, zero_division=0)
        rec = recall_score(y_true, y_pred_fixed,
                           pos_label=POSITIVE_LABEL, zero_division=0)
        f1 = f1_score(y_true, y_pred_fixed,
                      pos_label=POSITIVE_LABEL, zero_division=0)

        cm = confusion_matrix(y_true, y_pred_fixed, labels=LABELS)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        rows.append({
            "model": model_name,
            "n_total": n_total,
            "n_unknown": n_unknown,
            "accuracy": round(acc, 3),
            "precision_military": round(prec, 3),
            "recall_military": round(rec, 3),
            "f1_military": round(f1, 3),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        })

    return pd.DataFrame(rows).sort_values("f1_military", ascending=False)


def print_errors(df: pd.DataFrame) -> None:
    print("\n===== ОШИБКИ МОДЕЛЕЙ (нетривиальные случаи) =====")
    errors = df[df["true_label"] != df["predicted_label"]]

    if errors.empty:
        print("Ошибок нет.")
        return

    for _, row in errors.iterrows():
        print(
            f"[{row['model']}] {row['image']}: "
            f"истина={row['true_label']}, "
            f"предсказано={row['predicted_label']}, "
            f"ответ='{row['raw_response']}'"
        )