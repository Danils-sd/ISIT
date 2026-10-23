import pandas as pd
from config import MODELS, PREDICTIONS_CSV, METRICS_CSV
import pandas as pd

from config import MODELS, PREDICTIONS_CSV, METRICS_CSV
from dataset_loader import load_dataset
from inference import run_all_models
from metrics import compute_metrics, print_errors


def main():
    # 1. Датасет
    dataset = load_dataset()
    if not dataset:
        print("Датасет пуст. Проверь папки dataset/civil и dataset/military.")
        return

    # 2. Инференс (сначала проверка Cloudflare, потом Groq)
    all_records = run_all_models(dataset, MODELS)

    if not all_records:
        print("Нет данных для анализа. Проверь ключи и лимиты.")
        return

    # 3. Сохранение предсказаний
    df = pd.DataFrame(all_records)
    df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Предсказания сохранены: {PREDICTIONS_CSV}")

    # 4. Метрики
    metrics_df = compute_metrics(df)
    metrics_df.to_csv(METRICS_CSV, index=False, encoding="utf-8-sig")

    print("\n===== СРАВНЕНИЕ МОДЕЛЕЙ =====")
    print(metrics_df.to_string(index=False))

    # 5. Ошибки
    print_errors(df)


if __name__ == "__main__":
    main()