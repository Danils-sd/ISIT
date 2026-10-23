from pathlib import Path
from config import DATASET_DIR, LABELS

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp"}

def load_dataset() -> list[dict]:
    items = []
    for label in LABELS:
        folder = DATASET_DIR / label
        if not folder.exists():
            print(f"[WARN] Папка не найдена: {folder}")
            continue

        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in VALID_EXT:
                continue
            items.append({"path": str(img_path), "true_label": label})

    print(f"[INFO] Загружено изображений: {len(items)}")

    counts = {}
    for it in items:
        counts[it["true_label"]] = counts.get(it["true_label"], 0) + 1
    print(f"[INFO] Распределение по классам: {counts}")

    return items