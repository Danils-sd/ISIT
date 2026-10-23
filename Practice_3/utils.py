import base64
import io
from pathlib import Path

from PIL import Image


def encode_image_to_base64(image_path: str, max_size: int = 512, quality: int = 70) -> str:
    img = Image.open(image_path).convert("RGB")

    # Уменьшаем до max_size по большей стороне, сохраняя пропорции
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)

    # Сохраняем в буфер как JPEG с указанным качеством
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    raw = buffer.getvalue()

    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def normalize_prediction(text: str) -> str:
    if not text:
        return "unknown"

    t = text.lower()

    mil_keys = ["military", "военн", "боевой", "истребитель", "fighter", "combat"]
    civ_keys = ["civil", "гражданск", "civilian", "commercial", "passenger", "airliner", "пассажирск"]

    has_mil = any(k in t for k in mil_keys)
    has_civ = any(k in t for k in civ_keys)

    if has_mil and not has_civ:
        return "military"
    if has_civ and not has_mil:
        return "civil"

    if has_mil and has_civ:
        first_mil = min((t.find(k) for k in mil_keys if k in t), default=10**9)
        first_civ = min((t.find(k) for k in civ_keys if k in t), default=10**9)
        return "military" if first_mil < first_civ else "civil"

    return "unknown"