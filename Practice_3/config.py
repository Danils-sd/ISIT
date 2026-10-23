import os
from pathlib import Path

GROQ_API_KEY = "-"
CF_ACCOUNT_ID = "-"
CF_API_TOKEN = "-"

MODELS = {
    "Qwen3.8-27B (Groq)": "qwen/qwen3.8-27b",
    "Llama-3.2-Vision (Cloudflare)": "@cf/meta/llama-3.2-11b-vision-instruct",
}

PROVIDERS = {
    "Qwen3.8-27B (Groq)": "groq",
    "Llama-3.2-Vision (Cloudflare)": "cloudflare",
}

DATASET_DIR = Path("dataset")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

PREDICTIONS_CSV = RESULTS_DIR / "predictions.csv"
METRICS_CSV = RESULTS_DIR / "metrics.csv"

PROMPT = (
    "Посмотри на изображение. На нём изображён самолёт. "
    "Ответь одним словом, военный это самолёт или гражданский. "
    "Используй только слова 'военный' или 'гражданский'."
)

REQUEST_DELAY = 1.0
MAX_TOKENS = 16
TEMPERATURE = 0.0
RETRIES = 3

LABELS = ["civil", "military"]
POSITIVE_LABEL = "military"