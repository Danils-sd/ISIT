import time
from pathlib import Path

import requests
from groq import Groq

from config import (
    PROMPT, MAX_TOKENS, TEMPERATURE, RETRIES, REQUEST_DELAY,
    GROQ_API_KEY, CF_ACCOUNT_ID, CF_API_TOKEN,
    MODELS, PROVIDERS,
)
from utils import encode_image_to_base64, normalize_prediction

groq_client = Groq(api_key=GROQ_API_KEY)

def call_groq(model_id: str, image_b64: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": image_b64}},
        ],
    }]
    response = groq_client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    return response.choices[0].message.content or ""


def call_cloudflare(model_id: str, image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_bytes = list(f.read())

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{CF_ACCOUNT_ID}/ai/run/{model_id}"
    )

    payload = {
        "prompt": PROMPT,
        "image": image_bytes,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    headers = {
        "Authorization": f"Bearer {CF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()

    result = data.get("result", {})
    if isinstance(result, dict):
        return result.get("response", "") or ""
    return str(result)


def call_model(model_id: str, image_b64: str, image_path: str, provider: str) -> str:
    for attempt in range(RETRIES):
        try:
            if provider == "cloudflare":
                return call_cloudflare(model_id, image_path)
            elif provider == "groq":
                return call_groq(model_id, image_b64)
        except Exception as e:
            print(f"    [ERROR] попытка {attempt + 1}/{RETRIES}: {e}")
            time.sleep(2 * (attempt + 1))
    return ""

def check_provider(provider: str, model_id: str) -> bool:
    try:
        if provider == "groq":
            groq_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        elif provider == "cloudflare":
            url = (
                f"https://api.cloudflare.com/client/v4/accounts/"
                f"{CF_ACCOUNT_ID}/ai/run/{model_id}"
            )
            response = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {CF_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={"prompt": "ping", "max_tokens": 1},
                timeout=30,
            )
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
        return True
    except Exception as e:
        print(f"[FAIL] {provider} / {model_id}: {e}")
        return False


def check_providers(models: dict, providers: dict) -> dict:
    print("\n=== Проверка провайдеров ===")
    working = {}

    for model_name, model_id in models.items():
        provider = providers.get(model_name, "groq")
        print(f"  Проверка {model_name} через {provider} ...", end=" ")

        if check_provider(provider, model_id):
            print("OK")
            working[model_name] = model_id
        else:
            print("НЕДОСТУПНА")

    print(f"\n[INFO] Доступно моделей: {len(working)} из {len(models)}")
    return working

def run_inference(dataset: list[dict], model_name: str, model_id: str) -> list[dict]:
    provider = PROVIDERS.get(model_name, "groq")
    print(f"\n=== Модель: {model_name} через {provider} ===")

    records = []
    for i, item in enumerate(dataset, 1):
        img_name = Path(item["path"]).name
        print(f"  [{i}/{len(dataset)}] {img_name} ...", end=" ")

        image_b64 = encode_image_to_base64(item["path"])
        raw = call_model(model_id, image_b64, item["path"], provider)
        pred = normalize_prediction(raw)

        print(f"→ {pred}")

        records.append({
            "model": model_name,
            "image": img_name,
            "true_label": item["true_label"],
            "raw_response": raw.strip().replace("\n", " ")[:120],
            "predicted_label": pred,
        })

        time.sleep(REQUEST_DELAY)

    return records


def run_all_models(dataset: list[dict], models: dict) -> list[dict]:
    working_models = check_providers(models, PROVIDERS)

    if not working_models:
        print("\n[ERROR] Ни одна модель не доступна.")
        return []

    all_records = []
    for model_name, model_id in working_models.items():
        print(f"\n>>> Старт прогона: {model_name}")
        records = run_inference(dataset, model_name, model_id)
        all_records.extend(records)
        print(f">>> Завершён прогон: {model_name} ({len(records)} записей)")

    return all_records