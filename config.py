"""Project-wide configuration for the phone type classifier."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset_normalized.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Chosen by quality_check.py on the current normalized dataset.
K = 7

TARGET_COLUMN = "phone"
PHONE_LABELS = {
    0: "iPhone",
    1: "Android",
}

# The order is part of the data contract. The questionnaire must produce
# values in exactly this order before they are passed to KNN.
FEATURE_COLUMNS = (
    "attention_feel",
    "decision_style",
    "complex_tech",
    "everyday_choice",
    "battery_drain",
    "shoots_often",
    "design_vs_durability",
    "phone_lags",
    "apps_uniform",
    "likes_travel",
    "phone_change_years",
    "games_hours",
    "posts_social",
    "watches_on_phone",
    "eye_blue",
    "eye_brown",
    "eye_gray",
    "eye_gray_blue",
    "eye_green",
    "eye_other",
)

EYE_FEATURE_COLUMNS = (
    "eye_blue",
    "eye_brown",
    "eye_gray",
    "eye_gray_blue",
    "eye_green",
    "eye_other",
)
