"""Модуль конфигурации проекта: пути, параметры VK API и настройки визуализации."""
from pathlib import Path
import os

# Базовые директории
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
TESTS_DIR = BASE_DIR / "tests"

# Пути к файлам данных
GRAPH_PKL_PATH = DATA_DIR / "graph.pkl"
CENTRALITY_CSV_PATH = DATA_DIR / "centrality.csv"

# Пути к файлам отчётов и визуализаций
GRAPH_PNG_PATH = REPORTS_DIR / "graph.png"
CENTRALITY_BARS_PNG_PATH = REPORTS_DIR / "centrality_bars.png"
CENTRALITY_REPORT_JSON_PATH = REPORTS_DIR / "centrality_report.json"
GRAPH_SUMMARY_JSON_PATH = REPORTS_DIR / "graph_summary.json"

# Токен доступа VK (получить на https://dev.vk.com/ или через Standalone-приложение)
VK_TOKEN = os.getenv("VK_TOKEN", "")

# ID членов вашей группы (числовые VK ID либо screen_name)
GROUP_MEMBERS = [
    "id593281333",
    "id503794125",
    "id436106936",
    "id332058537",
    "id567281176",
]

# Ограничения VK API
MAX_FRIENDS_PER_USER = 30   # у некоторых тысячи друзей — берём срез для построения эго-сети
REQUESTS_PER_SECOND = 5      # VK API: ~5 запросов/сек для friends.get
VK_API_VERSION = "5.199"

# Параметры стилизации визуализации
# Уровни: 0 — члены группы, 1 — друзья, 2 — друзья друзей
LEVEL_COLORS = {
    0: "#e74c3c",  # Красный
    1: "#3498db",  # Синий
    2: "#2ecc71",  # Зелёный
}
DEFAULT_NODE_COLOR = "#95a5a6"

# Настройки отрисовки графа
BASE_NODE_SIZE = 60
BETWEENNESS_SIZE_SCALE = 4000
GRAPH_FIGURE_SIZE = (14, 12)
BARS_FIGURE_SIZE = (16, 6)
FIGURE_DPI = 140

# Цвета для гистограмм центральностей
METRIC_COLORS = {
    "betweenness": "#e74c3c",
    "closeness": "#3498db",
    "eigenvector": "#2ecc71",
}
