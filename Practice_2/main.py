"""Точка входа: сбор графа связей VK, расчёт метрик центральности и формирование отчётов."""
import argparse
import logging
import sys
from pathlib import Path

from config import (
    VK_TOKEN,
    GRAPH_PKL_PATH,
    CENTRALITY_CSV_PATH,
    CENTRALITY_REPORT_JSON_PATH,
    GRAPH_SUMMARY_JSON_PATH,
    GRAPH_PNG_PATH,
    CENTRALITY_BARS_PNG_PATH,
)
from vk_client import VKClient
from graph_builder import GraphBuilder
from centrality import CentralityAnalyzer
from visualize import GraphVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="Анализ социального графа учебной группы ВКонтакте")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Запросить свежие данные из VK API (требует установленного VK_TOKEN в окружении или config.py)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Пропустить построение графических визуализаций (только расчёт метрик)",
    )
    return parser.parse_args()


def run_pipeline(fetch_from_api: bool = False, skip_plots: bool = False) -> None:
    log.info("=== Запуск пайплайна сетевого анализа (Практика №2) ===")

    # 1. Получение графа (загрузка из кэша либо обращение к VK API)
    if fetch_from_api:
        log.info("Режим сбора данных: запрос к VK API...")
        if not VK_TOKEN:
            log.error("Для флага --fetch необходим VK_TOKEN! Укажите токен в config.py или переменной окружения.")
            sys.exit(1)
        client = VKClient(token=VK_TOKEN)
        builder = GraphBuilder(client)
        G, group_ids = builder.build_two_hop_graph()
        builder.save_graph(G, group_ids, filepath=GRAPH_PKL_PATH)
    else:
        log.info("Режим офлайн-анализа: загрузка сохраненного графа из %s...", GRAPH_PKL_PATH)
        if not GRAPH_PKL_PATH.exists():
            log.error("Файл графа %s не найден! Запустите скрипт с ключом --fetch или поместите graph.pkl в папку data/.", GRAPH_PKL_PATH)
            sys.exit(1)
        G, group_ids = GraphBuilder.load_graph(filepath=GRAPH_PKL_PATH)

    # 2. Расчет метрик центральности
    log.info("Инициализация анализатора центральностей...")
    analyzer = CentralityAnalyzer(G, group_ids)
    df = analyzer.compute_all()

    # Печать результатов в консоль
    print("\n" + "=" * 80)
    print(" МЕТРИКИ ЦЕНТРАЛЬНОСТИ ЧЛЕНОВ УЧЕБНОЙ ГРУППЫ ".center(80, "="))
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    # 3. Экспорт отчётов (CSV, JSON)
    analyzer.export_reports(
        df=df,
        csv_path=CENTRALITY_CSV_PATH,
        report_json_path=CENTRALITY_REPORT_JSON_PATH,
        summary_json_path=GRAPH_SUMMARY_JSON_PATH,
    )

    # 4. Построение графиков
    if not skip_plots:
        log.info("Генерация визуализаций...")
        visualizer = GraphVisualizer()
        visualizer.plot_centrality_bars(df, path=CENTRALITY_BARS_PNG_PATH)
        
        # Получаем betweenness по компоненте для масштабирования узлов
        betweenness_dict = analyzer.compute_betweenness()
        visualizer.plot_friends_graph(
            G=G,
            group_ids=group_ids,
            betweenness=betweenness_dict,
            path=GRAPH_PNG_PATH,
        )
    else:
        log.info("Построение графиков пропущено (флаг --no-plot).")

    log.info("Пайплайн успешно завершен!")


def main():
    args = parse_args()
    run_pipeline(fetch_from_api=args.fetch, skip_plots=args.no_plot)


if __name__ == "__main__":
    main()
