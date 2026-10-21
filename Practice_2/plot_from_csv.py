"""Скрипт для быстрой отрисовки гистограмм центральности из CSV-файла."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CENTRALITY_CSV_PATH, CENTRALITY_BARS_PNG_PATH
from visualize import GraphVisualizer

def main():
    csv_file = CENTRALITY_CSV_PATH if CENTRALITY_CSV_PATH.exists() else Path("centrality.csv")
    if not csv_file.exists():
        print(f"Файл {csv_file} не найден!")
        return

    df = pd.read_csv(csv_file)
    visualizer = GraphVisualizer()
    out_path = visualizer.plot_centrality_bars(df, path=CENTRALITY_BARS_PNG_PATH)
    print(f"Сохранено: {out_path}")

if __name__ == "__main__":
    main()
