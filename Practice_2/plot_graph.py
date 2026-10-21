"""Скрипт для быстрой отрисовки графа связей из сохранённого pickle-файла."""
from pathlib import Path
import pickle
import pandas as pd
import networkx as nx

from config import GRAPH_PKL_PATH, CENTRALITY_CSV_PATH, GRAPH_PNG_PATH
from visualize import GraphVisualizer

def main():
    pkl_file = GRAPH_PKL_PATH if GRAPH_PKL_PATH.exists() else Path("graph.pkl")
    if not pkl_file.exists():
        print(f"Файл {pkl_file} не найден!")
        return

    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    G = data["G"]
    group_ids = data["group_ids"]

    csv_file = CENTRALITY_CSV_PATH if CENTRALITY_CSV_PATH.exists() else Path("centrality.csv")
    betweenness = None
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        betweenness = dict(zip(df["user_id"], df["betweenness"]))

    visualizer = GraphVisualizer()
    out_path = visualizer.plot_friends_graph(G, group_ids, betweenness=betweenness, path=GRAPH_PNG_PATH)
    print(f"Сохранено: {out_path}")

if __name__ == "__main__":
    main()
