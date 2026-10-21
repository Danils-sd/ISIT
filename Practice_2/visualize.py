"""Модуль визуализации графа связей и сравнительных диаграмм центральности."""
from typing import List, Dict, Optional
from pathlib import Path
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from config import (
    LEVEL_COLORS,
    DEFAULT_NODE_COLOR,
    BASE_NODE_SIZE,
    BETWEENNESS_SIZE_SCALE,
    GRAPH_FIGURE_SIZE,
    BARS_FIGURE_SIZE,
    FIGURE_DPI,
    METRIC_COLORS,
    GRAPH_PNG_PATH,
    CENTRALITY_BARS_PNG_PATH,
)

log = logging.getLogger(__name__)


class GraphVisualizer:
    """Класс для генерации графических отчетов и визуализации сетевых структур."""

    def __init__(self, dpi: int = FIGURE_DPI):
        self.dpi = dpi

    def plot_friends_graph(
        self,
        G: nx.Graph,
        group_ids: List[int],
        betweenness: Optional[Dict[int, float]] = None,
        path: Path | str = GRAPH_PNG_PATH,
        iterations: int = 50,
        seed: int = 42,
    ) -> Path:
        """
        Строит и сохраняет диаграмму графа связей:
          - Узлы окрашиваются по уровню эго-сети (0: красные, 1: синие, 2: зеленые).
          - Размер узла пропорционален показателю betweenness centrality.
          - Числовыми подписями выделяются члены учебной группы.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Для связного расположения используем наибольшую компоненту
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            subgraph = G.subgraph(largest_cc).copy()
        else:
            subgraph = G

        if betweenness is None:
            log.info("Betweenness не передан, вычисляем для масштабирования узлов...")
            betweenness = nx.betweenness_centrality(subgraph, normalized=True)

        log.info("Расчёт позиций Fruchterman-Reingold (spring_layout)...")
        pos = nx.spring_layout(subgraph, seed=seed, k=0.15, iterations=iterations)

        levels = nx.get_node_attributes(subgraph, "level")
        node_colors = [LEVEL_COLORS.get(levels.get(n, 2), DEFAULT_NODE_COLOR) for n in subgraph.nodes()]
        node_sizes = [BASE_NODE_SIZE + BETWEENNESS_SIZE_SCALE * betweenness.get(n, 0.0) for n in subgraph.nodes()]

        plt.figure(figsize=GRAPH_FIGURE_SIZE)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.12, width=0.4, edge_color="#7f8c8d")
        nx.draw_networkx_nodes(
            subgraph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="white",
            linewidths=0.5,
        )

        # Выделяем подписями только членов группы
        labels = {n: str(n) for n in group_ids if n in subgraph}
        nx.draw_networkx_labels(
            subgraph,
            pos,
            labels=labels,
            font_size=10,
            font_weight="bold",
            font_color="#000000",
        )

        plt.title(
            "VK friends graph\n"
            "красные — группа, синие — друзья, зелёные — друзья друзей\n"
            "размер узла ∝ betweenness",
            fontsize=12,
            pad=14,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()

        log.info("Граф сохранен: %s", save_path)
        return save_path

    def plot_centrality_bars(
        self,
        df: pd.DataFrame,
        path: Path | str = CENTRALITY_BARS_PNG_PATH,
    ) -> Path:
        """
        Строит и сохраняет горизонтальную тройную гистограмму метрик центральности членов группы.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Сортируем по возрастанию betweenness, чтобы лидер был сверху
        plot_df = df.sort_values("betweenness", ascending=True)

        fig, axes = plt.subplots(1, 3, figsize=BARS_FIGURE_SIZE, sharey=True)

        metrics = [
            ("betweenness", METRIC_COLORS["betweenness"], "Betweenness"),
            ("closeness", METRIC_COLORS["closeness"], "Closeness"),
            ("eigenvector", METRIC_COLORS["eigenvector"], "Eigenvector"),
        ]

        y_labels = plot_df["user_id"].astype(str)

        for ax, (col, color, title) in zip(axes, metrics):
            ax.barh(y_labels, plot_df[col], color=color)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel("значение", fontsize=10)
            ax.grid(axis="x", linestyle="--", alpha=0.3)

        axes[0].set_ylabel("user_id", fontsize=10)
        plt.suptitle("Центральность членов группы", fontsize=14, y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi)
        plt.close()

        log.info("Гистограмма центральностей сохранена: %s", save_path)
        return save_path


def draw_graph(G: nx.Graph, group_ids: List[int], betweenness: Dict[int, float], path: str = "graph.png") -> None:
    """Функция-фасад для обратной совместимости."""
    visualizer = GraphVisualizer()
    visualizer.plot_friends_graph(G, group_ids, betweenness=betweenness, path=path)


def draw_centrality_bars(df: pd.DataFrame, path: str = "centrality_bars.png") -> None:
    """Функция-фасад для обратной совместимости."""
    visualizer = GraphVisualizer()
    visualizer.plot_centrality_bars(df, path=path)
