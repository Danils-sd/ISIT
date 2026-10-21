"""Расчёт метрик центральности графа: Betweenness, Closeness, Eigenvector и Degree."""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
import networkx as nx
import pandas as pd

from config import CENTRALITY_CSV_PATH, CENTRALITY_REPORT_JSON_PATH, GRAPH_SUMMARY_JSON_PATH

log = logging.getLogger(__name__)


class CentralityAnalyzer:
    """Анализатор метрик центральности вершин в графе социальных связей."""

    def __init__(self, graph: nx.Graph, group_ids: List[int]):
        """
        Инициализация анализатора.
        
        :param graph: Полный неориентированный граф NetworkX.
        :param group_ids: Список идентификаторов участников учебной группы.
        """
        self.raw_graph = graph
        self.group_ids = [int(uid) for uid in group_ids]

        # Для корректного расчёта расстояний берем наибольшую компоненту связности
        if nx.is_connected(self.raw_graph):
            self.graph = self.raw_graph
        else:
            largest_cc = max(nx.connected_components(self.raw_graph), key=len)
            self.graph = self.raw_graph.subgraph(largest_cc).copy()
            log.warning(
                "Исходный граф не является связным. Метрики рассчитываются по гигантской компоненте (%d узлов).",
                self.graph.number_of_nodes(),
            )

    def compute_betweenness(self, normalized: bool = True) -> Dict[int, float]:
        """
        Вычисляет центральность по посредничеству (Betweenness Centrality).
        Отражает долю кратчайших путей между всеми парами узлов, проходящих через данный узел.
        """
        log.info("Вычисление Betweenness Centrality (алгоритм Брандеса)...")
        return nx.betweenness_centrality(self.graph, normalized=normalized)

    def compute_closeness(self) -> Dict[int, float]:
        """
        Вычисляет центральность по близости (Closeness Centrality).
        Отражает обратную величину суммы расстояний от данного узла до всех остальных.
        """
        log.info("Вычисление Closeness Centrality...")
        return nx.closeness_centrality(self.graph)

    def compute_eigenvector(self, max_iter: int = 1000, tol: float = 1e-6) -> Dict[int, float]:
        """
        Вычисляет центральность по собственному вектору (Eigenvector Centrality).
        Отражает влиятельность узла с учётом влиятельности его соседей (степенной метод).
        """
        log.info("Вычисление Eigenvector Centrality...")
        try:
            return nx.eigenvector_centrality(self.graph, max_iter=max_iter, tol=tol)
        except nx.PowerIterationFailedConvergence:
            log.warning("Power iteration не сошелся. Применяется точный спектральный расчет nx.eigenvector_centrality_numpy.")
            return nx.eigenvector_centrality_numpy(self.graph)

    def compute_all(self) -> pd.DataFrame:
        """
        Вычисляет все три метрики центральности, ранжирует участников и формирует DataFrame.
        
        :return: pandas.DataFrame с колонками user_id, degree, betweenness, closeness,
                 eigenvector, rank_betweenness, rank_closeness, rank_eigenvector.
        """
        betweenness = self.compute_betweenness()
        closeness = self.compute_closeness()
        eigenvector = self.compute_eigenvector()

        rows = []
        for uid in self.group_ids:
            if uid not in self.graph:
                log.warning("Участник %d отсутствует в расчётной компоненте графа.", uid)
                continue

            rows.append({
                "user_id": uid,
                "degree": self.graph.degree(uid),
                "betweenness": float(betweenness.get(uid, 0.0)),
                "closeness": float(closeness.get(uid, 0.0)),
                "eigenvector": float(eigenvector.get(uid, 0.0)),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("Ни один из участников группы не найден в анализируемом графе!")

        # Добавляем ранги (1 — максимальная центральность)
        for col in ["betweenness", "closeness", "eigenvector"]:
            df[f"rank_{col}"] = df[col].rank(ascending=False, method="min").astype(int)

        # Сортируем по убыванию посредничества
        df = df.sort_values("betweenness", ascending=False).reset_index(drop=True)
        return df

    def get_graph_summary(self) -> Dict[str, Any]:
        """Возвращает общие топологические характеристики графа."""
        return {
            "total_nodes": self.raw_graph.number_of_nodes(),
            "total_edges": self.raw_graph.number_of_edges(),
            "component_nodes": self.graph.number_of_nodes(),
            "component_edges": self.graph.number_of_edges(),
            "is_connected": nx.is_connected(self.raw_graph),
            "number_of_components": nx.number_connected_components(self.raw_graph),
            "density": nx.density(self.graph),
            "group_members_analyzed": len([uid for uid in self.group_ids if uid in self.graph]),
        }

    def export_reports(
        self,
        df: pd.DataFrame,
        csv_path: Path | str = CENTRALITY_CSV_PATH,
        report_json_path: Path | str = CENTRALITY_REPORT_JSON_PATH,
        summary_json_path: Path | str = GRAPH_SUMMARY_JSON_PATH,
    ) -> None:
        """Сохраняет результаты расчетов в CSV и структурированный JSON."""
        csv_p = Path(csv_path)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_p, index=False)
        log.info("Таблица центральностей сохранена в %s", csv_p)

        rep_p = Path(report_json_path)
        rep_p.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "summary": self.get_graph_summary(),
            "members_centrality": df.to_dict(orient="records"),
        }
        with open(rep_p, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        log.info("Отчёт в формате JSON сохранен в %s", rep_p)

        sum_p = Path(summary_json_path)
        sum_p.parent.mkdir(parents=True, exist_ok=True)
        with open(sum_p, "w", encoding="utf-8") as f:
            json.dump(self.get_graph_summary(), f, ensure_ascii=False, indent=2)
        log.info("Сводка графа сохранена в %s", sum_p)


def compute_centralities(G: nx.Graph, group_ids: List[int]) -> pd.DataFrame:
    """Функция-фасад для обратной совместимости."""
    analyzer = CentralityAnalyzer(G, group_ids)
    return analyzer.compute_all()
