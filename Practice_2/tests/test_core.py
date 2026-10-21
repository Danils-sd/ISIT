"""Автоматические тесты для модулей сетевого анализа (Практика №2)."""
import unittest
from pathlib import Path
import tempfile
import networkx as nx
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import GRAPH_PKL_PATH, CENTRALITY_CSV_PATH, GROUP_MEMBERS
from graph_builder import GraphBuilder
from centrality import CentralityAnalyzer
from vk_client import VKClient
from visualize import GraphVisualizer


class MockVKClient(VKClient):
    """Мок-клиент VK API для детерминированного тестирования без сети."""

    def __init__(self, network_mock: dict):
        super().__init__(token="mock_token", requests_per_second=0)
        self.network_mock = network_mock

    def resolve_id(self, screen_name: str) -> int:
        if str(screen_name).isdigit():
            return int(screen_name)
        if str(screen_name).startswith("id") and screen_name[2:].isdigit():
            return int(screen_name[2:])
        name_map = {"alice": 101, "bob": 102}
        if screen_name in name_map:
            return name_map[screen_name]
        raise ValueError(f"Unknown mock user: {screen_name}")

    def get_friends(self, user_id: int, limit: int = 30) -> list[int]:
        return self.network_mock.get(user_id, [])[:limit]


class CoreTests(unittest.TestCase):
    """Набор базовых тестов графа, метрик центральности и вспомогательных модулей."""

    def test_saved_graph_contract(self) -> None:
        """Проверка контракта сохранённого графа (graph.pkl)."""
        self.assertTrue(GRAPH_PKL_PATH.exists(), f"Файл {GRAPH_PKL_PATH} должен существовать")
        G, group_ids = GraphBuilder.load_graph(GRAPH_PKL_PATH)

        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(G.number_of_nodes(), 1988)
        self.assertEqual(G.number_of_edges(), 2169)
        self.assertEqual(len(group_ids), 5)
        self.assertTrue(nx.is_connected(G), "Граф должен быть связным")

        # Проверка уровней узлов
        levels = nx.get_node_attributes(G, "level")
        self.assertEqual(len(levels), G.number_of_nodes())
        self.assertTrue(set(levels.values()).issubset({0, 1, 2}))

        # Все члены группы должны быть в графе
        for uid in group_ids:
            self.assertIn(uid, G)

    def test_centrality_csv_contract_and_invariants(self) -> None:
        """Проверка структуры и математических инвариантов таблицы centrality.csv."""
        self.assertTrue(CENTRALITY_CSV_PATH.exists(), f"Файл {CENTRALITY_CSV_PATH} должен существовать")
        df = pd.read_csv(CENTRALITY_CSV_PATH)

        expected_columns = [
            "user_id",
            "degree",
            "betweenness",
            "closeness",
            "eigenvector",
            "rank_betweenness",
            "rank_closeness",
            "rank_eigenvector",
        ]
        self.assertEqual(list(df.columns), expected_columns)
        self.assertEqual(len(df), 5)

        # Значения центральностей должны быть в диапазоне [0, 1]
        for col in ["betweenness", "closeness", "eigenvector"]:
            self.assertTrue(((df[col] >= 0.0) & (df[col] <= 1.0)).all())

        # Ранги должны быть целыми числами от 1 до 5
        for col in ["rank_betweenness", "rank_closeness", "rank_eigenvector"]:
            self.assertTrue((df[col] >= 1).all() and (df[col] <= 5).all())

    def test_vk_client_offline_resolution(self) -> None:
        """Проверка разрешения идентификаторов без обращения к сети."""
        client = VKClient(token="")
        self.assertEqual(client.resolve_id("593281333"), 593281333)
        self.assertEqual(client.resolve_id("id503794125"), 503794125)
        self.assertEqual(client.resolve_id(436106936), 436106936)

        # При попытке разрешить текстовый screen_name без токена должно быть исключение
        with self.assertRaises(ValueError):
            client.resolve_id("durov")

    def test_graph_builder_two_hop_bfs(self) -> None:
        """Проверка 2-шагового построения эго-сети через GraphBuilder с мок-клиентом."""
        # Граф связей:
        # Уровень 0: узел 1
        # Уровень 1 (друзья узла 1): узлы 2, 3
        # Уровень 2 (друзья узлов 2, 3): узлы 4, 5
        network_mock = {
            1: [2, 3],
            2: [1, 4],
            3: [1, 5],
            4: [2],
            5: [3],
        }
        client = MockVKClient(network_mock)
        builder = GraphBuilder(client=client, max_friends_per_user=10)
        G, group_ids = builder.build_two_hop_graph(members=["id1"])

        self.assertEqual(group_ids, [1])
        self.assertEqual(G.number_of_nodes(), 5)
        self.assertEqual(G.nodes[1]["level"], 0)
        self.assertEqual(G.nodes[2]["level"], 1)
        self.assertEqual(G.nodes[3]["level"], 1)
        self.assertEqual(G.nodes[4]["level"], 2)
        self.assertEqual(G.nodes[5]["level"], 2)
        self.assertTrue(G.has_edge(1, 2))
        self.assertTrue(G.has_edge(2, 4))

    def test_centrality_calculation_on_canonical_graph(self) -> None:
        """Проверка корректности расчета центральностей на каноническом графе-звезде (Star Graph)."""
        # В звезде S_5 узел 0 связан с 1, 2, 3, 4
        star = nx.star_graph(4)
        analyzer = CentralityAnalyzer(star, group_ids=[0, 1, 2])
        df = analyzer.compute_all()

        center_row = df[df["user_id"] == 0].iloc[0]
        leaf_row = df[df["user_id"] == 1].iloc[0]

        # Центральный узел звезды имеет максимальный betweenness = 1.0
        self.assertAlmostEqual(center_row["betweenness"], 1.0)
        self.assertAlmostEqual(leaf_row["betweenness"], 0.0)
        self.assertEqual(center_row["rank_betweenness"], 1)

        # Центральный узел ближе всех остальных ко всем узлам
        self.assertGreater(center_row["closeness"], leaf_row["closeness"])
        self.assertGreater(center_row["eigenvector"], leaf_row["eigenvector"])

    def test_visualizer_renders_files(self) -> None:
        """Проверка создания изображений модулем визуализации."""
        star = nx.star_graph(4)
        nx.set_node_attributes(star, {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}, "level")
        analyzer = CentralityAnalyzer(star, group_ids=[0, 1, 2, 3, 4])
        df = analyzer.compute_all()
        bw = analyzer.compute_betweenness()

        with tempfile.TemporaryDirectory() as tmpdir:
            bars_path = Path(tmpdir) / "test_bars.png"
            graph_path = Path(tmpdir) / "test_graph.png"

            visualizer = GraphVisualizer()
            visualizer.plot_centrality_bars(df, path=bars_path)
            visualizer.plot_friends_graph(star, group_ids=[0, 1], betweenness=bw, path=graph_path, iterations=10)

            self.assertTrue(bars_path.exists())
            self.assertGreater(bars_path.stat().st_size, 1000)
            self.assertTrue(graph_path.exists())
            self.assertGreater(graph_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
