"""Построение социального графа: группа, друзья и друзья друзей (BFS на 2 шага)."""
from typing import Tuple, List, Optional
from pathlib import Path
import logging
import pickle
import networkx as nx

from vk_client import VKClient
from config import GROUP_MEMBERS, MAX_FRIENDS_PER_USER, GRAPH_PKL_PATH

log = logging.getLogger(__name__)


class GraphBuilder:
    """Класс для построения и сериализации графа социальных связей через VK API."""

    def __init__(self, client: VKClient, max_friends_per_user: int = MAX_FRIENDS_PER_USER):
        """
        Инициализация строителя графа.
        
        :param client: Экземпляр VKClient для отправки запросов к API.
        :param max_friends_per_user: Максимальное количество друзей для каждого пользователя.
        """
        self.client = client
        self.max_friends_per_user = max_friends_per_user

    def build_two_hop_graph(self, members: Optional[List[str]] = None) -> Tuple[nx.Graph, List[int]]:
        """
        Строит неориентированный граф дружбы методом BFS на 2 шага:
          - Уровень 0: члены учебной группы
          - Уровень 1: друзья членов группы
          - Уровень 2: друзья друзей
          
        :param members: Список screen_name или числовых ID членов группы. По умолчанию config.GROUP_MEMBERS.
        :return: Кортеж (G, group_ids), где G — объект networkx.Graph, group_ids — список числовых ID.
        """
        if members is None:
            members = GROUP_MEMBERS

        G = nx.Graph()
        group_ids: List[int] = []

        # Шаг 0: Разрешаем screen_name -> user_id, добавляем участников (level=0)
        for name in members:
            try:
                uid = int(name) if str(name).isdigit() else self.client.resolve_id(name)
            except (ValueError, RuntimeError) as e:
                log.warning("Пропуск участника %s: %s", name, e)
                continue
            group_ids.append(uid)
            G.add_node(uid, level=0)

        log.info("Успешно добавлено членов группы (уровень 0): %d", len(group_ids))

        # Шаг 1: Получаем друзей каждого члена группы (level=1)
        level1_users = set()
        for uid in group_ids:
            friends = self.client.get_friends(uid, limit=self.max_friends_per_user)
            log.info("У участника %d получено друзей: %d", uid, len(friends))
            for friend_id in friends:
                # Если узел уже есть (например, другой член группы), не перезаписываем level=0
                if friend_id not in G:
                    G.add_node(friend_id, level=1)
                G.add_edge(uid, friend_id)
                level1_users.add(friend_id)

        log.info("Всего уникальных друзей уровня 1: %d", len(level1_users))

        # Шаг 2: Получаем друзей для каждого пользователя уровня 1 (level=2)
        level2_users = set()
        for friend_id in list(level1_users):
            friends_of_friend = self.client.get_friends(friend_id, limit=self.max_friends_per_user)
            for fof_id in friends_of_friend:
                if fof_id not in G:
                    G.add_node(fof_id, level=2)
                    level2_users.add(fof_id)
                G.add_edge(friend_id, fof_id)

        log.info(
            "Построение графа завершено. Узлов: %d (группа: %d, уровень 1: %d, уровень 2: %d), рёбер: %d",
            G.number_of_nodes(),
            len(group_ids),
            len(level1_users),
            len(level2_users),
            G.number_of_edges(),
        )
        return G, group_ids

    @staticmethod
    def save_graph(G: nx.Graph, group_ids: List[int], filepath: Path | str = GRAPH_PKL_PATH) -> None:
        """Сохраняет граф и идентификаторы группы в pickle-файл."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"G": G, "group_ids": group_ids}, f)
        log.info("Граф успешно сохранен в %s", path)

    @staticmethod
    def load_graph(filepath: Path | str = GRAPH_PKL_PATH) -> Tuple[nx.Graph, List[int]]:
        """Загружает граф и идентификаторы группы из pickle-файла."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Файл графа не найден: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        log.info("Граф успешно загружен из %s (узлов: %d, рёбер: %d)", path, data["G"].number_of_nodes(), data["G"].number_of_edges())
        return data["G"], data["group_ids"]


def build_friends_graph(client: VKClient) -> Tuple[nx.Graph, List[int]]:
    """Функция-фасад для обратной совместимости."""
    builder = GraphBuilder(client)
    return builder.build_two_hop_graph()
