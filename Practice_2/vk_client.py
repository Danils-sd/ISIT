"""Клиент для взаимодействия с VK API: сбор списка друзей и информации о пользователях."""
from typing import List, Dict, Any, Optional
import time
import logging
import vk_api
from vk_api.exceptions import ApiError

from config import VK_TOKEN, MAX_FRIENDS_PER_USER, REQUESTS_PER_SECOND

log = logging.getLogger(__name__)


class VKClient:
    """Обертка над VK API с контролем частоты запросов и обработкой исключений."""

    def __init__(self, token: Optional[str] = None, requests_per_second: float = REQUESTS_PER_SECOND):
        """
        Инициализация клиента VK API.
        
        :param token: Access token VK API. Если None, берется из переменной окружения VK_TOKEN.
        :param requests_per_second: Максимальное число запросов в секунду (rate limit).
        """
        self.token = token if token is not None else VK_TOKEN
        self.requests_per_second = requests_per_second
        self._last_request_time = 0.0
        self._api = None

        if self.token:
            session = vk_api.VkApi(token=self.token)
            self._api = session.get_api()
        else:
            log.warning("VKClient инициализирован без токена. Доступны только оффлайн-функции.")

    def _require_auth(self) -> None:
        """Проверяет наличие активной сессии API."""
        if self._api is None:
            raise ValueError(
                "VK_TOKEN не задан. Укажите токен при создании клиента, через config.py или переменную окружения."
            )

    def _throttle(self) -> None:
        """Ограничивает частоту вызовов к API в соответствии с лимитом."""
        if self.requests_per_second <= 0:
            return
        min_interval = 1.0 / self.requests_per_second
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def resolve_id(self, screen_name: str) -> int:
        """
        Преобразует screen_name или строку вида 'id12345' в числовой идентификатор пользователя.
        
        :param screen_name: Имя страницы или идентификатор (например, 'durov' или 'id593281333').
        :return: Числовой user_id.
        """
        cleaned = str(screen_name).strip()
        if cleaned.isdigit():
            return int(cleaned)
        if cleaned.startswith("id") and cleaned[2:].isdigit():
            return int(cleaned[2:])

        self._require_auth()
        self._throttle()
        try:
            resp = self._api.utils.resolveScreenName(screen_name=cleaned)
            if not resp or resp.get("type") != "user":
                raise ValueError(f"Не удалось разрешить {screen_name!r} в существующего пользователя.")
            return int(resp["object_id"])
        except ApiError as e:
            raise RuntimeError(f"Ошибка VK API при разрешении имени {screen_name}: {e}") from e

    def get_friends(self, user_id: int, limit: int = MAX_FRIENDS_PER_USER) -> List[int]:
        """
        Возвращает список идентификаторов друзей пользователя.
        
        :param user_id: Числовой идентификатор пользователя.
        :param limit: Максимальное количество друзей для выборки.
        :return: Список user_id друзей. При закрытом профиле или ошибке возвращает пустой список.
        """
        self._require_auth()
        self._throttle()
        try:
            resp = self._api.friends.get(user_id=user_id, count=limit)
            return resp.get("items", [])
        except ApiError as e:
            # Коды ошибок VK: 15 (доступ запрещен/приватный профиль), 18 (страница удалена/заблокирована)
            log.warning("Не удалось получить друзей для user_id=%s: %s", user_id, e)
            return []

    def get_users_info(self, user_ids: List[int], fields: str = "first_name,last_name") -> List[Dict[str, Any]]:
        """
        Пакетное получение информации о пользователях (до 1000 за один запрос).
        
        :param user_ids: Список идентификаторов пользователей.
        :param fields: Дополнительные запрашиваемые поля профиля.
        :return: Список словарей с данными пользователей.
        """
        if not user_ids:
            return []
        self._require_auth()

        result = []
        batch_size = 1000
        for i in range(0, len(user_ids), batch_size):
            chunk = user_ids[i:i + batch_size]
            self._throttle()
            try:
                users = self._api.users.get(user_ids=chunk, fields=fields)
                result.extend(users)
            except ApiError as e:
                log.warning("Ошибка при пакетном получении данных пользователей: %s", e)
        return result
