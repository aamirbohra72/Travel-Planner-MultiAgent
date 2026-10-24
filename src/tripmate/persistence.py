"""SQLite-backed LangGraph Store for long-term travel preferences."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)


class SQLitePreferenceStore(BaseStore):
    """Small durable key-value Store implementation for local deployments."""

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        if database_path != ":memory:":
            Path(database_path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        with self._connection() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS tripmate_store (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    PRIMARY KEY (namespace, key)
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _namespace(namespace: tuple[str, ...]) -> str:
        return json.dumps(namespace, separators=(",", ":"))

    @staticmethod
    def _item(row: sqlite3.Row) -> Item:
        return Item(
            value=json.loads(row["value"]),
            key=row["key"],
            namespace=tuple(json.loads(row["namespace"])),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any] | None,
        ttl: float | None = None,
    ) -> None:
        namespace_key = self._namespace(namespace)
        if value is None:
            self._delete(namespace, key)
            return
        now = datetime.now(UTC)
        expires_at = (now + timedelta(seconds=ttl)).isoformat() if ttl else None
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO tripmate_store
                   (namespace, key, value, created_at, updated_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(namespace, key) DO UPDATE SET
                     value=excluded.value,
                     updated_at=excluded.updated_at,
                     expires_at=excluded.expires_at""",
                (
                    namespace_key,
                    key,
                    json.dumps(value),
                    now.isoformat(),
                    now.isoformat(),
                    expires_at,
                ),
            )

    def _get(self, namespace: tuple[str, ...], key: str) -> Item | None:
        now = datetime.now(UTC).isoformat()
        with self._connection() as connection:
            row = connection.execute(
                """SELECT * FROM tripmate_store
                   WHERE namespace=? AND key=?
                   AND (expires_at IS NULL OR expires_at > ?)""",
                (self._namespace(namespace), key, now),
            ).fetchone()
        return self._item(row) if row else None

    def _delete(self, namespace: tuple[str, ...], key: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM tripmate_store WHERE namespace=? AND key=?",
                (self._namespace(namespace), key),
            )

    def _search(
        self,
        namespace_prefix: tuple[str, ...],
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> list[SearchItem]:
        now = datetime.now(UTC).isoformat()
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM tripmate_store
                   WHERE expires_at IS NULL OR expires_at > ?
                   ORDER BY updated_at DESC""",
                (now,),
            ).fetchall()
        matches: list[SearchItem] = []
        for row in rows:
            namespace = tuple(json.loads(row["namespace"]))
            if namespace[: len(namespace_prefix)] != namespace_prefix:
                continue
            value = json.loads(row["value"])
            if filter and any(value.get(key) != expected for key, expected in filter.items()):
                continue
            serialized = json.dumps(value, ensure_ascii=False)
            if query and query.casefold() not in serialized.casefold():
                continue
            matches.append(
                SearchItem(
                    namespace=namespace,
                    key=row["key"],
                    value=value,
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
            )
        return matches[offset : offset + limit]

    @staticmethod
    def _matches_condition(
        namespace: tuple[str, ...], condition: MatchCondition
    ) -> bool:
        path = condition.path
        if condition.match_type == "prefix":
            values = namespace[: len(path)]
        else:
            values = namespace[-len(path) :]
        return len(values) == len(path) and all(
            expected == "*" or expected == actual
            for expected, actual in zip(path, values)
        )

    def _list_namespaces(
        self,
        match_conditions: tuple[MatchCondition, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT namespace FROM tripmate_store ORDER BY namespace"
            ).fetchall()
        namespaces = [tuple(json.loads(row["namespace"])) for row in rows]
        if match_conditions:
            namespaces = [
                namespace
                for namespace in namespaces
                if all(
                    self._matches_condition(namespace, condition)
                    for condition in match_conditions
                )
            ]
        if max_depth is not None:
            namespaces = [namespace[:max_depth] for namespace in namespaces]
        return sorted(set(namespaces))[offset : offset + limit]

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None = None,
    ) -> None:
        self._put(namespace, key, value, ttl)

    async def aget(
        self, namespace: tuple[str, ...], key: str, *, refresh_ttl: bool | None = None
    ) -> Item | None:
        return self._get(namespace, key)

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        self._delete(namespace, key)

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        return self._search(namespace_prefix, query, filter, limit, offset)

    async def alist_namespaces(
        self,
        *,
        prefix: tuple[str | Literal["*"], ...] | None = None,
        suffix: tuple[str | Literal["*"], ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        conditions = tuple(
            condition
            for condition in (
                MatchCondition("prefix", prefix) if prefix else None,
                MatchCondition("suffix", suffix) if suffix else None,
            )
            if condition is not None
        )
        return self._list_namespaces(conditions, max_depth, limit, offset)

    def _dispatch(self, operation: Op) -> Result:
        if isinstance(operation, GetOp):
            return self._get(operation.namespace, operation.key)
        if isinstance(operation, PutOp):
            self._put(operation.namespace, operation.key, operation.value, operation.ttl)
            return None
        if isinstance(operation, SearchOp):
            return self._search(
                operation.namespace_prefix,
                operation.query,
                operation.filter,
                operation.limit,
                operation.offset,
            )
        if isinstance(operation, ListNamespacesOp):
            return self._list_namespaces(
                operation.match_conditions,
                operation.max_depth,
                operation.limit,
                operation.offset,
            )
        raise TypeError(f"Unsupported store operation: {type(operation).__name__}")

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute store operations synchronously in input order."""
        return [self._dispatch(operation) for operation in ops]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute store operations asynchronously in input order."""
        return [self._dispatch(operation) for operation in ops]