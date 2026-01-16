"""Ray actor-based cache utilities."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Hashable

import ray


@ray.remote
class CacheActor:
    """In-memory cache with optional LRU eviction."""

    def __init__(self, capacity: int | None = None) -> None:
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity must be a positive integer or None.")
        self._capacity = capacity
        self._data: OrderedDict[Hashable, Any] = OrderedDict()

    def get(self, key: Hashable) -> Any:
        if key not in self._data:
            raise KeyError(f"Cache miss for key: {key!r}")
        self._data.move_to_end(key)
        return self._data[key]

    def contains(self, key: Hashable) -> bool:
        return key in self._data

    def set(self, key: Hashable, value: Any) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        self._enforce_capacity()

    def delete(self, key: Hashable) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def clear(self) -> None:
        self._data.clear()

    def _enforce_capacity(self) -> None:
        if self._capacity is None:
            return
        while len(self._data) > self._capacity:
            self._data.popitem(last=False)
