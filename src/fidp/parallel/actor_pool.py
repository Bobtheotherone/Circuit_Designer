"""Actor pool utilities for Ray."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import ray
from ray.util import ActorPool

from fidp.parallel.resources import ResourceSpec


class RayActorPool:
    """Pool of identical actors for repeated evaluation workloads."""

    def __init__(
        self,
        actor_cls: type,
        *,
        size: int,
        resources: ResourceSpec | None = None,
        actor_args: Sequence[Any] | None = None,
        actor_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be >= 1.")
        self._resources = resources or ResourceSpec()
        actor_args = actor_args or ()
        actor_kwargs = actor_kwargs or {}

        remote_cls = ray.remote(**self._resources.ray_options())(actor_cls)
        self._actors = [remote_cls.remote(*actor_args, **actor_kwargs) for _ in range(size)]
        self._pool = ActorPool(self._actors)

    def map(self, fn: Callable[[Any, Any], Any], items: Iterable[Any]) -> list[Any]:
        """Map a function over items using pooled actors (ordered)."""
        return list(self._pool.map(fn, items))

    def map_unordered(self, fn: Callable[[Any, Any], Any], items: Iterable[Any]) -> list[Any]:
        """Map a function over items using pooled actors (unordered)."""
        return list(self._pool.map_unordered(fn, items))

    def submit(self, fn: Callable[[Any, Any], Any], item: Any) -> None:
        """Submit a single item to the actor pool."""
        self._pool.submit(fn, item)

    def get_next(self, timeout: float | None = None) -> Any:
        """Return the next available result in submission order."""
        return self._pool.get_next(timeout=timeout)

    def get_next_unordered(self, timeout: float | None = None) -> Any:
        """Return the next available result regardless of order."""
        return self._pool.get_next_unordered(timeout=timeout)

    def shutdown(self) -> None:
        """Terminate all actors in the pool."""
        for actor in self._actors:
            ray.kill(actor, no_restart=True)
