"""Ray runtime management utilities."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional

import ray


class RayRuntime:
    """Context manager for Ray initialization and shutdown."""

    def __init__(
        self,
        *,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        local_mode: bool = False,
        include_dashboard: bool = False,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._num_cpus = num_cpus
        self._num_gpus = num_gpus
        self._local_mode = local_mode
        self._include_dashboard = include_dashboard
        self._init_kwargs = init_kwargs or {}
        self._started = False

    def __enter__(self) -> "RayRuntime":
        if ray.is_initialized():
            return self
        init_args: Dict[str, Any] = {
            "include_dashboard": self._include_dashboard,
            "local_mode": self._local_mode,
        }
        if self._num_cpus is not None:
            init_args["num_cpus"] = self._num_cpus
        if self._num_gpus is not None:
            init_args["num_gpus"] = self._num_gpus
        init_args.update(self._init_kwargs)
        ray.init(**init_args)
        self._started = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._started:
            ray.shutdown()
            self._started = False


@contextmanager
def ray_session(
    *,
    num_cpus: int | None = None,
    num_gpus: int | None = None,
    local_mode: bool = False,
    include_dashboard: bool = False,
    init_kwargs: Optional[Dict[str, Any]] = None,
):
    """Convenience context manager around RayRuntime."""
    runtime = RayRuntime(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        local_mode=local_mode,
        include_dashboard=include_dashboard,
        init_kwargs=init_kwargs,
    )
    try:
        runtime.__enter__()
        yield runtime
    finally:
        runtime.__exit__(None, None, None)
