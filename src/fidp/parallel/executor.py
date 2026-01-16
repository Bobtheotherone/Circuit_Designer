"""Ray executor utilities for parallel evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, Sequence

import ray

from fidp.parallel.resources import ResourceSpec


class ResourceUnavailableError(RuntimeError):
    """Raised when a task requests unavailable Ray resources."""


class TaskExecutionError(RuntimeError):
    """Raised when a Ray task fails with context about the item."""

    def __init__(
        self,
        *,
        index: int | None = None,
        item: Any | None = None,
        context: str | None = None,
        original: BaseException | None = None,
    ) -> None:
        message = "Ray task failed"
        if index is not None:
            message += f" (index={index})"
        if context:
            message += f" [{context}]"
        if item is not None:
            message += f" item={item!r}"
        if original is not None:
            message += f": {original}"
        super().__init__(message)
        self.index = index
        self.item = item
        self.context = context
        self.original = original


class TaskBatchError(RuntimeError):
    """Raised when multiple Ray tasks fail."""

    def __init__(self, errors: Sequence[TaskExecutionError]) -> None:
        summary = "; ".join(str(error) for error in errors[:3])
        message = f"{len(errors)} task(s) failed. {summary}"
        super().__init__(message)
        self.errors = list(errors)


@dataclass(frozen=True)
class TaskHandle:
    """Handle for a submitted Ray task."""

    ref: ray.ObjectRef
    context: str | None = None
    item: Any | None = None


@dataclass(frozen=True)
class _TaskRef:
    ref: ray.ObjectRef
    indices: Sequence[int]
    item: Any
    cache_key: Hashable | None


_MISSING = object()


class RayExecutor:
    """High-level task executor on top of Ray."""

    def __init__(
        self,
        *,
        default_resources: ResourceSpec | None = None,
        cache_actor: Any | None = None,
        cache_key: Callable[[Any], Hashable] | None = None,
    ) -> None:
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Use RayRuntime or ray_session.")
        if (cache_actor is None) != (cache_key is None):
            raise ValueError("cache_actor and cache_key must be provided together.")
        self._default_resources = default_resources or ResourceSpec()
        self._cache_actor = cache_actor
        self._cache_key = cache_key

    def map(
        self,
        func: Callable[[Any], Any],
        items: Iterable[Any],
        *,
        resources: ResourceSpec | Sequence[ResourceSpec] | None = None,
        fail_fast: bool = True,
        use_cache: bool = True,
    ) -> list[Any]:
        """Apply func to items and return results in input order."""
        self._ensure_ray_initialized()
        items_list = list(items)
        if not items_list:
            return []
        specs = self._normalize_resources(items_list, resources)
        remote_func = ray.remote(func)
        results, tasks = self._prepare_tasks(
            remote_func,
            items_list,
            specs,
            use_cache=use_cache,
        )
        self._collect_results(tasks, results, fail_fast=fail_fast)
        return results

    def imap_unordered(
        self,
        func: Callable[[Any], Any],
        items: Iterable[Any],
        *,
        resources: ResourceSpec | Sequence[ResourceSpec] | None = None,
        fail_fast: bool = True,
        use_cache: bool = True,
    ) -> Iterable[Any]:
        """Yield results as tasks complete (unordered)."""
        self._ensure_ray_initialized()
        items_list = list(items)
        if not items_list:
            return iter(())
        specs = self._normalize_resources(items_list, resources)
        remote_func = ray.remote(func)
        results, tasks = self._prepare_tasks(
            remote_func,
            items_list,
            specs,
            use_cache=use_cache,
        )

        def _generator() -> Iterable[Any]:
            for value in results:
                if value is not _MISSING:
                    yield value
            yield from self._iter_task_results(tasks, fail_fast=fail_fast)

        return _generator()

    def submit(
        self,
        func: Callable[..., Any],
        *args: Any,
        resources: ResourceSpec | None = None,
        context: str | None = None,
        item: Any | None = None,
        **kwargs: Any,
    ) -> TaskHandle:
        """Submit a single task and return its handle."""
        self._ensure_ray_initialized()
        spec = resources or self._default_resources
        self._ensure_resources_available(spec)
        remote_func = ray.remote(func)
        ref = remote_func.options(**spec.ray_options()).remote(*args, **kwargs)
        return TaskHandle(ref=ref, context=context, item=item)

    def gather(
        self,
        tasks: Sequence[TaskHandle],
        *,
        fail_fast: bool = True,
    ) -> list[Any]:
        """Gather results for submitted tasks in submission order."""
        self._ensure_ray_initialized()
        results: list[Any] = [None] * len(tasks)
        errors: list[TaskExecutionError] = []
        for index, task in enumerate(tasks):
            try:
                results[index] = ray.get(task.ref)
            except Exception as exc:  # pragma: no cover - exercised by ray
                error = TaskExecutionError(
                    index=index,
                    item=task.item,
                    context=task.context,
                    original=exc,
                )
                if fail_fast:
                    for remaining in tasks[index + 1 :]:
                        ray.cancel(remaining.ref, force=True)
                    raise error from exc
                errors.append(error)
        if errors:
            raise TaskBatchError(errors)
        return results

    def _prepare_tasks(
        self,
        remote_func: Any,
        items: Sequence[Any],
        specs: Sequence[ResourceSpec],
        *,
        use_cache: bool,
    ) -> tuple[list[Any], list[_TaskRef]]:
        results: list[Any] = [_MISSING] * len(items)
        tasks: list[_TaskRef] = []
        cache_actor = self._cache_actor if use_cache else None
        if cache_actor is None or self._cache_key is None:
            for index, (item, spec) in enumerate(zip(items, specs)):
                self._ensure_resources_available(spec)
                ref = remote_func.options(**spec.ray_options()).remote(item)
                tasks.append(_TaskRef(ref=ref, indices=[index], item=item, cache_key=None))
            return results, tasks

        key_to_indices: dict[Hashable, list[int]] = {}
        keys_list: list[Hashable | None] = []
        for index, item in enumerate(items):
            key = self._cache_key(item)
            keys_list.append(key)
            if key is None:
                continue
            try:
                hash(key)
            except TypeError as exc:
                raise TypeError("cache_key must return a hashable key.") from exc
            key_to_indices.setdefault(key, []).append(index)

        cached_keys: set[Hashable] = set()
        if key_to_indices:
            keys = list(key_to_indices)
            contains_refs = [cache_actor.contains.remote(key) for key in keys]
            contains = ray.get(contains_refs)
            for key, hit in zip(keys, contains):
                if not hit:
                    continue
                value = ray.get(cache_actor.get.remote(key))
                cached_keys.add(key)
                for index in key_to_indices[key]:
                    results[index] = value

            for key, indices in key_to_indices.items():
                if key in cached_keys:
                    continue
                index = indices[0]
                spec = specs[index]
                self._ensure_resources_available(spec)
                ref = remote_func.options(**spec.ray_options()).remote(items[index])
                tasks.append(_TaskRef(ref=ref, indices=indices, item=items[index], cache_key=key))

        for index, key in enumerate(keys_list):
            if key is not None:
                continue
            spec = specs[index]
            self._ensure_resources_available(spec)
            ref = remote_func.options(**spec.ray_options()).remote(items[index])
            tasks.append(_TaskRef(ref=ref, indices=[index], item=items[index], cache_key=None))

        return results, tasks

    def _collect_results(
        self,
        tasks: list[_TaskRef],
        results: list[Any],
        *,
        fail_fast: bool,
    ) -> None:
        if not tasks:
            return
        errors: list[TaskExecutionError] = []
        set_refs: list[ray.ObjectRef] = []
        remaining = {task.ref: task for task in tasks}
        while remaining:
            ready, _ = ray.wait(list(remaining), num_returns=1)
            for ref in ready:
                task = remaining.pop(ref)
                try:
                    value = ray.get(ref)
                except Exception as exc:  # pragma: no cover - exercised by ray
                    error = TaskExecutionError(
                        index=task.indices[0],
                        item=task.item,
                        original=exc,
                    )
                    if fail_fast:
                        for other_ref in remaining:
                            ray.cancel(other_ref, force=True)
                        raise error from exc
                    errors.append(error)
                    continue
                for index in task.indices:
                    results[index] = value
                if self._cache_actor is not None and task.cache_key is not None:
                    set_refs.append(self._cache_actor.set.remote(task.cache_key, value))
        if set_refs:
            ray.get(set_refs)
        if errors:
            raise TaskBatchError(errors)

    def _iter_task_results(
        self,
        tasks: list[_TaskRef],
        *,
        fail_fast: bool,
    ) -> Iterable[Any]:
        if not tasks:
            return iter(())
        errors: list[TaskExecutionError] = []
        set_refs: list[ray.ObjectRef] = []
        remaining = {task.ref: task for task in tasks}

        def _generator() -> Iterable[Any]:
            nonlocal errors
            try:
                while remaining:
                    ready, _ = ray.wait(list(remaining), num_returns=1)
                    for ref in ready:
                        task = remaining.pop(ref)
                        try:
                            value = ray.get(ref)
                        except Exception as exc:  # pragma: no cover - exercised by ray
                            error = TaskExecutionError(
                                index=task.indices[0],
                                item=task.item,
                                original=exc,
                            )
                            if fail_fast:
                                for other_ref in remaining:
                                    ray.cancel(other_ref, force=True)
                                raise error from exc
                            errors.append(error)
                            continue
                        if self._cache_actor is not None and task.cache_key is not None:
                            set_refs.append(self._cache_actor.set.remote(task.cache_key, value))
                        for _ in task.indices:
                            yield value
            finally:
                if set_refs:
                    ray.get(set_refs)
            if errors:
                raise TaskBatchError(errors)

        return _generator()

    def _normalize_resources(
        self,
        items: Sequence[Any],
        resources: ResourceSpec | Sequence[ResourceSpec] | None,
    ) -> list[ResourceSpec]:
        if resources is None:
            return [self._default_resources] * len(items)
        if isinstance(resources, ResourceSpec):
            return [resources] * len(items)
        if isinstance(resources, Sequence) and not isinstance(resources, (str, bytes)):
            if len(resources) != len(items):
                raise ValueError("resources length must match items length.")
            specs = list(resources)
            for spec in specs:
                if not isinstance(spec, ResourceSpec):
                    raise TypeError("resources must contain ResourceSpec entries.")
            return specs
        raise TypeError("resources must be a ResourceSpec or a sequence of ResourceSpec.")

    def _ensure_resources_available(self, spec: ResourceSpec) -> None:
        if spec.requires_gpu():
            available = ray.cluster_resources().get("GPU", 0.0)
            if available <= 0:
                raise ResourceUnavailableError(
                    "Requested GPU resources but Ray reports 0 GPUs. "
                    "Initialize Ray with num_gpus>0 or run on a GPU machine."
                )
            if available < spec.num_gpus:
                raise ResourceUnavailableError(
                    f"Requested {spec.num_gpus} GPUs but only {available} available."
                )
        if spec.custom_resources:
            cluster = ray.cluster_resources()
            for name, value in spec.custom_resources.items():
                if cluster.get(name, 0.0) < value:
                    raise ResourceUnavailableError(
                        f"Requested {value} of resource '{name}' but only "
                        f"{cluster.get(name, 0.0)} available."
                    )

    @staticmethod
    def _ensure_ray_initialized() -> None:
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Use RayRuntime or ray_session.")
