import ray
import pytest

from fidp.parallel.cache import CacheActor
from fidp.parallel.executor import RayExecutor, ResourceUnavailableError, TaskExecutionError
from fidp.parallel.resources import ResourceSpec
from fidp.parallel.runtime import RayRuntime


CALLS: dict[str, int] = {"count": 0}


def _double(value: int) -> int:
    return value * 2


def _maybe_fail(value: int) -> int:
    if value == 2:
        raise ValueError("boom")
    return value


def _gpu_ids(_: int) -> list[int]:
    return ray.get_gpu_ids()


def _counted(value: int) -> int:
    CALLS["count"] += 1
    return value * 10


def test_executor_map_order_and_determinism() -> None:
    with RayRuntime(local_mode=True, num_cpus=2, num_gpus=0):
        executor = RayExecutor()
        items = [3, 1, 2]
        first = executor.map(_double, items)
        second = executor.map(_double, items)
        assert first == [6, 2, 4]
        assert second == first


def test_executor_exception_context() -> None:
    with RayRuntime(local_mode=True, num_cpus=1, num_gpus=0):
        executor = RayExecutor()
        with pytest.raises(TaskExecutionError) as exc:
            executor.map(_maybe_fail, [1, 2, 3])
        assert exc.value.index == 1
        assert exc.value.item == 2
        assert "index=1" in str(exc.value)
        assert "item=2" in str(exc.value)


def test_executor_cpu_resource_spec() -> None:
    with RayRuntime(local_mode=True, num_cpus=1, num_gpus=0):
        executor = RayExecutor()
        results = executor.map(
            _gpu_ids,
            [0, 1],
            resources=ResourceSpec(num_cpus=1, num_gpus=0),
        )
        assert results == [[], []]


def test_executor_gpu_unavailable_error() -> None:
    with RayRuntime(local_mode=True, num_cpus=1, num_gpus=0):
        executor = RayExecutor()
        with pytest.raises(ResourceUnavailableError) as exc:
            executor.map(_double, [1], resources=ResourceSpec(num_cpus=1, num_gpus=1))
        assert "GPU" in str(exc.value)


def test_executor_cache_deduplicates() -> None:
    CALLS["count"] = 0
    with RayRuntime(local_mode=True, num_cpus=1, num_gpus=0):
        cache = CacheActor.remote(capacity=8)
        executor = RayExecutor(cache_actor=cache, cache_key=lambda item: item)
        results = executor.map(_counted, [1, 1, 2])
        assert results == [10, 10, 20]
        assert CALLS["count"] == 2
