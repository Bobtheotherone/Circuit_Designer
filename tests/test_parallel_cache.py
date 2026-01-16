import ray
import pytest

from fidp.parallel.cache import CacheActor
from fidp.parallel.runtime import RayRuntime


def test_cache_actor_basic_ops() -> None:
    with RayRuntime(local_mode=True, num_cpus=1, num_gpus=0):
        cache = CacheActor.remote(capacity=2)
        assert not ray.get(cache.contains.remote("a"))
        with pytest.raises(KeyError):
            ray.get(cache.get.remote("a"))
        ray.get(cache.set.remote("a", 1))
        assert ray.get(cache.contains.remote("a"))
        assert ray.get(cache.get.remote("a")) == 1
        assert ray.get(cache.delete.remote("a"))
        assert not ray.get(cache.contains.remote("a"))
        assert not ray.get(cache.delete.remote("missing"))
        ray.get(cache.set.remote("b", 2))
        ray.get(cache.clear.remote())
        assert not ray.get(cache.contains.remote("b"))


def test_cache_actor_lru_eviction() -> None:
    with RayRuntime(local_mode=True, num_cpus=1, num_gpus=0):
        cache = CacheActor.remote(capacity=2)
        ray.get(cache.set.remote("a", 1))
        ray.get(cache.set.remote("b", 2))
        assert ray.get(cache.get.remote("a")) == 1
        ray.get(cache.set.remote("c", 3))
        assert not ray.get(cache.contains.remote("b"))
        assert ray.get(cache.contains.remote("a"))
        assert ray.get(cache.contains.remote("c"))
