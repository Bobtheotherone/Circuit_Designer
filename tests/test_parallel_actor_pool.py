from __future__ import annotations

from fidp.parallel.actor_pool import RayActorPool
from fidp.parallel.resources import ResourceSpec
from fidp.parallel.runtime import RayRuntime


class Adder:
    def __init__(self, bias: int) -> None:
        self._bias = bias

    def add(self, value: int) -> int:
        return value + self._bias


def _actor_add(actor: Adder, value: int):
    return actor.add.remote(value)


def test_actor_pool_map() -> None:
    with RayRuntime(local_mode=True, num_cpus=2, num_gpus=0):
        pool = RayActorPool(
            Adder,
            size=2,
            resources=ResourceSpec(num_cpus=1, num_gpus=0),
            actor_args=(5,),
        )
        results = pool.map(_actor_add, [1, 2, 3])
        pool.shutdown()
        assert results == [6, 7, 8]
