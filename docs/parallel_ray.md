# Ray Parallel Execution

FIDP ships a Ray-based parallel execution layer for high-throughput evaluation and training.

## Quick start

```python
from fidp.parallel import RayRuntime, RayExecutor, ResourceSpec


def evaluate(x: int) -> int:
    return x * 2

with RayRuntime(num_cpus=4, num_gpus=0, include_dashboard=False):
    executor = RayExecutor()
    results = executor.map(evaluate, [1, 2, 3])
    print(results)
```

## Resource routing

CPU-bound tasks use the default resource spec. GPU-bound tasks should request GPUs explicitly:

```python
from fidp.parallel import ResourceSpec

cpu_spec = ResourceSpec(num_cpus=1, num_gpus=0)
# gpu_spec = ResourceSpec(num_cpus=2, num_gpus=1)
```

If no GPUs are available, GPU-tagged tasks raise a clear error before scheduling.

## Actor pools and caching

Use actor pools for stateful evaluators and cache actors for repeatable results:

```python
from fidp.parallel import CacheActor, RayActorPool
```

## Tests

Tests run Ray in `local_mode=True` for deterministic, fast execution.
