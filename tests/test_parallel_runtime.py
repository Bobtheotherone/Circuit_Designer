import ray

from fidp.parallel.runtime import RayRuntime


def test_ray_runtime_idempotent() -> None:
    ray.shutdown()
    assert not ray.is_initialized()
    with RayRuntime(local_mode=True, num_cpus=1, num_gpus=0):
        assert ray.is_initialized()
        with RayRuntime(local_mode=True, num_cpus=1, num_gpus=0):
            assert ray.is_initialized()
        assert ray.is_initialized()
    assert not ray.is_initialized()
