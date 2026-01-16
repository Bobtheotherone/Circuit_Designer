"""Micro-benchmark for DSL expansion and canonicalization."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time

from fidp.circuits.canonical import CanonicalizationCache, canonicalize_circuit
from fidp.dsl import compile_dsl, parse_dsl


BENCH_DSL = """circuit Bench {
  ports: (P);
  body: series(R(100),C(1e-6),R(200),C(2e-6),R(300));
}
"""


def run_benchmark(iterations: int, seed: int, profile: bool) -> None:
    program = parse_dsl(BENCH_DSL)
    cache = CanonicalizationCache(max_items=256)

    def workload() -> None:
        start = time.perf_counter()
        for _ in range(iterations):
            circuit = compile_dsl(program, seed=seed)
            canonicalize_circuit(circuit, cache=cache)
        elapsed = time.perf_counter() - start
        per_min = iterations / elapsed * 60.0
        print(f"Iterations: {iterations}")
        print(f"Elapsed: {elapsed:.4f}s")
        print(f"Throughput: {per_min:.1f} circuits/min")

    if profile:
        profiler = cProfile.Profile()
        profiler.enable()
        workload()
        profiler.disable()
        stats = pstats.Stats(profiler).strip_dirs().sort_stats("tottime")
        stats.print_stats(20)
    else:
        workload()


def main() -> None:
    parser = argparse.ArgumentParser(description="FIDP DSL micro-benchmark")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    run_benchmark(args.iterations, args.seed, args.profile)


if __name__ == "__main__":
    main()
