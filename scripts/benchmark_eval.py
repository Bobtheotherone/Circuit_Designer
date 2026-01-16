#!/usr/bin/env python3
"""Quick micro-benchmark for MNA and MOR evaluators."""

from __future__ import annotations

import argparse
import time

from fidp.dsl.generators.ladder import domino_ladder
from fidp.evaluators.mna import MnaEvaluator
from fidp.evaluators.mor import PrimaEvaluator, MorOptions, PrimaConfig
from fidp.evaluators.types import EvalRequest, FrequencyGridSpec


def _bench_mna(stages: int, grid: FrequencyGridSpec) -> float:
    evaluator = MnaEvaluator()
    circuit = domino_ladder(stages=stages, r_value=50.0, c_value=1e-6)
    request = EvalRequest(grid=grid, fidelity="mid")
    start = time.perf_counter()
    evaluator.evaluate(circuit, request)
    return time.perf_counter() - start


def _bench_mor(stages: int, grid: FrequencyGridSpec) -> float:
    config = PrimaConfig(min_order=4, max_order=8, order_step=2, target_rel_error=0.1)
    evaluator = PrimaEvaluator(MorOptions(prima_config=config))
    circuit = domino_ladder(stages=stages, r_value=50.0, c_value=1e-6)
    request = EvalRequest(grid=grid, fidelity="mor")
    start = time.perf_counter()
    evaluator.evaluate(circuit, request)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MNA/MOR evaluators.")
    parser.add_argument("--stages", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--mor", action="store_true", help="Include MOR benchmark.")
    args = parser.parse_args()

    grid = FrequencyGridSpec(f_start_hz=10.0, f_stop_hz=1e5, points=40)
    print("MNA benchmark:")
    for stages in args.stages:
        elapsed = _bench_mna(stages, grid)
        print(f"  stages={stages} -> {elapsed:.4f}s")

    if args.mor:
        print("MOR benchmark:")
        for stages in args.stages:
            elapsed = _bench_mor(stages, grid)
            print(f"  stages={stages} -> {elapsed:.4f}s")


if __name__ == "__main__":
    main()
