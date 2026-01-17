import os
import shutil

import numpy as np
import pytest

from fidp.dsl import compile_dsl, parse_dsl
from fidp.evaluators.mna import MnaEvaluator
from fidp.evaluators.spice import SpiceEvaluator
from fidp.evaluators.types import EvalRequest, FrequencyGridSpec


RUN_SPICE = os.getenv("FIDP_RUN_SPICE_TESTS") == "1"
if not RUN_SPICE:
    pytest.skip("Set FIDP_RUN_SPICE_TESTS=1 to run real ngspice tests.", allow_module_level=True)
if shutil.which("ngspice") is None:
    pytest.skip("ngspice not available; install it or run scripts/bootstrap.sh.", allow_module_level=True)


@pytest.mark.spice
def test_mna_vs_real_ngspice_dsl_floating_oneport() -> None:
    dsl = """circuit FloatingDSL {
  ports: (P);
  body: series(R(10),L(1e-3),C(5e-6));
}
"""
    program = parse_dsl(dsl)
    circuit = compile_dsl(program, seed=0)

    grid = FrequencyGridSpec(f_start_hz=10.0, f_stop_hz=1000.0, points=6, spacing="linear")
    request_mna = EvalRequest(grid=grid, fidelity="mid")
    request_spice = EvalRequest(grid=grid, fidelity="truth", spice_simulator="ngspice", timeout_s=10.0)

    mna_result = MnaEvaluator().evaluate(circuit, request_mna)
    spice_result = SpiceEvaluator().evaluate(circuit, request_spice)

    assert mna_result.status == "ok"
    assert spice_result.status == "ok"
    assert np.allclose(mna_result.Z, spice_result.Z, rtol=1e-4, atol=1e-6)
