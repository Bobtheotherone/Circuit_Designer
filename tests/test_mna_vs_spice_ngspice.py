import os
import shutil

import numpy as np
import pytest

from fidp.circuits.ir import CircuitIR, Component, ParamValue, PortDef
from fidp.evaluators.mna import MnaEvaluator
from fidp.evaluators.spice import SpiceEvaluator
from fidp.evaluators.types import EvalRequest, FrequencyGridSpec


RUN_SPICE = os.getenv("FIDP_RUN_SPICE_TESTS") == "1"
if not RUN_SPICE:
    pytest.skip("Set FIDP_RUN_SPICE_TESTS=1 to run real ngspice tests.", allow_module_level=True)
if shutil.which("ngspice") is None:
    pytest.skip("ngspice not available; install it or run scripts/bootstrap.sh.", allow_module_level=True)


@pytest.mark.spice
def test_mna_vs_real_ngspice_oneport() -> None:
    grid = FrequencyGridSpec(f_start_hz=10.0, f_stop_hz=1000.0, points=6, spacing="linear")
    request_mna = EvalRequest(grid=grid, fidelity="mid")
    request_spice = EvalRequest(grid=grid, fidelity="truth", spice_simulator="ngspice", timeout_s=10.0)

    circuits = (
        CircuitIR(
            name="series_rlc",
            ports=[PortDef("P", "n1", "gnd")],
            components=[
                Component("R1", "R", "n1", "n2", ParamValue(10.0)),
                Component("L1", "L", "n2", "n3", ParamValue(1e-3)),
                Component("C1", "C", "n3", "gnd", ParamValue(5e-6)),
            ],
        ),
        CircuitIR(
            name="parallel_rc",
            ports=[PortDef("P", "n1", "gnd")],
            components=[
                Component("R1", "R", "n1", "gnd", ParamValue(50.0)),
                Component("C1", "C", "n1", "gnd", ParamValue(1e-6)),
            ],
        ),
    )

    mna = MnaEvaluator()
    spice = SpiceEvaluator()

    for circuit in circuits:
        mna_result = mna.evaluate(circuit, request_mna)
        spice_result = spice.evaluate(circuit, request_spice)
        assert mna_result.status == "ok"
        assert spice_result.status == "ok"
        # Looser tolerance to accommodate solver differences between MNA and ngspice.
        assert np.allclose(mna_result.Z, spice_result.Z, rtol=1e-4, atol=1e-6)
