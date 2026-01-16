from fidp.circuits.ir import CircuitIR, Component, ParamValue, PortDef
from fidp.evaluators.mna import MnaEvaluator
from fidp.evaluators.mor import PrimaEvaluator, MorOptions, PrimaConfig
from fidp.evaluators.spice import SpiceEvaluator
from fidp.evaluators.types import EvalRequest, FrequencyGridSpec


def test_mna_singular_matrix_returns_error() -> None:
    components = [
        Component("R1", "R", "n1", "gnd", ParamValue(10.0)),
        Component("R2", "R", "n2", "n3", ParamValue(5.0)),
    ]
    ports = [PortDef("P", "n1", "gnd")]
    circuit = CircuitIR(name="singular", ports=ports, components=components)

    grid = FrequencyGridSpec(f_start_hz=1.0, f_stop_hz=10.0, points=3, spacing="linear")
    request = EvalRequest(grid=grid, fidelity="mid")
    result = MnaEvaluator().evaluate(circuit, request)

    assert result.status == "error"
    assert any(err.code == "singular_matrix" for err in result.errors)


def test_spice_missing_executable(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)

    components = [Component("R1", "R", "n1", "gnd", ParamValue(10.0))]
    ports = [PortDef("P", "n1", "gnd")]
    circuit = CircuitIR(name="spice_missing", ports=ports, components=components)

    grid = FrequencyGridSpec(f_start_hz=1.0, f_stop_hz=10.0, points=3, spacing="linear")
    request = EvalRequest(grid=grid, fidelity="truth", spice_simulator="ngspice")
    result = SpiceEvaluator().evaluate(circuit, request)

    assert result.status == "error"
    assert any(err.code == "spice_not_available" for err in result.errors)


def test_mor_rank_deficiency_returns_error() -> None:
    components = [Component("R1", "R", "n1", "gnd", ParamValue(10.0))]
    ports = [PortDef("P", "n1", "gnd")]
    circuit = CircuitIR(name="mor_fail", ports=ports, components=components)

    grid = FrequencyGridSpec(f_start_hz=10.0, f_stop_hz=100.0, points=4, spacing="linear")
    request = EvalRequest(grid=grid, fidelity="mor")
    config = PrimaConfig(min_order=20, max_order=20, order_step=2, target_rel_error=0.1)
    evaluator = PrimaEvaluator(MorOptions(prima_config=config))
    result = evaluator.evaluate(circuit, request)

    assert result.status == "error"
    assert any(err.code == "mor_reduction_failed" for err in result.errors)
