"""SPICE evaluator for CircuitIR circuits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import tempfile
import time

import numpy as np

from fidp.circuits import CircuitIR
from fidp.circuits.ops import flatten_circuit
from fidp.circuits.ir import PortDef
from fidp.errors import CircuitIRValidationError, SpiceNotAvailableError, SpiceSimulationError
from fidp.evaluators.spice.spice import (
    AcAnalysisSpec,
    NgSpiceRunner,
    XyceRunner,
    _build_spice_node_map,
    _map_node,
    _mapped_measure_nodes,
    export_spice_netlist_ir,
)
from fidp.evaluators.types import EvalError, EvalRequest, EvalResult, FrequencyGridSpec


@dataclass(frozen=True)
class SpiceOptions:
    """Options for SPICE execution."""

    workdir: Optional[Path] = None
    keep_workdir: bool = False


class SpiceEvaluator:
    """Evaluator using external SPICE simulators."""

    def __init__(self, options: Optional[SpiceOptions] = None) -> None:
        self.options = options or SpiceOptions()

    def evaluate(self, circuit: CircuitIR, request: EvalRequest) -> EvalResult:
        freqs = request.grid.make_grid()
        try:
            circuit.validate()
        except CircuitIRValidationError as exc:
            return _error_result(freqs, "circuit_invalid", str(exc), {})
        try:
            ports = _resolve_ports(circuit, request.ports)
        except CircuitIRValidationError as exc:
            return _error_result(freqs, "port_resolution_failed", str(exc), {"stage": "ports"})
        start = time.perf_counter()

        analysis_spec = _analysis_spec_from_grid(request.grid)
        measure_nodes = _measure_nodes(ports)
        try:
            flat = flatten_circuit(circuit)
        except CircuitIRValidationError as exc:
            return _error_result(freqs, "netlist_invalid", str(exc), {})
        if flat.subcircuits:
            return _error_result(
                freqs,
                "netlist_invalid",
                "Circuit requires full flattening for SPICE export.",
                {"remaining_subcircuits": len(flat.subcircuits)},
            )
        node_map = _build_spice_node_map(flat)
        mapped_measure_nodes = _mapped_measure_nodes(measure_nodes, node_map)

        runner = _select_runner(request.spice_simulator)
        output_csv = "spice_output.csv"

        with _workdir(self.options.workdir, self.options.keep_workdir) as workdir:
            Z = np.zeros((freqs.size, len(ports), len(ports)), dtype=complex)
            for idx, port in enumerate(ports):
                try:
                    netlist = export_spice_netlist_ir(
                        circuit,
                        port,
                        analysis_spec,
                        output_csv=output_csv,
                        simulator=runner.name,
                        value_mode=request.value_mode,
                        measure_nodes=measure_nodes,
                        node_map=node_map,
                    )
                except CircuitIRValidationError as exc:
                    return _error_result(freqs, "netlist_invalid", str(exc), {})
                try:
                    freq_out, node_voltages = runner.run_nodes(
                        netlist,
                        mapped_measure_nodes,
                        workdir / f"port_{idx}",
                        output_csv=output_csv,
                        timeout_s=request.timeout_s,
                    )
                except SpiceNotAvailableError as exc:
                    return _error_result(freqs, "spice_not_available", str(exc), {})
                except SpiceSimulationError as exc:
                    return _error_result(
                        freqs,
                        _classify_spice_error(str(exc)),
                        str(exc),
                        {"port": port.name},
                    )
                except ValueError as exc:
                    return _error_result(freqs, "spice_parse_failed", str(exc), {})

                if not np.allclose(freq_out, freqs, rtol=0.0, atol=1e-12):
                    return _error_result(
                        freqs,
                        "frequency_mismatch",
                        "SPICE output frequencies do not match request.",
                        {"requested": freqs.tolist(), "received": freq_out.tolist()},
                    )

                for row, port_i in enumerate(ports):
                    pos_node = _map_node(port_i.pos, node_map)
                    neg_node = _map_node(port_i.neg, node_map)
                    v_pos = _node_voltage(node_voltages, pos_node, freqs)
                    v_neg = _node_voltage(node_voltages, neg_node, freqs)
                    Z[:, row, idx] = v_pos - v_neg

        if Z.shape[1] == 1:
            Z_out = Z[:, 0, 0]
        else:
            Z_out = Z

        meta = {
            "simulator": runner.name,
            "n_ports": len(ports),
            "analysis": {
                "sweep_type": analysis_spec.sweep_type,
                "points": analysis_spec.points,
                "f_start_hz": analysis_spec.f_start_hz,
                "f_stop_hz": analysis_spec.f_stop_hz,
            },
        }
        return EvalResult(
            freqs_hz=freqs,
            Z=Z_out,
            status="ok",
            meta=meta,
            timing_s={"total": time.perf_counter() - start},
        )


def _analysis_spec_from_grid(grid: FrequencyGridSpec) -> AcAnalysisSpec:
    sweep_type = "dec" if grid.spacing == "log" else "lin"
    return AcAnalysisSpec(
        sweep_type=sweep_type,
        points=int(grid.points),
        f_start_hz=float(grid.f_start_hz),
        f_stop_hz=float(grid.f_stop_hz),
    )


def _resolve_ports(circuit: CircuitIR, ports: Optional[list[str]]) -> list[PortDef]:
    by_name = {port.name: port for port in circuit.ports}
    if ports is None:
        return list(circuit.ports)
    resolved = []
    for name in ports:
        if name not in by_name:
            raise CircuitIRValidationError(f"Unknown port: {name}")
        resolved.append(by_name[name])
    return resolved


def _measure_nodes(ports: list[PortDef]) -> list[str]:
    nodes: list[str] = []
    for port in ports:
        nodes.extend([port.pos, port.neg])
    return nodes


def _select_runner(simulator: str):
    if simulator == "xyce":
        return XyceRunner()
    return NgSpiceRunner()


def _classify_spice_error(message: str) -> str:
    lower = message.lower()
    if "conver" in lower:
        return "spice_nonconvergence"
    return "spice_failed"


def _node_voltage(
    node_voltages: dict[str, np.ndarray],
    node: str,
    freqs: np.ndarray,
) -> np.ndarray:
    if node == "0":
        return np.zeros_like(freqs, dtype=complex)
    return node_voltages[node]


class _workdir:
    def __init__(self, root: Optional[Path], keep: bool) -> None:
        self.root = root
        self.keep = keep
        self._temp: Optional[tempfile.TemporaryDirectory[str]] = None
        self.path: Optional[Path] = None

    def __enter__(self) -> Path:
        if self.root is not None:
            self.path = self.root
            self.path.mkdir(parents=True, exist_ok=True)
            return self.path
        self._temp = tempfile.TemporaryDirectory()
        self.path = Path(self._temp.name)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._temp and not self.keep:
            self._temp.cleanup()


def _error_result(freqs: np.ndarray, code: str, message: str, details: dict[str, object]) -> EvalResult:
    Z = np.full_like(freqs, np.nan, dtype=complex)
    return EvalResult(
        freqs_hz=freqs,
        Z=Z,
        status="error",
        errors=[EvalError(code=code, message=message, details=details)],
    )
