"""SPICE netlist export and runner utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence
import csv
import shutil
import subprocess
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np

from fidp.circuits import CircuitGraph, CircuitIR, Port, PortDef, Resistor, Capacitor, Inductor
from fidp.circuits.ir_export import _sanitize_node, _sorted_components
from fidp.circuits.ops import flatten_circuit
from fidp.data import ImpedanceSweep
from fidp.errors import CircuitIRValidationError, SpiceNotAvailableError, SpiceSimulationError

_SPICE_GROUND_NAMES = {"0", "gnd", "ground"}


@dataclass(frozen=True)
class AcAnalysisSpec:
    """AC sweep specification for SPICE exports."""

    sweep_type: str
    points: int
    f_start_hz: float
    f_stop_hz: float


def export_spice_netlist(
    circuit: CircuitGraph,
    port: Port,
    analysis_spec: AcAnalysisSpec,
    output_csv: str = "spice_output.csv",
    simulator: str = "ngspice",
) -> str:
    """
    Export a SPICE netlist with a 1A AC current source between port nodes.

    The current source is oriented from port.neg -> port.pos so that +1A enters
    the network at port.pos and leaves at port.neg. The netlist writes a CSV
    with frequency and complex node voltages so that Z(s) = V(pos) - V(neg).
    """
    lines: List[str] = ["* FIDP impedance export"]
    node_map = _build_spice_node_map_graph(circuit, port)
    element_index = 1

    for comp in circuit.iter_components():
        node_a = _map_node(comp.node_a, node_map)
        node_b = _map_node(comp.node_b, node_map)
        if isinstance(comp, Resistor):
            lines.append(f"R{element_index} {node_a} {node_b} {comp.resistance_ohms}")
        elif isinstance(comp, Capacitor):
            lines.append(f"C{element_index} {node_a} {node_b} {comp.capacitance_f}")
        elif isinstance(comp, Inductor):
            lines.append(f"L{element_index} {node_a} {node_b} {comp.inductance_h}")
        element_index += 1

    port_pos = _map_node(port.pos, node_map)
    port_neg = _map_node(port.neg, node_map)
    lines.append(f"IIMP {port_neg} {port_pos} AC 1")
    lines.append(
        f".ac {analysis_spec.sweep_type} {analysis_spec.points}"
        f" {analysis_spec.f_start_hz} {analysis_spec.f_stop_hz}"
    )

    measure_nodes = _mapped_measure_nodes([port.pos, port.neg], node_map)
    if simulator.lower() == "ngspice":
        lines.extend(
            [
                ".control",
                "set filetype=csv",
                "set noaskquit",
                "run",
                f"wrdata {output_csv} frequency " + " ".join(f"v({node})" for node in measure_nodes),
                "quit",
                ".endc",
            ]
        )
    else:
        lines.append(
            f".print ac format=csv file={output_csv} " + " ".join(f"v({node})" for node in measure_nodes)
        )

    lines.append(".end")
    return "\n".join(lines) + "\n"


def parse_spice_csv(path: Path, port: Port) -> ImpedanceSweep:
    """Parse a CSV output containing frequency and complex node voltages."""
    pos_node = _normalize_spice_node(port.pos)
    neg_node = _normalize_spice_node(port.neg)

    measure_nodes: List[str] = []
    for node in (pos_node, neg_node):
        if node == "0" or node in measure_nodes:
            continue
        measure_nodes.append(node)

    missing_node: str | None = None
    try:
        freqs_arr, node_voltages = parse_spice_csv_nodes(path, measure_nodes)
    except ValueError:
        if len(measure_nodes) == 2:
            try:
                freqs_arr, node_voltages = parse_spice_csv_nodes(path, [pos_node])
                missing_node = neg_node
            except ValueError:
                freqs_arr, node_voltages = parse_spice_csv_nodes(path, [neg_node])
                missing_node = pos_node
        else:
            raise

    if missing_node == pos_node:
        pos_node = "0"
    if missing_node == neg_node:
        neg_node = "0"

    vpos_arr = _node_voltage(node_voltages, pos_node, freqs_arr)
    vneg_arr = _node_voltage(node_voltages, neg_node, freqs_arr)
    Z = vpos_arr - vneg_arr
    return ImpedanceSweep(freqs_hz=freqs_arr, Z=Z, meta={"source": str(path)})


def parse_spice_csv_nodes(path: Path, nodes: Sequence[str]) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Parse CSV output with voltages for multiple nodes."""
    text = path.read_text(encoding="utf-8").splitlines()
    data_lines = [
        line for line in text if line.strip() and not line.lstrip().startswith(("*", "#"))
    ]
    if not data_lines:
        raise ValueError("SPICE output file is empty.")
    if "," not in data_lines[0]:
        return _parse_ngspice_wrdata(text, nodes)

    node_set = {node.lower(): node for node in nodes}
    voltages: Dict[str, List[complex]] = {node: [] for node in nodes}
    freqs: List[float] = []

    reader = csv.reader(data_lines)
    header = next(reader)
    header_lower = [col.strip().lower() for col in header]
    freq_idx = _find_column(header_lower, ["frequency", "freq"])
    column_map = _resolve_node_columns(header_lower, node_set)

    for row in reader:
        if not row:
            continue
        freqs.append(float(row[freq_idx]))
        for node, (real_idx, imag_idx) in column_map.items():
            if imag_idx is None:
                voltages[node].append(_parse_complex_value(row[real_idx]))
            else:
                voltages[node].append(
                    float(row[real_idx]) + 1j * float(row[imag_idx])
                )

    freqs_arr = np.asarray(freqs, dtype=float)
    node_arrays = {node: np.asarray(values, dtype=complex) for node, values in voltages.items()}
    return freqs_arr, node_arrays


def _parse_ngspice_wrdata(
    lines: Sequence[str],
    nodes: Sequence[str],
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    freqs: List[float] = []
    voltages: Dict[str, List[complex]] = {node: [] for node in nodes}
    vector_count = 1 + len(nodes)
    expected_triple = vector_count * 3
    expected_compact = 1 + 2 * len(nodes)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("*") or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        token_count = len(tokens)
        if token_count == expected_triple:
            freqs.append(float(tokens[1]))
            for idx, node in enumerate(nodes, start=1):
                base = idx * 3
                real = float(tokens[base + 1])
                imag = float(tokens[base + 2])
                voltages[node].append(real + 1j * imag)
            continue
        if token_count == expected_compact:
            freqs.append(float(tokens[0]))
            for idx, node in enumerate(nodes):
                real = float(tokens[1 + 2 * idx])
                imag = float(tokens[1 + 2 * idx + 1])
                voltages[node].append(real + 1j * imag)
            continue
        raise ValueError(
            "Unexpected ngspice wrdata format: "
            f"token_count={token_count}, expected {expected_triple} "
            f"(3*(1+{len(nodes)})) or {expected_compact} "
            f"(1+2*{len(nodes)})."
        )

    freqs_arr = np.asarray(freqs, dtype=float)
    node_arrays = {node: np.asarray(values, dtype=complex) for node, values in voltages.items()}
    return freqs_arr, node_arrays


def _find_column(header: List[str], candidates: List[str]) -> int:
    for candidate in candidates:
        if candidate in header:
            return header.index(candidate)
    raise ValueError("Required column not found in SPICE CSV header.")


def _find_complex_columns(header: List[str], node: str) -> tuple[int, int]:
    node_lower = node.lower()
    patterns = [
        (f"v({node_lower})_real", f"v({node_lower})_imag"),
        (f"v({node_lower})#real", f"v({node_lower})#imag"),
        (f"v({node_lower})", f"v({node_lower})#imag"),
    ]
    for real_name, imag_name in patterns:
        if real_name in header and imag_name in header:
            return header.index(real_name), header.index(imag_name)
    raise ValueError("Complex voltage columns not found in SPICE CSV header.")


def _resolve_node_columns(header: List[str], node_set: Dict[str, str]) -> Dict[str, tuple[int, int | None]]:
    column_map: Dict[str, tuple[int, int | None]] = {}
    for node_lower, node in node_set.items():
        try:
            real_idx, imag_idx = _find_complex_columns(header, node_lower)
            column_map[node] = (real_idx, imag_idx)
            continue
        except ValueError:
            pass
        for idx, name in enumerate(header):
            if name.startswith(f"v({node_lower})"):
                column_map[node] = (idx, None)
                break
        if node not in column_map:
            raise ValueError(f"Voltage columns not found for node {node}.")
    return column_map


def _parse_complex_value(token: str) -> complex:
    value = token.strip()
    if "," in value:
        real_str, imag_str = value.split(",", maxsplit=1)
        return complex(float(real_str), float(imag_str))
    if value.endswith("j"):
        return complex(value)
    return complex(float(value), 0.0)


def export_spice_netlist_ir(
    circuit: CircuitIR,
    port: PortDef,
    analysis_spec: AcAnalysisSpec,
    output_csv: str = "spice_output.csv",
    simulator: str = "ngspice",
    value_mode: str = "snapped",
    measure_nodes: Sequence[str] | None = None,
    node_map: Dict[str, str] | None = None,
) -> str:
    """Export a CircuitIR impedance netlist with AC analysis."""
    circuit.validate()
    flat = flatten_circuit(circuit)
    if flat.subcircuits:
        raise CircuitIRValidationError("Circuit requires full flattening for SPICE export.")
    if node_map is None:
        node_map = _build_spice_node_map(flat)
    components = _sorted_components(flat.components, node_map, value_mode)
    lines = [f"* {circuit.name}"]
    type_counts: Dict[str, int] = {"R": 0, "C": 0, "L": 0}
    for kind, node_a, node_b, value in components:
        type_counts[kind] += 1
        name = f"{kind}{type_counts[kind]}"
        lines.append(f"{name} {node_a} {node_b} {value}")
    port_pos = _map_node(port.pos, node_map)
    port_neg = _map_node(port.neg, node_map)
    lines.append(f"IIMP {port_neg} {port_pos} AC 1")
    lines.append(
        f".ac {analysis_spec.sweep_type} {analysis_spec.points}"
        f" {analysis_spec.f_start_hz} {analysis_spec.f_stop_hz}"
    )
    nodes = list(measure_nodes) if measure_nodes is not None else [port.pos, port.neg]
    mapped_nodes = _mapped_measure_nodes(nodes, node_map)
    if simulator.lower() == "ngspice":
        lines.extend(
            [
                ".control",
                "set filetype=csv",
                "set noaskquit",
                "run",
                f"wrdata {output_csv} frequency " + " ".join(f"v({node})" for node in mapped_nodes),
                "quit",
                ".endc",
            ]
        )
    else:
        lines.append(
            f".print ac format=csv file={output_csv} " + " ".join(f"v({node})" for node in mapped_nodes)
        )
    lines.append(".end")
    return "\n".join(lines) + "\n"


def _map_node(node: str, node_map: Dict[str, str] | None) -> str:
    mapped = node_map.get(node, node) if node_map else node
    return _sanitize_node(mapped)


def _mapped_measure_nodes(nodes: Sequence[str], node_map: Dict[str, str]) -> List[str]:
    mapped: List[str] = []
    seen: set[str] = set()
    for node in (_map_node(node, node_map) for node in nodes):
        if node == "0":
            continue
        if node in seen:
            continue
        seen.add(node)
        mapped.append(node)
    return mapped


def _node_voltage(
    node_voltages: Dict[str, np.ndarray],
    node: str,
    freqs: np.ndarray,
) -> np.ndarray:
    if node == "0":
        return np.zeros_like(freqs, dtype=complex)
    return node_voltages[node]


def _normalize_spice_node(node: str) -> str:
    sanitized = _sanitize_node(node)
    return "0" if sanitized.lower() in _SPICE_GROUND_NAMES else sanitized


def _build_spice_node_map(circuit: CircuitIR) -> Dict[str, str]:
    nodes: set[str] = set()
    port_nodes: set[str] = set()
    for comp in circuit.components:
        nodes.update([comp.node_a, comp.node_b])
    for port in circuit.ports:
        nodes.update([port.pos, port.neg])
        port_nodes.update([port.pos, port.neg])
    preferred_neg = circuit.ports[0].neg if circuit.ports else None
    return _build_spice_node_map_from_nodes(nodes, preferred_neg, port_nodes)


def _build_spice_node_map_graph(circuit: CircuitGraph, port: Port) -> Dict[str, str]:
    nodes = set(circuit.nodes)
    nodes.update([port.pos, port.neg])
    return _build_spice_node_map_from_nodes(nodes, port.neg, {port.pos, port.neg})


def _build_spice_node_map_from_nodes(
    nodes: Sequence[str],
    preferred_neg: str | None,
    port_nodes: set[str],
) -> Dict[str, str]:
    if not nodes:
        raise CircuitIRValidationError("Circuit has no nodes for SPICE grounding.")

    nodes_sorted = sorted(set(nodes))
    sanitized_map = {node: _sanitize_node(node) for node in nodes_sorted}
    explicit_ground = {
        node for node, sanitized in sanitized_map.items() if sanitized.lower() in _SPICE_GROUND_NAMES
    }

    reference_node = None
    if not explicit_ground:
        reference_node = _choose_spice_reference_node(nodes_sorted, preferred_neg)

    node_map: Dict[str, str] = {}
    for node in explicit_ground:
        node_map[node] = "0"
    if reference_node is not None:
        node_map[reference_node] = "0"

    base_groups: Dict[str, List[str]] = {}
    for node in nodes_sorted:
        if node in node_map:
            continue
        base_groups.setdefault(sanitized_map[node], []).append(node)

    reserved_bases = set(base_groups.keys())
    used_names: set[str] = {"0"}
    for base in sorted(base_groups.keys()):
        group_nodes = sorted(
            base_groups[base],
            key=lambda node: (0 if node in port_nodes else 1, node),
        )
        for idx, node in enumerate(group_nodes):
            if idx == 0 and base not in used_names:
                candidate = base
            else:
                suffix = 1
                candidate = f"{base}__{suffix}"
                while candidate in used_names or candidate in reserved_bases:
                    suffix += 1
                    candidate = f"{base}__{suffix}"
            node_map[node] = candidate
            used_names.add(candidate)

    collisions: Dict[str, List[str]] = {}
    for node, mapped in node_map.items():
        if mapped == "0":
            continue
        collisions.setdefault(mapped, []).append(node)
    collision_details = {name: nodes for name, nodes in collisions.items() if len(nodes) > 1}
    if collision_details:
        details = "; ".join(
            f"{name} <- {sorted(nodes)}" for name, nodes in sorted(collision_details.items())
        )
        raise CircuitIRValidationError(f"SPICE node mapping collision detected: {details}")
    return node_map


def _choose_spice_reference_node(nodes: Sequence[str], preferred_neg: str | None) -> str:
    if preferred_neg and preferred_neg in nodes:
        return preferred_neg
    sanitized_nodes = sorted(
        ((_sanitize_node(node), node) for node in nodes),
        key=lambda item: (item[0], item[1]),
    )
    if not sanitized_nodes:
        raise CircuitIRValidationError("Circuit has no nodes for SPICE grounding.")
    return sanitized_nodes[0][1]


class SpiceRunner(ABC):
    """Base class for running SPICE simulations."""

    name: str = "spice"

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable

    def resolve_executable(self) -> str:
        executable = self.executable or self.name
        path = shutil.which(executable)
        if not path:
            raise SpiceNotAvailableError(f"{executable} not found in PATH.")
        return path

    @abstractmethod
    def build_command(self, netlist_path: Path, output_csv: str) -> List[str]:
        """Construct the SPICE command for the given netlist path."""
        ...

    def _run_netlist(
        self,
        netlist_text: str,
        workdir: Path,
        output_csv: str,
        timeout_s: float | None = None,
    ) -> tuple[Path, str, str]:
        workdir.mkdir(parents=True, exist_ok=True)
        netlist_path = workdir / "circuit.cir"
        netlist_path.write_text(netlist_text, encoding="utf-8")

        exe = self.resolve_executable()
        cmd = [exe, *self.build_command(netlist_path, output_csv)]
        try:
            result = subprocess.run(
                cmd,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise SpiceSimulationError(f"{self.name} timed out.") from exc

        if result.returncode != 0:
            raise SpiceSimulationError(
                f"{self.name} failed with code {result.returncode}: {result.stderr}"
            )

        output_path = workdir / output_csv
        if not output_path.exists():
            raise SpiceSimulationError(f"{self.name} did not produce output file {output_csv}.")
        return output_path, result.stdout, result.stderr

    def run(
        self,
        netlist_text: str,
        port: Port,
        workdir: Path,
        output_csv: str = "spice_output.csv",
        timeout_s: float | None = None,
    ) -> ImpedanceSweep:
        output_path, _, _ = self._run_netlist(
            netlist_text, workdir, output_csv, timeout_s=timeout_s
        )
        return parse_spice_csv(output_path, port)

    def run_nodes(
        self,
        netlist_text: str,
        nodes: Sequence[str],
        workdir: Path,
        output_csv: str = "spice_output.csv",
        timeout_s: float | None = None,
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        output_path, _, _ = self._run_netlist(
            netlist_text, workdir, output_csv, timeout_s=timeout_s
        )
        return parse_spice_csv_nodes(output_path, nodes)


class NgSpiceRunner(SpiceRunner):
    """Runner for ngspice in batch mode."""

    name = "ngspice"

    def build_command(self, netlist_path: Path, output_csv: str) -> List[str]:
        return ["-b", str(netlist_path)]


class XyceRunner(SpiceRunner):
    """Runner for Xyce."""

    name = "Xyce"

    def build_command(self, netlist_path: Path, output_csv: str) -> List[str]:
        return [str(netlist_path)]
