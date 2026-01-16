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
from fidp.circuits.ir_export import export_spice_netlist as export_ir_netlist
from fidp.data import ImpedanceSweep
from fidp.errors import SpiceNotAvailableError, SpiceSimulationError


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
    element_index = 1

    for comp in circuit.iter_components():
        if isinstance(comp, Resistor):
            lines.append(
                f"R{element_index} {comp.node_a} {comp.node_b} {comp.resistance_ohms}"
            )
        elif isinstance(comp, Capacitor):
            lines.append(
                f"C{element_index} {comp.node_a} {comp.node_b} {comp.capacitance_f}"
            )
        elif isinstance(comp, Inductor):
            lines.append(
                f"L{element_index} {comp.node_a} {comp.node_b} {comp.inductance_h}"
            )
        element_index += 1

    lines.append(f"IIMP {port.neg} {port.pos} AC 1")
    lines.append(
        f".ac {analysis_spec.sweep_type} {analysis_spec.points}"
        f" {analysis_spec.f_start_hz} {analysis_spec.f_stop_hz}"
    )

    if simulator.lower() == "ngspice":
        lines.extend(
            [
                ".control",
                "set filetype=csv",
                "set noaskquit",
                "run",
                f"wrdata {output_csv} frequency v({port.pos}) v({port.neg})",
                "quit",
                ".endc",
            ]
        )
    else:
        lines.append(
            f".print ac format=csv file={output_csv} v({port.pos}) v({port.neg})"
        )

    lines.append(".end")
    return "\n".join(lines) + "\n"


def parse_spice_csv(path: Path, port: Port) -> ImpedanceSweep:
    """Parse a CSV output containing frequency and complex node voltages."""
    freqs_arr, node_voltages = parse_spice_csv_nodes(path, [port.pos, port.neg])
    vpos_arr = node_voltages[port.pos]
    vneg_arr = node_voltages[port.neg]
    Z = vpos_arr - vneg_arr
    return ImpedanceSweep(freqs_hz=freqs_arr, Z=Z, meta={"source": str(path)})


def parse_spice_csv_nodes(path: Path, nodes: Sequence[str]) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Parse CSV output with voltages for multiple nodes."""
    node_set = {node.lower(): node for node in nodes}
    voltages: Dict[str, List[complex]] = {node: [] for node in nodes}
    freqs: List[float] = []

    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
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
) -> str:
    """Export a CircuitIR impedance netlist with AC analysis."""
    base = export_ir_netlist(circuit, title=circuit.name, canonicalize=True, value_mode=value_mode)
    lines = [line for line in base.splitlines() if line.strip() and line.strip().lower() != ".end"]
    lines.append(f"IIMP {port.neg} {port.pos} AC 1")
    lines.append(
        f".ac {analysis_spec.sweep_type} {analysis_spec.points}"
        f" {analysis_spec.f_start_hz} {analysis_spec.f_stop_hz}"
    )
    nodes = list(measure_nodes) if measure_nodes is not None else [port.pos, port.neg]
    if simulator.lower() == "ngspice":
        lines.extend(
            [
                ".control",
                "set filetype=csv",
                "set noaskquit",
                "run",
                f"wrdata {output_csv} frequency " + " ".join(f"v({node})" for node in nodes),
                "quit",
                ".endc",
            ]
        )
    else:
        lines.append(
            f".print ac format=csv file={output_csv} " + " ".join(f"v({node})" for node in nodes)
        )
    lines.append(".end")
    return "\n".join(lines) + "\n"


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
