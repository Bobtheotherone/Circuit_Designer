"""SPICE netlist export and runner scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np

from fidp.circuits import CircuitGraph, Port, Resistor, Capacitor, Inductor
from fidp.data import ImpedanceSweep
from fidp.errors import SpiceNotAvailableError


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

    The netlist writes a CSV with frequency and complex node voltages so that
    Z(s) = V(pos) - V(neg) for a +1A injection.
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

    lines.append(f"IIMP {port.pos} {port.neg} AC 1")
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
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        header_lower = [col.strip().lower() for col in header]

        freq_idx = _find_column(header_lower, ["frequency", "freq"])
        vpos_real_idx, vpos_imag_idx = _find_complex_columns(header_lower, port.pos)
        vneg_real_idx, vneg_imag_idx = _find_complex_columns(header_lower, port.neg)

        freqs: List[float] = []
        vpos: List[complex] = []
        vneg: List[complex] = []

        for row in reader:
            if not row:
                continue
            freqs.append(float(row[freq_idx]))
            vpos.append(
                float(row[vpos_real_idx]) + 1j * float(row[vpos_imag_idx])
            )
            vneg.append(
                float(row[vneg_real_idx]) + 1j * float(row[vneg_imag_idx])
            )

    freqs_arr = np.asarray(freqs, dtype=float)
    vpos_arr = np.asarray(vpos, dtype=complex)
    vneg_arr = np.asarray(vneg, dtype=complex)
    Z = vpos_arr - vneg_arr
    return ImpedanceSweep(freqs_hz=freqs_arr, Z=Z, meta={"source": str(path)})


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


class SpiceRunner:
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

    def build_command(self, netlist_path: Path, output_csv: str) -> List[str]:
        raise NotImplementedError

    def run(
        self,
        netlist_text: str,
        port: Port,
        workdir: Path,
        output_csv: str = "spice_output.csv",
    ) -> ImpedanceSweep:
        workdir.mkdir(parents=True, exist_ok=True)
        netlist_path = workdir / "circuit.cir"
        netlist_path.write_text(netlist_text, encoding="utf-8")

        exe = self.resolve_executable()
        cmd = self.build_command(netlist_path, output_csv)
        result = subprocess.run([exe, *cmd], cwd=workdir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{self.name} failed: {result.stderr}")

        output_path = workdir / output_csv
        return parse_spice_csv(output_path, port)


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
