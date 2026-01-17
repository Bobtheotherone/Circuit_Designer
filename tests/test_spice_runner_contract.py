import os
from pathlib import Path
import textwrap

import numpy as np

from fidp.circuits.ir import CircuitIR, Component, ParamValue, PortDef
from fidp.evaluators.spice import SpiceEvaluator
from fidp.evaluators.spice.evaluator import _select_runner
from fidp.evaluators.spice.spice import NgSpiceRunner, XyceRunner
from fidp.evaluators.types import EvalRequest, FrequencyGridSpec


def _write_fake_ngspice(tmp_path: Path) -> Path:
    script = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import csv
        import re
        import sys
        import math

        netlist_path = sys.argv[-1]
        text = open(netlist_path, "r", encoding="utf-8").read().splitlines()
        title = ""
        for line in text:
            if line.strip().startswith("*"):
                title = line.strip().lstrip("*").strip()
                break
        case = title.split()[0]

        ac_line = next(line for line in text if line.strip().lower().startswith(".ac"))
        tokens = ac_line.split()
        sweep = tokens[1].lower()
        points = int(tokens[2])
        f_start = float(tokens[3])
        f_stop = float(tokens[4])
        if sweep != "lin":
            raise SystemExit("Fake ngspice only supports lin sweep.")
        if points == 1:
            freqs = [f_start]
        else:
            step = (f_stop - f_start) / (points - 1)
            freqs = [f_start + step * idx for idx in range(points)]

        wr_line = next(line for line in text if "wrdata" in line.lower())
        wr_tokens = wr_line.split()
        output_csv = wr_tokens[1]
        nodes = [re.findall(r"v\\((.*?)\\)", tok, re.IGNORECASE)[0] for tok in wr_tokens[3:]]

        src_line = next(line for line in text if line.strip().upper().startswith("IIMP"))
        src_tokens = src_line.split()
        neg_node = src_tokens[1]
        pos_node = src_tokens[2]
        port_nodes = []
        for node in nodes:
            if node == neg_node:
                continue
            if node not in port_nodes:
                port_nodes.append(node)

        def series_rlc(s):
            return 10.0 + s * 1e-3 + 1.0 / (s * 5e-6)

        def parallel_rc(s):
            return 1.0 / (1.0 / 50.0 + s * 1e-6)

        def twoport_matrix():
            r1 = 100.0
            r2 = 150.0
            r12 = 50.0
            y11 = 1.0 / r1 + 1.0 / r12
            y22 = 1.0 / r2 + 1.0 / r12
            y12 = -1.0 / r12
            det = y11 * y22 - y12 * y12
            z11 = y22 / det
            z22 = y11 / det
            z12 = -y12 / det
            return [[z11, z12], [z12, z22]]

        z_twoport = twoport_matrix()
        with open(output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            header = ["frequency"]
            for node in nodes:
                header.append(f"v({node})_real")
                header.append(f"v({node})_imag")
            writer.writerow(header)
            for freq in freqs:
                s = 1j * 2.0 * math.pi * freq
                row = [f"{freq:.12g}"]
                for node in nodes:
                    v = 0.0 + 0.0j
                    if case == "case_rlc_series":
                        if node == pos_node:
                            v = series_rlc(s)
                    elif case == "case_rc_parallel":
                        if node == pos_node:
                            v = parallel_rc(s)
                    elif case == "case_twoport_res":
                        col = port_nodes.index(pos_node) if pos_node in port_nodes else 0
                        if node in port_nodes:
                            row_idx = port_nodes.index(node)
                            v = z_twoport[row_idx][col]
                    row.append(f"{v.real:.12g}")
                    row.append(f"{v.imag:.12g}")
                writer.writerow(row)
        """
    )
    path = tmp_path / "ngspice"
    path.write_text(script, encoding="utf-8")
    os.chmod(path, 0o755)
    return path


def _build_series_rlc() -> CircuitIR:
    components = [
        Component("R1", "R", "n1", "n2", ParamValue(10.0)),
        Component("L1", "L", "n2", "n3", ParamValue(1e-3)),
        Component("C1", "C", "n3", "gnd", ParamValue(5e-6)),
    ]
    ports = [PortDef("P", "n1", "gnd")]
    return CircuitIR(name="case_rlc_series", ports=ports, components=components)


def _build_parallel_rc() -> CircuitIR:
    components = [
        Component("R1", "R", "n1", "gnd", ParamValue(50.0)),
        Component("C1", "C", "n1", "gnd", ParamValue(1e-6)),
    ]
    ports = [PortDef("P", "n1", "gnd")]
    return CircuitIR(name="case_rc_parallel", ports=ports, components=components)


def _build_twoport_resistor() -> CircuitIR:
    components = [
        Component("R1", "R", "n1", "gnd", ParamValue(100.0)),
        Component("R2", "R", "n2", "gnd", ParamValue(150.0)),
        Component("R12", "R", "n1", "n2", ParamValue(50.0)),
    ]
    ports = [
        PortDef("P1", "n1", "gnd"),
        PortDef("P2", "n2", "gnd"),
    ]
    return CircuitIR(name="case_twoport_res", ports=ports, components=components)


def _expected_series_rlc(freqs: np.ndarray) -> np.ndarray:
    s = 1j * 2.0 * np.pi * freqs
    return 10.0 + s * 1e-3 + 1.0 / (s * 5e-6)


def _expected_parallel_rc(freqs: np.ndarray) -> np.ndarray:
    s = 1j * 2.0 * np.pi * freqs
    return 1.0 / (1.0 / 50.0 + s * 1e-6)


def _expected_twoport() -> np.ndarray:
    r1 = 100.0
    r2 = 150.0
    r12 = 50.0
    y11 = 1.0 / r1 + 1.0 / r12
    y22 = 1.0 / r2 + 1.0 / r12
    y12 = -1.0 / r12
    y = np.array([[y11, y12], [y12, y22]], dtype=float)
    return np.linalg.inv(y)


def test_spice_runner_contract_fake_ngspice(monkeypatch, tmp_path: Path) -> None:
    """Validate runner invocation + CSV parsing with a fake ngspice (not a truth baseline)."""
    fake = _write_fake_ngspice(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: str(fake))

    grid = FrequencyGridSpec(f_start_hz=10.0, f_stop_hz=1000.0, points=5, spacing="linear")
    request_spice = EvalRequest(grid=grid, fidelity="truth", spice_simulator="ngspice")

    spice = SpiceEvaluator()
    freqs = grid.make_grid()

    circuits = (
        (_build_series_rlc(), _expected_series_rlc(freqs)),
        (_build_parallel_rc(), _expected_parallel_rc(freqs)),
        (_build_twoport_resistor(), np.repeat(_expected_twoport()[None, :, :], freqs.size, axis=0)),
    )

    for circuit, expected in circuits:
        spice_result = spice.evaluate(circuit, request_spice)
        assert spice_result.status == "ok"
        assert np.allclose(spice_result.Z, expected, rtol=1e-6, atol=1e-8)


def test_select_runner_case_insensitive() -> None:
    assert isinstance(_select_runner("NGSPICE"), NgSpiceRunner)
    assert isinstance(_select_runner("XyCe"), XyceRunner)
