import os
from pathlib import Path
import textwrap

import numpy as np

from fidp.circuits.ir import CircuitIR, Component, ParamValue, PortDef
from fidp.evaluators.mna import MnaEvaluator
from fidp.evaluators.spice import SpiceEvaluator
from fidp.evaluators.types import EvalRequest, FrequencyGridSpec


def _write_fake_ngspice(tmp_path: Path) -> Path:
    script = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        import csv
        import re
        import sys
        import numpy as np

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
        freqs = np.linspace(f_start, f_stop, points)

        wr_line = next(line for line in text if "wrdata" in line.lower())
        wr_tokens = wr_line.split()
        output_csv = wr_tokens[1]
        nodes = [re.findall(r"v\\((.*?)\\)", tok, re.IGNORECASE)[0] for tok in wr_tokens[3:]]

        src_line = next(line for line in text if line.strip().upper().startswith("IIMP"))
        src_tokens = src_line.split()
        neg_node = src_tokens[1]
        pos_node = src_tokens[2]

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
            y = np.array([[y11, y12], [y12, y22]], dtype=float)
            return np.linalg.inv(y)

        z_twoport = twoport_matrix()
        with open(output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            header = ["frequency"]
            for node in nodes:
                header.append(f"v({node})_real")
                header.append(f"v({node})_imag")
            writer.writerow(header)
            for freq in freqs:
                s = 1j * 2.0 * np.pi * freq
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
                        col = 0 if pos_node == "n1" else 1
                        if node == "n1":
                            v = z_twoport[0, col]
                        if node == "n2":
                            v = z_twoport[1, col]
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


def test_mna_vs_spice_small(monkeypatch, tmp_path: Path) -> None:
    fake = _write_fake_ngspice(tmp_path)
    monkeypatch.setattr("shutil.which", lambda _: str(fake))

    grid = FrequencyGridSpec(f_start_hz=10.0, f_stop_hz=1000.0, points=5, spacing="linear")
    request_mna = EvalRequest(grid=grid, fidelity="mid")
    request_spice = EvalRequest(grid=grid, fidelity="truth", spice_simulator="ngspice")

    mna = MnaEvaluator()
    spice = SpiceEvaluator()

    for circuit in (_build_series_rlc(), _build_parallel_rc(), _build_twoport_resistor()):
        mna_result = mna.evaluate(circuit, request_mna)
        spice_result = spice.evaluate(circuit, request_spice)
        assert mna_result.status == "ok"
        assert spice_result.status == "ok"
        assert np.allclose(mna_result.Z, spice_result.Z, rtol=1e-6, atol=1e-8)
