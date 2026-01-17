from pathlib import Path

import numpy as np
import pytest

from fidp.evaluators.spice.spice import parse_spice_csv_nodes


def test_parse_wrdata_triple_group(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "* ngspice wrdata output",
            "   0 1.0 0.0 0 2.0 3.0 0 4.0 5.0",
            "",
            "# comment line",
            "0 10.0 0.0 0 -2.0 -3.0 0 6.0 7.0",
        ]
    )
    path = tmp_path / "wrdata_triple.txt"
    path.write_text(content, encoding="utf-8")

    freqs, node_voltages = parse_spice_csv_nodes(path, ["n1", "n2"])

    assert np.allclose(freqs, np.array([1.0, 10.0]))
    assert np.allclose(node_voltages["n1"], np.array([2.0 + 3.0j, -2.0 - 3.0j]))
    assert np.allclose(node_voltages["n2"], np.array([4.0 + 5.0j, 6.0 + 7.0j]))


def test_parse_wrdata_compact_pairs(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "1.0 2.0 3.0 4.0 5.0",
            "# another comment",
            "10.0 -1.0 0.5 6.0 -7.0",
        ]
    )
    path = tmp_path / "wrdata_compact.txt"
    path.write_text(content, encoding="utf-8")

    freqs, node_voltages = parse_spice_csv_nodes(path, ["v1", "v2"])

    assert np.allclose(freqs, np.array([1.0, 10.0]))
    assert np.allclose(node_voltages["v1"], np.array([2.0 + 3.0j, -1.0 + 0.5j]))
    assert np.allclose(node_voltages["v2"], np.array([4.0 + 5.0j, 6.0 - 7.0j]))


def test_parse_wrdata_unknown_format_raises(tmp_path: Path) -> None:
    path = tmp_path / "wrdata_bad.txt"
    path.write_text("1.0 2.0 3.0 4.0\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        parse_spice_csv_nodes(path, ["n1"])

    assert "token_count=4" in str(exc.value)
