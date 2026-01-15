import numpy as np

from fidp.modeling import RationalModel
from fidp.modeling.macromodel_io import (
    format_pole_residue_text,
    load_rational_model_json,
    save_rational_model_json,
)


def test_rational_model_json_roundtrip(tmp_path):
    model = RationalModel(
        poles=np.array([-1.0 + 2.0j, -1.0 - 2.0j]),
        residues=np.array([0.5 - 0.2j, 0.5 + 0.2j]),
        d=0.1 + 0.05j,
        h=0.01 - 0.02j,
        kind="impedance",
    )

    path = tmp_path / "model.json"
    save_rational_model_json(model, str(path))
    loaded = load_rational_model_json(str(path))

    freqs = np.array([10.0, 100.0, 1000.0])
    assert np.allclose(model.eval_freq(freqs), loaded.eval_freq(freqs), rtol=1e-10, atol=1e-12)

    text = format_pole_residue_text(loaded)
    assert "Poles/Residues" in text
