import numpy as np
import pytest

from fidp.analysis.fitting.vector_fitting import RationalModel
from fidp.analysis.fitting.passivity.tests import check_passivity
from fidp.analysis.fitting.passivity.enforce_residue_nnls import (
    enforce_passivity_nnls,
    NNLSPassivityConfig,
)
from fidp.analysis.fitting.passivity.enforce_qp import (
    enforce_passivity_qp,
    QPPassivityConfig,
)
from fidp.errors import PassivityViolationError


def _make_model() -> RationalModel:
    poles = np.array([-40.0 + 120.0j, -40.0 - 120.0j, -800.0], dtype=complex)
    residues = np.array([15.0 - 6.0j, 15.0 + 6.0j, 50.0], dtype=complex)
    return RationalModel(poles=poles, residues=residues, d=-0.02 + 0.0j, h=0.0 + 0.0j, kind="impedance")


def test_passivity_enforcement_nnls_and_qp():
    freq_hz = np.logspace(1, 5, 120)
    model = _make_model()

    report = check_passivity(freq_hz, model.eval_freq(freq_hz), tol=1e-9, refine=False)
    assert not report.is_passive

    nnls_model, nnls_report = enforce_passivity_nnls(
        model,
        freq_hz,
        NNLSPassivityConfig(tol=1e-9, max_rel_error_increase=0.2),
    )
    assert nnls_report.is_passive
    assert nnls_report.margin >= -nnls_report.tol
    assert nnls_report.details["rel_error_increase"] <= 0.2

    qp_model, qp_report = enforce_passivity_qp(
        model,
        freq_hz,
        QPPassivityConfig(tol=1e-9, max_rel_error_increase=0.2),
    )
    assert qp_report.is_passive
    assert qp_report.margin >= -qp_report.tol
    assert qp_report.details["rel_error_increase"] <= 0.2

    assert np.allclose(nnls_model.poles, qp_model.poles)


def test_passivity_enforcement_rejects_unstable_poles():
    freq_hz = np.logspace(1, 4, 60)
    model = RationalModel(
        poles=np.array([10.0 + 0.0j]),
        residues=np.array([1.0 + 0.0j]),
        d=0.0 + 0.0j,
        h=0.0 + 0.0j,
        kind="impedance",
    )

    with pytest.raises(PassivityViolationError):
        enforce_passivity_nnls(model, freq_hz)
