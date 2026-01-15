import numpy as np
import pytest

from fidp.modeling import RationalModel, check_oneport_passivity, passivate_oneport_min_offset


def test_passivity_check_passive_impedance():
    freqs = np.logspace(1, 3, 40)
    pole = -50.0
    residue = 25.0
    model = RationalModel(
        poles=np.array([pole]),
        residues=np.array([residue]),
        d=10.0 + 0.0j,
        h=0.0 + 0.0j,
        kind="impedance",
    )

    report = check_oneport_passivity(freqs, model.eval_freq(freqs), "impedance")

    assert report.is_passive
    assert report.min_real > 0.0


def test_passivation_adds_min_offset():
    freqs = np.logspace(1, 3, 60)
    model = RationalModel(
        poles=np.array([-100.0]),
        residues=np.array([5.0]),
        d=-2.0 + 0.0j,
        h=0.0 + 0.0j,
        kind="impedance",
    )

    report = check_oneport_passivity(freqs, model.eval_freq(freqs), "impedance")
    assert not report.is_passive

    updated, updated_report = passivate_oneport_min_offset(model, freqs, tol=1e-9)
    expected_delta = -(report.min_real) + report.tol

    assert updated.d.real == pytest.approx(model.d.real + expected_delta, rel=1e-8, abs=1e-10)
    assert updated_report.is_passive
    assert updated_report.n_violations == 0
