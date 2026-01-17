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
    assert report.margin > 0.0


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
    expected_delta = -(report.margin) + report.tol

    assert updated.d.real == pytest.approx(model.d.real + expected_delta, rel=1e-8, abs=1e-10)
    assert updated_report.is_passive
    assert updated_report.n_violations == 0


def test_passivity_admittance_and_passivation():
    freqs = np.logspace(1, 4, 50)
    model = RationalModel(
        poles=np.array([-60.0]),
        residues=np.array([12.0]),
        d=0.5 + 0.0j,
        h=0.0 + 0.0j,
        kind="admittance",
    )

    report = check_oneport_passivity(freqs, model.eval_freq(freqs), "admittance")

    assert report.is_passive
    assert report.margin > 0.0

    non_passive = RationalModel(
        poles=np.array([-60.0]),
        residues=np.array([12.0]),
        d=-1.0 + 0.0j,
        h=0.0 + 0.0j,
        kind="admittance",
    )

    bad_report = check_oneport_passivity(freqs, non_passive.eval_freq(freqs), "admittance")
    assert not bad_report.is_passive

    updated, updated_report = passivate_oneport_min_offset(non_passive, freqs, tol=1e-9)
    expected_delta = -(bad_report.margin) + bad_report.tol

    assert updated.d.real == pytest.approx(non_passive.d.real + expected_delta, rel=1e-8, abs=1e-10)
    assert updated_report.is_passive
    assert updated_report.details["delta_offset"] == pytest.approx(expected_delta, rel=1e-8, abs=1e-10)
