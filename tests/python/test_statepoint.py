"""Statepoint writing and reading (FR-OUT-1, FR-OUT-2)."""

from __future__ import annotations

import numpy as np
import pytest

import openndm


def test_statepoint_round_trip(tmp_path, iaea_model):
    result = iaea_model.solve()
    path = openndm.write_statepoint(
        tmp_path / "statepoint.h5",
        result,
        iaea_model,
        extra={"leakage_correction": "none"},
    )
    with openndm.StatePoint(path) as sp:
        assert sp.k_eff == pytest.approx(result.k_eff)
        assert sp.kernel == result.kernel
        assert sp.mode == "forward"
        assert sp.converged
        assert sp.version == openndm.__version__
        assert np.allclose(sp.flux, np.asarray(result.flux))
        assert np.allclose(sp.power, np.asarray(result.power))
        assert np.allclose(sp.radial_power, result.radial_power())
        assert np.allclose(sp.axial_power, result.axial_power())
        assert sp.f_q == pytest.approx(result.f_q)
        assert sp.lattice_shape == iaea_model.geometry.shape
        assert sp.history.shape[0] == result.outer_iterations


def test_statepoint_records_extra_metadata(tmp_path, iaea_model):
    """The leakage convention has to travel with the results.

    B1, P1 and a plain buckling-corrected D give measurably different group
    constants, so a cross-code comparison against a deck generated elsewhere
    is meaningless unless the file says which was used.
    """
    import h5py

    result = iaea_model.solve()
    path = openndm.write_statepoint(
        tmp_path / "sp.h5", result, iaea_model, extra={"leakage_correction": "b1"}
    )
    with h5py.File(path, "r") as f:
        assert f.attrs["leakage_correction"].decode() == "b1"


def test_reading_a_foreign_file_is_rejected(tmp_path):
    import h5py

    path = tmp_path / "not_ours.h5"
    with h5py.File(path, "w") as f:
        f.attrs["filetype"] = np.bytes_("statepoint")  # an OpenMC statepoint
    with pytest.raises(openndm.InputError, match="not an OpenNDM statepoint"):
        openndm.StatePoint(path)


def test_unknown_statepoint_version_is_rejected(tmp_path, iaea_model):
    import h5py

    result = iaea_model.solve()
    path = openndm.write_statepoint(tmp_path / "sp.h5", result, iaea_model)
    with h5py.File(path, "a") as f:
        f.attrs["format_version"] = 99
    with pytest.raises(openndm.InputError, match="format version 99"):
        openndm.StatePoint(path)
