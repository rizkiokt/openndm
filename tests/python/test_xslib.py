"""Cross section library storage, validation and branch interpolation."""

from __future__ import annotations

import numpy as np
import pytest

import openndm

from conftest import iaea_library


def test_scatter_matrix_orientation_is_from_to():
    """scatter[from][to]: an error here silently reverses the spectrum."""
    lib = openndm.XSLibrary(2, 1)
    lib.set_composition(
        0,
        D=[1.5, 0.4],
        absorption=[0.01, 0.08],
        scatter=[[0.0, 0.02], [0.0, 0.0]],
    )
    lib.finalize()
    comp = lib.composition(0)
    # Removal = absorption + out-scatter, so only the fast group loses to
    # scattering when the matrix means "from row to column".
    assert comp.removal[0] == pytest.approx(0.03)
    assert comp.removal[1] == pytest.approx(0.08)


def test_transport_is_converted_to_a_diffusion_coefficient():
    lib = openndm.XSLibrary(1, 1)
    lib.set_composition(0, transport=[0.2], absorption=[0.01], scatter=[[0.0]])
    assert lib.composition(0).D[0] == pytest.approx(1.0 / 0.6)


def test_giving_both_d_and_transport_is_rejected():
    lib = openndm.XSLibrary(1, 1)
    with pytest.raises(openndm.InputError, match="either D or transport"):
        lib.set_composition(0, D=[1.0], transport=[0.2])


def test_wrong_length_vector_is_rejected():
    lib = openndm.XSLibrary(2, 1)
    with pytest.raises(openndm.InputError, match="must have 2 entries"):
        lib.set_composition(0, D=[1.0])


def test_non_positive_diffusion_coefficient_fails_validation():
    lib = openndm.XSLibrary(1, 1)
    lib.set_composition(0, D=[0.0], absorption=[0.1], scatter=[[0.0]])
    with pytest.raises(openndm.LibraryError, match="non-positive diffusion"):
        lib.finalize()


def test_fission_spectrum_must_sum_to_one():
    lib = openndm.XSLibrary(2, 1)
    lib.set_composition(
        0,
        D=[1.5, 0.4],
        absorption=[0.01, 0.08],
        nu_fission=[0.0, 0.135],
        chi=[0.5, 0.0],
        scatter=[[0.0, 0.02], [0.0, 0.0]],
    )
    with pytest.raises(openndm.LibraryError, match="spectrum sums to"):
        lib.finalize()


def test_negative_scattering_warns_but_does_not_fail():
    """Monte Carlo noise produces these routinely (FR-XS-8)."""
    lib = openndm.XSLibrary(2, 1)
    lib.set_composition(
        0,
        D=[1.5, 0.4],
        absorption=[0.01, 0.08],
        scatter=[[0.0, 0.02], [-1.0e-5, 0.0]],
    )
    with pytest.warns(UserWarning, match="negative scattering transfer"):
        messages = lib.finalize()
    assert any("negative scattering" in m for m in messages)
    assert lib.finalized


def test_discontinuity_factors_default_to_one_and_round_trip():
    lib = iaea_library()
    assert np.allclose(lib.adf(0), 1.0)
    values = np.arange(12, dtype=float).reshape(6, 2) / 10.0 + 0.9
    lib.set_adf(0, values)
    assert np.allclose(lib.adf(0), values)
    # Untouched compositions keep the default.
    assert np.allclose(lib.adf(1), 1.0)


def test_scalar_adf_broadcasts_to_every_face():
    lib = iaea_library()
    lib.set_adf(0, [1.05, 0.95])
    assert np.allclose(lib.adf(0), np.tile([1.05, 0.95], (6, 1)))


def test_branch_axes_must_be_increasing():
    lib = openndm.XSLibrary(1, 1)
    with pytest.raises(openndm.LibraryError, match="strictly increasing"):
        lib.set_axes([("boron", [1000.0, 500.0])])


def branch_library():
    """Two-axis branch table whose data is linear in both state variables."""
    lib = openndm.XSLibrary(1, 1)
    lib.set_axes([("fuel_temperature", [500.0, 1500.0]), ("boron", [0.0, 2000.0])])
    for state, (temperature, boron) in enumerate(
        [(500.0, 0.0), (500.0, 2000.0), (1500.0, 0.0), (1500.0, 2000.0)]
    ):
        absorption = 0.01 + 1.0e-6 * temperature + 1.0e-6 * boron
        lib.set_composition(
            0,
            state=state,
            D=[1.0],
            absorption=[absorption],
            nu_fission=[0.1],
            kappa_fission=[0.1],
            chi=[1.0],
            scatter=[[0.0]],
        )
    lib.finalize()
    return lib


def test_branch_grid_shape_and_state_count():
    lib = branch_library()
    assert lib.n_states == 4
    assert [a.name for a in lib.axes] == ["fuel_temperature", "boron"]


@pytest.mark.parametrize(
    ("temperature", "boron"),
    [(500.0, 0.0), (1500.0, 2000.0), (1000.0, 1000.0), (750.0, 1750.0)],
)
def test_multilinear_interpolation_is_exact_on_linear_data(temperature, boron):
    """Multilinear interpolation must reproduce linear data everywhere."""
    lib = branch_library()
    interpolated = lib.interpolate(fuel_temperature=temperature, boron=boron)
    expected = 0.01 + 1.0e-6 * temperature + 1.0e-6 * boron
    assert interpolated.composition(0).absorption[0] == pytest.approx(expected)
    assert interpolated.n_states == 1
    assert interpolated.finalized


def test_interpolation_clamps_outside_the_grid_by_default():
    lib = branch_library()
    clamped = lib.interpolate(fuel_temperature=5000.0, boron=0.0)
    edge = lib.interpolate(fuel_temperature=1500.0, boron=0.0)
    assert clamped.composition(0).absorption[0] == pytest.approx(
        edge.composition(0).absorption[0]
    )


def test_interpolation_can_be_made_to_raise_outside_the_grid():
    lib = branch_library()
    lib.extrapolation = "error"
    with pytest.raises(openndm.InputError, match="outside axis"):
        lib.interpolate(fuel_temperature=5000.0, boron=0.0)


def test_interpolation_rejects_a_mismatched_state():
    lib = branch_library()
    with pytest.raises(openndm.InputError, match="branch state mismatch"):
        lib.interpolate(fuel_temperature=800.0)


def test_a_branch_library_cannot_be_solved_directly():
    lib = branch_library()
    geometry = openndm.Geometry.from_lattice(np.zeros((1, 1, 1), int), pitch=10.0)
    with pytest.raises(openndm.InputError, match="collapsed"):
        openndm.Model(geometry, lib)


def test_uncertainties_are_carried(tmp_path):
    lib = openndm.XSLibrary(2, 1)
    lib.set_composition(
        0,
        D=[1.5, 0.4],
        absorption=[0.01, 0.08],
        scatter=[[0.0, 0.02], [0.0, 0.0]],
        std={"absorption": [1.0e-4, 5.0e-4]},
    )
    lib.finalize()
    assert lib.has_uncertainty
    assert lib.composition(0).absorption_std == pytest.approx([1.0e-4, 5.0e-4])


def test_hdf5_round_trip_preserves_everything(tmp_path):
    lib = iaea_library()
    lib.set_adf(0, np.linspace(0.9, 1.1, 12).reshape(6, 2))
    lib.set_delayed([0.0002, 0.001, 0.0012], [0.0124, 0.0305, 0.111])
    path = tmp_path / "xslib.h5"
    lib.to_hdf5(path)
    restored = openndm.XSLibrary.from_hdf5(path)

    assert restored.n_groups == lib.n_groups
    assert restored.n_compositions == lib.n_compositions
    for c in range(lib.n_compositions):
        for field in ("D", "absorption", "nu_fission", "chi", "scatter"):
            assert np.allclose(
                getattr(restored.composition(c), field),
                getattr(lib.composition(c), field),
            )
        assert np.allclose(restored.adf(c), lib.adf(c))


def test_hdf5_branch_round_trip(tmp_path):
    lib = branch_library()
    path = tmp_path / "branch.h5"
    lib.to_hdf5(path)
    restored = openndm.XSLibrary.from_hdf5(path)
    assert restored.n_states == 4
    assert [a.name for a in restored.axes] == [a.name for a in lib.axes]
    a = restored.interpolate(fuel_temperature=1000.0, boron=1000.0)
    assert a.composition(0).absorption[0] == pytest.approx(0.012)


def test_hdf5_rejects_an_unknown_format_version(tmp_path):
    import h5py

    lib = iaea_library()
    path = tmp_path / "xslib.h5"
    lib.to_hdf5(path)
    with h5py.File(path, "a") as f:
        f.attrs["format_version"] = 999
    with pytest.raises(openndm.InputError, match="format version 999"):
        openndm.XSLibrary.from_hdf5(path)


def test_fissile_composition_without_kappa_fission_warns():
    """Otherwise the power distribution comes back silently zero."""
    lib = openndm.XSLibrary(2, 1)
    lib.set_composition(
        0,
        D=[1.5, 0.4],
        absorption=[0.01, 0.085],
        nu_fission=[0.0, 0.135],
        chi=[1.0, 0.0],
        scatter=[[0.0, 0.02], [0.0, 0.0]],
    )
    with pytest.warns(UserWarning, match="no kappa-fission"):
        lib.finalize()


def test_a_complete_composition_finalizes_without_warnings():
    lib = openndm.XSLibrary(2, 1)
    lib.set_composition(
        0,
        D=[1.5, 0.4],
        absorption=[0.01, 0.085],
        nu_fission=[0.0, 0.135],
        kappa_fission=[0.0, 0.135],
        chi=[1.0, 0.0],
        scatter=[[0.0, 0.02], [0.0, 0.0]],
    )
    assert lib.finalize() == []


def branch_over_temperature(n_compositions=3, points=(500.0, 1500.0)):
    """One axis, absorption rising linearly along it, same for every mixture."""
    lib = openndm.XSLibrary(2, n_compositions)
    lib.set_axes([("fuel_temperature", list(points))])
    for state, factor in enumerate(1.0 + 0.1 * np.arange(len(points))):
        for index in range(n_compositions):
            lib.set_composition(
                index,
                state=state,
                D=[1.5, 0.4],
                absorption=[0.010 * factor, 0.080 * factor * (1 + index)],
                nu_fission=[0.0, 0.135],
                kappa_fission=[0.0, 0.135],
                chi=[1.0, 0.0],
                scatter=[[0.0, 0.020], [0.0, 0.0]],
            )
    lib.finalize(warn=False)
    return lib


def test_a_uniform_state_matches_the_whole_library_interpolation():
    branch = branch_over_temperature()
    together = branch.interpolate(fuel_temperature=900.0)
    apart = branch.interpolate_by_composition({"fuel_temperature": 900.0})

    for index in range(branch.n_compositions):
        assert apart.composition(index).absorption == pytest.approx(
            together.composition(index).absorption
        )


def test_each_composition_lands_on_its_own_state():
    branch = branch_over_temperature()
    states = [500.0, 1000.0, 1500.0]
    collapsed = branch.interpolate_by_composition({"fuel_temperature": states})

    for index, temperature in enumerate(states):
        expected = branch.interpolate(fuel_temperature=temperature)
        assert collapsed.composition(index).absorption == pytest.approx(
            expected.composition(index).absorption
        )


def test_a_collapse_lands_on_the_grid_points_it_is_given():
    branch = branch_over_temperature()
    collapsed = branch.interpolate_by_composition(
        {"fuel_temperature": [500.0, 500.0, 1500.0]}
    )
    assert collapsed.composition(0).absorption[0] == pytest.approx(0.010)
    assert collapsed.composition(2).absorption[0] == pytest.approx(0.011)


def test_the_collapsed_library_is_single_state_and_finalized():
    collapsed = branch_over_temperature().interpolate_by_composition(
        {"fuel_temperature": [500.0, 1000.0, 1500.0]}
    )
    assert collapsed.n_states == 1
    assert collapsed.finalized
    assert collapsed.axes == []


def test_a_collapse_writes_into_an_existing_library_when_asked():
    branch = branch_over_temperature()
    target = branch.interpolate(fuel_temperature=500.0)
    before = id(target._lib)

    returned = branch.interpolate_by_composition(
        {"fuel_temperature": [1500.0, 1500.0, 1500.0]}, out=target
    )

    assert returned is target
    assert id(target._lib) == before
    assert target.composition(0).absorption[0] == pytest.approx(0.011)


def test_a_collapse_carries_the_discontinuity_factors_across():
    branch = branch_over_temperature()
    factors = np.arange(12, dtype=float).reshape(6, 2)
    branch.set_adf(1, factors)

    collapsed = branch.interpolate_by_composition({"fuel_temperature": 900.0})
    assert collapsed.adf(1) == pytest.approx(factors)


def test_a_collapse_carries_the_delayed_data_across():
    branch = branch_over_temperature()
    branch.set_delayed([0.0002, 0.001], [0.0124, 0.0305])

    collapsed = branch.interpolate_by_composition({"fuel_temperature": 900.0})
    assert collapsed._lib.delayed.beta == pytest.approx([0.0002, 0.001])
    assert collapsed._lib.delayed.lambda_ == pytest.approx([0.0124, 0.0305])


def test_repeated_states_cost_one_interpolation_each():
    branch = branch_over_temperature(n_compositions=6)
    collapsed = branch.interpolate_by_composition(
        {"fuel_temperature": [500.0] * 3 + [1500.0] * 3}
    )
    for index in range(3):
        assert collapsed.composition(index).absorption[0] == pytest.approx(0.010)
    for index in range(3, 6):
        assert collapsed.composition(index).absorption[0] == pytest.approx(0.011)


def test_collapsing_a_library_with_no_axes_is_an_error():
    lib = openndm.XSLibrary(1, 1)
    lib.set_composition(
        0, D=[1.0], absorption=[0.08], nu_fission=[0.1],
        kappa_fission=[0.1], chi=[1.0], scatter=[[0.0]],
    )
    lib.finalize()
    with pytest.raises(openndm.InputError, match="no branch axes"):
        lib.interpolate_by_composition({"fuel_temperature": 900.0})


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"boron": 900.0},
        {"fuel_temperature": 900.0, "boron": 500.0},
        {"fuel_temperature": [900.0, 900.0]},
    ],
)
def test_a_state_that_does_not_match_the_axes_is_an_error(state):
    with pytest.raises(openndm.InputError):
        branch_over_temperature().interpolate_by_composition(state)


def test_an_out_library_of_the_wrong_shape_is_an_error():
    branch = branch_over_temperature()
    with pytest.raises(openndm.InputError, match="compositions"):
        branch.interpolate_by_composition(
            {"fuel_temperature": 900.0}, out=openndm.XSLibrary(2, 5)
        )
    with pytest.raises(openndm.InputError, match="not branch-parameterised"):
        branch.interpolate_by_composition(
            {"fuel_temperature": 900.0}, out=branch_over_temperature()
        )
