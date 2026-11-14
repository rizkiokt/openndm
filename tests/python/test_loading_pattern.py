"""Loading pattern mutation by position label (FR-OPT-7)."""

from __future__ import annotations

import numpy as np
import pytest

import openndm

from conftest import iaea_geometry, iaea_library


def test_the_default_grid_letters_rows_and_numbers_columns():
    labels = openndm.default_labels(3, 4)
    assert labels.shape == (3, 4)
    assert labels[0, 0] == "A1"
    assert labels[0, 3] == "A4"
    assert labels[2, 0] == "C1"


def test_the_default_grid_takes_a_second_letter_past_z():
    labels = openndm.default_labels(28, 1)
    assert labels[25, 0] == "Z1"
    assert labels[26, 0] == "AA1"
    assert labels[27, 0] == "AB1"


def test_a_geometry_labels_its_radial_lattice():
    geometry = iaea_geometry()
    assert geometry.labels.shape == geometry.shape[1:]
    assert geometry.labels[0, 0] == "A1"
    assert geometry.position("A1") == 0
    assert geometry.position("B1") == geometry.shape[2]
    assert geometry.position(5) == 5


def test_labels_cover_the_lattice_subdivide_produced():
    geometry = iaea_geometry(subdivide=2)
    assert geometry.labels.shape == geometry.shape[1:]
    assert geometry.position(geometry.labels[-1, -1]) == (
        geometry.shape[1] * geometry.shape[2] - 1
    )


def test_labels_can_be_overridden():
    geometry = openndm.Geometry.from_lattice(
        np.zeros((2, 2, 2), dtype=int),
        pitch=20.0,
        labels=[[" H8 ", "H9"], ["G8", "G9"]],
    )
    assert geometry.labels[0, 0] == "H8"
    assert geometry.position("H8") == 0
    assert geometry.position("G9") == 3


def test_the_label_grid_is_a_copy():
    geometry = iaea_geometry()
    grid = geometry.labels
    grid[0, 0] = "ZZ"
    assert geometry.labels[0, 0] == "A1"
    assert geometry.position("A1") == 0


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ([["A1", "A2"]], "shape"),
        ([["A1", "A1"], ["B1", "B2"]], "unique"),
        ([["A1", " "], ["B1", "B2"]], "blank"),
    ],
)
def test_a_bad_label_grid_is_refused(labels, message):
    with pytest.raises(openndm.InputError, match=message):
        openndm.Geometry.from_lattice(
            np.zeros((1, 2, 2), dtype=int), pitch=20.0, labels=labels
        )


def test_an_unknown_label_names_the_grid_it_is_not_in():
    geometry = iaea_geometry()
    with pytest.raises(openndm.InputError, match="unknown position label 'K8'"):
        geometry.position("K8")


def test_an_index_outside_the_radial_lattice_is_refused():
    geometry = iaea_geometry()
    with pytest.raises(openndm.InputError, match="outside the 9x9 lattice"):
        geometry.position(81)


def test_a_label_swap_matches_the_same_swap_by_index():
    by_index = openndm.Model(
        iaea_geometry(planes=2), iaea_library(), openndm.Settings(verbosity=0)
    )
    by_label = openndm.Model(
        iaea_geometry(planes=2), iaea_library(), openndm.Settings(verbosity=0)
    )
    by_index.swap_assemblies(1, 2)
    by_label.swap_assemblies("A2", "A3")
    np.testing.assert_array_equal(
        by_index.geometry.compositions, by_label.geometry.compositions
    )


def test_set_composition_takes_the_whole_axial_column():
    model = openndm.Model(
        iaea_geometry(planes=3), iaea_library(), openndm.Settings(verbosity=0)
    )
    model.set_composition("A1", 3)
    nodes = model.geometry.lattice_to_node.reshape(model.geometry.shape)[:, 0, 0]
    np.testing.assert_array_equal(model.geometry.compositions[nodes], 3)


def test_set_rotation_is_absolute_and_rotate_assembly_is_cumulative():
    model = openndm.Model(
        iaea_geometry(planes=2), iaea_library(), openndm.Settings(verbosity=0)
    )
    nodes = model.geometry.lattice_to_node.reshape(model.geometry.shape)[:, 0, 1]
    model.set_rotation("A2", 2)
    np.testing.assert_array_equal(model.geometry.rotations[nodes], 2)
    model.set_rotation("A2", 1)
    np.testing.assert_array_equal(model.geometry.rotations[nodes], 1)
    model.rotate_assembly("A2")
    np.testing.assert_array_equal(model.geometry.rotations[nodes], 2)
    model.rotate_assembly("A2", 3)
    np.testing.assert_array_equal(model.geometry.rotations[nodes], 1)
    model.rotate_assembly("A2", -1)
    np.testing.assert_array_equal(model.geometry.rotations[nodes], 0)


def test_mutating_an_out_of_core_position_changes_nothing():
    model = openndm.Model(
        iaea_geometry(), iaea_library(), openndm.Settings(verbosity=0)
    )
    before = model.geometry.compositions.copy()
    with pytest.raises(openndm.InputError, match="'G9' is inactive"):
        model.set_composition("G9", 0)
    with pytest.raises(openndm.InputError, match="'G9' is inactive"):
        model.swap_assemblies("A1", "G9")
    np.testing.assert_array_equal(model.geometry.compositions, before)
