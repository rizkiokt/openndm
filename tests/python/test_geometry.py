"""Geometry construction and the core map conventions (FR-GEO)."""

from __future__ import annotations

import numpy as np
import pytest

import openndm

from conftest import IAEA_MAP, iaea_geometry


def test_uniform_lattice_node_and_surface_counts():
    core = np.zeros((4, 3, 2), dtype=int)
    g = openndm.Geometry.from_lattice(core, pitch=10.0)
    assert g.n_nodes == 24
    # Interior faces plus boundary faces: 3 * n + faces on the outside.
    expected = 3 * 24 + (3 * 4 + 2 * 4 + 2 * 3)
    assert g.n_surfaces == expected
    assert g.total_volume == pytest.approx(24 * 1000.0)


def test_two_dimensional_map_is_promoted_to_one_plane():
    g = openndm.Geometry.from_lattice(np.zeros((5, 5), dtype=int), pitch=20.0)
    assert g.shape == (1, 5, 5)
    assert g.n_nodes == 25


def test_inactive_positions_are_dropped():
    core = np.zeros((1, 3, 3), dtype=int)
    core[0, 2, 2] = openndm.INACTIVE
    g = openndm.Geometry.from_lattice(core, pitch=10.0)
    assert g.n_nodes == 8
    assert (g.lattice_to_node < 0).sum() == 1


def test_non_uniform_widths_are_respected():
    g = openndm.Geometry.from_lattice(
        np.zeros((3, 1, 1), dtype=int), pitch=10.0, dz=[5.0, 20.0, 15.0]
    )
    assert sorted(g.volumes) == pytest.approx([500.0, 1500.0, 2000.0])


def test_subdivision_multiplies_nodes_and_preserves_composition():
    core = np.array([[[0, 1], [1, 0]]])
    g = openndm.Geometry.from_lattice(core, pitch=20.0, subdivide=(2, 2, 1))
    assert g.n_nodes == 16
    assert np.bincount(g.compositions).tolist() == [8, 8]
    assert g.volumes == pytest.approx(np.full(16, 10.0 * 10.0 * 20.0))


def test_out_of_core_faces_do_not_inherit_a_symmetry_condition():
    """An inactive neighbour is a real outer boundary, not a symmetry plane.

    Getting this wrong on a quarter-core map reflects neutrons back into the
    core from positions that are outside it, which inflates k_eff by hundreds
    of pcm without any other symptom.
    """
    core = np.array([[[0, 0], [0, openndm.INACTIVE]]])
    reflective = openndm.Geometry.from_lattice(
        core,
        pitch=20.0,
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "reflective"
        ),
        outside="vacuum",
    )
    n_vacuum = sum(1 for s in reflective._g.surfaces if s.bc.name == "vacuum")
    assert n_vacuum == 2, "the two faces looking at the inactive cell must be vacuum"


def test_expand_scatters_node_values_back_onto_the_lattice():
    g = iaea_geometry()
    values = np.arange(g.n_nodes, dtype=float)
    lattice = g.expand(values, fill=-1.0)
    assert lattice.shape == g.shape
    assert (lattice[0] == -1.0).sum() == int((IAEA_MAP == 0).sum())
    assert lattice[lattice >= 0].sum() == pytest.approx(values.sum())


def test_albedo_boundary_requires_values():
    with pytest.raises(openndm.InputError, match="albedo"):
        openndm.Geometry.from_lattice(
            np.zeros((1, 2, 2), dtype=int),
            pitch=10.0,
            boundaries={"x_max": "albedo"},
        )


def test_unknown_boundary_name_is_rejected():
    with pytest.raises(ValueError, match="unknown boundary condition"):
        openndm.Geometry.from_lattice(
            np.zeros((1, 2, 2), dtype=int), pitch=10.0,
            boundaries={"x_max": "perfectly_matched"},
        )


def test_unknown_face_name_is_rejected():
    with pytest.raises(ValueError, match="unknown boundary face"):
        openndm.Geometry.from_lattice(
            np.zeros((1, 2, 2), dtype=int), pitch=10.0, boundaries={"radial": "vacuum"}
        )


def test_empty_core_map_is_rejected():
    with pytest.raises(openndm.InputError, match="no active positions"):
        openndm.Geometry.from_lattice(
            np.full((1, 2, 2), openndm.INACTIVE), pitch=10.0
        )


def test_negative_width_is_rejected():
    with pytest.raises(openndm.InputError, match="positive"):
        openndm.Geometry.from_lattice(
            np.zeros((2, 1, 1), dtype=int), pitch=10.0, dz=[10.0, -1.0]
        )


def test_iaea_map_has_the_expected_active_count():
    g = iaea_geometry()
    assert g.n_nodes == int((IAEA_MAP != 0).sum())
    assert g.n_compositions == 4  # the rodded reflector is unused in 2D
