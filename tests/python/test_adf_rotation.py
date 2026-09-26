"""Assembly rotation of discontinuity factors (FR-OPT-7).

OpenMC gives the discontinuity factors of an assembly in one orientation. The
same assembly loaded at a position rotated 90 degrees presents different faces
to its neighbours, so its factors have to be permuted to match.

Rotation belongs to the *position*, not to the composition, because one
assembly type is loaded at many positions in different orientations. The
oracle used throughout is that the two routes must agree exactly: turning an
assembly where it sits has to give the same answer as loading a composition
whose factors were permuted by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

import openndm

from conftest import ONE_GROUP

ASYMMETRIC = np.array([[0.7], [1.3], [0.9], [1.1], [1.0], [1.0]])
"""Discontinuity factors that differ on all four radial faces.

Every rotation of this set is distinguishable from every other. A set
symmetric in x or y would make half of these tests pass without the
permutation being applied at all.
"""

FACES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def library(adf_by_composition):
    """One-group library whose compositions differ only in their factors."""
    lib = openndm.XSLibrary(1, len(adf_by_composition))
    for index, adf in enumerate(adf_by_composition):
        lib.set_composition(
            index,
            D=[ONE_GROUP["D"]],
            absorption=[ONE_GROUP["absorption"]],
            nu_fission=[ONE_GROUP["nu_fission"]],
            kappa_fission=[ONE_GROUP["nu_fission"]],
            chi=[1.0],
            scatter=[[0.0]],
        )
        lib.set_adf(index, adf)
    lib.finalize()
    return lib


def core_map():
    """A 4x3 core, so the x and y directions are not interchangeable."""
    core = np.zeros((1, 3, 4), dtype=int)
    core[0, 1, 2] = 1
    return core


def centre_node(geometry):
    return int(geometry.lattice_to_node.reshape(geometry.shape)[0, 1, 2])


def build(adf_by_composition, rotation=None):
    geometry = openndm.Geometry.from_lattice(
        core_map(),
        pitch=20.0,
        boundaries=dict.fromkeys(FACES, "vacuum")
        | {"z_min": "reflective", "z_max": "reflective"},
        rotation=rotation,
    )
    return geometry, library(adf_by_composition)


def solve(geometry, lib, kernel="sanm"):
    settings = openndm.Settings(
        verbosity=0, k_tolerance=1.0e-13, fission_source_tolerance=1.0e-12
    )
    return openndm.Model(geometry, lib, settings).solve(kernel=kernel)


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_a_symmetric_set_is_invariant_under_every_rotation(turns):
    symmetric = np.array([[1.2], [1.2], [1.2], [1.2], [0.8], [0.8]])
    assert np.array_equal(openndm.rotate_adf(symmetric, turns), symmetric)


def test_four_quarter_turns_return_the_original_exactly():
    turned = ASYMMETRIC.copy()
    for _ in range(4):
        turned = openndm.rotate_adf(turned, 1)
    assert np.array_equal(turned, ASYMMETRIC)


def test_a_half_turn_swaps_each_radial_pair():
    turned = openndm.rotate_adf(ASYMMETRIC, 2)
    assert turned[0] == ASYMMETRIC[1] and turned[1] == ASYMMETRIC[0]
    assert turned[2] == ASYMMETRIC[3] and turned[3] == ASYMMETRIC[2]


def test_two_quarter_turns_equal_one_half_turn():
    once = openndm.rotate_adf(ASYMMETRIC, 1)
    assert np.array_equal(
        openndm.rotate_adf(once, 1), openndm.rotate_adf(ASYMMETRIC, 2)
    )


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_the_axial_faces_never_move(turns):
    axial = np.array([[1.0], [1.0], [1.0], [1.0], [0.3], [0.6]])
    turned = openndm.rotate_adf(axial, turns)
    assert turned[4] == 0.3 and turned[5] == 0.6


def test_a_quarter_turn_carries_plus_x_onto_plus_y():
    """The direction of the turn, which is the thing that inverts silently."""
    turned = openndm.rotate_adf(ASYMMETRIC, 1)
    assert turned[3] == ASYMMETRIC[1], "+y should show what +x carried"
    assert turned[0] == ASYMMETRIC[3], "-x should show what +y carried"


def test_rotate_adf_validates_its_arguments():
    with pytest.raises(openndm.InputError, match="0, 1, 2 or 3"):
        openndm.rotate_adf(ASYMMETRIC, 4)
    with pytest.raises(openndm.InputError, match="2 \\* n_axes"):
        openndm.rotate_adf(np.ones(6), 1)


def test_rotated_face_matches_the_array_permutation():
    for turns in range(4):
        expected = openndm.rotate_adf(ASYMMETRIC, turns)
        for face in range(6):
            assert expected[face] == ASYMMETRIC[openndm.rotated_face(face, turns)]


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_turning_a_node_matches_a_pre_rotated_composition(turns):
    """The oracle: the two routes to a rotated assembly must agree exactly.

    Turning the assembly where it sits and loading a composition whose factors
    were permuted by hand are the same physical core, so they must give the
    same eigenvalue to the last bit.
    """
    rotation = np.zeros((1, 3, 4), dtype=int)
    rotation[0, 1, 2] = turns
    turned_geometry, turned_library = build([np.ones((6, 1)), ASYMMETRIC], rotation)
    in_place = solve(turned_geometry, turned_library)

    pre_rotated_geometry, pre_rotated_library = build(
        [np.ones((6, 1)), openndm.rotate_adf(ASYMMETRIC, turns)]
    )
    by_hand = solve(pre_rotated_geometry, pre_rotated_library)

    assert in_place.k_eff == by_hand.k_eff
    assert np.array_equal(in_place.flux, by_hand.flux)


@pytest.mark.parametrize("turns", [1, 2, 3])
def test_turning_an_asymmetric_assembly_changes_the_answer(turns):
    """Without this the agreement above could hold for the wrong reason."""
    rotation = np.zeros((1, 3, 4), dtype=int)
    rotation[0, 1, 2] = turns
    unturned = solve(*build([np.ones((6, 1)), ASYMMETRIC]))
    turned = solve(*build([np.ones((6, 1)), ASYMMETRIC], rotation))
    assert turned.k_eff != unturned.k_eff


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_uniform_factors_leave_a_rotation_with_nothing_to_do(turns):
    rotation = np.full((1, 3, 4), turns, dtype=int)
    plain = solve(*build([np.ones((6, 1)), np.ones((6, 1))]))
    rotated = solve(*build([np.ones((6, 1)), np.ones((6, 1))], rotation))
    assert rotated.k_eff == plain.k_eff


def test_set_rotation_is_absolute_not_incremental():
    """Setting a position twice gives the same core as setting it once.

    This matches ``set_composition`` and ``ControlRods.insert``: a position is
    stated, not accumulated. An incremental reading would make four quarter
    turns a full circle, and make any repeated call drift.
    """
    geometry, lib = build([np.ones((6, 1)), ASYMMETRIC])
    node = centre_node(geometry)
    unturned = solve(geometry, lib).k_eff

    geometry.set_rotation(node, 1)
    once = solve(geometry, lib).k_eff
    for _ in range(3):
        geometry.set_rotation(node, 1)
    assert solve(geometry, lib).k_eff == once
    assert once != unturned

    geometry.set_rotation(node, 0)
    assert solve(geometry, lib).k_eff == unturned


@pytest.mark.parametrize("kernel", ["fdm", "nem", "sanm"])
def test_every_kernel_sees_the_rotation(kernel):
    """FDM reads the factors through the coupling, the nodal kernels also
    through the two-node problem, so both paths need the permutation."""
    rotation = np.zeros((1, 3, 4), dtype=int)
    rotation[0, 1, 2] = 1
    turned = solve(*build([np.ones((6, 1)), ASYMMETRIC], rotation), kernel)
    by_hand = solve(
        *build([np.ones((6, 1)), openndm.rotate_adf(ASYMMETRIC, 1)]), kernel
    )
    assert turned.k_eff == by_hand.k_eff


def test_a_rotation_map_matches_setting_each_node_by_hand():
    rotation = np.zeros((1, 3, 4), dtype=int)
    rotation[0, 1, 2] = 3
    from_map, _ = build([np.ones((6, 1)), ASYMMETRIC], rotation)
    by_hand, _ = build([np.ones((6, 1)), ASYMMETRIC])
    by_hand.set_rotation(centre_node(by_hand), 3)
    assert np.array_equal(from_map.rotations, by_hand.rotations)


def test_subdivision_carries_the_rotation_into_every_child():
    rotation = np.zeros((1, 3, 4), dtype=int)
    rotation[0, 1, 2] = 2
    geometry = openndm.Geometry.from_lattice(
        core_map(), pitch=20.0, rotation=rotation, subdivide=2
    )
    lattice = geometry.rotations[geometry.lattice_to_node].reshape(geometry.shape)
    assert set(np.unique(lattice)) == {0, 2}
    assert int((lattice == 2).sum()) == 8, "one cell splits into 2x2x2 nodes"


def test_an_unrotated_core_is_all_zeros():
    geometry, _ = build([np.ones((6, 1)), ASYMMETRIC])
    assert not geometry.rotations.any()


def test_the_rotation_map_is_validated():
    with pytest.raises(ValueError, match="0, 1, 2 or 3"):
        openndm.Geometry.from_lattice(
            core_map(), pitch=20.0, rotation=np.full((1, 3, 4), 4)
        )
    with pytest.raises(ValueError, match="must have shape"):
        openndm.Geometry.from_lattice(
            core_map(), pitch=20.0, rotation=np.zeros((1, 2, 2), dtype=int)
        )


def test_set_rotation_is_validated():
    geometry, _ = build([np.ones((6, 1)), ASYMMETRIC])
    with pytest.raises(openndm.InputError, match="quarter turns"):
        geometry.set_rotation(0, 5)
    with pytest.raises(openndm.InputError, match="out of range"):
        geometry.set_rotation(10_000, 1)


def test_rotated_adf_reads_back_through_the_library():
    lib = library([ASYMMETRIC])
    assert np.array_equal(lib.rotated_adf(0, 1), openndm.rotate_adf(ASYMMETRIC, 1))
