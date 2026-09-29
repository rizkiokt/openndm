"""A Geometry built from an ``openmc.RectLattice`` (FR-GEO-7).

Needs OpenMC's Python API only: no nuclear data and no ``openmc`` executable.
"""

from __future__ import annotations

import numpy as np
import pytest

import openndm
from openndm.gc import composition_map, lattice_universes

from conftest import IAEA_MAP, iaea_geometry, iaea_library

openmc = pytest.importorskip("openmc")

IAEA_PITCH = 20.0
IAEA_BOUNDARIES = {
    "x_min": "reflective",
    "y_min": "reflective",
    "x_max": "zero_flux",
    "y_max": "zero_flux",
    "z_min": "reflective",
    "z_max": "reflective",
}
OUT_OF_CORE = 0


def _lattice_from_map(core_map, pitch, universes, outer=None):
    """OpenMC lattice holding ``universes[v]`` where ``core_map`` holds ``v``.

    ``core_map`` has row 0 at the lowest y, as in OpenNDM. OpenMC lists the
    highest row first, so the rows are reversed on the way in.
    """
    lattice = openmc.RectLattice()
    lattice.pitch = pitch
    lattice.lower_left = [0.0] * len(pitch)
    if outer is not None:
        lattice.outer = outer
    lattice.universes = np.vectorize(universes.__getitem__, otypes=[object])(
        np.flip(core_map, axis=-2)
    )
    return lattice


def _iaea_lattice():
    """IAEA quarter core as an OpenMC lattice, and its universes by map value."""
    universes = {value: openmc.Universe(name=f"iaea-{value}") for value in range(5)}
    lattice = _lattice_from_map(
        IAEA_MAP, (IAEA_PITCH, IAEA_PITCH), universes, universes[OUT_OF_CORE]
    )
    return lattice, universes


def _iaea_domains(universes):
    """IAEA universes in library order: map value ``n`` is composition ``n - 1``."""
    return [universes[value] for value in range(1, 5)]


def test_a_lattice_reproduces_the_map_it_was_built_from():
    """The IAEA map is not symmetric under a flip in y, so this also pins the
    row order.
    """
    lattice, universes = _iaea_lattice()
    geometry = openndm.Geometry.from_openmc(
        lattice, domains=_iaea_domains(universes), dz=[IAEA_PITCH]
    )
    expected = np.where(IAEA_MAP == OUT_OF_CORE, openndm.INACTIVE, IAEA_MAP - 1)
    rebuilt = geometry.expand(geometry.compositions, fill=openndm.INACTIVE)
    assert np.array_equal(rebuilt[0], expected)


def test_domains_may_be_given_as_universe_ids():
    lattice, universes = _iaea_lattice()
    domains = _iaea_domains(universes)
    by_object = composition_map(lattice, domains)
    by_id = composition_map(lattice, [universe.id for universe in domains])
    assert np.array_equal(by_object, by_id)


def test_default_domain_order_is_first_appearance_from_the_lower_left():
    lattice, _ = _iaea_lattice()
    order = [universe.name for universe in lattice_universes(lattice)]
    assert order == ["iaea-3", "iaea-2", "iaea-1", "iaea-4"]
    composition = composition_map(lattice)
    assert composition[0, 0, 0] == 0
    assert np.all(composition[0][IAEA_MAP == OUT_OF_CORE] == openndm.INACTIVE)


def test_widths_are_the_lattice_pitch_and_the_given_axial_mesh():
    """A two-dimensional lattice is extruded through a non-uniform ``dz``."""
    fuel = openmc.Universe()
    lattice = _lattice_from_map(np.zeros((2, 3), dtype=int), (21.504, 19.25), [fuel])
    dz = [30.0, 17.5, 42.25]
    geometry = openndm.Geometry.from_openmc(lattice, dz=dz)
    assert geometry.shape == (3, 2, 3)
    assert np.array_equal(geometry.dx, np.full(3, 21.504))
    assert np.array_equal(geometry.dy, np.full(2, 19.25))
    assert np.array_equal(geometry.dz, dz)


def test_a_three_dimensional_lattice_takes_dz_from_its_z_pitch():
    """The map differs in every direction, so it also pins the plane order."""
    fuel, reflector = openmc.Universe(), openmc.Universe()
    core_map = np.zeros((4, 2, 2), dtype=int)
    core_map[0] = 1
    core_map[:, 1, 1] = 1
    lattice = _lattice_from_map(core_map, (21.5, 21.5, 36.6), [fuel, reflector])
    geometry = openndm.Geometry.from_openmc(lattice, domains=[fuel, reflector])
    assert np.array_equal(geometry.dz, np.full(4, 36.6))
    rebuilt = geometry.expand(geometry.compositions)
    assert np.array_equal(rebuilt, core_map)


def test_subdivision_passes_through_to_from_lattice():
    fuel = openmc.Universe()
    lattice = _lattice_from_map(np.zeros((2, 2), dtype=int), (20.0, 20.0), [fuel])
    geometry = openndm.Geometry.from_openmc(lattice, dz=[10.0], subdivide=(2, 2, 1))
    assert geometry.shape == (1, 4, 4)
    assert np.array_equal(geometry.dx, np.full(4, 10.0))


def test_the_same_core_gives_the_same_eigenvalue_as_the_hand_built_one(tight):
    lattice, universes = _iaea_lattice()
    from_openmc = openndm.Geometry.from_openmc(
        lattice,
        domains=_iaea_domains(universes),
        dz=[IAEA_PITCH],
        boundaries=IAEA_BOUNDARIES,
        outside="zero_flux",
    )
    k_from_openmc = openndm.Model(from_openmc, iaea_library(), tight).solve().k_eff
    k_by_hand = openndm.Model(iaea_geometry(), iaea_library(), tight).solve().k_eff
    assert k_from_openmc == k_by_hand


def test_a_universe_missing_from_the_domains_is_refused():
    lattice, universes = _iaea_lattice()
    with pytest.raises(openndm.InputError, match="not among the domains"):
        composition_map(lattice, _iaea_domains(universes)[:-1])


@pytest.mark.parametrize(
    ("dz", "message"), [(None, "needs dz"), ([10.0], "sets dz from its z pitch")]
)
def test_dz_is_required_in_2d_and_refused_in_3d(dz, message):
    fuel = openmc.Universe()
    two_d = _lattice_from_map(np.zeros((1, 1), dtype=int), (20.0, 20.0), [fuel])
    three_d = _lattice_from_map(
        np.zeros((1, 1, 1), dtype=int), (20.0, 20.0, 20.0), [fuel]
    )
    lattice = two_d if dz is None else three_d
    with pytest.raises(openndm.InputError, match=message):
        openndm.Geometry.from_openmc(lattice, dz=dz)


def test_the_lattice_pitch_cannot_be_overridden():
    fuel = openmc.Universe()
    lattice = _lattice_from_map(np.zeros((1, 1), dtype=int), (20.0, 20.0), [fuel])
    with pytest.raises(openndm.InputError, match="pitch"):
        openndm.Geometry.from_openmc(lattice, dz=[10.0], dx=[5.0])


def test_a_hexagonal_lattice_is_refused():
    with pytest.raises(openndm.InputError, match="RectLattice"):
        composition_map(openmc.HexLattice())
