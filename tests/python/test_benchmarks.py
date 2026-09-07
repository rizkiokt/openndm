"""Benchmark decks against their published references (NFR-QA-2).

The acceptance criterion in the specification is 100 pcm on k_eff for a static
benchmark with a published solution. Both IAEA decks meet it; the analytic
deck is checked against exact algebra instead. See ``benchmarks/README.md``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import openndm

from conftest import IAEA_2D_REFERENCE, IAEA_3D_REFERENCE, IAEA_MAP

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from common import (  # noqa: E402
    CORE_HEIGHT,
    IAEA_RADIAL_MAP,
    ROD_TIP_HEIGHT,
    check_radial_map,
)

#: Specification acceptance criterion for a static benchmark, in pcm.
ACCEPTANCE_PCM = 100.0


def _deck(name):
    """Import a deck's run.py under a unique module name.

    Every deck's entry point is called run.py, so importing them by bare name
    would return whichever one was imported first.
    """
    path = BENCHMARKS / name / "run.py"
    spec = importlib.util.spec_from_file_location(f"benchmark_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _solve(geometry, library, kernel="sanm"):
    settings = openndm.Settings(verbosity=0)
    return openndm.Model(geometry, library, settings).solve(kernel=kernel)


def pcm(k, reference):
    return 1.0e5 * (k - reference)


# ------------------------------------------------------------- the core map
def test_radial_map_invariants_hold():
    """Symmetry and an edge-connected fuel-1 band.

    A single dropped cell in the peripheral band is worth about 80 pcm and has
    no other visible symptom, which is exactly how it survived the first
    transcription of this deck.
    """
    check_radial_map()


def test_conftest_map_matches_the_benchmark_deck():
    """The verification suite and the deck must not drift apart."""
    assert np.array_equal(IAEA_MAP, IAEA_RADIAL_MAP)


def test_map_is_not_accidentally_symmetric_only_in_shape():
    """The fuel-1 band steps inward by exactly one cell per row on the taper."""
    counts = [int((row == 1).sum()) for row in IAEA_RADIAL_MAP]
    assert counts[4:7] == [2, 2, 2], counts


# ------------------------------------------------------------------ IAEA-2D
def test_iaea_2d_sanm_reproduces_the_reference_at_one_node_per_assembly():
    """The headline nodal result: one node per assembly, 20 cm mesh."""
    geometry, library = _deck("iaea2d").build(subdivide=1)
    result = _solve(geometry, library)
    error = pcm(result.k_eff, IAEA_2D_REFERENCE)
    assert abs(error) < ACCEPTANCE_PCM, f"{error:+.1f} pcm from the reference"


@pytest.mark.parametrize("kernel", ["fdm", "nem", "sanm"])
def test_iaea_2d_all_kernels_reproduce_the_reference_when_refined(kernel):
    geometry, library = _deck("iaea2d").build(subdivide=2)
    result = _solve(geometry, library, kernel)
    error = pcm(result.k_eff, IAEA_2D_REFERENCE)
    assert abs(error) < ACCEPTANCE_PCM, f"{kernel}: {error:+.1f} pcm"


@pytest.mark.slow
def test_iaea_2d_converges_under_refinement():
    """Refining must not walk the answer away from the reference."""
    deck = _deck("iaea2d")
    errors = []
    for sub in (2, 4, 8):
        geometry, library = deck.build(subdivide=sub)
        errors.append(pcm(_solve(geometry, library).k_eff, IAEA_2D_REFERENCE))
    assert all(abs(e) < 20.0 for e in errors), errors
    assert abs(errors[-1] - errors[-2]) < 2.0, errors


def test_iaea_2d_power_distribution_is_physical():
    geometry, library = _deck("iaea2d").build(subdivide=1)
    radial = _solve(geometry, library).radial_power()
    powered = radial[radial > 0]
    assert powered.mean() == pytest.approx(1.0)
    compositions = np.where(IAEA_RADIAL_MAP == 0, -1, IAEA_RADIAL_MAP - 1)
    rodded = compositions == 2
    fuel = np.isin(compositions, [0, 1])
    assert radial[rodded].mean() < radial[fuel].mean()
    assert radial[rodded].max() < 1.0


# ------------------------------------------------------------------ IAEA-3D
def test_iaea_3d_sanm_reproduces_the_reference():
    geometry, library = _deck("iaea3d").build(dz_target=20.0)
    result = _solve(geometry, library)
    error = pcm(result.k_eff, IAEA_3D_REFERENCE)
    assert abs(error) < ACCEPTANCE_PCM, f"{error:+.1f} pcm from the reference"


def test_iaea_3d_axial_mesh_converges():
    deck = _deck("iaea3d")
    coarse = _solve(*deck.build(dz_target=20.0)).k_eff
    fine = _solve(*deck.build(dz_target=10.0)).k_eff
    assert abs(pcm(fine, coarse)) < 50.0, (coarse, fine)
    assert abs(pcm(fine, IAEA_3D_REFERENCE)) < ACCEPTANCE_PCM


def test_iaea_3d_rod_tip_lands_on_a_plane_boundary():
    """Smearing a rod tip across a node is worth tens of pcm.

    It shows up as an axial mesh that refuses to converge monotonically, which
    is easy to mistake for a solver problem.
    """
    deck = _deck("iaea3d")
    for dz_target in (20.0, 10.0, 5.0, 4.0):
        dz, n_unrodded = deck.axial_mesh(dz_target)
        height_below_tip = sum(dz[1 : 1 + n_unrodded])
        assert height_below_tip == pytest.approx(ROD_TIP_HEIGHT), dz_target
        assert sum(dz[1:-1]) == pytest.approx(CORE_HEIGHT)


def test_iaea_3d_rods_occupy_the_upper_core_not_the_lower():
    """Guards the orientation of the 80 cm.

    Reading it as an insertion depth measured down from the top of the core,
    rather than as the height of the rod tips above the core bottom, leaves
    k_eff about 1700 pcm high and nothing else looks wrong.
    """
    deck = _deck("iaea3d")
    geometry, _ = deck.build(dz_target=20.0)
    nz, ny, nx = geometry.shape
    mapping = geometry.lattice_to_node.reshape(nz, ny, nx)
    compositions = geometry.compositions
    rodded_fuel = 2  # index of "fuel_2_rodded"

    def plane_has_rodded_fuel(k):
        nodes = mapping[k][mapping[k] >= 0]
        return bool((compositions[nodes] == rodded_fuel).any())

    # Plane 0 is the bottom reflector; planes 1..4 are the lower 80 cm.
    assert not plane_has_rodded_fuel(0)
    for k in range(1, 5):
        assert not plane_has_rodded_fuel(k), f"plane {k} should be below the tip"
    for k in range(5, nz - 1):
        assert plane_has_rodded_fuel(k), f"plane {k} should be rodded"


def test_iaea_3d_axial_profile_is_bottom_peaked():
    """Rods in the upper core push the power down, not up."""
    geometry, library = _deck("iaea3d").build(dz_target=20.0)
    axial = _solve(geometry, library).axial_power()
    assert axial[0] == 0.0 and axial[-1] == 0.0
    peak = int(np.argmax(axial))
    midplane = (len(axial) - 1) / 2.0
    assert 1 < peak < midplane, (peak, midplane, axial)


# ----------------------------------------------------------------- analytic
def test_analytic_deck_reports_the_exact_reference():
    assert _deck("analytic").analytic() == pytest.approx(1.205387388, abs=1.0e-8)
