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

from conftest import IAEA_2D_REFERENCE, IAEA_3D_REFERENCE, IAEA_MAP, chain_library

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


def test_performance_deck_library_matches_the_test_suite_copy():
    """The two synthetic eight-group libraries must not drift apart.

    `benchmarks/performance` needs one to size NFR-PERF-2 and the verification
    suite needs one to cover more than two groups, and neither can import the
    other. This is the same guard as the core map above, for the same reason:
    a correction applied to one copy and not the other is invisible.
    """
    deck = _deck("performance").eight_group_library()
    suite = chain_library()
    assert deck.n_groups == suite.n_groups
    assert deck.n_compositions == suite.n_compositions
    for index in range(suite.n_compositions):
        a = deck.composition(index)
        b = suite.composition(index)
        for field in ("D", "absorption", "nu_fission", "chi", "scatter"):
            np.testing.assert_allclose(
                np.asarray(getattr(a, field)),
                np.asarray(getattr(b, field)),
                err_msg=f"composition {index} field {field}",
            )


def test_performance_deck_reports_every_target():
    """Each NFR-PERF case must have a stated target, so none is quietly dropped."""
    targets = _deck("performance").TARGETS
    for case in range(1, 8):
        assert f"NFR-PERF-{case}" in targets
    assert "FR-OPT-3" in targets


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


# ------------------------------------------------------------------- biblis
def test_biblis_2d_sanm_reproduces_the_reference():
    """The deck is parsed from its source, not transcribed.

    An earlier attempt at this benchmark was written from memory and not
    shipped: the map used five of the eight compositions and put fuel where
    the reflector belongs, and the reference it was written against was
    1.02513 rather than the 1.02511 the source deck states. Both wrong, and
    tuning the first to reproduce the second would have looked like a pass.
    """
    deck = _deck("biblis2d")
    geometry, library = deck.build()
    error = pcm(_solve(geometry, library).k_eff, deck.PUBLISHED_K_EFF)
    assert abs(error) < ACCEPTANCE_PCM, f"SANM is {error:+.1f} pcm from the reference"


@pytest.mark.parametrize("kernel", ["fdm", "nem", "sanm"])
def test_biblis_2d_all_kernels_reproduce_the_reference_when_refined(kernel):
    deck = _deck("biblis2d")
    geometry, library = deck.build(subdivide=2)
    error = pcm(_solve(geometry, library, kernel).k_eff, deck.PUBLISHED_K_EFF)
    assert abs(error) < ACCEPTANCE_PCM, f"{kernel} is {error:+.1f} pcm"


def test_biblis_map_uses_every_composition():
    """The failure mode of the transcribed deck, asserted against directly.

    Eight compositions are defined and eight must appear. The map written
    from memory used five, which is the kind of thing that survives every
    check except comparing against the source.
    """
    deck = _deck("biblis2d")
    present = {int(v) for v in np.unique(deck.NODE_MAP) if v != 0}
    assert present == set(range(1, len(deck.COMPOSITIONS) + 1)), (
        f"map uses compositions {sorted(present)} of {len(deck.COMPOSITIONS)} defined"
    )


def test_biblis_symmetry_faces_carry_the_half_width_assemblies():
    """West and north are the symmetry cuts, so the map must be oriented.

    If the map were flipped the half-width assemblies would sit on the outer
    faces instead, which changes the core size without changing the node
    count.
    """
    deck = _deck("biblis2d")
    core = deck.NODE_MAP
    # The north-west corner is the core centre: fuel, never reflector.
    assert core[-1, 0] == 1
    # The south-east corner is outside the core entirely.
    assert core[0, -1] == 0

# ---------------------------------------------------------------------- LMW
def _lmw():
    return _deck("lmw")


def test_lmw_map_puts_fuel_on_the_symmetry_corner():
    """The map is printed north row first, so reading it in order inverts it.

    Getting this backwards puts the reflector where the core centre belongs
    and leaves the half-width assemblies on the vacuum faces instead of the
    symmetry cuts. It is the same convention the BIBLIS deck needed, and it
    is asserted rather than assumed because a flipped core still runs.
    """
    lmw = _lmw()
    core = lmw.node_map()
    middle = core[len(core) // 2]
    assert middle[0, 0] == 1, "inner core belongs on the south-west corner"
    assert middle[-1, -1] == 0, "the north-east corner is outside the core"
    assert (core[0] == core[-1]).all(), "both axial ends are reflector planes"


def test_lmw_symmetry_cuts_are_the_reflective_faces():
    lmw = _lmw()
    assert lmw.BOUNDARIES["x_min"] == "reflective"
    assert lmw.BOUNDARIES["y_min"] == "reflective"
    assert lmw.BOUNDARIES["x_max"] == "vacuum"
    assert lmw.BOUNDARIES["y_max"] == "vacuum"


def test_lmw_banks_reach_only_the_inner_core_and_the_reflector():
    """Every rodded column is inner core, which is why only it has increments.

    A bank that reached the outer core would need a rodded counterpart for it,
    and the deck gives none; an unsubstituted node looks exactly like a
    correctly withdrawn one, so this is checked rather than discovered.
    """
    lmw = _lmw()
    assert lmw.rod_bases() == [0, 2]
    nonzero = [
        index
        for index, delta in enumerate(lmw.ROD_DELTA)
        if any(any(v) for v in delta.values())
    ]
    assert nonzero == [0]


def test_lmw_bank_columns_match_the_deck():
    lmw = _lmw()
    assert int((lmw.BANK_MAP == 1).sum()) == 5
    assert int((lmw.BANK_MAP == 2).sum()) == 4


def test_lmw_rod_schedule_follows_the_deck():
    lmw = _lmw()
    assert lmw.position("bank_2", 0.0) == 100.0
    assert lmw.position("bank_2", 80.0 / 3.0) == pytest.approx(180.0)
    assert lmw.position("bank_2", 60.0) == 180.0
    assert lmw.position("bank_1", 7.5) == 180.0
    assert lmw.position("bank_1", 47.5) == pytest.approx(60.0)
    assert lmw.position("bank_1", 60.0) == 60.0


def test_lmw_steady_state_is_near_critical():
    """An operating core, so the initial state has to be close to critical."""
    lmw = _lmw()
    geometry, library, rods = lmw.build()
    model = openndm.Model(geometry, library, openndm.Settings(verbosity=0))
    rods.insert(lmw.positions_at(0.0))
    model.refresh()
    k_eff = rods.converge_cusping(model).k_eff
    assert abs(pcm(k_eff, 1.0)) < 500.0, k_eff


def test_lmw_withdrawing_adds_reactivity_and_inserting_removes_it():
    """The sign of every bank, which a flipped axis would silently invert."""
    lmw = _lmw()
    geometry, library, rods = lmw.build()
    model = openndm.Model(geometry, library, openndm.Settings(verbosity=0))

    def k_at(bank_1, bank_2):
        rods.insert({"bank_1": bank_1, "bank_2": bank_2})
        model.refresh()
        return rods.converge_cusping(model).k_eff

    initial = k_at(180.0, 100.0)
    withdrawn = k_at(180.0, 180.0)
    inserted = k_at(60.0, 180.0)
    assert withdrawn > initial, (withdrawn, initial)
    assert inserted < withdrawn, (inserted, withdrawn)


@pytest.mark.slow
def test_lmw_power_rises_then_falls():
    """The shape of the transient, which is what the scenario is designed for.

    Bank 2 withdraws from the start and bank 1 only begins inserting at
    7.5 s, so the power rises first; bank 1 overtakes it and the power ends
    below where it started. The peak height is step-size dependent and is not
    asserted -- see ``benchmarks/README.md`` for how far from converged it is.
    """
    lmw = _lmw()
    geometry, library, rods = lmw.build()
    model = openndm.Model(geometry, library, openndm.Settings(verbosity=0))
    _, history = lmw.run(model, rods, dt=0.5, total=40.0)
    time, power = history[:, 0], history[:, 1]

    peak = int(np.argmax(power))
    assert 10.0 < time[peak] < 30.0, time[peak]
    assert power[peak] > 1.2, power[peak]
    assert power[-1] < 1.0, power[-1]
    rising = power[: peak + 1]
    assert np.all(np.diff(rising) > 0.0), "power should rise monotonically first"
