"""Control rod banks (FR-MODE-5)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import openndm

from conftest import ONE_GROUP

FUEL, RODDED = 0, 1
"""Composition indices of plain fuel and of the same fuel with a rod in it."""

NPLANES = 10
"""Number of axial planes in the small test core."""

DZ = [10.0] * NPLANES
"""Axial mesh of the small test core, cm.

Ten 10 cm planes, so a plane boundary falls on every multiple of 10 cm.
"""

HEIGHT = sum(DZ)


def rod_library():
    library = openndm.XSLibrary(1, 2)
    for index, extra_absorption in ((FUEL, 0.0), (RODDED, 0.04)):
        library.set_composition(
            index,
            D=[ONE_GROUP["D"]],
            absorption=[ONE_GROUP["absorption"] + extra_absorption],
            nu_fission=[ONE_GROUP["nu_fission"]],
            kappa_fission=[ONE_GROUP["nu_fission"]],
            chi=[1.0],
            scatter=[[0.0]],
        )
    library.finalize()
    return library


def rod_core(banked=(1, 1)):
    """A 3x3 core of fuel, ten planes tall, with one banked column."""
    columns = np.zeros((3, 3), dtype=bool)
    columns[banked] = True
    core = np.full((len(DZ), 3, 3), FUEL, dtype=int)
    geometry = openndm.Geometry.from_lattice(
        core,
        pitch=(20.0, 20.0, 20.0),
        dz=DZ,
        boundaries=dict.fromkeys(
            ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"), "zero_flux"
        ),
    )
    bank = openndm.ControlRodBank(
        "A", columns=columns, rodded={FUEL: RODDED}, step_size=1.0
    )
    return geometry, openndm.ControlRods(geometry, [bank]), columns


def test_step_zero_is_fully_inserted_and_steps_withdraw():
    """KOMODO's convention: 0 steps is in, increasing steps pulls out."""
    geometry, rods, _ = rod_core()
    rods.insert(A=0.0)
    fully_in = int((geometry.compositions == RODDED).sum())
    rods.insert(A=HEIGHT)
    fully_out = int((geometry.compositions == RODDED).sum())
    assert fully_in == len(DZ), "step 0 must rod the whole banked column"
    assert fully_out == 0, "a tip at the top of the mesh must rod nothing"


def test_tip_on_a_plane_boundary_is_exact():
    """Every plane above the tip is rodded, every plane below is not."""
    geometry, rods, _ = rod_core()
    nz, ny, nx = geometry.shape
    mapping = geometry.lattice_to_node.reshape(nz, ny, nx)
    for planes_below in range(len(DZ) + 1):
        rods.insert(A=planes_below * 10.0)
        compositions = geometry.compositions
        for k in range(nz):
            node = mapping[k, 1, 1]
            expected = FUEL if k < planes_below else RODDED
            assert compositions[node] == expected, (
                f"tip at {planes_below * 10.0} cm: plane {k} is "
                f"{compositions[node]}, expected {expected}"
            )


def test_only_the_banked_columns_move():
    geometry, rods, columns = rod_core()
    nz, ny, nx = geometry.shape
    mapping = geometry.lattice_to_node.reshape(nz, ny, nx)
    rods.insert(A=0.0)
    compositions = geometry.compositions
    for j in range(ny):
        for i in range(nx):
            expected = RODDED if columns[j, i] else FUEL
            assert all(compositions[mapping[k, j, i]] == expected for k in range(nz)), (
                f"column ({j}, {i}) should be {expected} throughout"
            )


def test_positions_are_absolute_not_incremental():
    """Moving a bank twice equals setting the final position once."""
    geometry_a, rods_a, _ = rod_core()
    rods_a.insert(A=30.0)
    rods_a.insert(A=70.0)

    geometry_b, rods_b, _ = rod_core()
    rods_b.insert(A=70.0)

    assert np.array_equal(geometry_a.compositions, geometry_b.compositions)


def test_withdrawing_restores_the_unrodded_core():
    geometry, rods, _ = rod_core()
    base = geometry.compositions.copy()
    rods.insert(A=0.0)
    assert not np.array_equal(geometry.compositions, base)
    rods.withdraw()
    assert np.array_equal(geometry.compositions, base)
    assert rods.positions == {"A": None}


def test_a_withdrawn_bank_reports_the_top_of_the_mesh():
    _, rods, _ = rod_core()
    assert rods.tip_height("A") == pytest.approx(HEIGHT)


def test_inserting_the_bank_lowers_k_eff(tight):
    geometry, rods, _ = rod_core()
    model = openndm.Model(geometry, rod_library(), tight)
    out = model.solve().k_eff
    rods.insert(A=0.0)
    model.refresh()
    inserted = model.solve().k_eff
    worth_pcm = 1.0e5 * (out - inserted)
    assert worth_pcm > 100.0, f"bank worth only {worth_pcm:+.1f} pcm"


def test_worth_grows_monotonically_as_the_bank_inserts(tight):
    """On boundary-aligned positions the worth curve must be monotone."""
    geometry, rods, _ = rod_core()
    model = openndm.Model(geometry, rod_library(), tight)
    eigenvalues = []
    for planes_below in range(len(DZ), -1, -1):
        rods.insert(A=planes_below * 10.0)
        model.refresh()
        eigenvalues.append(model.solve().k_eff)
    differences = np.diff(eigenvalues)
    assert np.all(differences < 0.0), (
        f"k_eff must fall as the bank inserts, got steps {differences}"
    )


def test_a_composition_with_no_rodded_counterpart_is_an_error():
    """Silence here would look exactly like a correctly withdrawn bank."""
    columns = np.zeros((3, 3), dtype=bool)
    columns[1, 1] = True
    reflector_the_bank_cannot_substitute = 7
    core = np.full((len(DZ), 3, 3), FUEL, dtype=int)
    core[-1] = reflector_the_bank_cannot_substitute
    geometry = openndm.Geometry.from_lattice(core, pitch=20.0, dz=DZ)
    bank = openndm.ControlRodBank("A", columns=columns, rodded={FUEL: RODDED})
    rods = openndm.ControlRods(geometry, [bank])
    with pytest.raises(openndm.InputError, match="no rodded counterpart"):
        rods.insert(A=0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"columns": np.ones(3, dtype=bool)}, "2D"),
        ({"columns": np.zeros((3, 3), dtype=bool)}, "no columns selected"),
        ({"step_size": 0.0}, "step_size must be positive"),
        ({"max_steps": -1}, "max_steps must not be negative"),
    ],
)
def test_bank_definitions_are_validated(kwargs, match):
    defaults = {
        "name": "A",
        "columns": np.ones((3, 3), dtype=bool),
        "rodded": {FUEL: RODDED},
    }
    with pytest.raises(openndm.InputError, match=match):
        openndm.ControlRodBank(**{**defaults, **kwargs})


def test_positions_are_validated():
    _, rods, _ = rod_core()
    with pytest.raises(openndm.InputError, match="unknown bank"):
        rods.insert(B=0.0)
    with pytest.raises(openndm.InputError, match="must not be negative"):
        rods.insert(A=-1.0)


def test_max_steps_is_enforced():
    columns = np.zeros((3, 3), dtype=bool)
    columns[1, 1] = True
    core = np.full((len(DZ), 3, 3), FUEL, dtype=int)
    geometry = openndm.Geometry.from_lattice(core, pitch=20.0, dz=DZ)
    bank = openndm.ControlRodBank(
        "A", columns=columns, rodded={FUEL: RODDED}, max_steps=100
    )
    rods = openndm.ControlRods(geometry, [bank])
    with pytest.raises(openndm.InputError, match="exceeds max_steps"):
        rods.insert(A=101.0)


def test_duplicate_bank_names_are_rejected():
    columns = np.zeros((3, 3), dtype=bool)
    columns[1, 1] = True
    core = np.full((len(DZ), 3, 3), FUEL, dtype=int)
    geometry = openndm.Geometry.from_lattice(core, pitch=20.0, dz=DZ)
    bank = openndm.ControlRodBank("A", columns=columns, rodded={FUEL: RODDED})
    with pytest.raises(openndm.InputError, match="duplicate bank name"):
        openndm.ControlRods(geometry, [bank, bank])


def test_column_mask_must_match_the_lattice():
    core = np.full((len(DZ), 3, 3), FUEL, dtype=int)
    geometry = openndm.Geometry.from_lattice(core, pitch=20.0, dz=DZ)
    bank = openndm.ControlRodBank(
        "A", columns=np.ones((4, 4), dtype=bool), rodded={FUEL: RODDED}
    )
    with pytest.raises(openndm.InputError, match=r"lattice is \(3, 3\)"):
        openndm.ControlRods(geometry, [bank])


def _iaea3d():
    """Import the IAEA-3D deck under a unique module name."""
    benchmarks = Path(__file__).resolve().parents[2] / "benchmarks"
    if str(benchmarks) not in sys.path:
        sys.path.insert(0, str(benchmarks))
    path = benchmarks / "iaea3d" / "run.py"
    spec = importlib.util.spec_from_file_location("rods_iaea3d", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_bank_reproduces_the_iaea3d_deck_node_for_node():
    """The acceptance test for the bank API, against an exact oracle.

    ``benchmarks/iaea3d`` places its rods by hand, assigning a rodded
    composition plane by plane. Its axial mesh puts the rod tip exactly on a
    plane boundary, which is precisely where a bank at a continuous position
    must agree with it -- not to a tolerance, but node for node. Anything
    else means the bank covers the wrong planes or the wrong columns.
    """
    deck = _iaea3d()
    from common import (
        AXIAL_REFLECTOR,
        IAEA_BOUNDARIES,
        ROD_TIP_HEIGHT,
        radial_map_to_compositions,
    )

    by_hand, library = deck.build(dz_target=20.0)

    radial = radial_map_to_compositions()
    banked = radial == deck.FUEL_2_RODDED
    dz, _ = deck.axial_mesh(20.0)
    core = np.empty((len(dz), 9, 9), dtype=int)
    inactive = radial == openndm.INACTIVE
    core[0] = np.where(inactive, openndm.INACTIVE, deck.REFLECTOR)
    core[-1] = np.where(inactive, openndm.INACTIVE, deck.REFLECTOR)
    for k in range(1, len(dz) - 1):
        core[k] = np.where(banked, deck.FUEL_2, radial)

    withdrawn = openndm.Geometry.from_lattice(
        core,
        pitch=(20.0, 20.0, 20.0),
        dz=dz,
        boundaries={**IAEA_BOUNDARIES, "z_min": "vacuum", "z_max": "vacuum"},
        outside="zero_flux",
    )
    bank = openndm.ControlRodBank(
        "A",
        columns=banked,
        rodded={
            deck.FUEL_2: deck.FUEL_2_RODDED,
            deck.REFLECTOR: deck.REFLECTOR_RODDED,
        },
        zero_position=AXIAL_REFLECTOR,
    )
    rods = openndm.ControlRods(withdrawn, [bank])
    rods.insert(A=ROD_TIP_HEIGHT)

    assert rods.tip_height("A") == pytest.approx(AXIAL_REFLECTOR + ROD_TIP_HEIGHT)
    assert np.array_equal(withdrawn.compositions, by_hand.compositions), (
        "the bank built a different core than the deck places by hand"
    )

    settings = openndm.Settings(verbosity=0)
    from_bank = openndm.Model(withdrawn, library, settings).solve().k_eff
    from_deck = openndm.Model(by_hand, library, settings).solve().k_eff
    assert from_bank == pytest.approx(from_deck, abs=1.0e-12)


CUSP = 2


def cusp_library():
    library = openndm.XSLibrary(1, 3)
    for index, extra_absorption in ((FUEL, 0.0), (RODDED, 0.04), (CUSP, 0.0)):
        library.set_composition(
            index,
            D=[ONE_GROUP["D"]],
            absorption=[ONE_GROUP["absorption"] + extra_absorption],
            nu_fission=[ONE_GROUP["nu_fission"]],
            kappa_fission=[ONE_GROUP["nu_fission"]],
            chi=[1.0],
            scatter=[[0.0]],
        )
    library.finalize(warn=False)
    return library


def cusp_core(planes=NPLANES):
    """The same core as ``rod_core``, with a spare slot for the mixture."""
    columns = np.zeros((3, 3), dtype=bool)
    columns[1, 1] = True
    library = cusp_library()
    geometry = openndm.Geometry.from_lattice(
        np.full((planes, 3, 3), FUEL, dtype=int),
        pitch=(20.0, 20.0, 20.0),
        dz=[HEIGHT / planes] * planes,
        boundaries=dict.fromkeys(
            ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"), "zero_flux"
        ),
    )
    bank = openndm.ControlRodBank(
        "A", columns=columns, rodded={FUEL: RODDED}, cusp={FUEL: CUSP}
    )
    return geometry, library, openndm.ControlRods(geometry, [bank], library=library)


def _fine_mesh_reference(tips, planes=200):
    """k_eff with the tip always on a plane boundary, so no node is partial."""
    columns = np.zeros((3, 3), dtype=bool)
    columns[1, 1] = True
    geometry = openndm.Geometry.from_lattice(
        np.full((planes, 3, 3), FUEL, dtype=int),
        pitch=(20.0, 20.0, 20.0),
        dz=[HEIGHT / planes] * planes,
        boundaries=dict.fromkeys(
            ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"), "zero_flux"
        ),
    )
    bank = openndm.ControlRodBank("A", columns=columns, rodded={FUEL: RODDED})
    rods = openndm.ControlRods(geometry, [bank])
    model = openndm.Model(geometry, rod_library(), openndm.Settings(verbosity=0))
    out = []
    for tip in tips:
        rods.insert(A=float(tip))
        model.refresh()
        out.append(model.solve().k_eff)
    return np.array(out)


def test_insert_leaves_the_library_solvable(tight):
    """Writing a mixture unfinalizes the library; it must be finalized again."""
    geometry, library, rods = cusp_core()
    model = openndm.Model(geometry, library, tight)
    rods.insert(A=25.0)
    assert library.finalized, "insert must re-finalize after writing a mixture"
    model.refresh()
    assert model.solve().converged


def test_cusping_leaves_boundary_positions_alone(tight):
    """On a plane boundary no node is partial, so nothing may change."""
    plain_geometry, _, plain_rods = cusp_core()
    plain_model = openndm.Model(plain_geometry, rod_library(), tight)
    cusp_geometry, cusp_library_, cusp_rods = cusp_core()
    cusp_model = openndm.Model(cusp_geometry, cusp_library_, tight)

    columns = np.zeros((3, 3), dtype=bool)
    columns[1, 1] = True
    uncusped = openndm.ControlRodBank("A", columns=columns, rodded={FUEL: RODDED})
    plain_rods = openndm.ControlRods(plain_geometry, [uncusped])

    for tip in (20.0, 30.0, 40.0):
        plain_rods.insert(A=tip)
        plain_model.refresh()
        cusp_rods.insert(A=tip)
        cusp_model.refresh()
        assert cusp_model.solve().k_eff == pytest.approx(
            plain_model.solve().k_eff, abs=1.0e-12
        ), f"cusping changed the answer at a plane boundary, tip {tip}"


@pytest.mark.slow
def test_cusping_removes_most_of_the_staircase(tight):
    """The jump between neighbouring positions must collapse."""
    tips = np.arange(20.0, 40.01, 2.5)
    reference = _fine_mesh_reference(tips)

    geometry, library, rods = cusp_core()
    model = openndm.Model(geometry, library, tight)
    cusped = []
    for tip in tips:
        rods.insert(A=float(tip))
        model.refresh()
        cusped.append(rods.converge_cusping(model).k_eff)

    jump = np.max(np.abs(np.diff(1.0e5 * np.array(cusped))))
    reference_jump = np.max(np.abs(np.diff(1.0e5 * reference)))
    assert jump < 2.0 * reference_jump, (
        f"largest step {jump:.0f} pcm against a reference that itself moves "
        f"{reference_jump:.0f} pcm; uncusped is about 1561 pcm"
    )


@pytest.mark.slow
def test_flux_weighting_beats_volume_weighting(tight):
    """Volume weighting is the flat-flux limit and biased low.

    The flux is depressed on the rodded side, so weighting by volume alone
    over-counts the rodded absorption. Measured on this core: -103 pcm mean
    bias volume-weighted against -13 pcm flux-weighted.
    """
    tips = np.array([22.5, 25.0, 27.5, 32.5, 35.0, 37.5])
    reference = _fine_mesh_reference(tips)

    def curve(converge):
        geometry, library, rods = cusp_core()
        model = openndm.Model(geometry, library, tight)
        out = []
        for tip in tips:
            rods.insert(A=float(tip))
            model.refresh()
            out.append(
                rods.converge_cusping(model).k_eff if converge else model.solve().k_eff
            )
        return np.array(out)

    volume_bias = np.mean(1.0e5 * (curve(False) - reference))
    flux_bias = np.mean(1.0e5 * (curve(True) - reference))
    assert volume_bias < -40.0, (
        f"expected a low bias from volume weighting, got {volume_bias:+.1f}"
    )
    assert abs(flux_bias) < 0.5 * abs(volume_bias), (
        f"flux weighting left {flux_bias:+.1f} pcm against volume weighting's "
        f"{volume_bias:+.1f} pcm"
    )


def test_converge_cusping_is_one_solve_without_a_partial_node(tight):
    geometry, library, rods = cusp_core()
    model = openndm.Model(geometry, library, tight)
    rods.insert(A=30.0)
    model.refresh()
    assert rods.converge_cusping(model).k_eff == pytest.approx(
        model.solve().k_eff, abs=1.0e-12
    )


def test_cusping_needs_the_library():
    columns = np.zeros((3, 3), dtype=bool)
    columns[1, 1] = True
    geometry = openndm.Geometry.from_lattice(
        np.full((len(DZ), 3, 3), FUEL, dtype=int), pitch=20.0, dz=DZ
    )
    bank = openndm.ControlRodBank(
        "A", columns=columns, rodded={FUEL: RODDED}, cusp={FUEL: CUSP}
    )
    with pytest.raises(openndm.InputError, match="needs the library"):
        openndm.ControlRods(geometry, [bank])


def test_cusp_must_cover_every_composition_the_bank_reaches():
    """The banked column holds a second composition, which needs its own slot."""
    columns = np.zeros((3, 3), dtype=bool)
    columns[1, 1] = True
    core = np.full((len(DZ), 3, 3), FUEL, dtype=int)
    core[-1] = RODDED
    geometry = openndm.Geometry.from_lattice(core, pitch=20.0, dz=DZ)
    bank = openndm.ControlRodBank(
        "A",
        columns=columns,
        rodded={FUEL: RODDED, RODDED: RODDED},
        cusp={FUEL: CUSP},
    )
    with pytest.raises(openndm.InputError, match="no spare composition"):
        openndm.ControlRods(geometry, [bank], library=cusp_library())


def test_cusp_slots_must_be_distinct():
    columns = np.ones((3, 3), dtype=bool)
    with pytest.raises(openndm.InputError, match="not distinct"):
        openndm.ControlRodBank(
            "A",
            columns=columns,
            rodded={FUEL: RODDED, 3: RODDED},
            cusp={FUEL: CUSP, 3: CUSP},
        )


def test_worth_curve_matches_two_direct_solves(tight):
    """The curve must agree with worth computed the long way round."""
    geometry, library, rods = cusp_core()
    model = openndm.Model(geometry, library, tight)
    curve = rods.worth_curve(model, "A", [0.0, 50.0, 100.0])

    rods.withdraw()
    model.refresh()
    withdrawn = model.solve().k_eff
    rods.insert(A=0.0)
    model.refresh()
    inserted = rods.converge_cusping(model).k_eff
    direct = 1.0e5 * (1.0 / inserted - 1.0 / withdrawn)

    assert curve.integral[0] == pytest.approx(direct, abs=0.01)
    assert curve.integral[-1] == pytest.approx(0.0, abs=1.0e-9)


def test_inserting_gives_positive_worth(tight):
    geometry, library, rods = cusp_core()
    model = openndm.Model(geometry, library, tight)
    curve = rods.worth_curve(model, "A", [0.0, 100.0])
    assert curve.integral[0] > 0.0, "an inserted bank must have positive worth"


def test_worth_falls_monotonically_as_the_bank_withdraws(tight):
    geometry, library, rods = cusp_core()
    model = openndm.Model(geometry, library, tight)
    curve = rods.worth_curve(model, "A", np.arange(0.0, 100.1, 12.5))
    assert np.all(np.diff(curve.integral) < 0.0), (
        f"worth must fall as the bank withdraws, got {curve.integral}"
    )


def test_differential_integrates_back_to_the_integral(tight):
    """The two curves must describe the same function."""
    geometry, library, rods = cusp_core()
    model = openndm.Model(geometry, library, tight)
    steps = np.arange(0.0, 100.1, 5.0)
    curve = rods.worth_curve(model, "A", steps)

    rebuilt = curve.integral[0] + np.concatenate(
        [
            [0.0],
            np.cumsum(
                np.diff(steps)
                * 0.5
                * (curve.differential[:-1] + curve.differential[1:])
            ),
        ]
    )
    span = float(np.max(curve.integral) - np.min(curve.integral))
    assert np.max(np.abs(rebuilt - curve.integral)) < 0.02 * span, (
        "trapezoid-integrating the differential curve must return the "
        "integral curve to within the trapezoid error"
    )


def test_worth_curve_restores_the_bank(tight):
    """Measuring worth must not move the rods."""
    geometry, library, rods = cusp_core()
    model = openndm.Model(geometry, library, tight)
    rods.insert(A=40.0)
    before = geometry.compositions.copy()
    rods.worth_curve(model, "A", [0.0, 50.0, 100.0])
    assert rods.positions == {"A": 40.0}
    assert np.array_equal(geometry.compositions, before)


def test_warm_start_does_not_change_the_curve(tight):
    """Warm starting is an accelerator, not an approximation (FR-OPT-3)."""
    geometry, library, rods = cusp_core()
    model = openndm.Model(geometry, library, tight)
    steps = [0.0, 25.0, 50.0, 75.0, 100.0]
    warm = rods.worth_curve(model, "A", steps, warm_start=True)
    cold = rods.worth_curve(model, "A", steps, warm_start=False)
    assert np.max(np.abs(warm.integral - cold.integral)) < 0.5


def test_worth_curve_takes_an_explicit_reference(tight):
    geometry, library, rods = cusp_core()
    model = openndm.Model(geometry, library, tight)
    curve = rods.worth_curve(model, "A", [0.0, 50.0], reference=50.0)
    assert curve.integral[-1] == pytest.approx(0.0, abs=1.0e-9)
    assert curve.integral[0] > 0.0


def test_worth_curve_over_a_multi_bank_sequence(tight):
    """A prepared sequence models bank overlap."""
    columns_a = np.zeros((3, 3), dtype=bool)
    columns_a[0, 0] = True
    columns_b = np.zeros((3, 3), dtype=bool)
    columns_b[2, 2] = True
    geometry = openndm.Geometry.from_lattice(
        np.full((NPLANES, 3, 3), FUEL, dtype=int),
        pitch=(20.0, 20.0, 20.0),
        dz=DZ,
        boundaries=dict.fromkeys(
            ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"), "zero_flux"
        ),
    )
    banks = [
        openndm.ControlRodBank(name, columns=columns, rodded={FUEL: RODDED})
        for name, columns in (("A", columns_a), ("B", columns_b))
    ]
    rods = openndm.ControlRods(geometry, banks)
    model = openndm.Model(geometry, rod_library(), tight)

    sequence = [
        {"A": None, "B": None},
        {"A": 50.0, "B": None},
        {"A": 0.0, "B": None},
        {"A": 0.0, "B": 50.0},
        {"A": 0.0, "B": 0.0},
    ]
    curve = rods.worth_curve(
        model, positions=sequence, reference={"A": None, "B": None}
    )
    assert curve.bank is None
    assert np.array_equal(curve.steps, np.arange(5.0))
    assert curve.integral[0] == pytest.approx(0.0, abs=1.0e-9)
    assert np.all(np.diff(curve.integral) > 0.0), (
        f"each step of the sequence inserts more, so worth must rise: {curve.integral}"
    )


def test_worth_curve_validates_its_arguments(tight):
    geometry, library, rods = cusp_core()
    model = openndm.Model(geometry, library, tight)
    with pytest.raises(openndm.InputError, match="needs positions"):
        rods.worth_curve(model, "A")
    with pytest.raises(openndm.InputError, match="at least one position"):
        rods.worth_curve(model, "A", [])
    with pytest.raises(openndm.InputError, match="unknown bank"):
        rods.worth_curve(model, "B", [0.0])
    with pytest.raises(openndm.InputError, match="reference given as a mapping"):
        rods.worth_curve(model, positions=[{"A": 0.0}], reference=0.0)


def test_reweight_reaches_what_converge_cusping_reaches(tight):
    """Driving the sweep by hand converges on the same mixture.

    A transient cannot call :meth:`converge_cusping`, because the static solve
    inside it would discard the time-dependent flux. It re-weights against the
    flux it already has instead, so the two routes have to agree.
    """
    geometry, library, rods = cusp_core()
    model = openndm.Model(geometry, library, tight)
    rods.insert(A=25.5)
    model.refresh()
    converged = rods.converge_cusping(model).k_eff

    geometry, library, rods = cusp_core()
    model = openndm.Model(geometry, library, tight)
    rods.insert(A=25.5)
    model.refresh()
    result = model.solve()
    for _ in range(8):
        change = rods.reweight(result.flux)
        if change is None or change < 1.0e-6:
            break
        model.refresh()
        result = model.solve()
    assert result.k_eff == pytest.approx(converged, abs=1.0e-12)


def test_reweight_reports_nothing_to_do_on_a_plane_boundary(tight):
    geometry, library, rods = cusp_core()
    model = openndm.Model(geometry, library, tight)
    rods.insert(A=30.0)
    model.refresh()
    assert rods.reweight(model.solve().flux) is None
