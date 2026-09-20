"""Control rod banks (FR-MODE-5)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import openndm

from conftest import ONE_GROUP

#: Plain fuel, and the same fuel with a rod in it.
FUEL, RODDED = 0, 1
#: Number of axial planes in the small test core.
NPLANES = 10
#: Axial mesh of the small test core: ten 10 cm planes, so a plane boundary
#: falls on every multiple of 10 cm.
DZ = [10.0] * NPLANES
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
    core = np.full((len(DZ), 3, 3), FUEL, dtype=int)
    core[-1] = 7  # a reflector the bank travels through but cannot substitute
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


# --------------------------------------------- the shipped deck as an oracle
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


# ------------------------------------------------------------------ cusping
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

    # The uncusped bank on the same core, for the reference value.
    columns = np.zeros((3, 3), dtype=bool)
    columns[1, 1] = True
    bare = openndm.ControlRodBank("A", columns=columns, rodded={FUEL: RODDED})
    plain_rods = openndm.ControlRods(plain_geometry, [bare])

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
    columns = np.zeros((3, 3), dtype=bool)
    columns[1, 1] = True
    core = np.full((len(DZ), 3, 3), FUEL, dtype=int)
    core[-1] = RODDED  # a second composition in the banked column
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
