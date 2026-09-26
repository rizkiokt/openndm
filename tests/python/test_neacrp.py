"""The NEACRP deck, checked against the specification it was parsed from.

The deck is parsed from KOMODO; the numbers below come from NEACRP-L-335,
Finnemann and Galati 1991, which KOMODO transcribed. Agreement between the
two is what says the parse is right, and it is the check BIBLIS did not have
when it was first attempted from memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS / "neacrp"))

import neacrp_data as deck  # noqa: E402
import neacrp_reference as reference  # noqa: E402

QUARTER_CASES = ("A1", "A2", "B1", "B2")
HALF_CASES = ("C1", "C2")
ALL_CASES = QUARTER_CASES + HALF_CASES


def test_the_deck_holds_every_case():
    assert sorted(deck.CASES) == sorted(ALL_CASES)


def test_the_library_is_two_groups_over_eleven_compositions():
    """NEACRP-L-335 Section 2.4: eleven compositions, Section 2.2: two groups."""
    assert deck.N_GROUPS == 2
    assert deck.N_COMPOSITIONS == 11
    assert len(deck.BASE) == 11
    assert all(len(composition) == 2 for composition in deck.BASE)
    assert all(len(group) == len(deck.BASE_FIELDS) for c in deck.BASE for group in c)


@pytest.mark.parametrize(
    "table",
    ["BORON_DELTA", "FUEL_TEMPERATURE_DELTA", "MODERATOR_TEMPERATURE_DELTA",
     "COOLANT_DENSITY_DELTA"],
)
def test_every_derivative_table_covers_the_whole_library(table):
    values = getattr(deck, table)
    assert len(values) == deck.N_COMPOSITIONS
    assert all(len(composition) == deck.N_GROUPS for composition in values)
    assert all(
        len(group) == len(deck.DELTA_FIELDS) for c in values for group in c
    )


def test_the_reflector_compositions_feel_no_doppler():
    """Compositions 1 to 3 carry no fuel, so sqrt(T) cannot move them."""
    for index in range(3):
        assert deck.FUEL_TEMPERATURE_DELTA[index] == [[0.0] * 6, [0.0] * 6]


def test_the_fuel_compositions_do_feel_doppler():
    for index in range(3, deck.N_COMPOSITIONS):
        absorption = deck.FUEL_TEMPERATURE_DELTA[index][0][1]
        assert absorption > 0.0


def test_the_expansion_reference_points_are_the_decks_own():
    assert deck.BORON_REFERENCE == 1200.2
    assert deck.FUEL_TEMPERATURE_REFERENCE == 891.45
    assert deck.MODERATOR_TEMPERATURE_REFERENCE == 579.75
    assert deck.COOLANT_DENSITY_REFERENCE == 0.7125


def test_the_active_core_is_the_height_the_specification_states():
    """NEACRP-L-335 Section 2.1: active core 367.3 cm, plus two reflectors."""
    dz = deck.QUARTER_GEOMETRY["dz"]
    assert len(dz) == 18
    assert sum(dz) == pytest.approx(427.3)
    assert sum(dz[1:-1]) == pytest.approx(367.3)
    assert dz[0] == dz[-1] == 30.0


def test_the_half_core_has_the_same_axial_mesh_as_the_quarter():
    assert deck.HALF_GEOMETRY["dz"] == deck.QUARTER_GEOMETRY["dz"]
    assert deck.HALF_GEOMETRY["assignment"] == deck.QUARTER_GEOMETRY["assignment"]


def test_the_quarter_core_carries_its_symmetry_on_two_faces():
    """A half-width assembly on each symmetry cut, full width elsewhere."""
    geometry = deck.QUARTER_GEOMETRY
    assert geometry["shape"] == (9, 9, 18)
    assert geometry["dx"][0] == pytest.approx(10.803)
    assert geometry["dx"][1:] == [pytest.approx(21.606)] * 8
    assert geometry["dy"][-1] == pytest.approx(10.803)
    assert geometry["dy"][:-1] == [pytest.approx(21.606)] * 8


def test_the_half_core_is_twice_the_quarter_across():
    geometry = deck.HALF_GEOMETRY
    assert geometry["shape"] == (17, 9, 18)
    assert geometry["dx"] == [pytest.approx(21.606)] * 17


def test_the_outer_faces_lose_their_neutrons():
    """NEACRP-L-335 Section 2.2: flux vanishing at the outer reflector surface.

    KOMODO codes are 0 zero flux, 1 vacuum, 2 reflective, given as east,
    west, north, south, bottom, top.
    """
    assert deck.QUARTER_GEOMETRY["boundaries"] == [0, 2, 2, 0, 0, 0]
    assert deck.HALF_GEOMETRY["boundaries"] == [0, 0, 2, 0, 0, 0]


def test_there_are_three_planar_types_over_eighteen_layers():
    geometry = deck.QUARTER_GEOMETRY
    assert len(geometry["planar"]) == 3
    assert len(geometry["assignment"]) == 18
    assert set(geometry["assignment"]) == {1, 2, 3}
    for plane in geometry["planar"]:
        assert len(plane) == 9
        assert all(len(row) == 9 for row in plane)


def test_every_composition_used_by_the_map_exists_in_the_library():
    for label in ("QUARTER_GEOMETRY", "HALF_GEOMETRY"):
        for plane in getattr(deck, label)["planar"]:
            used = {value for row in plane for value in row}
            assert used <= set(range(deck.N_COMPOSITIONS + 1))
            assert used - {0}


@pytest.mark.parametrize("case", ALL_CASES)
def test_the_pin_geometry_matches_table_two_seven(case):
    """NEACRP-L-335 Table 2.7, data of the subassembly geometry."""
    card = deck.CASES[case]
    assert card["fuel_radius"] * 2.0 * 1e3 == pytest.approx(8.239, abs=1e-6)
    assert card["clad_thickness"] * 1e3 == pytest.approx(0.571, abs=1e-6)
    assert card["pin_pitch"] * 1e3 == pytest.approx(12.655, abs=1e-6)
    assert card["n_pins"] == 264
    assert card["n_guide_tubes"] == 25

    clad_outer = card["fuel_radius"] + card["gap_thickness"] + card["clad_thickness"]
    assert clad_outer * 2.0 * 1e3 == pytest.approx(9.517, abs=1e-6)


@pytest.mark.parametrize("case", ALL_CASES)
def test_the_energy_split_is_the_one_the_specification_states(case):
    """NEACRP-L-335 Section 2.9: 98.1% in the fuel, 1.9% in the coolant."""
    assert deck.CASES[case]["direct_heating"] == pytest.approx(0.019)


@pytest.mark.parametrize("case", ALL_CASES)
def test_the_inlet_is_the_specification_temperature(case):
    """NEACRP-L-335 Table 2.8: core inlet 286 C."""
    assert deck.CASES[case]["inlet_temperature"] - 273.15 == pytest.approx(286.0)


@pytest.mark.parametrize("case", ALL_CASES)
def test_the_modelled_fraction_scales_up_to_the_whole_core(case):
    """NEACRP-L-335 Table 2.8: core thermal output 2775 MW."""
    card = deck.CASES[case]
    fraction = 4 if card["geometry"] == "QUARTER" else 2
    assert card["power"] * fraction == pytest.approx(2775.0e6)


@pytest.mark.parametrize("case", ALL_CASES)
def test_the_rod_travel_is_the_one_the_specification_states(case):
    """NEACRP-L-335 Section 2.1: lower absorber edge 37.7 cm inserted,
    401.183 cm withdrawn, over 228 steps."""
    card = deck.CASES[case]
    assert card["zero_position"] == pytest.approx(37.7)
    assert card["max_steps"] == 228
    withdrawn = card["zero_position"] + card["max_steps"] * card["step_size"]
    assert withdrawn == pytest.approx(401.183, abs=1e-3)


@pytest.mark.parametrize("case", ALL_CASES)
def test_every_bank_sits_within_its_travel(case):
    card = deck.CASES[case]
    assert len(card["positions"]) == card["n_banks"]
    assert all(0.0 <= p <= card["max_steps"] for p in card["positions"])


def test_the_hot_zero_power_cases_are_at_essentially_no_power():
    for case in ("A1", "B1", "C1"):
        assert deck.CASES[case]["percent"] == pytest.approx(1.0e-4)
    for case in ("A2", "B2", "C2"):
        assert deck.CASES[case]["percent"] == pytest.approx(100.0)


def test_the_bank_map_covers_the_radial_plane():
    for case in QUARTER_CASES:
        bank_map = np.array(deck.CASES[case]["bank_map"])
        assert bank_map.shape == (9, 9)
        assert bank_map.max() <= deck.CASES[case]["n_banks"]
        assert bank_map.min() >= 0


def test_the_rod_increments_darken_the_thermal_group():
    """A control assembly can only add thermal absorption."""
    for case in ALL_CASES:
        for composition in deck.CASES[case]["rod_delta"]:
            assert composition[1][1] >= 0.0


def test_the_reference_covers_every_case():
    assert sorted(reference.INITIAL_STEADY_STATE) == sorted(ALL_CASES)
    assert sorted(reference.FINAL_STEADY_STATE) == sorted(ALL_CASES)


def test_the_reference_mesh_is_finer_than_the_deck_runs():
    """The reference is 4 radial nodes per assembly; the deck's map is 2x2."""
    assert reference.REFERENCE_MESH == (4, 16)


def test_the_hot_zero_power_reference_sits_at_the_inlet():
    """Nothing is heating, so the Doppler temperature is the coolant's."""
    for case in ("A1", "B1", "C1"):
        row = reference.INITIAL_STEADY_STATE[case]
        assert row["t_doppler"] == pytest.approx(286.0)
        assert row["t_centre"] == pytest.approx(286.0)
        assert row["power"] == pytest.approx(1.0e-6)


def test_the_full_power_reference_is_hot_in_the_middle():
    for case in ("A2", "B2", "C2"):
        row = reference.INITIAL_STEADY_STATE[case]
        assert row["t_centre"] > 1500.0
        assert 540.0 < row["t_doppler"] < 560.0
        assert row["power"] == pytest.approx(1.0)


def test_the_two_full_power_octant_cases_share_an_initial_state():
    """A2 and C2 differ only in which assembly is ejected, not in the state.

    Quarter core and half core, so this is also a check that two different
    geometries describe the same reactor.
    """
    a2 = reference.INITIAL_STEADY_STATE["A2"]
    c2 = reference.INITIAL_STEADY_STATE["C2"]
    for field in ("boron_ppm", "power", "f_xy", "f_q", "t_doppler", "t_centre"):
        assert a2[field] == c2[field]
    assert a2["rod_worth_pcm"] != c2["rod_worth_pcm"]


def test_the_reference_boron_is_in_the_range_the_expansion_covers():
    """Every case sits near the 1200.2 ppm the cross sections expand about."""
    for row in reference.INITIAL_STEADY_STATE.values():
        assert 500.0 < row["boron_ppm"] < 1300.0


def test_a_rod_ejection_is_worth_more_at_zero_power():
    for hot, cold in (("A2", "A1"), ("B2", "B1"), ("C2", "C1")):
        assert (
            reference.INITIAL_STEADY_STATE[cold]["rod_worth_pcm"]
            > reference.INITIAL_STEADY_STATE[hot]["rod_worth_pcm"]
        )


def test_the_two_geometries_describe_the_same_reactor():
    """The half core is the quarter reflected across its west face."""
    quarter, half = deck.QUARTER_GEOMETRY, deck.HALF_GEOMETRY
    assert sum(half["dx"]) == pytest.approx(2.0 * sum(quarter["dx"]))
    assert sum(half["dy"]) == pytest.approx(sum(quarter["dy"]))
    assert half["dy"] == quarter["dy"]
