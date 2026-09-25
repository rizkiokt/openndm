"""Radial fuel pin conduction and the Doppler temperature (FR-TH-2, FR-TH-4)."""

from __future__ import annotations

import numpy as np
import pytest

import openndm
from openndm.channel import ChannelModel, PinGeometry
from openndm.pin import PinConduction, PinState

COOLANT = 580.0
FUEL_K = 3.0
CLAD_K = 15.0
GAP_H = 1.0e4
FILM_H = 3.0e4
RATE = 2.0e4

PINS = PinGeometry(
    fuel_radius=4.11950e-3,
    gap_thickness=6.8e-5,
    clad_thickness=5.71e-4,
    pin_pitch=1.2655e-2,
    n_pins=264,
    n_guide_tubes=25,
)


def conduction(**kwargs):
    settings = {
        "fuel_conductivity": FUEL_K,
        "clad_conductivity": CLAD_K,
        "gap_conductance": GAP_H,
        "film_coefficient": FILM_H,
    }
    settings.update(kwargs)
    return PinConduction(PINS, **settings)


def solved(model=None, rate=RATE, coolant=COOLANT):
    model = conduction() if model is None else model
    return model.solve(np.array([rate]), np.array([coolant]))


def test_the_pin_api_is_importable_from_the_package():
    assert openndm.PinConduction is PinConduction
    assert openndm.PinState is PinState


def test_zero_power_leaves_every_radius_at_the_coolant_temperature():
    state = solved(rate=0.0)
    assert np.all(state.profile == COOLANT)
    assert float(state.clad_inner[0]) == COOLANT
    assert float(state.clad_outer[0]) == COOLANT
    assert float(state.average[0]) == COOLANT
    assert float(state.doppler[0]) == COOLANT


def test_the_pellet_profile_is_the_analytic_parabola_at_every_ring():
    model = conduction(n_rings=16)
    state = solved(model)

    generation = RATE / (np.pi * PINS.fuel_radius**2)
    radii = model.radii
    expected = state.surface[0] + generation * (
        PINS.fuel_radius**2 - radii**2
    ) / (4.0 * FUEL_K)
    assert state.profile[0] == pytest.approx(expected, rel=1.0e-13)


def test_the_centre_to_surface_rise_is_the_closed_form():
    state = solved()
    rise = float(state.centre[0] - state.surface[0])
    assert rise == pytest.approx(RATE / (4.0 * np.pi * FUEL_K), rel=1.0e-13)


@pytest.mark.parametrize("n_rings", [1, 2, 5, 40, 200])
def test_a_constant_conductivity_is_exact_at_any_ring_count(n_rings):
    state = solved(conduction(n_rings=n_rings))
    rise = float(state.centre[0] - state.surface[0])
    assert rise == pytest.approx(RATE / (4.0 * np.pi * FUEL_K), rel=1.0e-13)


def test_doubling_the_conductivity_halves_the_centre_to_surface_rise():
    single = solved()
    double = solved(conduction(fuel_conductivity=2.0 * FUEL_K))

    assert float(double.centre[0] - double.surface[0]) == pytest.approx(
        0.5 * float(single.centre[0] - single.surface[0])
    )


def test_the_gap_carries_the_jump_its_conductance_implies():
    state = solved()
    jump = float(state.surface[0] - state.clad_inner[0])
    expected = RATE / (2.0 * np.pi * PINS.fuel_radius * GAP_H)
    assert jump == pytest.approx(expected, rel=1.0e-13)


def test_the_cladding_carries_a_logarithmic_drop():
    state = solved()
    drop = float(state.clad_inner[0] - state.clad_outer[0])
    ratio = np.log(PINS.clad_outer_radius / PINS.clad_inner_radius)
    assert drop == pytest.approx(RATE * ratio / (2.0 * np.pi * CLAD_K), rel=1.0e-13)


def test_the_film_carries_the_rise_its_coefficient_implies():
    state = solved()
    rise = float(state.clad_outer[0] - COOLANT)
    expected = RATE / (2.0 * np.pi * PINS.clad_outer_radius * FILM_H)
    assert rise == pytest.approx(expected, rel=1.0e-13)


def test_the_four_resistances_add_up_to_the_centreline():
    state = solved()
    pellet = RATE / (4.0 * np.pi * FUEL_K)
    gap = RATE / (2.0 * np.pi * PINS.fuel_radius * GAP_H)
    ratio = np.log(PINS.clad_outer_radius / PINS.clad_inner_radius)
    clad = RATE * ratio / (2.0 * np.pi * CLAD_K)
    film = RATE / (2.0 * np.pi * PINS.clad_outer_radius * FILM_H)

    assert float(state.centre[0]) == pytest.approx(
        COOLANT + pellet + gap + clad + film, rel=1.0e-13
    )


def test_every_temperature_rise_is_linear_in_the_linear_heat_rate():
    single = solved(rate=RATE)
    double = solved(rate=2.0 * RATE)

    assert float(double.centre[0] - COOLANT) == pytest.approx(
        2.0 * float(single.centre[0] - COOLANT), rel=1.0e-13
    )


def test_the_volume_average_sits_midway_between_centre_and_surface():
    state = solved()
    midway = 0.5 * (float(state.centre[0]) + float(state.surface[0]))
    assert float(state.average[0]) == pytest.approx(midway, rel=1.0e-13)


def test_the_volume_average_weighting_reproduces_the_volume_average():
    state = solved(conduction(doppler_weight=0.5))
    assert float(state.doppler[0]) == pytest.approx(
        float(state.average[0]), rel=1.0e-13
    )


def test_the_doppler_weighting_is_the_users_to_choose():
    for weight in (0.0, 0.3, 0.7, 1.0):
        state = solved(conduction(doppler_weight=weight))
        expected = weight * float(state.surface[0]) + (1.0 - weight) * float(
            state.centre[0]
        )
        assert float(state.doppler[0]) == pytest.approx(expected, rel=1.0e-13)


def test_the_default_weighting_is_the_convention_the_requirement_names():
    state = solved()
    expected = 0.7 * float(state.surface[0]) + 0.3 * float(state.centre[0])
    assert float(state.doppler[0]) == pytest.approx(expected, rel=1.0e-13)


def test_a_conductivity_correlation_may_be_injected_instead_of_a_constant():
    def rising(temperature):
        return FUEL_K * (1.0 + 1.0e-4 * (np.asarray(temperature) - COOLANT))

    hotter = solved(conduction(fuel_conductivity=FUEL_K, n_rings=64))
    softer = solved(conduction(fuel_conductivity=rising, n_rings=64))

    assert float(softer.centre[0]) < float(hotter.centre[0])
    assert float(softer.surface[0]) == pytest.approx(float(hotter.surface[0]))


def test_a_temperature_dependent_conductivity_converges_with_the_ring_count():
    def rising(temperature):
        return FUEL_K * (1.0 + 1.0e-3 * (np.asarray(temperature) - COOLANT))

    def centre(n_rings):
        model = conduction(fuel_conductivity=rising, n_rings=n_rings)
        return float(solved(model).centre[0])

    reference = centre(8000)
    coarse = abs(centre(25) - reference)
    fine = abs(centre(50) - reference)
    assert coarse / fine == pytest.approx(2.0, rel=0.05)


def test_a_whole_core_of_pins_solves_in_one_call():
    rates = np.linspace(0.0, 3.0e4, 7)
    coolant = np.linspace(560.0, 600.0, 7)
    state = conduction().solve(rates, coolant)

    assert state.profile.shape == (7, 11)
    assert float(state.centre[0]) == pytest.approx(coolant[0])
    assert np.all(np.diff(state.centre) > 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fuel_conductivity": 0.0},
        {"clad_conductivity": -1.0},
        {"gap_conductance": 0.0},
        {"film_coefficient": -1.0},
        {"n_rings": 0},
        {"doppler_weight": -0.1},
        {"doppler_weight": 1.1},
    ],
)
def test_the_conduction_model_rejects_invalid_settings(kwargs):
    with pytest.raises(openndm.InputError):
        conduction(**kwargs)


@pytest.mark.parametrize(
    "rate, coolant",
    [
        (np.zeros(3), np.zeros(4)),
        (np.zeros((2, 2)), np.zeros((2, 2))),
        (np.full(3, -1.0), np.full(3, COOLANT)),
        (np.full(3, np.nan), np.full(3, COOLANT)),
    ],
)
def test_an_impossible_heat_rate_is_an_error(rate, coolant):
    with pytest.raises(openndm.InputError):
        conduction().solve(rate, coolant)


def test_the_pin_objects_are_described_by_their_reprs():
    assert "10 rings" in repr(conduction())
    assert "doppler_weight=0.7" in repr(conduction())
    assert "peak centre" in repr(solved())


def channel_with_conduction(**kwargs):
    geometry = openndm.Geometry.from_lattice(
        np.zeros((10, 1, 1), dtype=int), pitch=(21.606, 21.606, 30.0)
    )
    settings = {
        "mass_flow": 82.12102,
        "inlet_temperature": 560.0,
        "water": openndm.IF97Water(),
        "conduction": conduction(),
    }
    settings.update(kwargs)
    return geometry, ChannelModel(geometry, PINS, **settings)


def test_a_channel_without_conduction_reports_the_coolant_alone():
    geometry, channel = channel_with_conduction(conduction=None)
    channel.set_heat_source(np.full(geometry.n_nodes, 2.0e6))
    channel.solve()

    assert set(channel.get_temperatures()) == {"moderator_temperature"}
    assert channel.pin_state is None


def test_a_channel_with_conduction_reports_the_fuel_as_well():
    geometry, channel = channel_with_conduction()
    channel.set_heat_source(np.full(geometry.n_nodes, 2.0e6))
    channel.solve()

    fields = channel.get_temperatures()
    assert set(fields) == {
        "moderator_temperature",
        "fuel_temperature",
        "doppler_temperature",
    }
    assert np.all(fields["fuel_temperature"] > fields["moderator_temperature"])
    assert np.all(fields["doppler_temperature"] < fields["fuel_temperature"])


def test_the_fuel_starts_at_the_inlet_before_any_power_is_set():
    _, channel = channel_with_conduction()
    fields = channel.get_temperatures()
    assert fields["fuel_temperature"] == pytest.approx(560.0, abs=1.0e-8)
    assert fields["doppler_temperature"] == pytest.approx(560.0, abs=1.0e-8)


def test_direct_heating_cools_the_fuel_without_warming_the_outlet():
    geometry, cold = channel_with_conduction(direct_heating=0.0)
    _, warm = channel_with_conduction(direct_heating=0.019)
    power = np.full(geometry.n_nodes, 2.0e6)

    for channel in (cold, warm):
        channel.set_heat_source(power)
        channel.solve()

    assert float(warm.outlet_temperature[0]) == pytest.approx(
        float(cold.outlet_temperature[0]), rel=1.0e-14
    )
    cold_rise = cold.get_temperatures()["fuel_temperature"] - 560.0
    warm_rise = warm.get_temperatures()["fuel_temperature"] - 560.0
    assert np.all(warm_rise < cold_rise)


def test_a_conduction_model_built_on_other_pins_is_an_error():
    other = PinGeometry(5.0e-3, 1.0e-4, 6.0e-4, 1.3e-2, 264, 25)
    with pytest.raises(openndm.InputError):
        channel_with_conduction(
            conduction=PinConduction(
                other,
                fuel_conductivity=FUEL_K,
                clad_conductivity=CLAD_K,
                gap_conductance=GAP_H,
                film_coefficient=FILM_H,
            )
        )


def test_the_coupled_fields_drive_the_picard_loop():
    geometry, channel = channel_with_conduction()
    seen = {}

    def apply_state(temperatures, densities):
        seen.update(temperatures)

    coupling = openndm.PicardCoupling(channel, apply_state)
    result = coupling.solve(lambda: np.full(geometry.n_nodes, 2.0e6))

    assert result.converged
    assert seen["doppler_temperature"].min() > 560.0
    assert "fuel_temperature" in result.temperatures
