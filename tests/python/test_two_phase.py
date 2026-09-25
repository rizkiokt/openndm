"""Homogeneous equilibrium two-phase channel (FR-TH-5)."""

from __future__ import annotations

import numpy as np
import pytest

import openndm
from openndm.channel import ChannelModel, PinGeometry

BWR_PRESSURE = 7.0e6
INLET = 550.0
N_PLANES = 10
#: Per-node power that leaves the outlet below the saturated liquid enthalpy.
SUBCOOLED_POWER = 2.0e4

PINS = PinGeometry(
    fuel_radius=4.11950e-3,
    gap_thickness=6.8e-5,
    clad_thickness=5.71e-4,
    pin_pitch=1.2655e-2,
    n_pins=264,
    n_guide_tubes=25,
)


def geometry(n_planes=N_PLANES):
    return openndm.Geometry.from_lattice(
        np.zeros((n_planes, 1, 1), dtype=int), pitch=(15.0, 15.0, 36.0)
    )


def channel(two_phase=True, mass_flow=15.0, **kwargs):
    settings = {
        "mass_flow": mass_flow,
        "inlet_temperature": INLET,
        "pressure": BWR_PRESSURE,
        "water": openndm.IF97Water(),
        "two_phase": two_phase,
    }
    settings.update(kwargs)
    return ChannelModel(geometry(), PINS, **settings)


def boiling(power=1.5e6, **kwargs):
    model = channel(**kwargs)
    model.set_heat_source(np.full(N_PLANES, power))
    model.solve()
    return model


def saturation():
    water = openndm.IF97Water()
    return (
        float(water.saturated_liquid_enthalpy(BWR_PRESSURE)),
        float(water.saturated_vapour_enthalpy(BWR_PRESSURE)),
        float(water.saturated_liquid_density(BWR_PRESSURE)),
        float(water.saturated_vapour_density(BWR_PRESSURE)),
    )


def test_a_subcooled_channel_is_bit_identical_with_and_without_two_phase():
    power = np.full(N_PLANES, SUBCOOLED_POWER)

    single = channel(two_phase=False)
    single.set_heat_source(power)
    single.solve()

    both = channel(two_phase=True)
    both.set_heat_source(power)
    both.solve()

    assert np.all(both.quality == 0.0)
    assert np.array_equal(
        single.get_temperatures()["moderator_temperature"],
        both.get_temperatures()["moderator_temperature"],
    )
    assert np.array_equal(
        single.get_densities()["moderator_density"],
        both.get_densities()["moderator_density"],
    )


def test_a_single_phase_channel_reports_no_quality_or_void():
    model = channel(two_phase=False)
    model.set_heat_source(np.full(N_PLANES, SUBCOOLED_POWER))
    model.solve()
    assert np.all(model.quality == 0.0)
    assert np.all(model.void_fraction == 0.0)


def test_quality_is_zero_below_saturation_and_rises_above_it():
    model = boiling()
    liquid_enthalpy, _, _, _ = saturation()
    enthalpy = model.enthalpy

    subcooled = enthalpy <= liquid_enthalpy
    assert np.all(model.quality[subcooled] == 0.0)
    assert np.all(model.quality[~subcooled] > 0.0)
    assert np.all(np.diff(model.quality) > 0.0)


def test_void_fraction_is_zero_below_saturation_and_rises_above_it():
    model = boiling()
    assert np.all(model.void_fraction[model.quality == 0.0] == 0.0)
    assert np.all(np.diff(model.void_fraction) > 0.0)
    assert np.all((model.void_fraction >= 0.0) & (model.void_fraction < 1.0))


def test_quality_is_the_enthalpy_fraction_between_the_two_phases():
    model = boiling()
    liquid_enthalpy, vapour_enthalpy, _, _ = saturation()
    expected = np.clip(
        (model.enthalpy - liquid_enthalpy) / (vapour_enthalpy - liquid_enthalpy),
        0.0,
        None,
    )
    assert model.quality == pytest.approx(expected, rel=1.0e-12)


def test_the_mixture_density_is_bracketed_by_the_two_phases():
    model = boiling()
    _, _, liquid_density, vapour_density = saturation()
    density = model.get_densities()["moderator_density"]

    assert np.all(density > vapour_density)
    assert np.all(density <= liquid_density * (1.0 + 1.0e-12))
    assert np.all(np.diff(density) < 0.0)


def test_the_mixture_density_is_the_homogeneous_one_at_unit_slip():
    model = boiling()
    _, _, liquid_density, vapour_density = saturation()
    quality = model.quality
    boiling_nodes = quality > 0.0

    specific_volume = quality / vapour_density + (1.0 - quality) / liquid_density
    expected = 1.0 / specific_volume
    density = model.get_densities()["moderator_density"]
    assert density[boiling_nodes] == pytest.approx(
        expected[boiling_nodes], rel=1.0e-12
    )


def test_the_coolant_stops_warming_once_it_boils():
    model = boiling()
    water = openndm.IF97Water()
    boiling_temperature = float(water.saturation_temperature(BWR_PRESSURE))

    temperatures = model.get_temperatures()["moderator_temperature"]
    hot = model.quality > 0.0
    assert temperatures[hot] == pytest.approx(boiling_temperature, rel=1.0e-12)
    assert np.all(temperatures[~hot] < boiling_temperature)


def test_boiling_does_not_break_the_energy_balance():
    model = boiling()
    liquid_enthalpy, vapour_enthalpy, _, _ = saturation()
    water = openndm.IF97Water()

    inlet = float(water.enthalpy(BWR_PRESSURE, INLET))
    carried = 15.0 * (float(model.outlet_enthalpy[0]) - inlet)
    assert carried == pytest.approx(N_PLANES * 1.5e6, rel=1.0e-12)
    assert liquid_enthalpy < float(model.outlet_enthalpy[0]) < vapour_enthalpy


def test_a_larger_slip_ratio_holds_less_void_at_the_same_quality():
    homogeneous = boiling()
    slipping = boiling(slip_ratio=2.0)

    assert slipping.quality == pytest.approx(homogeneous.quality)
    voided = homogeneous.quality > 0.0
    assert np.all(
        slipping.void_fraction[voided] < homogeneous.void_fraction[voided]
    )


def test_the_slip_ratio_leaves_a_subcooled_channel_alone():
    """Nothing boils, so there is no void for the slip ratio to act on."""
    homogeneous = boiling(power=SUBCOOLED_POWER)
    slipping = boiling(power=SUBCOOLED_POWER, slip_ratio=3.0)
    assert np.array_equal(
        homogeneous.get_densities()["moderator_density"],
        slipping.get_densities()["moderator_density"],
    )


def test_void_fraction_follows_the_homogeneous_relation():
    model = boiling()
    _, _, liquid_density, vapour_density = saturation()
    quality = model.quality
    voided = quality > 0.0

    x = quality[voided]
    expected = 1.0 / (1.0 + (1.0 - x) / x * vapour_density / liquid_density)
    assert model.void_fraction[voided] == pytest.approx(expected, rel=1.0e-12)


def test_more_power_drives_the_outlet_to_a_higher_quality():
    outlets = []
    for power in (0.8e6, 1.2e6, 1.6e6):
        model = boiling(power=power)
        outlets.append(float(model.quality.max()))
    assert outlets == sorted(outlets)
    assert outlets[0] > 0.0
    assert outlets[-1] < 1.0


def test_a_channel_that_boils_dry_is_an_error():
    model = channel()
    model.set_heat_source(np.full(N_PLANES, 1.0e7))
    with pytest.raises(openndm.InputError, match="boils dry"):
        model.solve()


def test_two_phase_needs_a_backend_that_knows_both_sides():
    with pytest.raises(openndm.InputError, match="saturation line"):
        channel(two_phase=True, water=openndm.ConstantWater())


def test_a_single_phase_channel_accepts_any_backend():
    model = channel(two_phase=False, water=openndm.ConstantWater())
    assert not model.two_phase


@pytest.mark.parametrize("slip", [0.0, -1.0])
def test_an_impossible_slip_ratio_is_an_error(slip):
    with pytest.raises(openndm.InputError, match="slip_ratio"):
        channel(slip_ratio=slip)


def test_the_reported_quality_and_void_do_not_alias_the_buffers():
    model = boiling()
    quality = model.quality
    quality[:] = 0.0
    assert np.any(model.quality > 0.0)

    void = model.void_fraction
    void[:] = 0.0
    assert np.any(model.void_fraction > 0.0)


def test_a_boiling_channel_still_drives_the_pin_and_the_picard_loop():
    conduction = openndm.PinConduction(
        PINS,
        fuel_conductivity=3.0,
        clad_conductivity=15.0,
        gap_conductance=1.0e4,
        film_coefficient=3.0e4,
    )
    model = boiling(conduction=conduction)
    fields = model.get_temperatures()

    assert np.all(fields["fuel_temperature"] > fields["moderator_temperature"])

    seen = {}
    coupling = openndm.PicardCoupling(model, lambda t, d: seen.update(d))
    result = coupling.solve(lambda: np.full(N_PLANES, 1.5e6))
    assert result.converged
    assert seen["moderator_density"].min() < 400.0
