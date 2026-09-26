"""Water and steam property backends (FR-TH-3, FR-TH-8)."""

from __future__ import annotations

import numpy as np
import pytest

import openndm
import openndm.water
from openndm.water import ConstantWater, WaterProperties

PWR_PRESSURE = 15.5e6


def test_a_backend_satisfies_the_protocol():
    assert isinstance(ConstantWater(), WaterProperties)
    assert not isinstance(object(), WaterProperties)


@pytest.mark.parametrize(
    "kwargs", [{"density": 0.0}, {"density": -1.0}, {"specific_heat": 0.0}]
)
def test_constant_water_rejects_unphysical_constants(kwargs):
    with pytest.raises(openndm.InputError):
        ConstantWater(**kwargs)


def test_constant_water_reproduces_its_own_constants():
    water = ConstantWater(density=812.0, specific_heat=5321.0)
    assert float(water.density(PWR_PRESSURE, 580.0)) == 812.0
    assert float(water.saturation_temperature(PWR_PRESSURE)) == 618.0


def test_constant_water_enthalpy_is_linear_in_temperature():
    water = ConstantWater(
        specific_heat=5000.0, reference_temperature=560.0, reference_enthalpy=1.3e6
    )
    assert float(water.enthalpy(PWR_PRESSURE, 560.0)) == 1.3e6
    assert float(water.enthalpy(PWR_PRESSURE, 561.0)) == 1.3e6 + 5000.0


def test_constant_water_temperature_inverts_enthalpy_exactly():
    water = ConstantWater()
    temperatures = np.linspace(500.0, 610.0, 23)
    enthalpies = water.enthalpy(PWR_PRESSURE, temperatures)
    np.testing.assert_allclose(
        water.temperature(PWR_PRESSURE, enthalpies), temperatures, rtol=1e-14
    )


def test_constant_water_broadcasts_over_arrays():
    water = ConstantWater()
    pressures = np.full(7, PWR_PRESSURE)
    temperatures = np.linspace(550.0, 600.0, 7)

    assert water.density(pressures, temperatures).shape == (7,)
    assert water.enthalpy(pressures, temperatures).shape == (7,)
    assert water.saturation_temperature(pressures).shape == (7,)


def test_constant_water_returns_a_fresh_array_each_call():
    water = ConstantWater()
    first = water.density(np.full(4, PWR_PRESSURE), np.full(4, 580.0))
    first[0] = -1.0
    second = water.density(np.full(4, PWR_PRESSURE), np.full(4, 580.0))
    assert second[0] == 750.0


def test_constant_water_is_described_by_its_repr():
    assert "ConstantWater" in repr(ConstantWater())


def has_external_backend():
    try:
        openndm.water.external_backend()
    except openndm.InputError:
        return False
    return True


IF97_TABLE5 = [
    (300.0, 3.0e6, 0.100215168e-2, 0.115331273e3, 0.417301218e1),
    (300.0, 80.0e6, 0.971180894e-3, 0.184142828e3, 0.401008987e1),
    (500.0, 3.0e6, 0.120241800e-2, 0.975542239e3, 0.465580682e1),
]
"""IAPWS R7-97(2012) Table 5: (T [K], p [Pa], v [m^3/kg], h [kJ/kg], cp [kJ/kg K])."""

IF97_TABLE35 = [
    (300.0, 0.353658941e-2),
    (500.0, 0.263889776e1),
    (600.0, 0.123443146e2),
]
"""IAPWS R7-97(2012) Table 35: (T [K], saturation pressure [MPa])."""

IF97_TABLE36 = [
    (0.1, 0.372755919e3),
    (1.0, 0.453035632e3),
    (10.0, 0.584149488e3),
]
"""IAPWS R7-97(2012) Table 36: (p [MPa], saturation temperature [K])."""

IF97_TABLE7 = [
    (3.0, 500.0, 0.391798509e3),
    (80.0, 500.0, 0.378108626e3),
    (80.0, 1500.0, 0.611041229e3),
]
"""IAPWS R7-97(2012) Table 7: (p [MPa], h [kJ/kg], T [K]).

From the backward equation, which the release permits to differ from the basic
equation by up to 25 mK. This package inverts the basic equation instead, so it
is held to that same tolerance rather than to the digits printed.
"""

BACKWARD_TOLERANCE_K = 25.0e-3


@pytest.mark.parametrize(("temperature", "pressure", "v", "h", "cp"), IF97_TABLE5)
def test_if97_reproduces_the_release_verification_values(
    temperature, pressure, v, h, cp
):
    """IAPWS R7-97(2012) Table 5, the release's own program-verification values.

    This is what makes a mistranscribed coefficient visible: the table is
    printed to nine significant figures and a single wrong digit anywhere in
    Table 2 moves one of these.
    """
    water = openndm.IF97Water()
    assert float(water.specific_volume(pressure, temperature)) == pytest.approx(
        v, rel=1e-8
    )
    assert float(water.enthalpy(pressure, temperature)) == pytest.approx(
        h * 1e3, rel=1e-8
    )
    assert float(water.specific_heat(pressure, temperature)) == pytest.approx(
        cp * 1e3, rel=1e-8
    )


@pytest.mark.parametrize(("temperature", "pressure_mpa"), IF97_TABLE35)
def test_if97_saturation_pressure_matches_the_release(temperature, pressure_mpa):
    """IAPWS R7-97(2012) Table 35."""
    water = openndm.IF97Water()
    assert float(water.saturation_pressure(temperature)) == pytest.approx(
        pressure_mpa * 1e6, rel=1e-8
    )


@pytest.mark.parametrize(("pressure_mpa", "temperature"), IF97_TABLE36)
def test_if97_saturation_temperature_matches_the_release(pressure_mpa, temperature):
    """IAPWS R7-97(2012) Table 36."""
    water = openndm.IF97Water()
    assert float(water.saturation_temperature(pressure_mpa * 1e6)) == pytest.approx(
        temperature, rel=1e-8
    )


@pytest.mark.parametrize(("pressure_mpa", "enthalpy", "temperature"), IF97_TABLE7)
def test_if97_inversion_agrees_with_the_release_backward_equation(
    pressure_mpa, enthalpy, temperature
):
    """Within the 25 mK the release itself allows between the two."""
    water = openndm.IF97Water()
    computed = float(water.temperature(pressure_mpa * 1e6, enthalpy * 1e3))
    assert abs(computed - temperature) < BACKWARD_TOLERANCE_K


def test_if97_inversion_is_exact_against_its_own_enthalpy():
    """Newton on the basic equation, so the round trip has no tolerance gap."""
    water = openndm.IF97Water()
    pressures = np.array([1.0e6, 15.5e6, 50.0e6, 90.0e6])
    temperatures = np.array([400.0, 560.0, 600.0, 620.0])

    enthalpies = water.enthalpy(pressures, temperatures)
    np.testing.assert_allclose(
        water.temperature(pressures, enthalpies), temperatures, atol=1e-9
    )


def test_if97_saturation_round_trips():
    temperatures = np.linspace(300.0, 640.0, 29)
    water = openndm.IF97Water()
    pressures = water.saturation_pressure(temperatures)
    np.testing.assert_allclose(
        water.saturation_temperature(pressures), temperatures, rtol=1e-9
    )


def test_if97_density_falls_with_temperature_and_rises_with_pressure():
    water = openndm.IF97Water()
    hotter = water.density(15.5e6, np.array([500.0, 560.0, 600.0]))
    assert np.all(np.diff(hotter) < 0.0)
    squeezed = water.density(np.array([15.5e6, 50.0e6, 90.0e6]), 560.0)
    assert np.all(np.diff(squeezed) > 0.0)


def test_if97_is_in_the_right_place_for_a_pwr():
    """A sanity check a reactor physicist would make without opening a table."""
    water = openndm.IF97Water()
    density = float(water.density(15.5e6, 583.0))
    assert 690.0 < density < 730.0
    assert 615.0 < float(water.saturation_temperature(15.5e6)) < 620.0


@pytest.mark.parametrize(
    ("pressure", "temperature"),
    [
        (15.5e6, 270.0),
        (15.5e6, 700.0),
        (200.0e6, 500.0),
        (-1.0e6, 500.0),
        (1.0e5, 500.0),
    ],
)
def test_if97_refuses_states_outside_region_one(pressure, temperature):
    """Outside region 1 the equation is not merely inaccurate, it is wrong."""
    with pytest.raises(openndm.InputError):
        openndm.IF97Water().density(pressure, temperature)


def test_if97_refuses_a_saturation_state_that_does_not_exist():
    """Above the critical pressure there is no phase boundary to report."""
    water = openndm.IF97Water()
    with pytest.raises(openndm.InputError, match="critical pressure"):
        water.saturation_temperature(50.0e6)
    with pytest.raises(openndm.InputError, match="critical temperature"):
        water.saturation_pressure(700.0)


def test_if97_broadcasts_and_keeps_shape():
    water = openndm.IF97Water()
    pressures = np.full((3, 4), 15.5e6)
    temperatures = np.linspace(500.0, 600.0, 4)

    assert water.density(pressures, temperatures).shape == (3, 4)
    assert water.enthalpy(pressures, temperatures).shape == (3, 4)
    assert water.temperature(
        pressures, water.enthalpy(pressures, temperatures)
    ).shape == (3, 4)


def test_if97_satisfies_the_protocol_and_is_exported():
    assert isinstance(openndm.IF97Water(), WaterProperties)
    assert openndm.IF97Water is openndm.water.IF97Water


@pytest.mark.parametrize("kwargs", [{"tolerance": 0.0}, {"max_iterations": 0}])
def test_if97_rejects_invalid_iteration_settings(kwargs):
    with pytest.raises(openndm.InputError):
        openndm.IF97Water(**kwargs)


def test_an_unknown_external_backend_is_an_error():
    with pytest.raises(openndm.InputError, match="unknown backend"):
        openndm.water.external_backend("steam_tables_from_memory")


@pytest.mark.skipif(
    has_external_backend(), reason="an external property package is installed"
)
def test_a_missing_external_backend_says_so():
    with pytest.raises(openndm.InputError, match="not installed"):
        openndm.water.external_backend("coolprop")


@pytest.mark.skipif(
    not has_external_backend(), reason="needs iapws or CoolProp installed"
)
def test_the_external_backend_agrees_with_the_built_in_formulation():
    """An independent implementation of the same release, as a cross-check.

    The release's own verification values pin three states; this pins the
    whole PWR range against code that shares none of ours.
    """
    external = openndm.water.external_backend()
    builtin = openndm.IF97Water()
    pressures = np.array([1.0e6, 15.5e6, 15.5e6, 20.0e6])
    temperatures = np.array([400.0, 500.0, 583.0, 600.0])

    np.testing.assert_allclose(
        external.density(pressures, temperatures),
        builtin.density(pressures, temperatures),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        external.enthalpy(pressures, temperatures),
        builtin.enthalpy(pressures, temperatures),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        external.saturation_temperature(pressures),
        builtin.saturation_temperature(pressures),
        rtol=1e-9,
    )


IF97_TABLE15 = [
    (300.0, 0.0035e6, 0.394913866e2, 0.254991145e4, 0.191300162e1),
    (700.0, 0.0035e6, 0.923015898e2, 0.333568375e4, 0.208141274e1),
    (700.0, 30.0e6, 0.542946619e-2, 0.263149474e4, 0.103505092e2),
]
"""IAPWS R7-97(2012) Table 15, region 2.

Columns are (T [K], p [Pa], v [m^3/kg], h [kJ/kg], cp [kJ/kg K]).
"""

B23_VERIFICATION = (0.623150000e3, 0.165291643e2)
"""IAPWS R7-97(2012) Section 4: the B23 boundary must pass through this point."""


@pytest.mark.parametrize(("temperature", "pressure", "v", "h", "cp"), IF97_TABLE15)
def test_if97_reproduces_the_region_two_verification_values(
    temperature, pressure, v, h, cp
):
    """IAPWS R7-97(2012) Table 15, the release's own verification values.

    Printed to nine significant figures, so one wrong digit anywhere in the
    fifty-two coefficients of Tables 10 and 11 moves one of these.
    """
    water = openndm.IF97Water()
    assert float(
        water.vapour_specific_volume(pressure, temperature)
    ) == pytest.approx(v, rel=1e-8)
    assert float(water.vapour_enthalpy(pressure, temperature)) == pytest.approx(
        h * 1e3, rel=1e-8
    )
    assert float(
        water.vapour_specific_heat(pressure, temperature)
    ) == pytest.approx(cp * 1e3, rel=1e-8)


def test_the_region_boundary_passes_through_its_verification_point():
    temperature, pressure_mpa = B23_VERIFICATION
    boundary = float(openndm.IF97Water().region23_pressure(temperature))
    assert boundary == pytest.approx(pressure_mpa * 1e6, rel=1e-8)


def test_the_region_boundary_spans_the_range_the_release_states():
    water = openndm.IF97Water()
    low, high = openndm.water.REGION23_TEMPERATURE_RANGE
    assert float(water.region23_pressure(low)) == pytest.approx(16.5292e6, rel=1e-5)
    assert float(water.region23_pressure(high)) == pytest.approx(100.0e6, rel=1e-6)


@pytest.mark.parametrize("temperature", [623.14, 863.16, 300.0, 1073.15])
def test_the_region_boundary_refuses_temperatures_it_does_not_span(temperature):
    with pytest.raises(openndm.InputError, match="region 2/3 boundary"):
        openndm.IF97Water().region23_pressure(temperature)


def test_the_region_three_pressure_is_the_saturation_line_at_region_ones_ceiling():
    water = openndm.IF97Water()
    ceiling = openndm.water.REGION1_TEMPERATURE_RANGE[1]
    assert float(water.saturation_pressure(ceiling)) == pytest.approx(
        openndm.water.SATURATION_REGION3_PRESSURE, rel=1e-12
    )
    assert openndm.water.SATURATION_REGION3_PRESSURE == pytest.approx(
        B23_VERIFICATION[1] * 1e6, rel=1e-8
    )


@pytest.mark.parametrize(
    ("pressure", "temperature"),
    [
        (0.0035e6, 272.0),
        (0.0035e6, 1074.0),
        (101.0e6, 900.0),
        (0.0, 700.0),
        (15.5e6, 500.0),
        (30.0e6, 640.0),
    ],
)
def test_if97_refuses_states_outside_region_two(pressure, temperature):
    with pytest.raises(openndm.InputError):
        openndm.IF97Water().vapour_density(pressure, temperature)


def test_steam_is_lighter_and_hotter_in_enthalpy_than_its_liquid():
    water = openndm.IF97Water()
    for pressure in (0.1e6, 1.0e6, 7.0e6, 15.5e6):
        assert float(water.saturated_vapour_enthalpy(pressure)) > float(
            water.saturated_liquid_enthalpy(pressure)
        )
        assert float(water.saturated_liquid_density(pressure)) > float(
            water.saturated_vapour_density(pressure)
        )
        assert float(water.latent_heat(pressure)) > 0.0


def test_the_latent_heat_shrinks_as_the_critical_point_is_approached():
    water = openndm.IF97Water()
    pressures = np.geomspace(0.1e6, openndm.water.SATURATION_REGION3_PRESSURE, 12)
    assert np.all(np.diff(water.latent_heat(pressures)) < 0.0)


def test_the_saturated_phases_are_the_two_regions_at_the_boiling_point():
    water = openndm.IF97Water()
    pressure = 7.0e6
    boiling = float(water.saturation_temperature(pressure))

    assert float(water.saturated_liquid_enthalpy(pressure)) == pytest.approx(
        float(water.enthalpy(pressure, boiling)), rel=1e-9
    )
    assert float(water.saturated_vapour_enthalpy(pressure)) == pytest.approx(
        float(water.vapour_enthalpy(pressure, boiling)), rel=1e-9
    )


def test_a_pwr_and_a_bwr_sit_where_they_should_on_the_saturation_line():
    water = openndm.IF97Water()
    assert 615.0 < float(water.saturation_temperature(PWR_PRESSURE)) < 620.0
    assert 555.0 < float(water.saturation_temperature(7.0e6)) < 562.0
    assert 30.0 < float(water.saturated_vapour_density(7.0e6)) < 40.0
    assert 700.0 < float(water.saturated_liquid_density(7.0e6)) < 780.0


@pytest.mark.parametrize("pressure", [20.0e6, 22.0e6, 30.0e6])
def test_the_saturation_line_stops_where_region_three_starts(pressure):
    water = openndm.IF97Water()
    for accessor in (
        water.saturated_liquid_enthalpy,
        water.saturated_vapour_enthalpy,
        water.saturated_liquid_density,
        water.saturated_vapour_density,
    ):
        with pytest.raises(openndm.InputError, match="region 3"):
            accessor(pressure)


def test_the_saturated_accessors_broadcast_and_keep_shape():
    water = openndm.IF97Water()
    pressures = np.array([[1.0e6, 3.0e6], [7.0e6, 15.0e6]])
    for values in (
        water.saturated_liquid_enthalpy(pressures),
        water.saturated_vapour_density(pressures),
        water.latent_heat(pressures),
    ):
        assert values.shape == pressures.shape
        assert np.all(np.isfinite(values))


def test_only_a_backend_with_both_phases_satisfies_the_saturation_protocol():
    assert isinstance(openndm.IF97Water(), openndm.water.SaturationProperties)
    assert not isinstance(ConstantWater(), openndm.water.SaturationProperties)
    assert isinstance(ConstantWater(), WaterProperties)
    assert openndm.SaturationProperties is openndm.water.SaturationProperties


def test_the_repr_says_which_regions_are_covered():
    assert "regions 1 and 2" in repr(openndm.IF97Water())


@pytest.mark.skipif(
    not has_external_backend(), reason="needs iapws or CoolProp installed"
)
def test_the_saturation_line_agrees_with_the_external_package():
    iapws = pytest.importorskip("iapws")
    water = openndm.IF97Water()
    pressures = np.geomspace(
        1.0e4, openndm.water.SATURATION_REGION3_PRESSURE, 25
    )
    liquid = [iapws.IAPWS97(P=float(p) / 1e6, x=0.0) for p in pressures]
    vapour = [iapws.IAPWS97(P=float(p) / 1e6, x=1.0) for p in pressures]

    for ours, theirs in (
        (water.saturated_liquid_enthalpy(pressures) / 1e3, [s.h for s in liquid]),
        (water.saturated_vapour_enthalpy(pressures) / 1e3, [s.h for s in vapour]),
        (water.saturated_liquid_density(pressures), [s.rho for s in liquid]),
        (water.saturated_vapour_density(pressures), [s.rho for s in vapour]),
    ):
        assert ours == pytest.approx(np.array(theirs), rel=1e-8)
