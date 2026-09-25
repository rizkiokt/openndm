"""Single-phase closed-channel coolant model (FR-TH-1)."""

from __future__ import annotations

import numpy as np
import pytest

import openndm
from openndm.channel import ChannelModel, PinGeometry, absolute_power
from openndm.thermal import ThermalSolver

INLET = 560.0
PRESSURE = 15.5e6
SPECIFIC_HEAT = 5000.0
NEACRP_PINS = PinGeometry(
    fuel_radius=4.11950e-3,
    gap_thickness=6.8e-5,
    clad_thickness=5.71e-4,
    pin_pitch=1.2655e-2,
    n_pins=264,
    n_guide_tubes=25,
)


def constant_water():
    """Fixed properties, so the channel has a closed-form answer."""
    return openndm.ConstantWater(
        density=750.0,
        specific_heat=SPECIFIC_HEAT,
        reference_temperature=INLET,
        reference_enthalpy=1.3e6,
    )


def single_channel(n_planes=10, height=30.0):
    """One column of ``n_planes`` nodes, each ``height`` cm tall."""
    return openndm.Geometry.from_lattice(
        np.zeros((n_planes, 1, 1), dtype=int), pitch=(21.606, 21.606, height)
    )


def quarter_core():
    """A map with inactive corners, so channels differ in length."""
    radial = np.array(
        [
            [0, 0, 0],
            [0, 0, openndm.INACTIVE],
            [0, openndm.INACTIVE, openndm.INACTIVE],
        ]
    )
    core = np.repeat(radial[np.newaxis, :, :], 4, axis=0)
    return openndm.Geometry.from_lattice(core, pitch=(21.606, 21.606, 30.0))


def channel_model(geometry, water=None, **kwargs):
    settings = {
        "mass_flow": 82.12102,
        "inlet_temperature": INLET,
        "pressure": PRESSURE,
        "water": constant_water() if water is None else water,
    }
    settings.update(kwargs)
    return ChannelModel(geometry, NEACRP_PINS, **settings)


def test_the_channel_api_is_importable_from_the_package():
    assert openndm.ChannelModel is ChannelModel
    assert openndm.PinGeometry is PinGeometry
    assert openndm.absolute_power is absolute_power


def test_a_channel_model_satisfies_the_thermal_solver_protocol():
    assert isinstance(channel_model(single_channel()), ThermalSolver)


def test_one_column_of_the_map_is_one_channel():
    assert channel_model(quarter_core()).n_channels == 6


def test_the_pin_geometry_stacks_the_radii_outwards():
    assert NEACRP_PINS.clad_inner_radius == pytest.approx(4.18750e-3)
    assert NEACRP_PINS.clad_outer_radius == pytest.approx(4.75850e-3)
    assert NEACRP_PINS.n_rods == 289


def test_the_flow_area_is_the_lattice_cells_less_the_rods():
    cell = 1.2655e-2**2 - np.pi * 4.75850e-3**2
    assert NEACRP_PINS.flow_area == pytest.approx(289 * cell)


def test_only_the_fuel_pins_are_heated_but_every_rod_is_wetted():
    circumference = 2.0 * np.pi * 4.75850e-3
    assert NEACRP_PINS.heated_perimeter == pytest.approx(264 * circumference)
    assert NEACRP_PINS.wetted_perimeter == pytest.approx(289 * circumference)


def test_the_hydraulic_diameter_follows_from_the_area_and_perimeter():
    expected = 4.0 * NEACRP_PINS.flow_area / NEACRP_PINS.wetted_perimeter
    assert NEACRP_PINS.hydraulic_diameter == pytest.approx(expected)


@pytest.mark.parametrize(
    "override",
    [
        {"fuel_radius": 0.0},
        {"gap_thickness": -1.0e-5},
        {"clad_thickness": 0.0},
        {"pin_pitch": -1.0e-2},
        {"n_pins": 0},
        {"n_guide_tubes": -1},
        {"pin_pitch": 5.0e-3},
    ],
)
def test_an_impossible_pin_geometry_is_an_error(override):
    fields = {
        "fuel_radius": 4.11950e-3,
        "gap_thickness": 6.8e-5,
        "clad_thickness": 5.71e-4,
        "pin_pitch": 1.2655e-2,
        "n_pins": 264,
        "n_guide_tubes": 25,
        **override,
    }
    with pytest.raises(openndm.InputError):
        PinGeometry(**fields)


def test_zero_power_leaves_every_node_at_the_inlet():
    channel = channel_model(single_channel())
    channel.set_heat_source(np.zeros(10))
    channel.solve()

    temperatures = channel.get_temperatures()["moderator_temperature"]
    assert np.all(temperatures == INLET)
    assert float(channel.outlet_temperature[0]) == INLET


def test_the_enthalpy_rise_carries_exactly_the_power_put_in():
    geometry = single_channel()
    channel = channel_model(geometry, water=openndm.IF97Water())
    power = np.linspace(1.0e6, 3.0e6, geometry.n_nodes)
    channel.set_heat_source(power)
    channel.solve()

    water = openndm.IF97Water()
    inlet = float(water.enthalpy(PRESSURE, INLET))
    outlet = float(water.enthalpy(PRESSURE, channel.outlet_temperature[0]))
    carried = 82.12102 * (outlet - inlet)
    assert carried == pytest.approx(float(power.sum()), rel=1.0e-9)


def test_uniform_power_raises_the_temperature_linearly():
    geometry = single_channel()
    channel = channel_model(geometry)
    per_node = 2.0e6
    channel.set_heat_source(np.full(geometry.n_nodes, per_node))
    channel.solve()

    step = per_node / (82.12102 * SPECIFIC_HEAT)
    expected = INLET + step * (np.arange(geometry.n_nodes) + 0.5)
    temperatures = channel.get_temperatures()["moderator_temperature"]
    assert temperatures == pytest.approx(expected, rel=1.0e-12)


def test_a_cosine_power_shape_follows_the_integral_of_the_cosine():
    geometry = single_channel(n_planes=20)
    channel = channel_model(geometry)

    edges = np.linspace(-0.5, 0.5, geometry.n_nodes + 1)
    integral = np.sin(np.pi * edges) / np.pi
    per_node = 6.0e7 * np.diff(integral) / (integral[-1] - integral[0])
    channel.set_heat_source(per_node)
    channel.solve()

    below = np.concatenate(([0.0], np.cumsum(per_node)[:-1]))
    expected = INLET + (below + 0.5 * per_node) / (82.12102 * SPECIFIC_HEAT)
    temperatures = channel.get_temperatures()["moderator_temperature"]
    assert temperatures == pytest.approx(expected, rel=1.0e-12)


def test_the_outlet_does_not_depend_on_how_finely_the_channel_is_meshed():
    outlets = []
    for n_planes in (5, 10, 40):
        geometry = single_channel(n_planes=n_planes, height=300.0 / n_planes)
        channel = channel_model(geometry, water=openndm.IF97Water())
        channel.set_heat_source(np.full(n_planes, 2.0e7 / n_planes))
        channel.solve()
        outlets.append(float(channel.outlet_temperature[0]))
    assert outlets[1] == pytest.approx(outlets[0], rel=1.0e-12)
    assert outlets[2] == pytest.approx(outlets[0], rel=1.0e-12)


def test_direct_heating_moves_power_off_the_pin_without_warming_the_outlet():
    geometry = single_channel()
    power = np.full(geometry.n_nodes, 2.0e6)

    outlets = []
    rates = []
    for fraction in (0.0, 0.019, 0.5):
        channel = channel_model(geometry, direct_heating=fraction)
        channel.set_heat_source(power)
        channel.solve()
        outlets.append(float(channel.outlet_temperature[0]))
        rates.append(float(channel.linear_heat_rate[0]))

    assert outlets[1] == pytest.approx(outlets[0], rel=1.0e-14)
    assert outlets[2] == pytest.approx(outlets[0], rel=1.0e-14)
    assert rates[1] == pytest.approx(0.981 * rates[0])
    assert rates[2] == pytest.approx(0.500 * rates[0])


def test_the_linear_heat_rate_is_the_pin_power_over_the_pin_length():
    geometry = single_channel(n_planes=4, height=25.0)
    channel = channel_model(geometry)
    channel.set_heat_source(np.full(4, 1.0e6))
    channel.solve()

    expected = 1.0e6 / (264 * 0.25)
    assert channel.linear_heat_rate == pytest.approx(np.full(4, expected))


def test_each_channel_sees_only_its_own_power():
    geometry = quarter_core()
    channel = channel_model(geometry)
    power = np.zeros(geometry.n_nodes)
    heated = geometry.lattice_to_node.reshape(geometry.shape)[:, 0, 0]
    power[heated] = 2.0e6
    channel.set_heat_source(power)
    channel.solve()

    rises = channel.outlet_temperature - INLET
    assert rises[0] == pytest.approx(4 * 2.0e6 / (82.12102 * SPECIFIC_HEAT))
    assert np.all(rises[1:] == 0.0)


def test_a_colder_channel_is_the_one_carrying_more_flow():
    geometry = quarter_core()
    flows = np.full(6, 80.0)
    flows[0] = 160.0
    channel = channel_model(geometry, mass_flow=flows)
    channel.set_heat_source(np.full(geometry.n_nodes, 2.0e6))
    channel.solve()

    rises = channel.outlet_temperature - INLET
    assert rises[0] == pytest.approx(0.5 * rises[1])
    assert channel.mass_flow.tolist() == flows.tolist()


def test_density_falls_as_the_coolant_heats_up():
    geometry = single_channel()
    channel = channel_model(geometry, water=openndm.IF97Water())
    channel.set_heat_source(np.full(geometry.n_nodes, 2.0e6))
    channel.solve()

    densities = channel.get_densities()["moderator_density"]
    assert np.all(np.diff(densities) < 0.0)
    assert 600.0 < densities.min() < densities.max() < 800.0


def test_the_reported_state_does_not_alias_the_model_buffers():
    channel = channel_model(single_channel())
    channel.set_heat_source(np.full(10, 1.0e6))
    channel.solve()

    temperatures = channel.get_temperatures()["moderator_temperature"]
    temperatures[:] = 0.0
    assert np.all(channel.get_temperatures()["moderator_temperature"] > INLET)

    densities = channel.get_densities()["moderator_density"]
    densities[:] = 0.0
    assert np.all(channel.get_densities()["moderator_density"] > 0.0)


def test_the_mass_flux_is_the_flow_over_the_free_area():
    channel = channel_model(single_channel())
    expected = 82.12102 / NEACRP_PINS.flow_area
    assert float(channel.mass_flux[0]) == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass_flow": 0.0},
        {"mass_flow": -1.0},
        {"mass_flow": [1.0, 2.0]},
        {"inlet_temperature": 0.0},
        {"pressure": 0.0},
        {"direct_heating": -0.1},
        {"direct_heating": 1.1},
    ],
)
def test_the_model_rejects_invalid_settings(kwargs):
    with pytest.raises(openndm.InputError):
        channel_model(single_channel(), **kwargs)


@pytest.mark.parametrize(
    "power",
    [np.zeros(3), np.full(10, -1.0), np.full(10, np.nan)],
)
def test_an_impossible_heat_source_is_an_error(power):
    channel = channel_model(single_channel())
    with pytest.raises(openndm.InputError):
        channel.set_heat_source(power)


def test_a_channel_is_described_by_its_repr():
    text = repr(channel_model(quarter_core()))
    assert "6 channels" in text
    assert "560 K" in text
    assert "15.5 MPa" in text


def test_absolute_power_weights_by_volume_and_keeps_the_total():
    relative = np.array([0.5, 1.0, 1.5])
    volumes = np.array([100.0, 200.0, 300.0])
    watts = absolute_power(relative, volumes, 3.0e9)

    assert float(watts.sum()) == pytest.approx(3.0e9)
    assert watts[2] / watts[0] == pytest.approx(9.0)


def test_absolute_power_scales_with_the_percent_asked_for():
    relative = np.ones(4)
    volumes = np.ones(4)
    assert float(absolute_power(relative, volumes, 1.0e9, 0.0001).sum()) == (
        pytest.approx(1.0e3)
    )


@pytest.mark.parametrize(
    "args",
    [
        (np.ones(3), np.ones(4), 1.0e9, 100.0),
        (np.ones(3), np.ones(3), -1.0, 100.0),
        (np.ones(3), np.ones(3), 1.0e9, 101.0),
        (np.zeros(3), np.ones(3), 1.0e9, 100.0),
    ],
)
def test_absolute_power_rejects_an_impossible_request(args):
    with pytest.raises(openndm.InputError):
        absolute_power(*args)


def test_a_solved_channel_drives_the_picard_loop():
    geometry = single_channel()
    channel = channel_model(geometry, water=openndm.IF97Water())
    seen = {}

    def apply_state(temperatures, densities):
        seen["moderator_temperature"] = temperatures["moderator_temperature"]
        seen["moderator_density"] = densities["moderator_density"]

    coupling = openndm.PicardCoupling(channel, apply_state)
    result = coupling.solve(lambda: np.full(geometry.n_nodes, 2.0e6))

    assert result.converged
    assert result.iterations == 1
    assert seen["moderator_temperature"].min() > INLET
    assert seen["moderator_density"].max() < 800.0


#: NEACRP-L-335 Table 2.7 gives the guide tube its own diameter, 12.259 mm.
NEACRP_GUIDE_TUBE_RADIUS = 12.259e-3 / 2
NEACRP_SPEC_PINS = PinGeometry(
    fuel_radius=4.11950e-3,
    gap_thickness=6.8e-5,
    clad_thickness=5.71e-4,
    pin_pitch=1.2655e-2,
    n_pins=264,
    n_guide_tubes=25,
    guide_tube_radius=NEACRP_GUIDE_TUBE_RADIUS,
)


def test_a_guide_tube_defaults_to_the_cladding_radius():
    assert NEACRP_PINS.guide_tube_radius is None
    assert NEACRP_PINS.guide_tube_outer_radius == NEACRP_PINS.clad_outer_radius


def test_the_default_leaves_every_derived_quantity_where_it_was():
    cell = 1.2655e-2**2 - np.pi * 4.75850e-3**2
    circumference = 2.0 * np.pi * 4.75850e-3
    assert NEACRP_PINS.flow_area == pytest.approx(289 * cell)
    assert NEACRP_PINS.wetted_perimeter == pytest.approx(289 * circumference)


def test_a_fatter_guide_tube_displaces_more_coolant():
    assert NEACRP_SPEC_PINS.guide_tube_outer_radius == pytest.approx(
        NEACRP_GUIDE_TUBE_RADIUS
    )
    assert NEACRP_SPEC_PINS.flow_area < NEACRP_PINS.flow_area
    assert NEACRP_SPEC_PINS.wetted_perimeter > NEACRP_PINS.wetted_perimeter
    assert NEACRP_SPEC_PINS.hydraulic_diameter < NEACRP_PINS.hydraulic_diameter


def test_the_specification_geometry_gives_the_area_the_specification_states():
    assert NEACRP_SPEC_PINS.flow_area * 1.0e4 == pytest.approx(245.523, abs=1.0e-3)
    assert NEACRP_SPEC_PINS.hydraulic_diameter * 1.0e3 == pytest.approx(
        11.0895, abs=1.0e-4
    )


def test_only_the_fuel_pins_are_heated_whatever_the_guide_tube_is():
    assert NEACRP_SPEC_PINS.heated_perimeter == pytest.approx(
        NEACRP_PINS.heated_perimeter
    )


def test_the_two_rod_populations_are_counted_separately():
    pins = 264 * (1.2655e-2**2 - np.pi * 4.75850e-3**2)
    tubes = 25 * (1.2655e-2**2 - np.pi * NEACRP_GUIDE_TUBE_RADIUS**2)
    assert NEACRP_SPEC_PINS.flow_area == pytest.approx(pins + tubes)

    wetted = 2.0 * np.pi * (264 * 4.75850e-3 + 25 * NEACRP_GUIDE_TUBE_RADIUS)
    assert NEACRP_SPEC_PINS.wetted_perimeter == pytest.approx(wetted)


def test_a_channel_with_no_guide_tubes_does_not_care_about_their_radius():
    bare = PinGeometry(4.11950e-3, 6.8e-5, 5.71e-4, 1.2655e-2, 264, 0)
    fat = PinGeometry(
        4.11950e-3, 6.8e-5, 5.71e-4, 1.2655e-2, 264, 0, guide_tube_radius=6.0e-3
    )
    assert fat.flow_area == bare.flow_area
    assert fat.wetted_perimeter == bare.wetted_perimeter
    assert fat.hydraulic_diameter == bare.hydraulic_diameter


@pytest.mark.parametrize("radius", [0.0, -1.0e-3, 7.0e-3])
def test_an_impossible_guide_tube_is_an_error(radius):
    with pytest.raises(openndm.InputError):
        PinGeometry(
            4.11950e-3, 6.8e-5, 5.71e-4, 1.2655e-2, 264, 25,
            guide_tube_radius=radius,
        )


def test_a_channel_accepts_the_specification_geometry():
    geometry = single_channel()
    model = ChannelModel(
        geometry,
        NEACRP_SPEC_PINS,
        mass_flow=82.12102,
        inlet_temperature=INLET,
        pressure=PRESSURE,
        water=constant_water(),
    )
    expected = 82.12102 / NEACRP_SPEC_PINS.flow_area
    assert float(model.mass_flux[0]) == pytest.approx(expected)
