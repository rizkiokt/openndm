"""Transient with thermal-hydraulic feedback (FR-MODE-7)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import openndm
from openndm.channel import ChannelModel, PinGeometry
from openndm.pin import PinConduction

PK = {
    "velocity": 2.2e5,
    "absorption": 0.08,
    "nu_fission": 0.1,
    "beta": 0.0065,
    "lambda": 0.0785,
}
"""One-group data for the leakage-free box, as in the V-5 kinetics checks."""

GENERATION = 1.0 / (PK["velocity"] * PK["absorption"])
"""Prompt neutron generation time, s."""

INSERTED = 1.2 * PK["beta"]
"""Reactivity of the step insertion: prompt supercritical, a Doppler burst."""

GAMMA = 1.0e-3
"""Square-root Doppler coefficient on absorption, K^-1/2."""

REFERENCE_TEMPERATURE = 600.0
"""Fuel temperature at which the feedback is zero, K."""

INITIAL_POWER = 1.0
"""Power at the start of the burst, W."""

HEAT_CAPACITY = 100.0
"""Heat capacity of the adiabatic fuel lump, J/K."""

BURST_END = 0.8
"""Time the burst is followed to, past its peak, s."""

ALL_REFLECTIVE = dict.fromkeys(
    ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"), "reflective"
)

PINS = PinGeometry(4.1195e-3, 6.8e-5, 5.71e-4, 1.2655e-2, 264, 25)


class AdiabaticLump:
    """A thermal solver with no heat removal: temperature integrates power."""

    def __init__(self, n_nodes=1):
        self.temperature = np.full(n_nodes, REFERENCE_TEMPERATURE)
        self.power = np.zeros(n_nodes)

    def set_heat_source(self, q):
        self.power = np.asarray(q, dtype=float).copy()

    def solve(self):
        pass

    def advance(self, dt):
        self.temperature = self.temperature + self.power * dt / HEAT_CAPACITY

    def get_temperatures(self):
        return {"fuel_temperature": self.temperature.copy()}

    def get_densities(self):
        return {}


class SteadyOnly:
    """A thermal solver that cannot be stepped in time."""

    def set_heat_source(self, q):
        pass

    def solve(self):
        pass

    def get_temperatures(self):
        return {}

    def get_densities(self):
        return {}


def _one_group(library, absorption):
    library.set_composition(
        0,
        D=[1.0],
        absorption=[absorption],
        nu_fission=[PK["nu_fission"]],
        kappa_fission=[PK["nu_fission"]],
        chi=[1.0],
        scatter=[[0.0]],
        inv_velocity=[1.0 / PK["velocity"]],
    )


def _point_kinetics_model():
    library = openndm.XSLibrary(1, 1)
    _one_group(library, PK["absorption"])
    library.set_delayed(beta=[PK["beta"]], decay_constant=[PK["lambda"]])
    library.finalize(warn=False)
    geometry = openndm.Geometry.from_lattice(
        np.zeros((1, 1, 1), dtype=int), pitch=20.0, boundaries=ALL_REFLECTIVE
    )
    settings = openndm.Settings(
        verbosity=0,
        k_tolerance=1.0e-13,
        fission_source_tolerance=1.0e-12,
        inner_tolerance=1.0e-13,
        theta=1.0,
        max_step_iterations=500,
        step_tolerance=1.0e-14,
    )
    return openndm.Model(geometry, library, settings), library


def _burst(dt):
    """Doppler-terminated burst in the leakage-free box: (times, powers, T)."""
    model, library = _point_kinetics_model()
    model.solve()
    _one_group(library, PK["absorption"] * (1.0 - INSERTED))
    library.finalize(warn=False)
    doppler = openndm.DopplerFeedback(
        library, [0], gamma=GAMMA, reference_temperature=REFERENCE_TEMPERATURE
    )
    lump = AdiabaticLump()
    coupled = model.start_coupled_transient(
        lump,
        lambda temperatures, _: doppler.apply(temperatures["fuel_temperature"]),
        total_power=INITIAL_POWER,
    )
    times, powers = [0.0], [INITIAL_POWER]
    for _ in range(round(BURST_END / dt)):
        record = coupled.step(dt)
        times.append(record.time)
        powers.append(record.power)
    return np.array(times), np.array(powers), float(lump.temperature[0])


def _reference_burst(h=1.0e-5):
    """Point kinetics with adiabatic Doppler feedback, by RK4: (peak, T end).

    In the leakage-free box the spatial solve is point kinetics exactly, and
    the absorption scaling makes the reactivity
    1 - (1 - rho_inserted)(1 + gamma (sqrt T - sqrt T0)).
    """

    def reactivity(temperature):
        doppler = 1.0 + GAMMA * (
            np.sqrt(temperature) - np.sqrt(REFERENCE_TEMPERATURE)
        )
        return 1.0 - (1.0 - INSERTED) * doppler

    def rate(state):
        power, precursors, temperature = state
        return np.array(
            [
                (reactivity(temperature) - PK["beta"]) / GENERATION * power
                + PK["lambda"] * precursors,
                PK["beta"] / GENERATION * power - PK["lambda"] * precursors,
                INITIAL_POWER * power / HEAT_CAPACITY,
            ]
        )

    state = np.array(
        [1.0, PK["beta"] / (GENERATION * PK["lambda"]), REFERENCE_TEMPERATURE]
    )
    peak = 1.0
    for _ in range(round(BURST_END / h)):
        k1 = rate(state)
        k2 = rate(state + 0.5 * h * k1)
        k3 = rate(state + 0.5 * h * k2)
        k4 = rate(state + h * k3)
        state = state + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        peak = max(peak, state[0])
    return peak * INITIAL_POWER, state[2]


@pytest.mark.slow
def test_a_doppler_burst_converges_on_point_kinetics_at_first_order():
    """The feedback is lagged one step, so the driver is first order in dt.

    A prompt-supercritical insertion in a leakage-free box with an adiabatic
    fuel lump is point kinetics with Doppler feedback exactly, so an RK4
    integration of the three ODEs is the reference.
    """
    peak, final_temperature = _reference_burst()
    peak_errors, temperature_errors = [], []
    for dt in (4.0e-3, 2.0e-3, 1.0e-3):
        _, powers, temperature = _burst(dt)
        peak_errors.append(abs(powers.max() / peak - 1.0))
        temperature_errors.append(abs(temperature - final_temperature))

    for errors in (peak_errors, temperature_errors):
        orders = [np.log2(a / b) for a, b in itertools.pairwise(errors)]
        assert min(orders) > 0.9, (errors, orders)
    assert peak_errors[-1] < 0.01
    assert temperature_errors[-1] < 1.0


def test_feedback_turns_the_burst_over():
    _, powers, temperature = _burst(4.0e-3)
    assert powers.argmax() < powers.size - 1
    assert powers[-1] < 0.5 * powers.max()
    assert temperature > REFERENCE_TEMPERATURE + 100.0


def _column_model():
    n_planes = 10
    library = openndm.XSLibrary(2, n_planes)
    for composition in range(n_planes):
        library.set_composition(
            composition,
            D=[1.4, 0.4],
            absorption=[0.010, 0.085],
            nu_fission=[0.007, 0.13],
            kappa_fission=[0.007, 0.13],
            chi=[1.0, 0.0],
            scatter=[[0.0, 0.018], [0.0, 0.0]],
            inv_velocity=[1.0 / 2.0e7, 1.0 / 2.2e5],
        )
    library.set_delayed(beta=[0.0065], decay_constant=[0.08])
    library.finalize(warn=False)
    boundaries = {**ALL_REFLECTIVE, "z_min": "vacuum", "z_max": "vacuum"}
    geometry = openndm.Geometry.from_lattice(
        np.arange(n_planes).reshape(n_planes, 1, 1),
        pitch=(21.606, 21.606, 36.6),
        boundaries=boundaries,
    )
    settings = openndm.Settings(
        verbosity=0,
        k_tolerance=1.0e-10,
        fission_source_tolerance=1.0e-9,
        inner_tolerance=1.0e-10,
        max_outer=2000,
        max_step_iterations=500,
        step_tolerance=1.0e-13,
    )
    conduction = PinConduction(
        PINS,
        fuel_conductivity=3.0,
        clad_conductivity=15.0,
        gap_conductance=1.0e4,
        film_coefficient=3.0e4,
        fuel_density=10400.0,
        clad_density=6550.0,
        fuel_heat_capacity=300.0,
        clad_heat_capacity=330.0,
    )
    channel = ChannelModel(
        geometry,
        PINS,
        mass_flow=82.12102,
        inlet_temperature=560.0,
        conduction=conduction,
    )
    doppler = openndm.DopplerFeedback(
        library,
        list(range(n_planes)),
        gamma=2.0e-3,
        reference_temperature=560.0,
        groups=[0],
    )

    def apply_state(temperatures, _):
        doppler.apply(temperatures["doppler_temperature"])

    return openndm.Model(geometry, library, settings), channel, apply_state


ASSEMBLY_POWER = 17.67e6
"""Thermal power of one assembly column, W."""


def test_a_coupled_steady_state_stays_put_when_nothing_changes():
    """A null transient from solve_coupled must not move.

    It checks that the driver starts from the state solve_coupled left, that
    the power scale is fixed against it, and that the transient channel at
    constant power is the steady channel.
    """
    model, channel, apply_state = _column_model()
    model.solve_coupled(
        channel,
        apply_state,
        total_power=ASSEMBLY_POWER,
        tolerance=1.0e-9,
        max_iterations=200,
    )
    coupled = model.start_coupled_transient(
        channel, apply_state, total_power=ASSEMBLY_POWER
    )
    for _ in range(20):
        record = coupled.step(0.1)
        assert record.power == pytest.approx(ASSEMBLY_POWER, rel=1.0e-8)
    assert coupled.node_power.sum() == pytest.approx(record.power)
    assert record.peak_node_power == pytest.approx(coupled.node_power.max())


def test_the_node_fission_power_sums_to_the_step_total():
    model, _ = _point_kinetics_model()
    model.solve()
    transient = model.start_transient()
    record = transient.step(1.0e-3)
    assert transient.fission_power.sum() == pytest.approx(record.total_power)


def test_a_thermal_solver_that_cannot_advance_is_refused():
    model, _ = _point_kinetics_model()
    model.solve()
    with pytest.raises(openndm.InputError, match="advance"):
        model.start_coupled_transient(SteadyOnly(), lambda *_: None, total_power=1.0)


@pytest.mark.parametrize(
    ("total_power", "percent"), [(0.0, 100.0), (1.0, 0.0), (1.0, 101.0)]
)
def test_a_coupled_transient_needs_a_positive_starting_power(total_power, percent):
    model, _ = _point_kinetics_model()
    model.solve()
    with pytest.raises(openndm.InputError, match="positive power"):
        model.start_coupled_transient(
            AdiabaticLump(),
            lambda *_: None,
            total_power=total_power,
            percent=percent,
        )


def test_the_driver_api_is_importable_from_the_package():
    assert openndm.CoupledTransient is not None
    assert openndm.CoupledTransientStep is not None
    assert issubclass(openndm.TransientThermalSolver, openndm.ThermalSolver)
