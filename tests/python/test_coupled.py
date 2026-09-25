"""Coupled steady state with thermal-hydraulic feedback (FR-MODE-6)."""

from __future__ import annotations

import numpy as np
import pytest

import openndm
from openndm.channel import ChannelModel, PinGeometry, absolute_power
from openndm.pin import PinConduction
from openndm.thermal import CompositionMapping

N_PLANES = 8
PLANE_HEIGHT = 20.0
INLET = 560.0
TOTAL_POWER = 1.0e7
COLD, HOT = 500.0, 1500.0

PINS = PinGeometry(
    fuel_radius=4.11950e-3,
    gap_thickness=6.8e-5,
    clad_thickness=5.71e-4,
    pin_pitch=1.2655e-2,
    n_pins=264,
    n_guide_tubes=25,
)


def slab_geometry(n_compositions=N_PLANES):
    """One axial column, one composition per plane, leaking only axially."""
    core = np.arange(N_PLANES).reshape(N_PLANES, 1, 1) % n_compositions
    return openndm.Geometry.from_lattice(
        core,
        pitch=(21.606, 21.606, PLANE_HEIGHT),
        boundaries={
            "x_min": "reflective",
            "x_max": "reflective",
            "y_min": "reflective",
            "y_max": "reflective",
            "z_min": "zero_flux",
            "z_max": "zero_flux",
        },
    )


def branch_library(doppler=0.05, n_compositions=N_PLANES):
    """Two groups over one fuel temperature axis, absorption rising with it."""
    lib = openndm.XSLibrary(2, n_compositions)
    lib.set_axes([("fuel_temperature", [COLD, HOT])])
    for state, factor in enumerate((1.0, 1.0 + doppler)):
        for index in range(n_compositions):
            lib.set_composition(
                index,
                state=state,
                D=[1.5, 0.4],
                absorption=[0.010 * factor, 0.080 * factor],
                nu_fission=[0.0, 0.135],
                kappa_fission=[0.0, 0.135],
                chi=[1.0, 0.0],
                scatter=[[0.0, 0.020], [0.0, 0.0]],
            )
    lib.finalize(warn=False)
    return lib


def thermal_model(geometry, **kwargs):
    conduction = PinConduction(
        PINS,
        fuel_conductivity=3.0,
        clad_conductivity=15.0,
        gap_conductance=1.0e4,
        film_coefficient=3.0e4,
    )
    settings = {
        "mass_flow": 82.12102,
        "inlet_temperature": INLET,
        "conduction": conduction,
        "water": openndm.IF97Water(),
    }
    settings.update(kwargs)
    return ChannelModel(geometry, PINS, **settings)


def settings():
    """Tight enough to resolve the feedback, loose enough to run in the suite.

    The eigenvalue moves thousands of pcm here, so 1e-9 leaves four orders of
    margin, and every solve in the loop starts cold.
    """
    return openndm.Settings(
        verbosity=0, k_tolerance=1.0e-9, fission_source_tolerance=1.0e-8
    )


class Recorder:
    """Passes a thermal solver through, keeping every heat source it saw."""

    def __init__(self, inner):
        self.inner = inner
        self.heat_sources = []

    def set_heat_source(self, q):
        self.heat_sources.append(np.array(q, dtype=float))
        self.inner.set_heat_source(q)

    def solve(self):
        self.inner.solve()

    def get_temperatures(self):
        return self.inner.get_temperatures()

    def get_densities(self):
        return self.inner.get_densities()


def coupled_case(doppler=0.05, **kwargs):
    """A model, its thermal solver and the apply_state that joins them."""
    geometry = slab_geometry()
    branch = branch_library(doppler)
    model = openndm.Model(
        geometry, branch.interpolate(fuel_temperature=COLD), settings()
    )
    thermal = Recorder(thermal_model(geometry, **kwargs))
    mapping = CompositionMapping(geometry)

    def apply_state(temperatures, densities):
        fuel = mapping.average(temperatures["doppler_temperature"])
        branch.interpolate_by_composition(
            {"fuel_temperature": fuel}, out=model.library
        )

    return model, thermal, apply_state


def test_the_coupled_api_is_importable_from_the_package():
    assert openndm.CoupledResult is not None
    assert openndm.CompositionMapping is CompositionMapping


def test_a_library_with_no_temperature_dependence_converges_in_one_iteration():
    geometry = slab_geometry()
    model = openndm.Model(geometry, branch_library(doppler=0.0).interpolate(
        fuel_temperature=COLD
    ), settings())
    uncoupled = model.solve().k_eff

    thermal = Recorder(thermal_model(geometry))
    coupled = model.solve_coupled(
        thermal, lambda t, d: None, total_power=TOTAL_POWER
    )

    assert coupled.iterations == 1
    assert coupled.k_eff == pytest.approx(uncoupled, abs=1.0e-12)


def test_a_flat_branch_library_also_gives_the_uncoupled_eigenvalue():
    model, thermal, apply_state = coupled_case(doppler=0.0)
    uncoupled = model.solve().k_eff

    coupled = model.solve_coupled(
        thermal, apply_state, total_power=TOTAL_POWER
    )

    assert coupled.iterations == 1
    assert coupled.k_eff == pytest.approx(uncoupled, abs=1.0e-12)


def test_feedback_lowers_the_eigenvalue_below_the_cold_one():
    model, thermal, apply_state = coupled_case()
    cold = model.solve().k_eff

    coupled = model.solve_coupled(
        thermal, apply_state, total_power=TOTAL_POWER
    )

    assert coupled.k_eff < cold
    assert coupled.iterations > 1


def test_raising_the_power_lowers_the_eigenvalue_monotonically():
    eigenvalues = []
    for percent in (0.001, 50.0, 100.0):
        model, thermal, apply_state = coupled_case()
        coupled = model.solve_coupled(
            thermal, apply_state, total_power=TOTAL_POWER, percent=percent
        )
        eigenvalues.append(coupled.k_eff)

    assert np.all(np.diff(eigenvalues) < 0.0)


def test_the_converged_power_is_the_one_the_last_solve_produced():
    model, thermal, apply_state = coupled_case()
    coupled = model.solve_coupled(
        thermal, apply_state, total_power=TOTAL_POWER, tolerance=1.0e-8
    )

    from_result = absolute_power(
        coupled.result.power, model.geometry.volumes, TOTAL_POWER
    )
    assert coupled.power == pytest.approx(from_result, rel=1.0e-14)

    last_seen = thermal.heat_sources[-1]
    scale = float(np.mean(np.abs(from_result)))
    assert float(np.max(np.abs(last_seen - from_result))) / scale < 1.0e-8


def test_the_converged_power_sums_to_the_thermal_power_asked_for():
    model, thermal, apply_state = coupled_case()
    coupled = model.solve_coupled(
        thermal, apply_state, total_power=TOTAL_POWER, percent=60.0
    )
    assert float(coupled.power.sum()) == pytest.approx(0.6 * TOTAL_POWER)


def test_the_history_carries_one_eigenvalue_per_iteration():
    model, thermal, apply_state = coupled_case()
    coupled = model.solve_coupled(
        thermal, apply_state, total_power=TOTAL_POWER
    )

    assert len(coupled.history) == coupled.iterations
    assert [step.iteration for step in coupled.history] == list(
        range(1, coupled.iterations + 1)
    )
    assert all(step.k_eff is not None for step in coupled.history)
    assert coupled.history[-1].k_eff == pytest.approx(coupled.k_eff)
    assert coupled.power_change < 1.0e-5


def test_the_power_change_falls_as_the_loop_proceeds():
    model, thermal, apply_state = coupled_case()
    coupled = model.solve_coupled(
        thermal, apply_state, total_power=TOTAL_POWER, tolerance=1.0e-10
    )
    changes = [step.power_change for step in coupled.history]
    assert changes[-1] < changes[0]


def test_an_exhausted_iteration_cap_raises_rather_than_returning():
    model, thermal, apply_state = coupled_case()
    with pytest.raises(openndm.ConvergenceError):
        model.solve_coupled(
            thermal,
            apply_state,
            total_power=TOTAL_POWER,
            tolerance=1.0e-14,
            max_iterations=2,
        )


def test_the_reported_temperatures_are_the_converged_ones():
    model, thermal, apply_state = coupled_case()
    coupled = model.solve_coupled(
        thermal, apply_state, total_power=TOTAL_POWER
    )

    fields = coupled.temperatures
    assert set(fields) == {
        "moderator_temperature",
        "fuel_temperature",
        "doppler_temperature",
    }
    assert np.all(fields["moderator_temperature"] > INLET)
    assert np.all(fields["fuel_temperature"] > fields["moderator_temperature"])
    assert set(coupled.densities) == {"moderator_density"}


def test_replacing_the_library_instead_of_writing_into_it_is_an_error():
    geometry = slab_geometry()
    branch = branch_library()
    model = openndm.Model(
        geometry, branch.interpolate(fuel_temperature=COLD), settings()
    )
    thermal = thermal_model(geometry)

    def replace(temperatures, densities):
        model.library = branch.interpolate(fuel_temperature=HOT)

    with pytest.raises(openndm.InputError):
        model.solve_coupled(thermal, replace, total_power=TOTAL_POWER)


def test_a_coupled_result_is_described_by_its_repr():
    model, thermal, apply_state = coupled_case()
    coupled = model.solve_coupled(
        thermal, apply_state, total_power=TOTAL_POWER
    )
    text = repr(coupled)
    assert "k_eff=" in text
    assert f"iterations={coupled.iterations}" in text


def test_under_relaxation_takes_more_iterations_to_the_same_answer():
    model, thermal, apply_state = coupled_case()
    full = model.solve_coupled(
        thermal, apply_state, total_power=TOTAL_POWER, tolerance=1.0e-9
    )

    model, thermal, apply_state = coupled_case()
    damped = model.solve_coupled(
        thermal,
        apply_state,
        total_power=TOTAL_POWER,
        tolerance=1.0e-9,
        relaxation=0.5,
    )

    assert damped.iterations > full.iterations
    assert damped.k_eff == pytest.approx(full.k_eff, abs=1.0e-8)


def test_settings_overrides_reach_every_solve_in_the_loop():
    model, thermal, apply_state = coupled_case()
    coupled = model.solve_coupled(
        thermal, apply_state, total_power=TOTAL_POWER, kernel="fdm"
    )
    assert coupled.result.kernel == "fdm"
