"""Coupling interface, Picard driver and field mapping (FR-TH-6, FR-TH-7)."""

from __future__ import annotations

import numpy as np
import pytest

import openndm
from openndm.thermal import (
    AxialMapping,
    CompositionMapping,
    PicardCoupling,
    ThermalSolver,
)

INLET = 560.0
HEATING = 1.0e-4


class FixedSolver:
    """A solver whose answer never depends on the power given to it."""

    def __init__(self, n_nodes=4, temperature=INLET):
        self.n_nodes = n_nodes
        self.temperature = temperature
        self.heat_source = None
        self.solves = 0

    def set_heat_source(self, q):
        self.heat_source = np.asarray(q, dtype=float)

    def solve(self):
        self.solves += 1

    def get_temperatures(self):
        return {"fuel_temperature": np.full(self.n_nodes, self.temperature)}

    def get_densities(self):
        return {"moderator_density": np.full(self.n_nodes, 750.0)}


class HeatingSolver:
    """Temperature rises linearly with the power deposited in each node."""

    def __init__(self, n_nodes=4, gain=HEATING):
        self.n_nodes = n_nodes
        self.gain = gain
        self.heat_source = np.zeros(n_nodes)
        self.temperature = np.full(n_nodes, INLET)

    def set_heat_source(self, q):
        self.heat_source = np.asarray(q, dtype=float)

    def solve(self):
        self.temperature = INLET + self.gain * self.heat_source

    def get_temperatures(self):
        return {"fuel_temperature": self.temperature}

    def get_densities(self):
        return {"moderator_density": 750.0 - 0.3 * (self.temperature - INLET)}


def ignore_state(temperatures, densities):
    pass


def test_a_plain_object_satisfies_the_protocol():
    assert isinstance(FixedSolver(), ThermalSolver)
    assert not isinstance(object(), ThermalSolver)


def test_the_thermal_api_is_importable_from_the_package():
    assert openndm.PicardCoupling is PicardCoupling
    assert openndm.AxialMapping is AxialMapping
    assert openndm.ThermalSolver is ThermalSolver


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relaxation": 0.0},
        {"relaxation": 1.5},
        {"relaxation": -0.2},
        {"tolerance": 0.0},
        {"tolerance": -1e-6},
        {"max_iterations": 0},
    ],
)
def test_the_driver_rejects_invalid_settings(kwargs):
    with pytest.raises(openndm.InputError):
        PicardCoupling(FixedSolver(), ignore_state, **kwargs)


def test_a_fixed_solver_converges_in_one_iteration():
    thermal = FixedSolver()
    power = np.array([1.0e6, 2.0e6, 2.0e6, 1.0e6])
    coupling = PicardCoupling(thermal, ignore_state)

    result = coupling.solve(lambda: power)

    assert result.converged
    assert result.iterations == 1
    assert thermal.solves == 1
    assert result.history[0].power_change == 0.0
    np.testing.assert_allclose(result.power, power)


def test_the_driver_hands_the_thermal_state_straight_on():
    thermal = FixedSolver()
    seen = {}

    def apply_state(temperatures, densities):
        seen.update(temperatures=temperatures, densities=densities)

    PicardCoupling(thermal, apply_state).solve(lambda: np.ones(4))

    np.testing.assert_allclose(thermal.heat_source, np.ones(4))
    np.testing.assert_allclose(seen["temperatures"]["fuel_temperature"], INLET)
    np.testing.assert_allclose(seen["densities"]["moderator_density"], 750.0)


def test_the_result_does_not_alias_the_solver_buffers():
    thermal = HeatingSolver()
    result = PicardCoupling(thermal, ignore_state).solve(lambda: np.ones(4))

    thermal.temperature[:] = 9999.0

    assert np.all(result.temperatures["fuel_temperature"] < 1000.0)


def test_feedback_drives_the_loop_to_a_fixed_point():
    thermal = HeatingSolver()
    state = {}

    def apply_state(temperatures, densities):
        state["fuel_temperature"] = temperatures["fuel_temperature"]

    def node_power():
        if "fuel_temperature" not in state:
            return np.full(4, 1.0e5)
        return np.full(4, 1.0e5) * (
            1.0 - 1.0e-4 * (state["fuel_temperature"] - INLET)
        )

    result = PicardCoupling(thermal, apply_state, tolerance=1e-10).solve(node_power)

    assert result.converged
    assert 1 < result.iterations < 50
    settled = np.full(4, 1.0e5) * (
        1.0 - 1.0e-4 * (INLET + HEATING * result.power - INLET)
    )
    np.testing.assert_allclose(result.power, settled, rtol=1e-8)


def test_the_history_records_every_iteration_and_the_eigenvalue():
    thermal = HeatingSolver()
    eigenvalues = iter([1.05, 1.04, 1.03, 1.02, 1.01])
    last = {"k": 1.0}

    def k_eff_source():
        last["k"] = next(eigenvalues, last["k"])
        return last["k"]

    result = PicardCoupling(thermal, ignore_state).solve(
        lambda: np.array([1.0, 2.0, 3.0, 4.0]), k_eff_source=k_eff_source
    )

    assert [step.iteration for step in result.history] == list(
        range(1, result.iterations + 1)
    )
    assert result.history[0].k_eff == 1.05


def test_under_relaxation_moves_only_part_of_the_way():
    thermal = FixedSolver()
    powers = iter([np.zeros(4), np.full(4, 100.0), np.full(4, 100.0)])
    coupling = PicardCoupling(thermal, ignore_state, relaxation=0.25, max_iterations=2)

    with pytest.raises(openndm.ConvergenceError):
        coupling.solve(lambda: next(powers))

    np.testing.assert_allclose(thermal.heat_source, 25.0)


def unstable_feedback(thermal, alpha, base=1.0e5):
    """A neutronics step steep enough in temperature to diverge undamped."""
    state = {"temperature": np.full(thermal.n_nodes, INLET)}

    def apply_state(temperatures, densities):
        state["temperature"] = temperatures["fuel_temperature"]

    def node_power():
        return base * (1.0 - alpha * (state["temperature"] - INLET))

    return apply_state, node_power


def test_a_full_step_diverges_where_a_damped_one_converges():
    diverging = HeatingSolver()
    apply_state, node_power = unstable_feedback(diverging, alpha=3.0e-1)
    with pytest.raises(openndm.ConvergenceError):
        PicardCoupling(
            diverging, apply_state, relaxation=1.0, max_iterations=20
        ).solve(node_power)

    damped = HeatingSolver()
    apply_state, node_power = unstable_feedback(damped, alpha=3.0e-1)
    result = PicardCoupling(
        damped, apply_state, relaxation=0.25, max_iterations=20
    ).solve(node_power)

    assert result.converged


def test_an_oscillation_raises_rather_than_returning_silently():
    thermal = FixedSolver()
    powers = np.array([1.0e6, 2.0e6])
    calls = {"n": 0}

    def node_power():
        calls["n"] += 1
        return np.full(4, powers[calls["n"] % 2])

    coupling = PicardCoupling(thermal, ignore_state, max_iterations=12)
    with pytest.raises(openndm.ConvergenceError) as excinfo:
        coupling.solve(node_power)

    assert excinfo.value.iterations == 12
    assert excinfo.value.residual > 0.0


def test_a_power_field_that_changes_shape_is_an_error():
    shapes = iter([np.ones(4), np.ones(6)])

    coupling = PicardCoupling(FixedSolver(), ignore_state)
    with pytest.raises(openndm.InputError, match="changed shape"):
        coupling.solve(lambda: next(shapes))


def test_the_coupling_reprs_are_readable():
    coupling = PicardCoupling(FixedSolver(), ignore_state)
    result = coupling.solve(lambda: np.ones(4))
    assert "PicardCoupling" in repr(coupling)
    assert "converged=True" in repr(result)


NODAL_EDGES = np.array([0.0, 30.0, 70.0, 100.0])
FINE_EDGES = np.linspace(0.0, 100.0, 21)


def test_a_uniform_field_stays_uniform_on_any_mesh():
    rng = np.random.default_rng(0)
    coarse = np.sort(rng.uniform(0.0, 100.0, 7))
    coarse[0], coarse[-1] = 0.0, 100.0
    fine = np.sort(rng.uniform(0.0, 100.0, 31))
    fine[0], fine[-1] = 0.0, 100.0

    mapping = AxialMapping(coarse, fine)
    uniform = np.full(mapping.n_source, 583.15)

    onto_fine = mapping.average(uniform)
    np.testing.assert_allclose(onto_fine, 583.15)
    np.testing.assert_allclose(mapping.reverse().average(onto_fine), 583.15)


def test_distributing_power_keeps_its_total():
    mapping = AxialMapping(NODAL_EDGES, FINE_EDGES)
    power = np.array([1.2e6, 3.4e6, 0.9e6])

    fine = mapping.distribute(power)

    assert fine.shape == (mapping.n_target,)
    assert np.sum(fine) == pytest.approx(np.sum(power), rel=1e-15)


def test_a_round_trip_conserves_the_integral_to_round_off():
    mapping = AxialMapping(NODAL_EDGES, FINE_EDGES)
    power = np.array([1.2e6, 3.4e6, 0.9e6])

    back = mapping.reverse().distribute(mapping.distribute(power))

    assert np.sum(back) == pytest.approx(np.sum(power), rel=1e-14)


def test_a_round_trip_onto_a_refinement_is_exact():
    mapping = AxialMapping(NODAL_EDGES, [0.0, 15.0, 30.0, 50.0, 70.0, 85.0, 100.0])
    power = np.array([1.2e6, 3.4e6, 0.9e6])

    np.testing.assert_allclose(
        mapping.reverse().distribute(mapping.distribute(power)), power, rtol=1e-14
    )


def test_averaging_keeps_the_volume_weighted_mean():
    mapping = AxialMapping(NODAL_EDGES, FINE_EDGES)
    temperature = np.array([570.0, 600.0, 620.0])
    widths = np.diff(NODAL_EDGES)

    fine = mapping.average(temperature)

    expected = float(np.sum(temperature * widths) / np.sum(widths))
    fine_widths = np.diff(FINE_EDGES)
    assert float(np.sum(fine * fine_widths) / np.sum(fine_widths)) == pytest.approx(
        expected, rel=1e-14
    )


def test_averaging_a_linear_profile_onto_a_refinement_is_exact():
    mapping = AxialMapping([0.0, 50.0, 100.0], [0.0, 25.0, 50.0, 75.0, 100.0])
    coarse = np.array([500.0, 700.0])

    np.testing.assert_allclose(mapping.average(coarse), [500.0, 500.0, 700.0, 700.0])


def test_leading_axes_map_a_whole_core_at_once():
    mapping = AxialMapping(NODAL_EDGES, FINE_EDGES)
    core = np.array([[1.0e6, 2.0e6, 1.5e6], [0.4e6, 0.8e6, 0.6e6]])

    fine = mapping.distribute(core)

    assert fine.shape == (2, mapping.n_target)
    np.testing.assert_allclose(
        fine, np.stack([mapping.distribute(row) for row in core])
    )
    np.testing.assert_allclose(np.sum(fine, axis=1), np.sum(core, axis=1), rtol=1e-14)


def test_the_extensive_and_intensive_transfers_differ():
    mapping = AxialMapping(NODAL_EDGES, FINE_EDGES)
    values = np.array([1.0, 1.0, 1.0])

    assert not np.allclose(mapping.distribute(values), mapping.average(values))


def test_a_mesh_is_described_by_its_repr():
    mapping = AxialMapping(NODAL_EDGES, FINE_EDGES)
    assert mapping.n_source == 3
    assert mapping.n_target == 20
    assert "3 -> 20" in repr(mapping)


@pytest.mark.parametrize(
    "source,target",
    [
        ([0.0, 50.0, 100.0], [0.0, 50.0, 120.0]),
        ([0.0, 50.0, 100.0], [-5.0, 50.0, 100.0]),
        ([0.0, 50.0, 100.0], [0.0, 60.0, 50.0, 100.0]),
        ([0.0, 50.0, 50.0, 100.0], [0.0, 100.0]),
        ([0.0], [0.0, 100.0]),
    ],
)
def test_an_inconsistent_pair_of_meshes_is_an_error(source, target):
    with pytest.raises(openndm.InputError):
        AxialMapping(source, target)


def test_a_field_of_the_wrong_length_is_an_error():
    mapping = AxialMapping(NODAL_EDGES, FINE_EDGES)
    with pytest.raises(openndm.InputError, match="source cells"):
        mapping.distribute([1.0, 2.0])
    with pytest.raises(openndm.InputError, match="source cells"):
        mapping.average(np.ones((2, 5)))


def two_composition_geometry(widths=(20.0, 20.0)):
    """One plane, two positions, so composition 0 and 1 hold one node each."""
    return openndm.Geometry.from_lattice(
        np.array([[[0, 1]]]), pitch=20.0, dx=list(widths)
    )


def test_a_composition_holding_one_node_takes_that_nodes_value():
    mapping = CompositionMapping(two_composition_geometry())
    assert mapping.average([600.0, 900.0]).tolist() == [600.0, 900.0]


def test_nodes_sharing_a_composition_are_averaged_by_volume():
    geometry = openndm.Geometry.from_lattice(
        np.array([[[0, 0]]]), pitch=20.0, dx=[10.0, 30.0]
    )
    mapping = CompositionMapping(geometry)
    assert float(mapping.average([600.0, 1000.0])[0]) == pytest.approx(900.0)


def test_a_uniform_field_survives_the_reduction_whatever_the_weights():
    geometry = openndm.Geometry.from_lattice(
        np.array([[[0, 0, 1]]]), pitch=20.0, dx=[10.0, 30.0, 5.0]
    )
    mapping = CompositionMapping(geometry)
    assert mapping.average(np.full(3, 700.0)) == pytest.approx(np.full(2, 700.0))


def test_the_weighting_is_the_callers_to_choose():
    geometry = openndm.Geometry.from_lattice(np.array([[[0, 0]]]), pitch=20.0)
    by_power = CompositionMapping(geometry, weights=[3.0, 1.0])
    assert float(by_power.average([600.0, 1000.0])[0]) == pytest.approx(700.0)


def test_a_composition_no_node_carries_still_gets_a_usable_coordinate():
    geometry = openndm.Geometry.from_lattice(np.array([[[0, 2]]]), pitch=20.0)
    mapping = CompositionMapping(geometry)

    assert mapping.n_compositions == 3
    assert mapping.occupied.tolist() == [True, False, True]

    states = mapping.average([600.0, 900.0])
    assert np.all(np.isfinite(states))
    assert states.tolist() == [600.0, 750.0, 900.0]


def test_expanding_a_composition_field_puts_it_back_on_the_nodes():
    mapping = CompositionMapping(two_composition_geometry())
    assert mapping.expand([600.0, 900.0]).tolist() == [600.0, 900.0]


def test_a_reduction_followed_by_an_expansion_is_a_projection():
    geometry = openndm.Geometry.from_lattice(
        np.array([[[0, 0, 1]]]), pitch=20.0
    )
    mapping = CompositionMapping(geometry)
    once = mapping.expand(mapping.average([600.0, 1000.0, 800.0]))
    twice = mapping.expand(mapping.average(once))
    assert twice == pytest.approx(once)


def test_the_reduction_is_volume_conservative():
    geometry = openndm.Geometry.from_lattice(
        np.array([[[0, 0, 1]]]), pitch=20.0, dx=[10.0, 30.0, 5.0]
    )
    mapping = CompositionMapping(geometry)
    field = np.array([600.0, 1000.0, 800.0])
    volumes = geometry.volumes
    reduced = mapping.expand(mapping.average(field))
    assert float(reduced @ volumes) == pytest.approx(float(field @ volumes))


def test_a_composition_mapping_rejects_impossible_input():
    mapping = CompositionMapping(two_composition_geometry())
    with pytest.raises(openndm.InputError, match="nodes"):
        mapping.average([600.0])
    with pytest.raises(openndm.InputError, match="compositions"):
        mapping.expand([600.0, 700.0, 800.0])
    with pytest.raises(openndm.InputError, match="negative"):
        CompositionMapping(two_composition_geometry(), weights=[-1.0, 1.0])
    with pytest.raises(openndm.InputError, match="weight"):
        CompositionMapping(two_composition_geometry(), weights=[1.0])


def test_a_composition_mapping_is_described_by_its_repr():
    text = repr(CompositionMapping(two_composition_geometry()))
    assert "2 nodes" in text
    assert "2 of 2 compositions" in text
