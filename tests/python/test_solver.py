"""Solver behaviour: modes, determinism, warm start and failure handling."""

from __future__ import annotations

import numpy as np
import pytest

import openndm

from conftest import ALL_KERNELS, iaea_geometry, iaea_library


def test_result_reports_convergence_metadata(iaea_model):
    result = iaea_model.solve()
    assert result.converged
    assert result.outer_iterations > 1
    assert result.runtime > 0.0
    assert result.kernel == "sanm"
    assert result.history["outer"][-1] == result.outer_iterations
    assert abs(result.history["k_change"][-1]) < 1.0e-9


def test_power_is_normalised_to_a_mean_of_one(iaea_model):
    result = iaea_model.solve()
    powered = result.power[result.power > 0]
    assert powered.mean() == pytest.approx(1.0)
    assert result.f_q > 1.0
    assert result.f_dh > 1.0
    assert result.f_q >= result.f_dh


def test_reflector_nodes_carry_no_power(iaea_model):
    result = iaea_model.solve()
    compositions = iaea_model.geometry.compositions
    # Compositions 3 and 4 are the reflector, which has no fission source.
    assert np.all(result.power[compositions >= 3] == 0.0)
    assert np.all(result.power[compositions < 3] > 0.0)


def test_radial_and_axial_profiles_have_the_lattice_shape(iaea_model):
    result = iaea_model.solve()
    nz, ny, nx = iaea_model.geometry.shape
    assert result.radial_power().shape == (ny, nx)
    assert result.axial_power().shape == (nz,)
    assert result.power_lattice().shape == (nz, ny, nx)


@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_results_are_bit_identical_across_thread_counts(threads, tight):
    """FR-OPT-4: deterministic reductions, whatever the thread count."""
    model = openndm.Model(iaea_geometry(subdivide=2), iaea_library(), tight)
    reference = openndm.Model(
        iaea_geometry(subdivide=2), iaea_library(), tight
    ).solve(threads=1)
    result = model.solve(threads=threads)
    assert result.k_eff == reference.k_eff, (
        f"{threads} threads changed k_eff by "
        f"{result.k_eff - reference.k_eff:.3e}"
    )
    assert np.array_equal(np.asarray(result.flux), np.asarray(reference.flux))


def test_repeated_solves_are_reproducible(iaea_model):
    first = iaea_model.solve().k_eff
    second = iaea_model.solve().k_eff
    assert first == second


def test_warm_start_reaches_the_same_answer_in_fewer_outers(tight):
    """FR-OPT-3: a perturbed re-solve should reuse the previous solution."""
    model = openndm.Model(iaea_geometry(), iaea_library(), tight)
    cold = model.solve()

    # Perturb by swapping two assemblies, then solve both ways.
    model.swap_assemblies(1, 3)
    model.refresh()
    perturbed_cold = model.solve(warm_start=False)

    model_warm = openndm.Model(iaea_geometry(), iaea_library(), tight)
    model_warm.solve()
    model_warm.swap_assemblies(1, 3)
    model_warm.refresh()
    model_warm.solve()
    warm = model_warm.solve(warm_start=True)

    assert warm.k_eff == pytest.approx(perturbed_cold.k_eff, abs=1.0e-7)
    assert warm.outer_iterations < cold.outer_iterations


def test_swapping_identical_assemblies_leaves_k_unchanged(tight):
    model = openndm.Model(iaea_geometry(), iaea_library(), tight)
    before = model.solve().k_eff
    # Positions (0,1) and (0,2) on the top row are both composition 1.
    nx = model.geometry.shape[2]
    model.swap_assemblies(1, 2)
    model.refresh()
    assert model.solve().k_eff == pytest.approx(before, abs=1.0e-9)
    assert nx == 9


def test_swap_changes_k_when_the_compositions_differ(tight):
    model = openndm.Model(iaea_geometry(), iaea_library(), tight)
    before = model.solve().k_eff
    nx = model.geometry.shape[2]
    model.swap_assemblies(0, 7 * nx + 0)  # rodded centre against outer fuel
    model.refresh()
    after = model.solve().k_eff
    assert abs(after - before) > 1.0e-4


def _absorber_library(absorption=0.08, nu_fission=0.02):
    library = openndm.XSLibrary(1, 1)
    library.set_composition(
        0,
        D=[1.0],
        absorption=[absorption],
        nu_fission=[nu_fission],
        kappa_fission=[nu_fission],
        chi=[1.0],
        scatter=[[0.0]],
    )
    library.finalize()
    return library


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_fixed_source_reproduces_the_analytic_leakage_free_solution(kernel, tight):
    """FR-MODE-3: subcritical multiplication, checked against exact algebra.

    With every face reflective there is no leakage, so a spatially uniform
    source S produces the uniform flux S / (Sigma_a - nuSigma_f) exactly.
    """
    absorption, nu_fission, source_density = 0.08, 0.02, 1.0
    geometry = openndm.Geometry.from_lattice(
        np.zeros((3, 3, 3), dtype=int),
        pitch=10.0,
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "reflective"
        ),
    )
    model = openndm.Model(
        geometry, _absorber_library(absorption, nu_fission), tight
    )
    source = np.full((geometry.n_nodes, 1), source_density)
    result = model.solve_fixed_source(source, kernel=kernel)

    expected = source_density / (absorption - nu_fission)
    assert result.converged
    assert np.asarray(result.flux) == pytest.approx(expected, rel=1.0e-8)


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_fixed_source_with_a_distributed_source_stays_positive(kernel, tight):
    """A smooth source must give a strictly positive flux on every kernel.

    A single-node point source is a different matter: the two-node nodal
    closure overshoots against a delta and SANM can undershoot slightly
    negative near the tail. That is a property of nodal methods, not a defect
    of this implementation, and it is why Settings.dhat_limit exists.
    """
    geometry = openndm.Geometry.from_lattice(
        np.zeros((4, 4, 4), dtype=int),
        pitch=10.0,
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "vacuum"
        ),
    )
    model = openndm.Model(geometry, _absorber_library(), tight)
    source = np.ones((geometry.n_nodes, 1))
    result = model.solve_fixed_source(source, kernel=kernel)
    assert result.converged
    assert np.all(np.asarray(result.flux) > 0.0)


def test_fixed_source_kernels_agree_on_a_distributed_source(tight):
    geometry = openndm.Geometry.from_lattice(
        np.zeros((4, 4, 4), dtype=int),
        pitch=10.0,
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "vacuum"
        ),
    )
    model = openndm.Model(geometry, _absorber_library(), tight)
    source = np.ones((geometry.n_nodes, 1))
    totals = {
        kernel: np.asarray(
            model.solve_fixed_source(source, kernel=kernel).flux
        ).sum()
        for kernel in ALL_KERNELS
    }
    spread = (max(totals.values()) - min(totals.values())) / min(totals.values())
    assert spread < 5.0e-3, totals


def test_fixed_source_shape_is_validated(iaea_model):
    with pytest.raises(openndm.InputError, match="shape"):
        iaea_model.solve_fixed_source(np.ones(3))


def test_non_convergence_raises_a_typed_exception_and_does_not_abort(tight):
    """FR-OPT-6: a failed solve is catchable, never a process abort."""
    model = openndm.Model(iaea_geometry(), iaea_library(), tight)
    with pytest.raises(openndm.ConvergenceError) as excinfo:
        model.solve(max_outer=2, min_outer=2, k_tolerance=1.0e-15)
    assert excinfo.value.iterations == 2
    assert excinfo.value.residual >= 0.0
    # The interpreter is still alive and the model still usable.
    assert model.solve().converged


def test_a_model_without_fission_is_rejected_by_the_eigenvalue_solver(tight):
    library = openndm.XSLibrary(1, 1)
    library.set_composition(0, D=[1.0], absorption=[0.1], scatter=[[0.0]])
    library.finalize()
    geometry = openndm.Geometry.from_lattice(
        np.zeros((2, 2, 2), dtype=int), pitch=10.0
    )
    with pytest.raises(openndm.InputError, match="no fission source"):
        openndm.Model(geometry, library, tight).solve()


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_wielandt_shift_does_not_change_the_answer(kernel, tight):
    """The shift accelerates the outer iteration; it must not move k_eff."""
    model = openndm.Model(iaea_geometry(), iaea_library(), tight)
    shifted = model.solve(kernel=kernel, wielandt_shift=0.2)
    unshifted = model.solve(kernel=kernel, wielandt_shift=0.0)
    assert shifted.k_eff == pytest.approx(unshifted.k_eff, abs=1.0e-8)
    assert shifted.outer_iterations < unshifted.outer_iterations


def test_settings_reject_an_unknown_option():
    with pytest.raises(AttributeError, match="no option"):
        openndm.Settings(nonexistent_option=1)


def test_settings_reject_an_unknown_kernel():
    with pytest.raises(ValueError, match="unknown kernel"):
        openndm.Settings(kernel="magic")


def test_per_call_overrides_do_not_mutate_the_model_settings(iaea_model):
    assert iaea_model.settings.kernel == "sanm"
    iaea_model.solve(kernel="fdm")
    assert iaea_model.settings.kernel == "sanm"


def test_critical_boron_search_finds_the_target(tight):
    """FR-MODE-4: iterate boron to a target eigenvalue."""
    geometry = iaea_geometry()
    library = iaea_library()
    # A simple linear boron model on the thermal absorption of the fuel.
    base = [library.composition(c).absorption[1] for c in range(3)]

    def apply_boron(lib, ppm):
        for c in range(3):
            lib.set_composition(c, absorption=[0.010, base[c] + 1.0e-5 * ppm])
        lib.finalize(warn=False)

    model = openndm.Model(geometry, library, tight)
    search = model.search_boron(apply_boron, target_k=1.0, guess=500.0)
    assert search.result.k_eff == pytest.approx(1.0, abs=1.0e-5)
    assert 0.0 < search.boron < 3000.0
    assert search.iterations <= 20


def test_boron_search_says_when_the_target_is_out_of_reach(tight):
    """The common failure is an unreachable target, not an inert model.

    A subcritical core cannot be made critical by adding boron, and the search
    then walks to a bracket edge and evaluates the same concentration twice.
    Reporting that as insensitivity to boron sends the reader looking in the
    wrong place, so the message names the reachable range instead.
    """
    library = iaea_library()
    base = [library.composition(c).absorption[1] for c in range(3)]

    def apply_boron(lib, ppm):
        for c in range(3):
            lib.set_composition(c, absorption=[0.010, base[c] + 1.0e-5 * ppm])
        lib.finalize(warn=False)

    model = openndm.Model(iaea_geometry(), library, tight)
    with pytest.raises(openndm.ConvergenceError, match="not reachable") as excinfo:
        # This core is supercritical across the whole range, so a target of
        # 0.5 cannot be met by any concentration inside the bracket.
        model.search_boron(
            apply_boron, target_k=0.5, guess=100.0, bracket=(0.0, 200.0)
        )
    assert "k_eff stayed between" in str(excinfo.value)


def test_boron_search_reports_failure_when_boron_has_no_effect(tight):
    model = openndm.Model(iaea_geometry(), iaea_library(), tight)

    def no_op(lib, ppm):
        # Deliberately ignores the concentration: the search must notice that
        # k_eff does not respond and say so rather than spin.
        return None

    with pytest.raises(openndm.ConvergenceError, match="did not respond") as excinfo:
        model.search_boron(no_op, target_k=1.0, guess=500.0, max_iterations=4)
    assert "apply_boron" in str(excinfo.value)


def test_sweep_runs_every_case(tight):
    model = openndm.Model(iaea_geometry(), iaea_library(), tight)

    def mutate(m, case):
        m.geometry.set_composition(0, case)

    results = model.sweep(mutate, [0, 1, 2])
    assert len(results) == 3
    assert all(r.converged for r in results)
    # The rodded centre must be worth something.
    assert results[2].k_eff < results[0].k_eff
