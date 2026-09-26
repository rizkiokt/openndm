"""Solver behaviour: modes, determinism, warm start and failure handling."""

from __future__ import annotations

import numpy as np
import pytest

import openndm

from conftest import (
    ALL_KERNELS,
    NODAL_KERNELS,
    chain_library,
    iaea_geometry,
    iaea_library,
)


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
    assert result.f_q >= result.f_dh * (1.0 - 1.0e-12)


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


IDENTICAL_ASSEMBLIES = (1, 3)
UNLIKE_ASSEMBLIES = (4, 5)
RODDED_CENTRE_AGAINST_OUTER_FUEL = (0, 63)


def _swapped_model(settings, swap, *, solve_first=True):
    """IAEA core with two assemblies swapped and the model refreshed."""
    model = openndm.Model(iaea_geometry(), iaea_library(), settings)
    if solve_first:
        model.solve()
    model.swap_assemblies(*swap)
    model.refresh()
    return model


def test_refresh_keeps_the_flux_for_a_warm_start(tight):
    """FR-OPT-3: ``refresh`` used to clear the flux, so a warm start after any
    change to the model took as many outers as a cold one: 104 here.
    """
    cold = _swapped_model(tight, IDENTICAL_ASSEMBLIES).solve(warm_start=False)
    warm = _swapped_model(tight, IDENTICAL_ASSEMBLIES).solve(warm_start=True)
    assert warm.k_eff == pytest.approx(cold.k_eff, abs=1.0e-9)
    assert warm.outer_iterations < cold.outer_iterations / 10


def test_warm_start_after_a_real_swap_reaches_the_cold_answer(tight):
    """The retained Dhat belongs to the old loading pattern; the nodal update
    must still converge it to the new one.
    """
    cold = _swapped_model(tight, UNLIKE_ASSEMBLIES).solve(warm_start=False)
    warm = _swapped_model(tight, UNLIKE_ASSEMBLIES).solve(warm_start=True)
    assert warm.k_eff == pytest.approx(cold.k_eff, abs=1.0e-8)


def test_cold_solve_after_refresh_matches_a_fresh_model(tight):
    """A cold solve discards what refresh kept, so it retraces a fresh solve."""
    refreshed = _swapped_model(tight, UNLIKE_ASSEMBLIES).solve(warm_start=False)
    fresh = _swapped_model(tight, UNLIKE_ASSEMBLIES, solve_first=False).solve()
    assert refreshed.k_eff == fresh.k_eff
    assert refreshed.outer_iterations == fresh.outer_iterations
    assert np.array_equal(np.asarray(refreshed.flux), np.asarray(fresh.flux))


def test_adjoint_after_refresh_reconverges_the_forward_coupling(tight):
    """The adjoint reuses the forward Dhat, which belongs to the old model.

    Keeping it through a swap of the rodded centre with outer fuel left the
    adjoint eigenvalue 569 pcm away from the forward one.
    """
    model = openndm.Model(iaea_geometry(), iaea_library(), tight)
    model.solve()
    model.solve_adjoint()
    model.swap_assemblies(*RODDED_CENTRE_AGAINST_OUTER_FUEL)
    model.refresh()
    adjoint = model.solve_adjoint(warm_start=True)

    fresh = _swapped_model(
        tight, RODDED_CENTRE_AGAINST_OUTER_FUEL, solve_first=False
    ).solve()
    assert adjoint.k_eff == pytest.approx(fresh.k_eff, abs=1.0e-6)


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


def vacuum_box_total_flux(n_per_side, kernel, settings):
    """Volume-integrated flux in a 40 cm vacuum-bounded box of pure absorber.

    Volume-integrated rather than summed, so meshes are comparable. The
    diffusion length is 4.1 cm, which makes the coarsest mesh here 2.4
    diffusion lengths per node: the regime where the boundary treatment
    decides the answer.
    """
    pitch = 40.0 / n_per_side
    geometry = openndm.Geometry.from_lattice(
        np.zeros((n_per_side,) * 3, dtype=int),
        pitch=pitch,
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "vacuum"
        ),
    )
    model = openndm.Model(geometry, _absorber_library(), settings)
    source = np.ones((geometry.n_nodes, 1))
    flux = np.asarray(model.solve_fixed_source(source, kernel=kernel).flux)
    return float(flux.sum()) * pitch**3


def test_fixed_source_nodal_kernels_beat_fdm_at_a_vacuum_boundary(tight):
    """Two diffusion lengths per node is where the boundary treatment shows.

    The reference is FDM's own Richardson limit, so it borrows nothing from
    either nodal kernel. Asserting instead that all three kernels agree on
    the coarse mesh would hold the nodal ones to FDM's 13% error, which is
    what they made themselves while boundary faces kept a finite difference
    coupling.
    """
    fine = vacuum_box_total_flux(16, "fdm", tight)
    finer = vacuum_box_total_flux(32, "fdm", tight)
    reference = finer + (finer - fine) / 3.0

    coarse = {k: vacuum_box_total_flux(4, k, tight) for k in ALL_KERNELS}
    error = {k: abs(v - reference) / reference for k, v in coarse.items()}

    assert error["fdm"] > 0.10, coarse
    for kernel in ("nem", "sanm"):
        assert error[kernel] < 0.02, (kernel, coarse, reference)
    assert abs(coarse["nem"] - coarse["sanm"]) / coarse["sanm"] < 5.0e-3, coarse


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


# ------------------------------------------- mutating a finalized library
def _mutable_model(absorption, tight):
    library = openndm.XSLibrary(1, 1)
    library.set_composition(
        0,
        D=[1.0],
        absorption=[absorption],
        nu_fission=[0.1],
        kappa_fission=[0.1],
        chi=[1.0],
        scatter=[[0.0]],
    )
    library.finalize(warn=False)
    geometry = openndm.Geometry.from_lattice(
        np.zeros((8, 8, 8), dtype=int),
        pitch=12.5,
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "zero_flux"
        ),
    )
    return openndm.Model(geometry, library, tight), library


def _rewrite_absorption(library, absorption):
    library.set_composition(
        0,
        D=[1.0],
        absorption=[absorption],
        nu_fission=[0.1],
        kappa_fission=[0.1],
        chi=[1.0],
        scatter=[[0.0]],
    )


@pytest.mark.parametrize("call", ["solve", "refresh", "solve_fixed_source"])
def test_solving_a_mutated_library_is_refused(call, tight):
    """``removal`` is derived at finalize time and the kernels read it.

    Writing a composition afterwards leaves the cached removal in place, so
    solving anyway uses the *old* absorption while every accessor reports the
    new one. That is silent and worth about 39000 pcm on this problem.
    """
    model, library = _mutable_model(0.08, tight)
    model.solve()
    _rewrite_absorption(library, 0.12)
    arguments = ([np.zeros((model.geometry.n_nodes, 1))] if "fixed" in call else [])
    with pytest.raises(openndm.InputError, match="stale"):
        getattr(model, call)(*arguments)


def test_refinalizing_makes_the_change_take_effect(tight):
    model, library = _mutable_model(0.08, tight)
    before = model.solve().k_eff
    _rewrite_absorption(library, 0.12)
    library.finalize(warn=False)
    model.refresh()
    after = model.solve().k_eff
    assert after < before - 0.1, (
        f"raising absorption by half must drop k_eff; got {before} -> {after}"
    )


# ------------------------------------------- changing the model mid-transient
def _kinetic_model(tight):
    """A leaky cuboid with kinetics data, so the leakage operator matters."""
    library = openndm.XSLibrary(1, 1)
    library.set_composition(
        0,
        D=[1.0],
        absorption=[0.08],
        nu_fission=[0.1],
        kappa_fission=[0.1],
        chi=[1.0],
        scatter=[[0.0]],
        inv_velocity=[1.0e-6],
    )
    library.set_delayed(beta=[0.0065], decay_constant=[0.08])
    library.finalize(warn=False)
    geometry = openndm.Geometry.from_lattice(
        np.zeros((8, 8, 8), dtype=int),
        pitch=12.5,
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "zero_flux"
        ),
    )
    return openndm.Model(geometry, library, tight), library


def _powers(model, steps=4, change=None):
    transient = model.start_transient(theta=1.0)
    history = []
    for index in range(steps):
        if change is not None and index == 1:
            change()
        history.append(transient.step(0.05).total_power)
    return np.array(history)


def test_a_change_in_diffusion_reaches_the_next_step(tight):
    """Dtilde is derived from D, and a step that refreshed only the cached
    cross sections left the leakage operator reading the old one.

    Nothing else in the operator reads D, so halving it across the whole core
    used to change the power history by exactly nothing -- bit for bit, which
    is the one symptom that cannot be mistaken for physics.
    """
    model, library = _kinetic_model(tight)
    model.solve()
    unchanged = _powers(model)

    model, library = _kinetic_model(tight)
    model.solve()

    def halve_diffusion():
        library.set_composition(0, D=[0.5])
        library.finalize(warn=False)

    changed = _powers(model, change=halve_diffusion)

    assert not np.array_equal(unchanged, changed)
    assert changed[-1] > 1.5 * unchanged[-1], (unchanged, changed)


def test_refreshing_during_a_transient_is_refused(tight):
    """``refresh`` is refused while a transient runs.

    It once emptied the flux, and the next step returned plausible-looking
    power by reading past the end of the vector. A step re-reads the model by
    itself, so there is nothing for ``refresh`` to do.
    """
    model, _ = _kinetic_model(tight)
    model.solve()
    model.start_transient()
    with pytest.raises(openndm.InputError, match="transient is in progress"):
        model.refresh()


def test_a_static_solve_ends_the_transient_and_frees_refresh(tight):
    model, _ = _kinetic_model(tight)
    model.solve()
    transient = model.start_transient()
    transient.step(0.05)
    model.solve()
    model.refresh()
    with pytest.raises(openndm.InputError, match="start_transient"):
        transient.step(0.05)


def test_a_transient_that_changes_nothing_is_unaffected_by_the_refresh(tight):
    """Rebuilding Dtilde every step must not perturb a step that needs it not."""
    model, _ = _kinetic_model(tight)
    model.solve()
    first = _powers(model, steps=6)
    assert np.allclose(first, first[0], rtol=1.0e-10), first


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_a_long_down_scatter_chain_converges(kernel):
    """Eight groups against a zero flux boundary, which once made a two-cycle.

    The one-node boundary problem writes a face current as Dhat times the
    node-average flux. In an intermediate group of a chain that ratio is badly
    conditioned at a boundary, because the flux there is small next to the
    within-node source driving it. Undamped, the update saturated against
    `dhat_limit` and alternated sign forever at about 1e-4 in k.

    Nothing else in the suite has more than two groups, which is why this went
    unnoticed until an eight-group deck was built for the performance cases.
    """
    model = openndm.Model(
        iaea_geometry(planes=4),
        chain_library(),
        openndm.Settings(verbosity=0, max_outer=400),
    )
    result = model.solve(kernel=kernel)
    assert result.converged
    assert result.outer_iterations < 100


@pytest.mark.parametrize("kernel", NODAL_KERNELS)
def test_a_refined_multi_group_core_converges(kernel):
    """The same chain on a finer radial mesh, which damping alone did not fix.

    Under-relaxing the boundary Dhat removed the two-cycle on one node per
    assembly but not here. The cause was the clamp: an update the coarse mesh
    cannot express is untrustworthy, and saturating it at `dhat_limit` puts a
    large wrong number where the old good one was. A boundary update outside
    the band is discarded instead, which falls back to the finite difference
    coupling that face started with.
    """
    model = openndm.Model(
        iaea_geometry(subdivide=2, planes=4),
        chain_library(),
        openndm.Settings(verbosity=0, max_outer=400),
    )
    result = model.solve(kernel=kernel)
    assert result.converged
    assert result.outer_iterations < 100


def test_boundary_relaxation_does_not_move_the_answer():
    """Damping changes the path to the fixed point, not the fixed point.

    At convergence the update is a no-op whatever the factor, so damping
    cannot move the answer it converges to. It is held to 1 pcm rather than to
    round-off because the outer iteration stops on the eigenvalue and the
    fission source and does not watch the change in Dhat, so it can stop while
    Dhat is still moving; how far it has got by then does depend on the path.
    That is a property of the nonlinear scheme and not of the damping, and
    tightening the source tolerance by two decades does not alter the spread.
    """
    eigenvalues = []
    for relaxation in (0.3, 0.5, 0.8):
        model = openndm.Model(
            iaea_geometry(planes=4),
            chain_library(),
            openndm.Settings(
                verbosity=0,
                max_outer=400,
                k_tolerance=1.0e-11,
                fission_source_tolerance=1.0e-10,
                boundary_relaxation=relaxation,
            ),
        )
        eigenvalues.append(model.solve(kernel="sanm").k_eff)
    assert max(eigenvalues) - min(eigenvalues) < 1.0e-5, eigenvalues
