"""Code verification against solutions known independently of OpenNDM.

These are the tests that pin down correctness. Everything else checks
plumbing; these check physics.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import openndm

from conftest import (
    ALL_KERNELS,
    NODAL_KERNELS,
    ONE_GROUP,
    analytic_k,
    cuboid,
    iaea_geometry,
    iaea_library,
    one_group_library,
)


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_v1_bare_cuboid_matches_analytic_buckling(kernel, tight):
    """V-1: every kernel reproduces k = nuSf / (Sa + D B^2) on a fine mesh."""
    side = 100.0
    model = openndm.Model(cuboid(32, side), one_group_library(), tight)
    result = model.solve(kernel=kernel)
    reference = analytic_k(side=side, **ONE_GROUP)
    error_pcm = 1.0e5 * (result.k_eff - reference)
    assert abs(error_pcm) < 5.0, f"{kernel}: {error_pcm:+.2f} pcm from analytic"


#: Lower bound on the observed spatial convergence order per kernel. FDM is
#: second order by construction. The nodal kernels are exact inside a node, so
#: their residual error comes from the finite difference treatment of the
#: outer boundary faces and converges faster; they are held to a floor rather
#: than to a band.
EXPECTED_ORDER = {"fdm": (1.8, 2.3), "nem": (1.8, 4.0), "sanm": (1.8, 4.0)}


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_v2_spatial_convergence_order(kernel, tight):
    """V-2: refining the mesh reduces the error at the expected order."""
    side = 100.0
    reference = analytic_k(side=side, **ONE_GROUP)
    errors = []
    for n in (8, 16, 32):
        model = openndm.Model(cuboid(n, side), one_group_library(), tight)
        errors.append(abs(model.solve(kernel=kernel).k_eff - reference))

    low, high = EXPECTED_ORDER[kernel]
    for coarse, fine in itertools.pairwise(errors):
        order = np.log2(coarse / fine)
        assert low < order < high, (
            f"{kernel}: observed convergence order {order:.2f}, expected in "
            f"({low}, {high}); errors {errors}"
        )


@pytest.mark.parametrize("kernel", NODAL_KERNELS)
def test_v1_nodal_kernels_beat_finite_difference(kernel, tight):
    """A nodal kernel must be more accurate than FDM on the same mesh.

    If it is not, the two-node solve is contributing nothing and the nonlinear
    iteration has silently degenerated to plain CMFD.
    """
    side = 100.0
    reference = analytic_k(side=side, **ONE_GROUP)
    model = openndm.Model(cuboid(8, side), one_group_library(), tight)
    fdm_error = abs(model.solve(kernel="fdm").k_eff - reference)
    nodal_error = abs(model.solve(kernel=kernel).k_eff - reference)
    assert nodal_error < 0.6 * fdm_error, (
        f"{kernel} error {nodal_error:.3e} is not better than FDM "
        f"{fdm_error:.3e}"
    )


#: Mesh-converged IAEA-2D eigenvalue. SANM and NEM both sit on this to within
#: 0.1 pcm from four nodes per assembly onward, and FDM approaches it from
#: below at second order.
IAEA_2D_CONVERGED = 1.0295271

#: Tolerance in pcm on how far a nodal kernel may sit from the converged
#: eigenvalue when run at one node per assembly.
COARSE_MESH_TOLERANCE = {"sanm": 25.0, "nem": 150.0}


@pytest.mark.parametrize("kernel", NODAL_KERNELS)
def test_v3_coarse_mesh_nodal_matches_fine_mesh_fdm(kernel, tight):
    """V-3: coarse SANM and NEM agree with a refined finite difference solve.

    This is the check that catches a wrong transverse leakage, a wrong
    discontinuity factor convention or a broken two-node closure: those leave
    the nodal kernels converging to a different answer than the reference
    kernel, which mesh refinement alone would not reveal. FDM is the reference
    because it shares no machinery with either nodal kernel.
    """
    library = iaea_library()
    reference = (
        openndm.Model(iaea_geometry(subdivide=8), library, tight)
        .solve(kernel="fdm")
        .k_eff
    )
    coarse = openndm.Model(iaea_geometry(subdivide=1), library, tight)
    error_pcm = 1.0e5 * (coarse.solve(kernel=kernel).k_eff - reference)
    assert abs(error_pcm) < COARSE_MESH_TOLERANCE[kernel], (
        f"{kernel} on one node per assembly is {error_pcm:+.1f} pcm from the "
        f"fine-mesh FDM reference {reference:.6f}"
    )


def test_v3_sanm_and_nem_agree_to_a_fraction_of_a_pcm(tight):
    """The two nodal kernels share only the transverse leakage fit.

    NEM closes its two-node problem with quartic polynomials, two moment
    equations and the coarse-mesh outer-face currents; SANM closes it with
    analytic basis functions and the node-average constraint alone. They have
    no reason to agree this closely unless both are right, which makes this
    the sharpest single check in the suite.
    """
    library = iaea_library()
    geometry = iaea_geometry(subdivide=4)
    nem = openndm.Model(geometry, library, tight).solve(kernel="nem").k_eff
    sanm = openndm.Model(geometry, library, tight).solve(kernel="sanm").k_eff
    assert abs(1.0e5 * (nem - sanm)) < 1.0, (nem, sanm)
    assert abs(1.0e5 * (sanm - IAEA_2D_CONVERGED)) < 2.0, sanm


def test_v3_kernels_converge_to_the_same_limit(tight):
    """Under refinement all three kernels must agree to a few pcm."""
    library = iaea_library()
    results = {
        kernel: openndm.Model(iaea_geometry(subdivide=8), library, tight)
        .solve(kernel=kernel)
        .k_eff
        for kernel in ALL_KERNELS
    }
    spread_pcm = 1.0e5 * (max(results.values()) - min(results.values()))
    assert spread_pcm < 25.0, f"kernels disagree by {spread_pcm:.1f} pcm: {results}"
    for kernel, k in results.items():
        assert abs(1.0e5 * (k - IAEA_2D_CONVERGED)) < 25.0, (kernel, k)


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_v4_adjoint_eigenvalue_matches_forward(kernel, tight):
    """V-4: the adjoint operator is the transpose, so it shares the spectrum."""
    model = openndm.Model(iaea_geometry(), iaea_library(), tight)
    forward = model.solve(kernel=kernel)
    adjoint = model.solve_adjoint(kernel=kernel)
    difference_pcm = 1.0e5 * abs(forward.k_eff - adjoint.k_eff)
    assert difference_pcm < 1.0, (
        f"{kernel}: adjoint k_eff differs from forward by "
        f"{difference_pcm:.3f} pcm"
    )


def test_adjoint_flux_is_not_the_forward_flux(tight):
    """A transposed solve must actually produce a different flux shape.

    Without this, an adjoint that quietly ran the forward problem would pass
    the eigenvalue check above.
    """
    model = openndm.Model(iaea_geometry(), iaea_library(), tight)
    forward = np.asarray(model.solve().flux)
    adjoint = np.asarray(model.solve_adjoint().flux)
    # Compare group-wise shapes, each normalised, so the difference cannot be
    # explained by an overall scaling.
    forward_shape = forward / forward.sum(axis=0)
    adjoint_shape = adjoint / adjoint.sum(axis=0)
    assert not np.allclose(forward_shape, adjoint_shape, atol=1.0e-4)


def test_infinite_medium_reproduces_k_infinity(tight):
    """A fully reflected node must return the infinite-medium eigenvalue.

    For the two-group IAEA fuel this is nuSf2 * Ss12 / (Sa2 * (Sa1 + Ss12)).
    """
    library = iaea_library()
    geometry = openndm.Geometry.from_lattice(
        np.zeros((1, 1, 1), dtype=int),
        pitch=20.0,
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "reflective"
        ),
    )
    result = openndm.Model(geometry, library, tight).solve()
    a1, a2, f2, s12 = 0.010, 0.080, 0.135, 0.020
    k_inf = f2 * s12 / (a2 * (a1 + s12))
    assert result.k_eff == pytest.approx(k_inf, abs=1.0e-10)


# ------------------------------------------------- adjoint perturbation theory
def _fission_inner_product(model, forward, adjoint):
    r"""\langle \phi^\dagger, F \phi \rangle over the whole core."""
    geometry, library = model.geometry, model.library
    compositions = geometry.compositions
    nu_fission = library.array("nu_fission")[compositions]
    chi = library.array("chi")[compositions]
    volume = geometry.volumes
    production = np.einsum("ng,ng->n", nu_fission, forward)
    return float(np.einsum("ng,ng,n->", adjoint, chi, production * volume))


def test_v4_first_order_perturbation_theory_matches_a_direct_resolve(tight):
    r"""V-4: the adjoint flux weighted against a perturbation.

    Checking that the adjoint eigenvalue equals the forward one only confirms
    that the operator was transposed; it says nothing about the adjoint flux
    *shape*, which is what the adjoint is actually for. First-order
    perturbation theory does test the shape:

    .. math::

        \delta(1/k) = \frac{\langle \phi^\dagger, \delta A \phi \rangle}
                           {\langle \phi^\dagger, F \phi \rangle}

    where the term in :math:`\delta\phi` drops out precisely because
    :math:`\phi^\dagger` is the adjoint eigenfunction. A wrong adjoint shape
    leaves the estimate biased by an amount that does not shrink with the
    perturbation, so the error must fall linearly as the perturbation does.

    Run on the finite difference kernel, so that the operator is a fixed
    linear system and the nonlinear coupling coefficients cannot shift
    underneath the perturbation.
    """
    geometry = iaea_geometry()
    library = iaea_library()
    model = openndm.Model(geometry, library, tight)

    forward = np.asarray(model.solve(kernel="fdm").flux)
    adjoint = np.asarray(model.solve_adjoint(kernel="fdm").flux)
    k0 = model.solve(kernel="fdm").k_eff
    denominator = _fission_inner_product(model, forward, adjoint)

    # Perturb the thermal absorption of the inner fuel, composition 1.
    base = library.composition(1).absorption[1]
    target = geometry.compositions == 1
    volume = geometry.volumes

    errors = []
    for delta in (1.0e-3, 5.0e-4, 2.5e-4):
        numerator = float(
            np.sum(adjoint[target, 1] * delta * volume[target] * forward[target, 1])
        )
        predicted = 1.0 / k0 + numerator / denominator

        library.set_composition(1, absorption=[0.010, base + delta])
        library.finalize(warn=False)
        model.refresh()
        actual = 1.0 / model.solve(kernel="fdm").k_eff
        library.set_composition(1, absorption=[0.010, base])
        library.finalize(warn=False)
        model.refresh()

        errors.append(abs(predicted - actual) / abs(actual - 1.0 / k0))

    # First-order theory is exact to O(delta^2), so halving the perturbation
    # must halve the relative error. A wrong adjoint shape gives a floor.
    assert errors[-1] < 0.05, errors
    for coarse, fine in itertools.pairwise(errors):
        assert fine < 0.65 * coarse, errors


def test_adjoint_weighting_differs_from_flux_weighting(tight):
    """The adjoint must actually be doing something.

    If the adjoint flux were quietly the forward flux, perturbation theory
    above would still work for a uniform perturbation. Weighting a *localised*
    perturbation is where the two diverge.
    """
    model = openndm.Model(iaea_geometry(), iaea_library(), tight)
    forward = np.asarray(model.solve(kernel="fdm").flux)
    adjoint = np.asarray(model.solve_adjoint(kernel="fdm").flux)
    ratio = adjoint[:, 1] / forward[:, 1]
    assert ratio.max() / ratio.min() > 1.05, ratio.max() / ratio.min()


# ------------------------------------------------------ symmetry invariance
def test_symmetric_core_map_gives_a_symmetric_power_distribution(tight):
    """The IAEA map is symmetric about the diagonal, so its power must be too.

    This is the cheapest possible check on the x and y indexing paths, and it
    fails on any asymmetry between them: a transposed lattice index, a face
    ordering mistake, or a boundary condition applied to the wrong axis.
    """
    model = openndm.Model(iaea_geometry(), iaea_library(), tight)
    radial = model.solve().radial_power()
    assert np.allclose(radial, radial.T, atol=1.0e-10), np.abs(
        radial - radial.T
    ).max()


def _asymmetric_core():
    """A deliberately lopsided core, so a rotation is a real change."""
    core = np.full((3, 6, 6), 1, dtype=int)
    core[:, :4, :4] = 0
    core[:, 0, 0] = 2
    core[:, 5, :] = openndm.INACTIVE
    core[:, :, 5] = openndm.INACTIVE
    return core


def _rotation_library():
    lib = openndm.XSLibrary(2, 3)
    for index, (a2, f2) in enumerate(((0.080, 0.135), (0.100, 0.120), (0.130, 0.135))):
        lib.set_composition(
            index,
            D=[1.5, 0.4],
            absorption=[0.010, a2],
            nu_fission=[0.0, f2],
            kappa_fission=[0.0, f2],
            chi=[1.0, 0.0],
            scatter=[[0.0, 0.020], [0.0, 0.0]],
        )
    lib.finalize(warn=False)
    return lib


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_rotating_the_core_rotates_the_power_and_leaves_k_unchanged(kernel, tight):
    """Rotating the whole problem must be a relabelling and nothing more."""
    library = _rotation_library()
    boundaries = dict.fromkeys(
        ["x_min", "x_max", "y_min", "y_max"], "vacuum"
    ) | {"z_min": "reflective", "z_max": "reflective"}

    def solve(core):
        geometry = openndm.Geometry.from_lattice(
            core, pitch=20.0, boundaries=boundaries, outside="vacuum"
        )
        result = openndm.Model(geometry, library, tight).solve(kernel=kernel)
        return result.k_eff, result.radial_power()

    core = _asymmetric_core()
    k0, power0 = solve(core)
    k90, power90 = solve(np.rot90(core, k=1, axes=(1, 2)))

    # The two solves take different iteration paths through the same problem,
    # so they agree to the outer convergence tolerance rather than to machine
    # precision. That is still 0.0001 pcm.
    assert k90 == pytest.approx(k0, abs=1.0e-9)
    assert np.allclose(np.rot90(power0, k=1), power90, atol=1.0e-7)


def test_mirroring_the_core_mirrors_the_power(tight):
    library = _rotation_library()
    boundaries = dict.fromkeys(
        ["x_min", "x_max", "y_min", "y_max"], "vacuum"
    ) | {"z_min": "reflective", "z_max": "reflective"}

    def solve(core):
        geometry = openndm.Geometry.from_lattice(
            core, pitch=20.0, boundaries=boundaries, outside="vacuum"
        )
        result = openndm.Model(geometry, library, tight).solve()
        return result.k_eff, result.radial_power()

    core = _asymmetric_core()
    k0, power0 = solve(core)
    k1, power1 = solve(core[:, ::-1, :])
    assert k1 == pytest.approx(k0, abs=1.0e-9)
    assert np.allclose(power0[::-1, :], power1, atol=1.0e-7)


# --------------------------------------------------------- neutron balance
@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_node_neutron_balance_closes(kernel, tight):
    """Every node must conserve neutrons to the iteration tolerance.

    The standard internal consistency check for a nodal code. It fails on a
    wrong coupling coefficient, a mis-assembled scattering term, or a boundary
    condition applied to the wrong face, none of which need move k_eff far
    enough to be obvious.
    """
    model = openndm.Model(iaea_geometry(planes=4), iaea_library(), tight)
    model.solve(kernel=kernel)
    residual = np.abs(model.neutron_balance())
    assert residual.max() < 1.0e-8, residual.max()


def test_neutron_balance_needs_a_solve_first(tight):
    model = openndm.Model(iaea_geometry(), iaea_library(), tight)
    with pytest.raises(openndm.InputError, match="no solution"):
        model.neutron_balance()


def test_global_balance_relates_leakage_absorption_and_production(tight):
    """Core-wide: leakage plus absorption equals production over k.

    Summing the node balance collapses every scattering term, so this is an
    independent statement about the boundary treatment in particular.
    """
    geometry = iaea_geometry()
    library = iaea_library()
    model = openndm.Model(geometry, library, tight)
    result = model.solve()

    flux = np.asarray(result.flux)
    volume = geometry.volumes
    compositions = geometry.compositions
    absorption = library.array("absorption")[compositions]
    nu_fission = library.array("nu_fission")[compositions]

    total_absorption = float(np.einsum("ng,ng,n->", absorption, flux, volume))
    production = float(np.einsum("ng,ng,n->", nu_fission, flux, volume))

    currents = model.surface_currents()
    leakage = 0.0
    for index, surface in enumerate(geometry._g.surfaces):
        if not surface.is_boundary():
            continue
        # Outward normal points along +axis when the node is on the low side.
        sign = 1.0 if surface.lo >= 0 else -1.0
        leakage += sign * float(currents[index].sum()) * surface.area

    assert leakage > 0.0, "a core with vacuum and zero-flux faces must leak"
    balance = leakage + total_absorption - production / result.k_eff
    assert abs(balance) / production < 1.0e-9, balance / production
