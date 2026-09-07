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


def test_v3_coarse_mesh_nodal_matches_fine_mesh_fdm(tight):
    """V-3: coarse SANM and NEM agree with a refined FDM solve.

    This is the check that catches a wrong transverse leakage, a wrong
    discontinuity factor convention or a broken two-node closure: those leave
    the nodal kernels converging to a different answer than the reference
    kernel, which mesh refinement alone would not reveal.
    """
    library = iaea_library()
    fine = openndm.Model(iaea_geometry(subdivide=8), library, tight)
    reference = fine.solve(kernel="fdm").k_eff

    for kernel in NODAL_KERNELS:
        coarse = openndm.Model(iaea_geometry(subdivide=1), library, tight)
        error_pcm = 1.0e5 * (coarse.solve(kernel=kernel).k_eff - reference)
        assert abs(error_pcm) < 150.0, (
            f"{kernel} on one node per assembly is {error_pcm:+.1f} pcm from "
            f"the fine-mesh FDM reference {reference:.6f}"
        )


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
