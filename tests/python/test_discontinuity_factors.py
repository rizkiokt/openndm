"""Verification of the discontinuity factor path (FR-XS-4).

Until now the only tests of this path were that factors default to 1.0 and
survive an HDF5 round trip. Neither says anything about whether the coupling
coefficient built from them is right, and a wrong one is easy to write: the
factor can be applied to one side only, folded into the wrong term, or read
from the wrong face index. Any of those changes k_eff without producing a
single other symptom.

The central test here is the equivalence theorem itself. Homogenise a
heterogeneous problem by flux-volume weighting, take the discontinuity factors
implied by the reference solution, and the coarse solve must reproduce the
reference eigenvalue and node-average fluxes *exactly* — not approximately,
which is the whole point of generalised equivalence theory. It holds for any
choice of homogenised diffusion coefficient, so a deliberately poor one is
used to show the factors absorbing it.
"""

from __future__ import annotations

import numpy as np
import pytest

import openndm

TIGHT = {
    "verbosity": 0,
    "k_tolerance": 1.0e-13,
    "fission_source_tolerance": 1.0e-12,
    "inner_tolerance": 1.0e-12,
    "max_inner": 800,
    "max_outer": 20000,
}

#: One period of the heterogeneous slab: six fuel nodes then four moderator.
#:
#: The fuel differs from period to period on purpose. A periodic stack with
#: reflective outer faces has identical coarse nodes and a flat coarse flux,
#: so its eigenvalue is fixed by the homogenised absorption and production
#: ratio alone and no coupling coefficient, right or wrong, can move it. That
#: makes it useless as a test of discontinuity factors. Tilting the fuel
#: across the stack puts a real current between coarse nodes.
FUEL_ABSORPTION = (0.016, 0.019, 0.022, 0.025, 0.028, 0.031)
FUEL_NU_FISSION = (0.030, 0.029, 0.028, 0.027, 0.026, 0.025)
MODERATOR = {"D": 0.6, "absorption": 0.005, "nu_fission": 0.0}
FINE_WIDTH = 2.0
FUEL_NODES = 6
MODERATOR_NODES = 4
PERIODS = len(FUEL_ABSORPTION)

#: Composition index of the moderator, which follows the per-period fuels.
MODERATOR_INDEX = PERIODS

PERIOD_NODES = FUEL_NODES + MODERATOR_NODES
#: Face index of the low and high z faces, in the order -x +x -y +y -z +z.
Z_MIN_FACE, Z_MAX_FACE = 4, 5


def one_group_library(materials):
    lib = openndm.XSLibrary(1, len(materials))
    for index, m in enumerate(materials):
        lib.set_composition(
            index,
            D=[m["D"]],
            absorption=[m["absorption"]],
            nu_fission=[m["nu_fission"]],
            kappa_fission=[m["nu_fission"]],
            chi=[1.0] if m["nu_fission"] > 0.0 else [0.0],
            scatter=[[0.0]],
        )
    lib.finalize(warn=False)
    return lib


def stack_geometry(widths, compositions):
    """A z-stack of nodes, reflective on every face."""
    return openndm.Geometry.from_lattice(
        np.asarray(compositions, dtype=int).reshape(-1, 1, 1),
        pitch=1.0,
        dx=[1.0],
        dy=[1.0],
        dz=list(widths),
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "reflective"
        ),
    )


def fine_materials():
    materials = [
        {"D": 1.2, "absorption": a, "nu_fission": f}
        for a, f in zip(FUEL_ABSORPTION, FUEL_NU_FISSION, strict=True)
    ]
    materials.append(MODERATOR)
    return materials


def fine_problem():
    compositions = []
    for period in range(PERIODS):
        compositions += [period] * FUEL_NODES
        compositions += [MODERATOR_INDEX] * MODERATOR_NODES
    widths = [FINE_WIDTH] * len(compositions)
    return (
        stack_geometry(widths, compositions),
        one_group_library(fine_materials()),
        np.asarray(compositions),
    )


def interface_current(model, geometry, lo_node):
    """Net current on the z surface between ``lo_node`` and the next node."""
    currents = model.surface_currents()
    for index, surface in enumerate(geometry._g.surfaces):
        if surface.axis == 2 and surface.lo == lo_node and surface.hi == lo_node + 1:
            return float(currents[index, 0])
    raise AssertionError(f"no z surface above node {lo_node}")


def test_fine_reference_is_heterogeneous_enough_to_matter():
    """A flat flux would make every discontinuity factor trivially one."""
    geometry, library, _ = fine_problem()
    model = openndm.Model(geometry, library, openndm.Settings(**TIGHT))
    flux = np.asarray(model.solve(kernel="fdm").flux).ravel()
    assert flux.max() / flux.min() > 1.2, flux.max() / flux.min()


def build_equivalent_coarse(d_choice="volume"):
    r"""Homogenise the fine problem and derive the exact discontinuity factors.

    Returns the coarse geometry, its library, the reference eigenvalue and the
    reference coarse node-average fluxes.

    The factors follow from their definition. On each side of an interface the
    homogeneous node's surface flux is what the finite difference relation
    :math:`J = 2D(\bar\phi - \phi_s)/h` gives for the reference current, and
    the discontinuity factor is the ratio of the true heterogeneous surface
    flux to that. Both sides are divided into the *same* heterogeneous surface
    flux, which is what makes the ADF-weighted continuity condition hold.
    """
    geometry, library, compositions = fine_problem()
    model = openndm.Model(geometry, library, openndm.Settings(**TIGHT))
    reference = model.solve(kernel="fdm")
    k_reference = reference.k_eff
    fine_flux = np.asarray(reference.flux).ravel()
    fine_volume = geometry.volumes
    fine_D = np.array(
        [library.composition(int(c)).D[0] for c in compositions]
    )

    n_coarse = PERIODS
    coarse_flux = np.empty(n_coarse)
    coarse_D = np.empty(n_coarse)
    coarse_absorption = np.empty(n_coarse)
    coarse_nu_fission = np.empty(n_coarse)
    coarse_width = np.empty(n_coarse)

    for c in range(n_coarse):
        s = slice(c * PERIOD_NODES, (c + 1) * PERIOD_NODES)
        weight = fine_flux[s] * fine_volume[s]
        volume = fine_volume[s].sum()
        coarse_width[c] = volume  # unit transverse area
        coarse_flux[c] = weight.sum() / volume
        absorption = np.array(
            [library.composition(int(i)).absorption[0] for i in compositions[s]]
        )
        production = np.array(
            [library.composition(int(i)).nu_fission[0] for i in compositions[s]]
        )
        coarse_absorption[c] = (absorption * weight).sum() / weight.sum()
        coarse_nu_fission[c] = (production * weight).sum() / weight.sum()
        if d_choice == "volume":
            # Deliberately not the "best" homogenised D. The discontinuity
            # factors have to absorb whatever this is.
            coarse_D[c] = (fine_D[s] * fine_volume[s]).sum() / volume
        else:
            coarse_D[c] = (fine_D[s] * weight).sum() / weight.sum()

    # Reference current and heterogeneous surface flux at each coarse interface.
    adf = np.ones((n_coarse, 6))
    for c in range(n_coarse - 1):
        lo_fine = (c + 1) * PERIOD_NODES - 1
        current = interface_current(model, geometry, lo_fine)
        surface_flux = (
            fine_flux[lo_fine]
            - current * fine_volume[lo_fine] / (2.0 * fine_D[lo_fine])
        )
        homogeneous_lo = (
            coarse_flux[c] - current * coarse_width[c] / (2.0 * coarse_D[c])
        )
        homogeneous_hi = (
            coarse_flux[c + 1]
            + current * coarse_width[c + 1] / (2.0 * coarse_D[c + 1])
        )
        adf[c, Z_MAX_FACE] = surface_flux / homogeneous_lo
        adf[c + 1, Z_MIN_FACE] = surface_flux / homogeneous_hi

    coarse_library = openndm.XSLibrary(1, n_coarse)
    for c in range(n_coarse):
        coarse_library.set_composition(
            c,
            D=[coarse_D[c]],
            absorption=[coarse_absorption[c]],
            nu_fission=[coarse_nu_fission[c]],
            kappa_fission=[coarse_nu_fission[c]],
            chi=[1.0],
            scatter=[[0.0]],
        )
        coarse_library.set_adf(c, adf[c].reshape(6, 1))
    coarse_library.finalize(warn=False)

    coarse_geometry = stack_geometry(coarse_width, list(range(n_coarse)))
    return coarse_geometry, coarse_library, k_reference, coarse_flux, adf


@pytest.mark.parametrize("d_choice", ["volume", "flux"])
def test_equivalence_theory_reproduces_the_reference_exactly(d_choice):
    """The defining property of a discontinuity factor.

    With flux-volume homogenised cross sections and the factors implied by the
    reference solution, the coarse solve must return the reference eigenvalue
    and node-average fluxes exactly, for *any* homogenised diffusion
    coefficient. Getting this to hold is what the factors are for; if it does
    not, the coupling coefficient is wrong.
    """
    geometry, library, k_reference, flux_reference, _ = build_equivalent_coarse(
        d_choice
    )
    result = openndm.Model(geometry, library, openndm.Settings(**TIGHT)).solve(
        kernel="fdm"
    )
    error_pcm = 1.0e5 * (result.k_eff - k_reference)
    assert abs(error_pcm) < 0.01, f"{error_pcm:+.4f} pcm from the reference"

    computed = np.asarray(result.flux).ravel()
    computed = computed / computed.mean() * flux_reference.mean()
    relative = np.abs(computed - flux_reference) / flux_reference
    assert relative.max() < 1.0e-9, relative.max()


def test_the_derived_factors_are_not_all_one():
    """Otherwise the test above would pass on a no-op implementation."""
    *_, adf = build_equivalent_coarse()
    interior = np.concatenate([adf[:-1, Z_MAX_FACE], adf[1:, Z_MIN_FACE]])
    assert np.abs(interior - 1.0).max() > 0.02, np.abs(interior - 1.0).max()


def test_without_the_factors_the_coarse_solve_is_wrong():
    """Homogenisation alone does not reproduce the reference.

    This is the control: it shows the previous test is measuring the factors
    and not merely a well-behaved homogenisation.
    """
    geometry, library, k_reference, _, _adf = build_equivalent_coarse()
    for c in range(library.n_compositions):
        library.set_adf(c, np.ones((6, 1)))
    library.finalize(warn=False)
    result = openndm.Model(geometry, library, openndm.Settings(**TIGHT)).solve(
        kernel="fdm"
    )
    error_pcm = abs(1.0e5 * (result.k_eff - k_reference))
    assert error_pcm > 10.0, (
        f"only {error_pcm:.2f} pcm without discontinuity factors, so this "
        f"problem is too easy to be a control"
    )


# ------------------------------------------------------ structural identities
def test_a_uniform_factor_on_every_face_changes_nothing():
    r"""Scaling every factor by a constant is not a physical change.

    The interface condition is :math:`f_L\phi_{s,L} = f_R\phi_{s,R}`, so a
    common factor cancels. With reflective outer boundaries the boundary
    coupling is zero regardless, so the eigenvalue must be untouched. A test
    that fails here means the factor is entering one side only.
    """
    geometry, library, _compositions = fine_problem()
    settings = openndm.Settings(**TIGHT)
    base = openndm.Model(geometry, library, settings).solve(kernel="fdm").k_eff

    for scale in (0.5, 2.0, 7.0):
        scaled = one_group_library(fine_materials())
        for c in range(scaled.n_compositions):
            scaled.set_adf(c, np.full((6, 1), scale))
        scaled.finalize(warn=False)
        k = openndm.Model(geometry, scaled, settings).solve(kernel="fdm").k_eff
        assert k == pytest.approx(base, abs=1.0e-11), (scale, k, base)


def test_unit_factors_match_a_library_with_none_set():
    geometry, library, _ = fine_problem()
    explicit = one_group_library(fine_materials())
    for c in range(explicit.n_compositions):
        explicit.set_adf(c, np.ones((6, 1)))
    explicit.finalize(warn=False)
    settings = openndm.Settings(**TIGHT)
    a = openndm.Model(geometry, library, settings).solve(kernel="fdm").k_eff
    b = openndm.Model(geometry, explicit, settings).solve(kernel="fdm").k_eff
    assert a == pytest.approx(b, abs=1.0e-13)


def test_two_node_coupling_matches_the_analytic_algebra():
    r"""Direct check of the coupling coefficient, faces included.

    Two absorbing nodes with a fixed source and reflective outer faces reduce
    to a 2x2 system that can be written down:

    .. math::
        \Lambda (f_L\phi_L - f_R\phi_R)/h + \Sigma_a \phi_L = S_L

    with :math:`\Lambda = 2 D_L D_R / (f_L h_L D_R + f_R h_R D_L)`. Using
    different factors on the two faces also pins the face indexing: reading
    the low face where the high one was meant would silently substitute 1.0.
    """
    D, absorption, h = 1.0, 0.05, 10.0
    f_lo, f_hi = 1.3, 0.8
    source = np.array([[2.0], [0.5]])

    library = openndm.XSLibrary(1, 2)
    for index in range(2):
        library.set_composition(
            index, D=[D], absorption=[absorption], scatter=[[0.0]]
        )
    lo_faces = np.ones((6, 1))
    lo_faces[Z_MAX_FACE] = f_lo
    hi_faces = np.ones((6, 1))
    hi_faces[Z_MIN_FACE] = f_hi
    library.set_adf(0, lo_faces)
    library.set_adf(1, hi_faces)
    library.finalize(warn=False)

    geometry = stack_geometry([h, h], [0, 1])
    result = openndm.Model(
        geometry, library, openndm.Settings(**TIGHT)
    ).solve_fixed_source(source, kernel="fdm")
    computed = np.asarray(result.flux).ravel()

    lam = 2.0 * D * D / (f_lo * h * D + f_hi * h * D)
    matrix = np.array(
        [
            [lam * f_lo / h + absorption, -lam * f_hi / h],
            [-lam * f_lo / h, lam * f_hi / h + absorption],
        ]
    )
    expected = np.linalg.solve(matrix, source.ravel())
    assert computed == pytest.approx(expected, rel=1.0e-10)


def test_swapping_the_two_face_factors_changes_the_answer():
    """Guards against the face index being read from the wrong side."""
    D, absorption, h = 1.0, 0.05, 10.0
    source = np.array([[2.0], [0.5]])
    settings = openndm.Settings(**TIGHT)

    def solve(lo_value, hi_value):
        library = openndm.XSLibrary(1, 2)
        for index in range(2):
            library.set_composition(
                index, D=[D], absorption=[absorption], scatter=[[0.0]]
            )
        lo_faces = np.ones((6, 1))
        lo_faces[Z_MAX_FACE] = lo_value
        hi_faces = np.ones((6, 1))
        hi_faces[Z_MIN_FACE] = hi_value
        library.set_adf(0, lo_faces)
        library.set_adf(1, hi_faces)
        library.finalize(warn=False)
        geometry = stack_geometry([h, h], [0, 1])
        return np.asarray(
            openndm.Model(geometry, library, settings)
            .solve_fixed_source(source, kernel="fdm")
            .flux
        ).ravel()

    forward = solve(1.3, 0.8)
    swapped = solve(0.8, 1.3)
    assert not np.allclose(forward, swapped, rtol=1.0e-6)


def test_factors_are_applied_per_group():
    """A two-group factor must act on its own group only."""
    library = openndm.XSLibrary(2, 1)
    library.set_composition(
        0,
        D=[1.5, 0.4],
        absorption=[0.010, 0.085],
        nu_fission=[0.0, 0.135],
        kappa_fission=[0.0, 0.135],
        chi=[1.0, 0.0],
        scatter=[[0.0, 0.020], [0.0, 0.0]],
    )
    values = np.ones((6, 2))
    values[Z_MAX_FACE, 1] = 1.15
    library.set_adf(0, values)
    library.finalize(warn=False)
    assert library.adf(0)[Z_MAX_FACE, 1] == pytest.approx(1.15)
    assert library.adf(0)[Z_MAX_FACE, 0] == pytest.approx(1.0)
