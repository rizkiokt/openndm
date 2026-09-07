"""Verification against exactly solvable reactor-physics problems.

Every reference here is a closed-form or transcendental solution of the
diffusion equation, independent of OpenNDM. Unlike the benchmark decks, none
of these depends on a transcribed core map, and unlike the mesh-refinement
studies, none of them uses the code to check itself.

The problems are the standard set a nodal diffusion code is verified against:
a reflected slab, an albedo boundary, and a method of manufactured solutions
for the full multi-group operator including scattering and fission.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import openndm

from conftest import ALL_KERNELS


def bisect(f, lo, hi, tolerance=1.0e-14, max_iterations=200):
    """Smallest root of ``f`` bracketed by ``(lo, hi)``.

    Written out rather than pulled from scipy so the test suite keeps the same
    dependencies as the package.
    """
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(f"root not bracketed: f({lo})={f_lo}, f({hi})={f_hi}")
    for _ in range(max_iterations):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tolerance or (hi - lo) < tolerance:
            return mid
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def slab_geometry(dz, *, top="zero_flux", albedo=None):
    """A one-dimensional slab along z, reflective on every transverse face.

    ``z_min`` is a symmetry plane, so the analytic solution is even about it.
    """
    boundaries = {
        "x_min": "reflective",
        "x_max": "reflective",
        "y_min": "reflective",
        "y_max": "reflective",
        "z_min": "reflective",
        "z_max": top,
    }
    kwargs = {}
    if albedo is not None:
        kwargs["albedo"] = {"z_max": albedo}
    return openndm.Geometry.from_lattice(
        np.zeros((len(dz), 1, 1), dtype=int),
        pitch=1.0,
        dx=[1.0],
        dy=[1.0],
        dz=list(dz),
        boundaries=boundaries,
        **kwargs,
    )


def one_group(D, absorption, nu_fission, n_compositions=1):
    lib = openndm.XSLibrary(1, n_compositions)
    for c in range(n_compositions):
        lib.set_composition(
            c,
            D=[D[c]],
            absorption=[absorption[c]],
            nu_fission=[nu_fission[c]],
            kappa_fission=[nu_fission[c]],
            chi=[1.0] if nu_fission[c] > 0 else [0.0],
            scatter=[[0.0]],
        )
    lib.finalize(warn=False)
    return lib


TIGHT = {
    "verbosity": 0,
    "k_tolerance": 1.0e-12,
    "fission_source_tolerance": 1.0e-11,
    "inner_tolerance": 1.0e-11,
    "max_inner": 600,
    "max_outer": 8000,
}


# --------------------------------------------------- reflected slab, 1 group
#: Core and reflector data for the reflected slab. The core half-width is
#: measured from the symmetry plane.
CORE = {"D": 1.2, "absorption": 0.02, "nu_fission": 0.025, "half_width": 50.0}
REFLECTOR = {"D": 0.4, "absorption": 0.01, "thickness": 20.0}


def reflected_slab_reference():
    r"""Analytic eigenvalue of a slab reactor with one reflector.

    In the core the flux is :math:`A\cos(Bz)`, even about the symmetry plane;
    in the reflector it is :math:`C\sinh(\kappa(a+b-z))`, vanishing at the
    outer face. Matching flux and current at the interface eliminates the
    amplitudes and leaves

    .. math::

        D_1 B \tan(B a) = D_2 \kappa \coth(\kappa b)

    which has exactly one root in :math:`(0, \pi/2a)`. The eigenvalue then
    follows from the core buckling.
    """
    a, b = CORE["half_width"], REFLECTOR["thickness"]
    kappa = math.sqrt(REFLECTOR["absorption"] / REFLECTOR["D"])
    right = REFLECTOR["D"] * kappa / math.tanh(kappa * b)

    def residual(B):
        return CORE["D"] * B * math.tan(B * a) - right

    # tan(Ba) sweeps 0 to infinity across the interval, so the root is unique.
    B = bisect(residual, 1.0e-9, 0.5 * math.pi / a - 1.0e-9)
    k = CORE["nu_fission"] / (CORE["absorption"] + CORE["D"] * B * B)
    return k, B


def build_reflected_slab(n_core, n_reflector):
    dz = [CORE["half_width"] / n_core] * n_core
    dz += [REFLECTOR["thickness"] / n_reflector] * n_reflector
    geometry = openndm.Geometry.from_lattice(
        np.array([0] * n_core + [1] * n_reflector, dtype=int).reshape(-1, 1, 1),
        pitch=1.0,
        dx=[1.0],
        dy=[1.0],
        dz=dz,
        boundaries={
            "x_min": "reflective",
            "x_max": "reflective",
            "y_min": "reflective",
            "y_max": "reflective",
            "z_min": "reflective",
            "z_max": "zero_flux",
        },
    )
    library = one_group(
        D=[CORE["D"], REFLECTOR["D"]],
        absorption=[CORE["absorption"], REFLECTOR["absorption"]],
        nu_fission=[CORE["nu_fission"], 0.0],
        n_compositions=2,
    )
    return geometry, library


def test_reflected_slab_reference_is_self_consistent():
    """Sanity-check the transcendental root before trusting it."""
    k, B = reflected_slab_reference()
    a, b = CORE["half_width"], REFLECTOR["thickness"]
    kappa = math.sqrt(REFLECTOR["absorption"] / REFLECTOR["D"])
    assert CORE["D"] * B * math.tan(B * a) == pytest.approx(
        REFLECTOR["D"] * kappa / math.tanh(kappa * b), rel=1.0e-10
    )
    assert 1.0 < k < 1.5


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_reflected_slab_converges_to_the_analytic_eigenvalue(kernel):
    """A reflected slab is the classic heterogeneous verification problem.

    It exercises a material interface, a reflector and two different boundary
    conditions at once, against a reference that owes nothing to this code.
    """
    reference, _ = reflected_slab_reference()
    geometry, library = build_reflected_slab(50, 20)
    result = openndm.Model(geometry, library, openndm.Settings(**TIGHT)).solve(
        kernel=kernel
    )
    error = 1.0e5 * (result.k_eff - reference)
    assert abs(error) < 20.0, f"{kernel}: {error:+.2f} pcm from analytic"


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_reflected_slab_error_falls_under_refinement(kernel):
    reference, _ = reflected_slab_reference()
    errors = []
    for factor in (1, 2, 4):
        geometry, library = build_reflected_slab(5 * factor, 2 * factor)
        result = openndm.Model(
            geometry, library, openndm.Settings(**TIGHT)
        ).solve(kernel=kernel)
        errors.append(abs(result.k_eff - reference))
    for coarse, fine in itertools.pairwise(errors):
        assert fine < coarse, errors
    order = math.log2(errors[0] / errors[-1]) / 2.0
    assert order > 1.5, f"{kernel}: observed order {order:.2f}, errors {errors}"


def test_reflected_slab_nodal_kernels_beat_finite_difference():
    """On a coarse mesh the nodal kernels must earn their extra work."""
    reference, _ = reflected_slab_reference()
    geometry, library = build_reflected_slab(5, 2)
    settings = openndm.Settings(**TIGHT)
    errors = {
        kernel: abs(
            openndm.Model(geometry, library, settings).solve(kernel=kernel).k_eff
            - reference
        )
        for kernel in ALL_KERNELS
    }
    assert errors["sanm"] < 0.5 * errors["fdm"], errors
    assert errors["nem"] < errors["fdm"], errors


def test_reflected_slab_flux_shape_matches_the_analytic_solution():
    """The eigenvalue can be right while the shape is wrong; check both."""
    _, B = reflected_slab_reference()
    n_core, n_reflector = 100, 40
    geometry, library = build_reflected_slab(n_core, n_reflector)
    result = openndm.Model(geometry, library, openndm.Settings(**TIGHT)).solve()

    widths = geometry.volumes  # unit transverse area, so volume is the width
    edges = np.concatenate(([0.0], np.cumsum(widths)))
    a, b = CORE["half_width"], REFLECTOR["thickness"]
    kappa = math.sqrt(REFLECTOR["absorption"] / REFLECTOR["D"])

    # Node-average of the analytic solution, integrated exactly.
    exact = np.empty(geometry.n_nodes)
    for i in range(n_core):
        lo, hi = edges[i], edges[i + 1]
        exact[i] = (math.sin(B * hi) - math.sin(B * lo)) / (B * (hi - lo))
    scale = math.cos(B * a) / math.sinh(kappa * b)
    for i in range(n_core, geometry.n_nodes):
        lo, hi = edges[i], edges[i + 1]
        exact[i] = (
            scale
            * (math.cosh(kappa * (a + b - lo)) - math.cosh(kappa * (a + b - hi)))
            / (kappa * (hi - lo))
        )

    computed = np.asarray(result.flux).ravel()
    computed = computed / computed[0] * exact[0]
    relative = np.abs(computed - exact) / exact.max()
    assert relative.max() < 2.0e-3, relative.max()


# ------------------------------------------------------- albedo boundary
ALBEDO_SLAB = {
    "D": 1.0,
    "absorption": 0.02,
    "nu_fission": 0.025,
    "half_width": 50.0,
}


def albedo_slab_reference(beta):
    r"""Analytic eigenvalue of a bare slab closed by an albedo boundary.

    Writing the boundary condition as :math:`J = \gamma\phi_s` with
    :math:`\gamma = (1-\beta)/2(1+\beta)`, the flux :math:`A\cos(Bz)` gives

    .. math::  D B \tan(B a) = \gamma

    so the whole albedo formulation reduces to a one-line root find. The two
    limits are recognisable: :math:`\beta = 1` gives :math:`\gamma = 0` and a
    reflective face, :math:`\beta = -1` gives :math:`\gamma = \infty` and a
    zero-flux face.
    """
    D, a = ALBEDO_SLAB["D"], ALBEDO_SLAB["half_width"]
    gamma = (1.0 - beta) / (2.0 * (1.0 + beta))

    def residual(B):
        return D * B * math.tan(B * a) - gamma

    B = bisect(residual, 1.0e-12, 0.5 * math.pi / a - 1.0e-12)
    k = ALBEDO_SLAB["nu_fission"] / (
        ALBEDO_SLAB["absorption"] + D * B * B
    )
    return k


def build_albedo_slab(n_nodes, *, top="albedo", beta=None):
    dz = [ALBEDO_SLAB["half_width"] / n_nodes] * n_nodes
    geometry = slab_geometry(
        dz, top=top, albedo=None if beta is None else [beta]
    )
    library = one_group(
        D=[ALBEDO_SLAB["D"]],
        absorption=[ALBEDO_SLAB["absorption"]],
        nu_fission=[ALBEDO_SLAB["nu_fission"]],
    )
    return geometry, library


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_albedo_boundary_matches_the_analytic_eigenvalue(beta, kernel):
    """Verifies the albedo coupling coefficient, not just its limits."""
    reference = albedo_slab_reference(beta)
    geometry, library = build_albedo_slab(200, beta=beta)
    result = openndm.Model(geometry, library, openndm.Settings(**TIGHT)).solve(
        kernel=kernel
    )
    error = 1.0e5 * (result.k_eff - reference)
    assert abs(error) < 15.0, f"beta={beta} {kernel}: {error:+.2f} pcm"


def test_albedo_of_one_is_a_reflective_face():
    """beta = 1 means every neutron returns, so there is no leakage at all."""
    geometry, library = build_albedo_slab(20, beta=1.0)
    settings = openndm.Settings(**TIGHT)
    k_albedo = openndm.Model(geometry, library, settings).solve().k_eff
    reflective, _ = build_albedo_slab(20, top="reflective")
    k_reflective = openndm.Model(reflective, library, settings).solve().k_eff
    k_infinity = ALBEDO_SLAB["nu_fission"] / ALBEDO_SLAB["absorption"]
    assert k_albedo == pytest.approx(k_reflective, abs=1.0e-12)
    assert k_albedo == pytest.approx(k_infinity, abs=1.0e-10)


def test_albedo_of_zero_is_a_vacuum_face():
    """beta = 0 means nothing returns, which is the Marshak condition."""
    geometry, library = build_albedo_slab(40, beta=0.0)
    settings = openndm.Settings(**TIGHT)
    k_albedo = openndm.Model(geometry, library, settings).solve().k_eff
    vacuum, _ = build_albedo_slab(40, top="vacuum")
    k_vacuum = openndm.Model(vacuum, library, settings).solve().k_eff
    assert k_albedo == pytest.approx(k_vacuum, abs=1.0e-12)


def test_albedo_of_minus_one_is_a_zero_flux_face():
    geometry, library = build_albedo_slab(40, beta=-1.0)
    settings = openndm.Settings(**TIGHT)
    k_albedo = openndm.Model(geometry, library, settings).solve().k_eff
    zero_flux, _ = build_albedo_slab(40, top="zero_flux")
    k_zero = openndm.Model(zero_flux, library, settings).solve().k_eff
    assert k_albedo == pytest.approx(k_zero, abs=1.0e-12)


def test_eigenvalue_decreases_monotonically_with_the_albedo():
    """More leakage must mean less reactivity, with no crossings."""
    settings = openndm.Settings(**TIGHT)
    values = []
    for beta in (1.0, 0.9, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0):
        geometry, library = build_albedo_slab(40, beta=beta)
        values.append(openndm.Model(geometry, library, settings).solve().k_eff)
    assert all(a > b for a, b in itertools.pairwise(values)), values


# --------------------------------------- method of manufactured solutions
#: Subcritical two-group medium for the manufactured solution, chosen so that
#: both group sources come out positive.
MMS = {
    "D": (1.5, 0.4),
    "absorption": (0.01, 0.05),
    "down_scatter": 0.02,
    "nu_fission": (0.0, 0.05),
    "chi": (1.0, 0.0),
    "amplitude": (1.0, 0.6),
    "side": 100.0,
}


def mms_library():
    lib = openndm.XSLibrary(2, 1)
    lib.set_composition(
        0,
        D=list(MMS["D"]),
        absorption=list(MMS["absorption"]),
        nu_fission=list(MMS["nu_fission"]),
        kappa_fission=list(MMS["nu_fission"]),
        chi=list(MMS["chi"]),
        scatter=[[0.0, MMS["down_scatter"]], [0.0, 0.0]],
    )
    lib.finalize(warn=False)
    return lib


def _mean_sine(lo, hi, side):
    """Exact average of sin(pi x / side) over ``[lo, hi]``."""
    k = math.pi / side
    return (math.cos(k * lo) - math.cos(k * hi)) / (k * (hi - lo))


def mms_source_amplitudes():
    r"""Constant multiplying the shape function in each group's source.

    Substituting :math:`\phi_g = c_g \sin(\pi x/L)\sin(\pi y/L)\sin(\pi z/L)`
    into the multi-group diffusion operator leaves the same shape function
    times a constant, because the Laplacian of the product is
    :math:`-3(\pi/L)^2` times itself. That is what makes this manufactured
    solution usable: the source is the analytic shape, node-averaged exactly.
    """
    side = MMS["side"]
    buckling = 3.0 * (math.pi / side) ** 2
    c = MMS["amplitude"]
    removal = (
        MMS["absorption"][0] + MMS["down_scatter"],
        MMS["absorption"][1],
    )
    fission = sum(MMS["nu_fission"][g] * c[g] for g in range(2))
    s0 = (MMS["D"][0] * buckling + removal[0]) * c[0] - MMS["chi"][0] * fission
    s1 = (
        (MMS["D"][1] * buckling + removal[1]) * c[1]
        - MMS["down_scatter"] * c[0]
        - MMS["chi"][1] * fission
    )
    return s0, s1


def mms_case(n_nodes):
    """Geometry, node-average source and the exact node-average flux."""
    side = MMS["side"]
    h = side / n_nodes
    geometry = openndm.Geometry.from_lattice(
        np.zeros((n_nodes, n_nodes, n_nodes), dtype=int),
        pitch=h,
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "zero_flux"
        ),
    )
    edges = np.linspace(0.0, side, n_nodes + 1)
    axis_mean = np.array(
        [_mean_sine(edges[i], edges[i + 1], side) for i in range(n_nodes)]
    )
    # Node index order is (k, j, i) flattened, matching Geometry.expand.
    shape = (
        axis_mean[:, None, None] * axis_mean[None, :, None]
        * axis_mean[None, None, :]
    ).ravel()

    s0, s1 = mms_source_amplitudes()
    source = np.column_stack([s0 * shape, s1 * shape])
    exact = np.column_stack(
        [MMS["amplitude"][0] * shape, MMS["amplitude"][1] * shape]
    )
    return geometry, source, exact


def mms_error(n_nodes, kernel):
    geometry, source, exact = mms_case(n_nodes)
    result = openndm.Model(
        geometry, mms_library(), openndm.Settings(**TIGHT)
    ).solve_fixed_source(source, kernel=kernel)
    computed = np.asarray(result.flux)
    return float(
        np.linalg.norm(computed - exact) / np.linalg.norm(exact)
    ), computed, exact


def test_mms_source_amplitudes_are_positive():
    """A negative manufactured source would still be solvable but unphysical."""
    s0, s1 = mms_source_amplitudes()
    assert s0 > 0.0 and s1 > 0.0, (s0, s1)


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_mms_recovers_the_manufactured_flux(kernel):
    """V-2: the full multi-group operator, scattering and fission included.

    Everything the eigenvalue tests exercise is exercised here too, but
    against a solution chosen in advance rather than one the code produces,
    so a compensating pair of errors cannot hide.
    """
    error, computed, _ = mms_error(16, kernel)
    assert error < 5.0e-3, f"{kernel}: relative L2 error {error:.3e}"
    assert np.all(computed > 0.0)


#: Observed order of accuracy band per kernel on the manufactured solution.
#: FDM is second order exactly. The nodal kernels do better because the
#: analytic basis resolves the within-node shape, and the external source is
#: expanded quadratically rather than treated as flat.
MMS_ORDER = {"fdm": (1.8, 2.2), "nem": (1.8, 3.5), "sanm": (1.8, 3.5)}


@pytest.mark.parametrize("kernel", ALL_KERNELS)
def test_mms_converges_at_the_expected_order(kernel):
    """V-2: the observed spatial order of accuracy of each kernel."""
    errors = [mms_error(n, kernel)[0] for n in (8, 16, 32)]
    low, high = MMS_ORDER[kernel]
    for coarse, fine in itertools.pairwise(errors):
        order = math.log2(coarse / fine)
        assert low < order < high, (
            f"{kernel}: observed order {order:.2f}, errors {errors}"
        )


@pytest.mark.parametrize("kernel", ["nem", "sanm"])
def test_mms_nodal_kernels_see_the_external_source(kernel):
    """The two-node problem must include the external source.

    Without it a fixed-source nodal solve reconstructs the within-node shape
    from the scattering and fission sources alone, and is markedly *worse*
    than plain finite difference rather than better. That is the signature to
    watch for: a nodal kernel losing to FDM on a smooth problem means its
    local problem is missing a term.
    """
    fdm = mms_error(16, "fdm")[0]
    nodal = mms_error(16, kernel)[0]
    assert nodal < fdm, (
        f"{kernel} error {nodal:.3e} is worse than FDM {fdm:.3e}, which means "
        f"the two-node problem is not seeing the external source"
    )


def test_mms_kernels_agree_with_each_other():
    """All three solve the same fixed-source problem, so all three must agree."""
    solutions = {k: mms_error(16, k)[1] for k in ALL_KERNELS}
    reference = solutions["sanm"]
    for kernel, values in solutions.items():
        relative = np.linalg.norm(values - reference) / np.linalg.norm(reference)
        assert relative < 5.0e-3, (kernel, relative)
