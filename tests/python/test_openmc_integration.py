"""Opt-in integration tests against a real OpenMC installation (§6.3).

Skipped unless OpenMC is importable, the ``openmc`` executable is on PATH and
a nuclear data library is configured. The full validation cases live in
``tests/validation/``; this module runs a deliberately small version of C-1 so
that a developer with OpenMC set up gets the check for free, without pulling a
gigabyte of nuclear data into the default suite.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

VALIDATION = Path(__file__).resolve().parents[1] / "validation"

_reasons = []
if importlib.util.find_spec("openmc") is None:
    _reasons.append("OpenMC is not installed")
if shutil.which("openmc") is None:
    _reasons.append("the openmc executable is not on PATH")
if not os.environ.get("OPENMC_CROSS_SECTIONS"):
    _reasons.append("OPENMC_CROSS_SECTIONS is not set")

pytestmark = [
    pytest.mark.openmc,
    pytest.mark.slow,
    pytest.mark.skipif(bool(_reasons), reason="; ".join(_reasons)),
]


@pytest.fixture(scope="module")
def c1(tmp_path_factory):
    """Run C-1 once and share the results across the tests below."""
    import openmc

    sys.path.insert(0, str(VALIDATION))
    from c1_homogeneous import GROUP_EDGES_2, build_mgxs_library, build_model


    model = build_model(particles=5000, batches=40, inactive=10)
    model.tallies = openmc.Tallies()
    library = build_mgxs_library(model, GROUP_EDGES_2)
    workdir = tmp_path_factory.mktemp("c1")
    statepoint = model.run(cwd=str(workdir), output=False)
    with openmc.StatePoint(statepoint) as sp:
        library.load_from_statepoint(sp)
        keff = sp.keff
    return library, float(keff.nominal_value), float(keff.std_dev)


def _solve(library, **kwargs):
    import openndm
    from openndm.gc import from_mgxs_library

    xslib = from_mgxs_library(library, **kwargs)
    geometry = openndm.Geometry.from_lattice(
        np.zeros((1, 1, 1), dtype=int),
        pitch=20.0,
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "reflective"
        ),
    )
    settings = openndm.Settings(
        verbosity=0, k_tolerance=1.0e-12, fission_source_tolerance=1.0e-11
    )
    return xslib, openndm.Model(geometry, xslib, settings).solve()


def test_c1_translation_reproduces_openmc_k_infinity(c1):
    """C-1: MGXS in, k_eff out, with nothing else in the way.

    The tolerance is loose because this runs only 150k histories to stay
    inside a unit test; ``tests/validation/c1_homogeneous.py`` is the real
    case.
    """
    library, k_openmc, sigma = c1
    _, result = _solve(library)
    difference = 1.0e5 * (result.k_eff - k_openmc)
    assert abs(difference) < max(300.0, 4.0e5 * sigma), (
        f"{difference:+.1f} pcm from OpenMC ({1e5 * sigma:.1f} pcm sigma)"
    )


def test_c1_multiplicity_correction_moves_k_the_right_way(c1):
    """Disabling the (n,xn) correction must lower k by the (n,xn) production.

    This is the defect C-1 found: without it the translation silently loses
    every neutron produced by (n,2n) and (n,3n).
    """
    library, _, _ = c1
    _, corrected = _solve(library)
    _, ignored = _solve(library, scattering_multiplicity="ignore")
    shift = 1.0e5 * (corrected.k_eff - ignored.k_eff)
    assert shift > 50.0, f"correction moved k by only {shift:+.1f} pcm"


def test_c1_diffusion_coefficient_does_not_affect_an_infinite_medium(c1):
    """With no leakage, D must not touch the eigenvalue at all.

    If it does, the geometry is not actually infinite and every other
    conclusion from this case is suspect.
    """
    library, _, _ = c1
    _, from_dc = _solve(library, prefer="diffusion-coefficient")
    _, from_tr = _solve(library, prefer="transport")
    assert from_dc.k_eff == pytest.approx(from_tr.k_eff, abs=1.0e-9)


def test_c1_carries_uncertainties_and_a_sane_spectrum(c1):
    """FR-XS-9 and FR-OMC-6, plus a physical sanity check on the result."""
    library, _, _ = c1
    xslib, result = _solve(library)
    assert xslib.has_uncertainty
    comp = xslib.composition(0)
    # Group 1 is the fast group, so thermal absorption dominates and every
    # fission neutron is born fast.
    assert comp.absorption[1] > comp.absorption[0]
    assert comp.chi[0] == pytest.approx(1.0, abs=1.0e-6)
    flux = np.asarray(result.flux).ravel()
    assert flux[0] > flux[1] > 0.0


# ------------------------------------------------------------------ C-2
@pytest.fixture(scope="module")
def c2(tmp_path_factory):
    """A small heterogeneous pin lattice, run once for the tests below."""
    import openmc
    import openmc.mgxs

    sys.path.insert(0, str(VALIDATION))
    from c2_assembly_adf import GROUP_EDGES, build_model

    from openndm.gc import add_adf_tallies, compute_adf

    slab_fraction = 0.05
    model, lattice = build_model(True, particles=4000, batches=40)
    groups = openmc.mgxs.EnergyGroups(GROUP_EDGES)
    add_adf_tallies(model, lattice, groups, slab_fraction=slab_fraction)
    workdir = tmp_path_factory.mktemp("c2")
    statepoint = model.run(cwd=str(workdir), output=False)
    with openmc.StatePoint(statepoint) as sp:
        return compute_adf(sp, slab_fraction=slab_fraction, warn_sigma=1.0)


def test_c2_pin_lattice_thermal_factor_exceeds_one(c2):
    """The assembly surface is water, where the thermal flux peaks.

    Together with the fast test below this pins the group ordering, which is
    otherwise silent: reversing it leaves plausible values whose sense is
    simply inverted.
    """
    thermal = c2.values[:4, 1]
    assert thermal.min() > 1.0, thermal


def test_c2_pin_lattice_fast_factor_falls_below_one(c2):
    """Fast neutrons are born in the fuel, not at the assembly surface."""
    fast = c2.values[:4, 0]
    assert fast.max() < 1.0, fast


def test_c2_factors_are_symmetric_across_opposite_faces(c2):
    """A reflected square lattice has no preferred direction."""
    for group in range(2):
        values = c2.values[:4, group]
        assert values.max() - values.min() < 0.02, values
