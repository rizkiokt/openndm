"""Regression baselines for the benchmark decks (NFR-QA-2).

These lock in the values this code currently produces so that a change which
moves them has to be deliberate. Where a deck reproduces a published
reference, the tolerance is against that reference; where it does not, the
tolerance is against the recorded baseline and the test says so. See
``benchmarks/README.md`` for which is which.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import openndm

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

from common import iaea_library, radial_map_to_compositions  # noqa: E402

#: Values this implementation currently produces on the shipped decks. These
#: are regression baselines, not benchmark agreement: iaea2d and iaea3d do not
#: yet reproduce their published eigenvalues.
BASELINE_IAEA_2D = {"fdm": 1.032610, "nem": 1.027552, "sanm": 1.028753}
BASELINE_IAEA_2D_FINE = 1.028581


def _iaea_2d_geometry(subdivide=1):
    core = radial_map_to_compositions()[np.newaxis, :, :]
    return openndm.Geometry.from_lattice(
        core,
        pitch=(20.0, 20.0, 20.0),
        subdivide=(subdivide, subdivide, 1),
        boundaries={
            "x_min": "reflective",
            "y_min": "reflective",
            "x_max": "zero_flux",
            "y_max": "zero_flux",
            "z_min": "reflective",
            "z_max": "reflective",
        },
        outside="zero_flux",
    )


@pytest.mark.parametrize("kernel", ["fdm", "nem", "sanm"])
def test_iaea_2d_matches_its_recorded_baseline(kernel):
    model = openndm.Model(
        _iaea_2d_geometry(), iaea_library(), openndm.Settings(verbosity=0)
    )
    result = model.solve(kernel=kernel)
    assert result.k_eff == pytest.approx(BASELINE_IAEA_2D[kernel], abs=1.0e-5)


@pytest.mark.slow
def test_iaea_2d_fine_mesh_limit_is_stable():
    """The mesh-converged eigenvalue is what the coarse kernels are judged on."""
    model = openndm.Model(
        _iaea_2d_geometry(subdivide=8),
        iaea_library(),
        openndm.Settings(verbosity=0),
    )
    result = model.solve(kernel="sanm")
    assert result.k_eff == pytest.approx(BASELINE_IAEA_2D_FINE, abs=2.0e-5)


def test_iaea_2d_power_distribution_is_physical():
    model = openndm.Model(
        _iaea_2d_geometry(), iaea_library(), openndm.Settings(verbosity=0)
    )
    result = model.solve()
    radial = result.radial_power()
    powered = radial[radial > 0]
    assert powered.mean() == pytest.approx(1.0)
    # The rodded positions must be local depressions.
    compositions = radial_map_to_compositions()
    rodded = compositions == 2
    fuel = np.isin(compositions, [0, 1])
    assert radial[rodded].mean() < radial[fuel].mean()


def _load_deck(name):
    """Import a deck's run.py under a unique module name.

    Every deck's entry point is called run.py, so importing them by bare name
    would return whichever one was imported first.
    """
    import importlib.util

    path = BENCHMARKS / name / "run.py"
    spec = importlib.util.spec_from_file_location(f"benchmark_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_analytic_deck_reports_the_exact_reference():
    analytic_deck = _load_deck("analytic")
    assert analytic_deck.analytic() == pytest.approx(1.205387388, abs=1.0e-8)


def test_iaea3d_deck_builds_a_physical_geometry():
    deck = _load_deck("iaea3d")

    geometry, library = deck.build(planes=19)
    assert geometry.shape[0] == 19
    assert library.n_groups == 2
    result = openndm.Model(
        geometry, library, openndm.Settings(verbosity=0)
    ).solve()
    axial = result.axial_power()
    # The reflector planes carry no power and the profile peaks in the interior.
    assert axial[0] == 0.0
    assert axial[-1] == 0.0
    assert 1 < int(np.argmax(axial)) < len(axial) - 2
