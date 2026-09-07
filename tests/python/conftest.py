"""Shared fixtures for the OpenNDM test suite."""

from __future__ import annotations

import numpy as np
import pytest

import openndm

ALL_KERNELS = ["fdm", "nem", "sanm"]
NODAL_KERNELS = ["nem", "sanm"]

#: One-group data for the analytic bare-cuboid verification case (V-1).
ONE_GROUP = {"D": 1.0, "absorption": 0.08, "nu_fission": 0.1}

#: IAEA 2D PWR benchmark group constants: D1, D2, Sa1, Sa2, nuSf2, Ss12.
IAEA_XS = [
    (1.5, 0.4, 0.010, 0.080, 0.135, 0.020),  # fuel 1
    (1.5, 0.4, 0.010, 0.085, 0.135, 0.020),  # fuel 2
    (1.5, 0.4, 0.010, 0.130, 0.135, 0.020),  # fuel 2 + rod
    (2.0, 0.3, 0.000, 0.010, 0.000, 0.040),  # reflector
    (2.0, 0.3, 0.000, 0.055, 0.000, 0.040),  # reflector + rod
]

#: IAEA quarter-core radial map. Row 0 and column 0 lie on the symmetry lines;
#: 0 marks an out-of-core position.
IAEA_MAP = np.array(
    [
        [3, 2, 2, 2, 3, 2, 2, 1, 4],
        [2, 2, 2, 2, 2, 2, 2, 1, 4],
        [2, 2, 2, 2, 2, 2, 2, 1, 4],
        [2, 2, 2, 2, 2, 2, 2, 1, 4],
        [3, 2, 2, 2, 3, 2, 1, 1, 4],
        [2, 2, 2, 2, 2, 2, 1, 4, 4],
        [2, 2, 2, 2, 1, 1, 4, 4, 0],
        [1, 1, 1, 1, 1, 4, 4, 0, 0],
        [4, 4, 4, 4, 4, 0, 0, 0, 0],
    ]
)


def analytic_k(D, absorption, nu_fission, side, n_dimensions=3):
    """Analytic eigenvalue of a bare homogeneous cuboid with zero-flux faces."""
    buckling = n_dimensions * (np.pi / side) ** 2
    return nu_fission / (absorption + D * buckling)


def one_group_library(**overrides):
    """Single-composition, single-group library for the analytic case."""
    data = {**ONE_GROUP, **overrides}
    lib = openndm.XSLibrary(1, 1)
    lib.set_composition(
        0,
        D=[data["D"]],
        absorption=[data["absorption"]],
        nu_fission=[data["nu_fission"]],
        kappa_fission=[data["nu_fission"]],
        chi=[1.0],
        scatter=[[0.0]],
    )
    lib.finalize()
    return lib


def cuboid(n_per_side, side=100.0):
    """Bare cuboid geometry with zero flux on every face."""
    return openndm.Geometry.from_lattice(
        np.zeros((n_per_side,) * 3, dtype=int),
        pitch=side / n_per_side,
        boundaries=dict.fromkeys(
            ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"], "zero_flux"
        ),
    )


def iaea_library():
    """Two-group IAEA benchmark library."""
    lib = openndm.XSLibrary(2, len(IAEA_XS))
    for index, (d1, d2, a1, a2, f2, s12) in enumerate(IAEA_XS):
        lib.set_composition(
            index,
            D=[d1, d2],
            absorption=[a1, a2],
            nu_fission=[0.0, f2],
            kappa_fission=[0.0, f2],
            chi=[1.0, 0.0] if f2 > 0 else [0.0, 0.0],
            scatter=[[0.0, s12], [0.0, 0.0]],
        )
    lib.finalize()
    return lib


def iaea_geometry(subdivide=1, planes=1):
    """IAEA 2D core map, optionally extruded and radially subdivided."""
    core = np.where(IAEA_MAP == 0, openndm.INACTIVE, IAEA_MAP - 1)
    core = np.repeat(core[np.newaxis, :, :], planes, axis=0)
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


@pytest.fixture
def tight():
    """Settings tight enough that iteration error does not mask a bug."""
    return openndm.Settings(
        verbosity=0,
        k_tolerance=1.0e-11,
        fission_source_tolerance=1.0e-10,
        inner_tolerance=1.0e-9,
        max_inner=400,
        max_outer=3000,
    )


@pytest.fixture
def iaea_model(tight):
    return openndm.Model(iaea_geometry(), iaea_library(), tight)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: verification runs over a few seconds")
    config.addinivalue_line("markers", "openmc: requires OpenMC")
