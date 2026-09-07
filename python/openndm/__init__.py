"""OpenNDM: Open Nodal Diffusion Method.

A three-dimensional, multi-group nodal diffusion solver for reactor core
analysis, with a C++17 core and a group-constant generation path built on the
OpenMC stack.

The Python API is the primary interface (DP-3). A minimal static solve::

    import numpy as np
    import openndm

    lib = openndm.XSLibrary(n_groups=2, n_compositions=1)
    lib.set_composition(0, D=[1.5, 0.4], absorption=[0.01, 0.085],
                        nu_fission=[0.0, 0.135], kappa_fission=[0.0, 0.135],
                        chi=[1.0, 0.0], scatter=[[0.0, 0.02], [0.0, 0.0]])
    lib.finalize()

    geom = openndm.Geometry.from_lattice(np.zeros((10, 9, 9), int), pitch=20.0)
    result = openndm.Model(geom, lib).solve()

``openndm.gc`` holds the OpenMC-native group constant generation path and is
the only part of the package that needs OpenMC installed (FR-OMC-14).
"""

from __future__ import annotations

from . import _core

# The exception module must be importable before the extension loads, because
# the extension resolves ConvergenceError out of it.
from .exceptions import (
    ConvergenceError,
    InputError,
    LibraryError,
    OpenNDMError,
)
from .geometry import INACTIVE, Geometry
from .model import BoronSearchResult, Model, Result
from .settings import Settings
from .statepoint import StatePoint, write_statepoint
from .xslib import XSLibrary

__version__ = _core.__version__

__all__ = [
    "INACTIVE",
    "BoronSearchResult",
    "ConvergenceError",
    "Geometry",
    "InputError",
    "LibraryError",
    "Model",
    "OpenNDMError",
    "Result",
    "Settings",
    "StatePoint",
    "XSLibrary",
    "__version__",
    "write_statepoint",
]


def __getattr__(name):
    # Import openndm.gc lazily so that `import openndm` never pulls in OpenMC.
    if name == "gc":
        import importlib

        module = importlib.import_module("openndm.gc")
        globals()["gc"] = module
        return module
    raise AttributeError(f"module 'openndm' has no attribute {name!r}")
