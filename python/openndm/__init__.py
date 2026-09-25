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
from .channel import ChannelModel, PinGeometry, absolute_power
from .exceptions import (
    ConvergenceError,
    InputError,
    LibraryError,
    OpenNDMError,
)
from .feedback import DopplerFeedback
from .geometry import INACTIVE, Geometry
from .model import (
    BoronSearchResult,
    CoupledResult,
    Model,
    Result,
    Transient,
    TransientStep,
)
from .pin import PinConduction, PinState
from .rods import ControlRodBank, ControlRods, RodWorth
from .settings import Settings
from .statepoint import StatePoint, write_statepoint
from .thermal import (
    AxialMapping,
    CompositionMapping,
    CouplingResult,
    CouplingStep,
    PicardCoupling,
    ThermalSolver,
)
from .vtk import write_vtk
from .water import (
    ConstantWater,
    IF97Water,
    SaturationProperties,
    WaterProperties,
)
from .xslib import XSLibrary, rotate_adf, rotated_face

__version__ = _core.__version__

__all__ = [
    "INACTIVE",
    "AxialMapping",
    "BoronSearchResult",
    "ChannelModel",
    "CompositionMapping",
    "ConstantWater",
    "ControlRodBank",
    "ControlRods",
    "ConvergenceError",
    "CoupledResult",
    "CouplingResult",
    "CouplingStep",
    "DopplerFeedback",
    "Geometry",
    "IF97Water",
    "InputError",
    "LibraryError",
    "Model",
    "OpenNDMError",
    "PicardCoupling",
    "PinConduction",
    "PinGeometry",
    "PinState",
    "Result",
    "RodWorth",
    "SaturationProperties",
    "Settings",
    "StatePoint",
    "ThermalSolver",
    "Transient",
    "TransientStep",
    "WaterProperties",
    "XSLibrary",
    "__version__",
    "absolute_power",
    "rotate_adf",
    "rotated_face",
    "write_statepoint",
    "write_vtk",
]


def __getattr__(name):
    """Import ``openndm.gc`` only when it is asked for (FR-OMC-14).

    Importing it eagerly would make ``import openndm`` pull in OpenMC.
    """
    if name == "gc":
        import importlib

        module = importlib.import_module("openndm.gc")
        globals()["gc"] = module
        return module
    raise AttributeError(f"module 'openndm' has no attribute {name!r}")
