"""Group constant generation from OpenMC (FR-OMC).

This is the differentiating capability of OpenNDM: everything needed to turn
an OpenMC lattice calculation into a nodal library, with no user-written
format conversion (DP-2).

OpenMC is an optional runtime dependency. Importing :mod:`openndm` never pulls
it in; only this subpackage needs it, and only the functions that actually
touch OpenMC objects (FR-OMC-14).

.. rubric:: What lives where

============================ ===========================================
Module                       Responsibility
============================ ===========================================
:mod:`openndm.gc.mgxs`       ``openmc.mgxs.Library`` / ``mgxs.h5`` ingestion
:mod:`openndm.gc.adf`        assembly discontinuity factor tallies
:mod:`openndm.gc.leakage`    B1 / P1 critical spectrum and buckling search
:mod:`openndm.gc.kinetics`   adjoint-weighted beta_eff and Lambda
:mod:`openndm.gc.formfunc`   pin form functions for power reconstruction
:mod:`openndm.gc.driver`     branch library generation and assembly
============================ ===========================================

The subpackage orchestrates OpenMC; it does not reimplement lattice physics.
"""

from __future__ import annotations

from .adf import add_adf_tallies, compute_adf
from .driver import BranchDriver, BranchGrid
from .formfunc import compute_form_functions
from .kinetics import KineticsParameters, compute_kinetics
from .leakage import CriticalSpectrum, critical_spectrum
from .mgxs import (
    from_mgxs_file,
    from_mgxs_library,
    from_statepoint,
    require_openmc,
)

__all__ = [
    "BranchDriver",
    "BranchGrid",
    "CriticalSpectrum",
    "KineticsParameters",
    "add_adf_tallies",
    "compute_adf",
    "compute_form_functions",
    "compute_kinetics",
    "critical_spectrum",
    "from_mgxs_file",
    "from_mgxs_library",
    "from_statepoint",
    "require_openmc",
]
