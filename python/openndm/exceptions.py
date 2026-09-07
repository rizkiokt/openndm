"""Typed exceptions raised by OpenNDM (FR-OPT-6).

A solve never aborts the interpreter. Every failure path in the C++ core maps
onto one of these, so a caller driving thousands of perturbed models can catch
a non-converged solve and carry on.
"""

from __future__ import annotations

__all__ = [
    "ConvergenceError",
    "InputError",
    "LibraryError",
    "NotImplementedError_",
    "OpenNDMError",
]


class OpenNDMError(RuntimeError):
    """Base class for every error raised by OpenNDM."""


class InputError(OpenNDMError):
    """Invalid or inconsistent geometry, settings or library input."""


class LibraryError(OpenNDMError):
    """A cross section library failed validation (FR-XS-8)."""


class NotImplementedError_(OpenNDMError):
    """A requested capability exists in the API but is not implemented yet."""


class ConvergenceError(OpenNDMError):
    """An iterative solve exhausted its budget without converging.

    Attributes
    ----------
    iterations : int
        Number of outer iterations performed before giving up.
    residual : float
        The last convergence measure reached.
    """

    def __init__(self, message: str, iterations: int = 0, residual: float = 0.0):
        super().__init__(message)
        self.iterations = iterations
        self.residual = residual
