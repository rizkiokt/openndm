"""Solver configuration (FR-SOL-6, FR-SOL-8)."""

from __future__ import annotations

from . import _core

__all__ = ["Settings"]

_KERNELS = {
    "fdm": _core.KernelType.fdm,
    "nem": _core.KernelType.nem,
    "sanm": _core.KernelType.sanm,
}
_MODES = {
    "forward": _core.SolveMode.forward,
    "adjoint": _core.SolveMode.adjoint,
    "fixed_source": _core.SolveMode.fixed_source,
}


class Settings:
    """Run-time solver settings.

    Every option is a keyword argument; the defaults solve a typical PWR
    quarter core with the SANM kernel to 1 pcm.

    Parameters
    ----------
    kernel : {'sanm', 'nem', 'fdm'}
        Nodal kernel. Selected at run time, never at build time (FR-SOL-8).
    mode : {'forward', 'adjoint', 'fixed_source'}
        Calculation mode (FR-MODE-1..3).
    k_tolerance, fission_source_tolerance : float
        Outer convergence criteria on the eigenvalue and on the node-wise
        fission source.
    wielandt_shift : float
        Additive Wielandt shift, ``k_shift = k + wielandt_shift``. Smaller
        accelerates the outer iteration but makes the inner systems harder;
        set to 0 to disable.
    warm_start : bool
        Reuse the previous flux and coupling coefficients (FR-OPT-3).
    threads : int
        OpenMP thread count; 0 leaves the environment default alone. Results
        are bit-identical regardless of this value (FR-OPT-4).

    Examples
    --------
    >>> Settings(kernel="nem", k_tolerance=1e-8, verbosity=2).kernel
    'nem'
    """

    __slots__ = ("_s",)

    def __init__(self, **kwargs):
        object.__setattr__(self, "_s", _core.Settings())
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def kernel(self) -> str:
        return self._s.kernel.name

    @kernel.setter
    def kernel(self, value) -> None:
        if isinstance(value, str):
            try:
                value = _KERNELS[value.lower()]
            except KeyError:
                raise ValueError(
                    f"unknown kernel {value!r}; choose from {sorted(_KERNELS)}"
                ) from None
        self._s.kernel = value

    @property
    def mode(self) -> str:
        return self._s.mode.name

    @mode.setter
    def mode(self, value) -> None:
        if isinstance(value, str):
            try:
                value = _MODES[value.lower()]
            except KeyError:
                raise ValueError(
                    f"unknown mode {value!r}; choose from {sorted(_MODES)}"
                ) from None
        self._s.mode = value

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return getattr(self._s, name)
        except AttributeError:
            raise AttributeError(f"Settings has no option {name!r}") from None

    def __setattr__(self, name, value):
        if name in ("kernel", "mode"):
            object.__setattr__(self, name, value)
            return
        if not hasattr(self._s, name):
            raise AttributeError(f"Settings has no option {name!r}")
        setattr(self._s, name, value)

    def __repr__(self) -> str:
        return (
            f"Settings(kernel={self.kernel!r}, mode={self.mode!r}, "
            f"k_tolerance={self._s.k_tolerance:g}, max_outer={self._s.max_outer})"
        )
