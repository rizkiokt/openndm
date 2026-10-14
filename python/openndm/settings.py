"""Solver configuration (FR-SOL-6, FR-SOL-8).

Every option of the C++ ``Settings`` struct is reachable by name through
:class:`Settings`, which resolves a kernel or mode given as a string and
passes every other value straight through.
"""

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

    Every option is a keyword argument to the constructor and an attribute
    afterwards. The defaults solve a typical PWR quarter core with the SANM
    kernel to 1 pcm.

    Parameters
    ----------
    kernel : {'sanm', 'nem', 'fdm'}, optional
        Nodal kernel. Selected at run time, never at build time (FR-SOL-8).
        Default ``'sanm'``.
    mode : {'forward', 'adjoint', 'fixed_source'}, optional
        Calculation mode (FR-MODE-1..3). Default ``'forward'``.
    k_tolerance : float, optional
        Change in k_eff between outers below which the solve has converged,
        in the units of k_eff, where 1e-5 is 1 pcm. Default 1e-9.
    fission_source_tolerance : float, optional
        Largest node-wise relative change in the normalised fission source
        below which the solve has converged, tested together with
        ``k_tolerance``. Dimensionless, default 1e-8.
    max_outer, min_outer : int, optional
        Outer iteration cap, and the outers run before either tolerance is
        tested. Defaults 500 and 2.
    inner_tolerance : float, optional
        Residual relative to the right hand side at which one within-group
        BiCGSTAB solve stops. Dimensionless, default 1e-5.
    max_inner : int, optional
        BiCGSTAB iteration cap per group solve. Default 50.
    group_sweeps : int, optional
        Gauss-Seidel sweeps over energy groups per outer iteration. Default 50.
    group_sweep_tolerance : float, optional
        Relative flux change between group sweeps below which the sweep stops.
        Dimensionless, default 1e-8.
    wielandt_shift : float, optional
        Additive Wielandt shift, ``k_shift = k_eff + wielandt_shift``, in the
        units of k_eff. Smaller accelerates the outer iteration but makes the
        inner systems harder; 0 disables it. Default 0.05.
    wielandt_start : int, optional
        Outer iteration at which the shift starts being applied. Default 3.
    nodal_update_interval, nodal_start : int, optional
        Outers between two-node kernel updates, and the first outer the kernel
        runs at. Defaults 1 and 2.
    two_node_sweeps : int, optional
        Gauss-Seidel sweeps over groups inside one two-node problem. Default 2.
    dhat_limit : float, optional
        Cap on the magnitude of Dhat relative to Dtilde, dimensionless. An
        interior face clamps to it, a boundary face discards the update
        instead. Default 10.
    boundary_relaxation : float, optional
        Under-relaxation on the boundary faces' Dhat, in ``(0, 1]``.
        Default 0.5.
    theta : float, optional
        Time integration weighting (FR-KIN-2): 1 is fully implicit, 0.5 is
        Crank-Nicolson. Default 1.
    max_step_iterations : int, optional
        Iterations within one time step, over the implicit fission source.
        Default 50.
    step_tolerance : float, optional
        Relative flux change between step iterations below which a step stops.
        Dimensionless, default 1e-9.
    warm_start : bool, optional
        Reuse the previous flux and coupling coefficients (FR-OPT-3).
        Default False.
    verbosity : int, optional
        0 silent, 1 summary, 2 per-outer iteration history (FR-OUT-7).
        Default 1.
    threads : int, optional
        OpenMP thread count; 0 leaves the environment default alone. Results
        are bit-identical regardless of this value (FR-OPT-4). Default 0.

    Raises
    ------
    AttributeError
        On an option the C++ settings struct does not carry.
    ValueError
        On an unknown kernel or mode name.

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
        """Nodal kernel in use: ``sanm``, ``nem`` or ``fdm``."""
        return self._s.kernel.name

    @kernel.setter
    def kernel(self, value) -> None:
        """Set the kernel from its name or a ``KernelType``.

        Parameters
        ----------
        value : {'sanm', 'nem', 'fdm'} or KernelType
            Kernel to run. A name is matched without regard to case.

        Raises
        ------
        ValueError
            On a name no kernel answers to.
        """
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
        """Calculation mode in use: ``forward``, ``adjoint`` or ``fixed_source``."""
        return self._s.mode.name

    @mode.setter
    def mode(self, value) -> None:
        """Set the calculation mode from its name or a ``SolveMode``.

        Parameters
        ----------
        value : {'forward', 'adjoint', 'fixed_source'} or SolveMode
            Mode to solve in. A name is matched without regard to case.

        Raises
        ------
        ValueError
            On a name no mode answers to.
        """
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
