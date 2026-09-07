"""The top level model object and the calculation modes (FR-MODE, FR-OPT)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from . import _core
from .exceptions import ConvergenceError, InputError
from .geometry import Geometry
from .settings import Settings
from .xslib import XSLibrary

__all__ = ["BoronSearchResult", "Model", "Result"]


class Result:
    """Results of one solve.

    Arrays are zero-copy views onto the C++ buffers where possible
    (FR-OPT-5); they stay valid as long as this object does.
    """

    def __init__(self, raw, geometry: Geometry, library: XSLibrary):
        self._raw = raw
        self._geom = geometry
        self._lib = library

    # ------------------------------------------------------------- scalars
    @property
    def k_eff(self) -> float:
        return self._raw.k_eff

    @property
    def converged(self) -> bool:
        return self._raw.converged

    @property
    def outer_iterations(self) -> int:
        return self._raw.outer_iterations

    @property
    def runtime(self) -> float:
        """Wall-clock time of the solve, in seconds."""
        return self._raw.runtime_seconds

    @property
    def kernel(self) -> str:
        return self._raw.kernel

    # -------------------------------------------------------------- arrays
    @property
    def flux(self) -> np.ndarray:
        """Scalar flux, shape ``(n_nodes, n_groups)``."""
        return self._raw.flux

    @property
    def power(self) -> np.ndarray:
        """Relative node power, normalised to a mean of 1 over powered nodes."""
        return self._raw.power

    @property
    def history(self) -> np.ndarray:
        """Outer iteration history as a structured array (FR-OUT-7)."""
        records = self._raw.history
        out = np.zeros(
            len(records),
            dtype=[
                ("outer", "i4"),
                ("k_eff", "f8"),
                ("k_change", "f8"),
                ("source_change", "f8"),
                ("inner_iterations", "i4"),
            ],
        )
        for i, r in enumerate(records):
            out[i] = (
                r.outer,
                r.k_eff,
                r.k_change,
                r.source_change,
                r.inner_iterations,
            )
        return out

    # ---------------------------------------------------- derived quantities
    def power_lattice(self) -> np.ndarray:
        """Node power scattered onto the ``(nz, ny, nx)`` lattice."""
        return self._geom.expand(self.power)

    def radial_power(self) -> np.ndarray:
        """Volume-weighted radial (2D) power map, ``(ny, nx)`` (FR-OUT-3).

        Normalised to a mean of 1 over the columns that carry power.
        """
        p = self._geom.expand(self.power, fill=0.0)
        v = self._geom.expand(self._geom.volumes, fill=0.0)
        radial = np.einsum("kji,kji->ji", p, v)
        area = v.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            radial = np.where(area > 0, radial / np.where(area > 0, area, 1), 0.0)
        active = radial > 0
        if active.any():
            radial = radial / radial[active].mean()
        return radial

    def axial_power(self) -> np.ndarray:
        """Volume-weighted axial (1D) power profile, ``(nz,)`` (FR-OUT-3)."""
        p = self._geom.expand(self.power, fill=0.0)
        v = self._geom.expand(self._geom.volumes, fill=0.0)
        axial = np.einsum("kji,kji->k", p, v)
        volume = v.sum(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            axial = np.where(volume > 0, axial / np.where(volume > 0, volume, 1), 0.0)
        active = axial > 0
        if active.any():
            axial = axial / axial[active].mean()
        return axial

    @property
    def f_q(self) -> float:
        """Total peaking factor: peak node power over the core average."""
        p = self.power
        return float(p.max()) if p.size else 0.0

    @property
    def f_dh(self) -> float:
        """Radial enthalpy-rise peaking factor: peak radial power."""
        radial = self.radial_power()
        return float(radial.max()) if radial.size else 0.0

    def __repr__(self) -> str:
        return (
            f"<Result k_eff={self.k_eff:.6f} kernel={self.kernel!r} "
            f"outers={self.outer_iterations} F_q={self.f_q:.3f}>"
        )


class BoronSearchResult:
    """Outcome of a critical boron search (FR-MODE-4)."""

    def __init__(self, boron, result, iterations, history):
        self.boron = boron
        self.result = result
        self.iterations = iterations
        self.history = history

    @property
    def k_eff(self) -> float:
        return self.result.k_eff

    def __repr__(self) -> str:
        return (
            f"<BoronSearchResult boron={self.boron:.1f} ppm "
            f"k_eff={self.k_eff:.6f} iterations={self.iterations}>"
        )


class Model:
    """A geometry, a cross section library and the settings that solve them.

    The model owns a persistent solver, so a perturbed re-solve can reuse the
    converged flux and coupling coefficients rather than starting cold
    (FR-OPT-3). Nothing here touches the filesystem: build, solve, read
    results, mutate, re-solve is a pure in-memory loop (FR-OPT-1).

    Parameters
    ----------
    geometry : Geometry
    library : XSLibrary
        Must be finalized. A branch-parameterised library must be collapsed
        with :meth:`XSLibrary.interpolate` first.
    settings : Settings, optional

    Examples
    --------
    >>> import numpy as np, openndm
    >>> lib = openndm.XSLibrary(1, 1)
    >>> lib.set_composition(0, D=[1.0], absorption=[0.08],
    ...                     nu_fission=[0.1], kappa_fission=[0.1], chi=[1.0],
    ...                     scatter=[[0.0]])
    >>> _ = lib.finalize()
    >>> geom = openndm.Geometry.from_lattice(
    ...     np.zeros((8, 8, 8), int), pitch=12.5,
    ...     boundaries=dict.fromkeys(
    ...         ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"],
    ...         "zero_flux"))
    >>> model = openndm.Model(geom, lib, openndm.Settings(verbosity=0))
    >>> round(model.solve().k_eff, 4)
    1.2059
    """

    def __init__(
        self,
        geometry: Geometry,
        library: XSLibrary,
        settings: Settings | None = None,
    ):
        if not library.finalized:
            raise InputError(
                "library must be finalized before use; call library.finalize()"
            )
        if library.n_states > 1:
            raise InputError(
                "a branch-parameterised library must be collapsed first; "
                "call library.interpolate(**state)"
            )
        self.geometry = geometry
        self.library = library
        self.settings = settings if settings is not None else Settings()
        self._solver = _core.Solver(geometry._g, library._lib)

    # ---------------------------------------------------------------- solves
    def solve(self, settings: Settings | None = None, **overrides) -> Result:
        """Run a static eigenvalue solve (FR-MODE-1, FR-MODE-2).

        Parameters
        ----------
        settings : Settings, optional
            Overrides the model's settings for this call only.
        **overrides
            Individual settings overridden for this call, e.g.
            ``model.solve(kernel='nem', warm_start=True)``.

        Returns
        -------
        Result

        Raises
        ------
        ConvergenceError
            If the outer iteration exhausts ``max_outer``. The exception
            carries the iteration count and the last residual.
        """
        return Result(
            self._solver.solve(self._settings(settings, overrides)._s),
            self.geometry,
            self.library,
        )

    def solve_adjoint(self, **overrides) -> Result:
        """Adjoint static eigenvalue (FR-MODE-2).

        The adjoint reuses the nonlinear coupling coefficients from a forward
        solve, so a cold call runs the forward problem first.
        """
        return self.solve(mode="adjoint", **overrides)

    def solve_fixed_source(self, source, **overrides) -> Result:
        """Fixed external source, including subcritical multiplication.

        Parameters
        ----------
        source : array_like, shape (n_nodes, n_groups)
            External source density per node and group.
        """
        arr = np.asarray(source, dtype=float)
        expected = (self.geometry.n_nodes, self.library.n_groups)
        if arr.shape != expected:
            raise InputError(
                f"source must have shape {expected}, got {arr.shape}"
            )
        settings = self._settings(None, overrides)
        return Result(
            self._solver.solve_fixed_source(arr.ravel(order="C"), settings._s),
            self.geometry,
            self.library,
        )

    # ------------------------------------------------------------ parametric
    def search_boron(
        self,
        apply_boron: Callable[[XSLibrary, float], None],
        *,
        target_k: float = 1.0,
        guess: float = 800.0,
        bracket: tuple[float, float] = (0.0, 3000.0),
        tolerance: float = 1.0e-6,
        max_iterations: int = 20,
        **overrides,
    ) -> BoronSearchResult:
        """Critical boron search (FR-MODE-4).

        Iterates the boron concentration to ``target_k`` using the secant
        method, falling back to bisection whenever a secant step leaves the
        bracket. ``apply_boron`` is called with the library and a candidate
        concentration and must mutate the library in place; this keeps the
        boron model in the caller's hands rather than hard-coding a
        correlation.

        Parameters
        ----------
        apply_boron : callable
            ``apply_boron(library, ppm)``, mutating and re-finalizing.
        target_k : float
            Eigenvalue to search for.
        guess : float
            Starting concentration in ppm.
        bracket : (float, float)
            Hard limits on the search.

        Returns
        -------
        BoronSearchResult

        Raises
        ------
        ConvergenceError
            If ``max_iterations`` is reached without meeting ``tolerance``.
        """
        lo, hi = bracket
        if not lo <= guess <= hi:
            raise InputError(f"guess {guess} is outside the bracket {bracket}")

        history = []

        def evaluate(ppm: float) -> tuple[float, Result]:
            apply_boron(self.library, ppm)
            self._solver.reset()
            result = self.solve(**overrides)
            history.append((ppm, result.k_eff))
            return result.k_eff - target_k, result

        f0, r0 = evaluate(guess)
        if abs(f0) < tolerance:
            return BoronSearchResult(guess, r0, 1, history)

        # A second point one hundred ppm away seeds the secant; the sign of the
        # first residual chooses the direction that should move k toward target.
        x0, x1 = guess, min(max(guess + (100.0 if f0 > 0 else -100.0), lo), hi)
        f1, r1 = evaluate(x1)

        for iteration in range(2, max_iterations + 1):
            if abs(f1) < tolerance:
                return BoronSearchResult(x1, r1, iteration, history)
            if f1 == f0:
                raise ConvergenceError(
                    "boron search stalled: k_eff is insensitive to boron",
                    iterations=iteration,
                    residual=abs(f1),
                )
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            if not lo <= x2 <= hi or not np.isfinite(x2):
                x2 = 0.5 * (lo + hi)
            x0, f0 = x1, f1
            x1 = x2
            f1, r1 = evaluate(x1)
            # Keep the bracket honest so the bisection fallback stays valid.
            if f1 > 0:
                lo = max(lo, x1) if f0 < 0 else lo
            else:
                hi = min(hi, x1) if f0 > 0 else hi

        raise ConvergenceError(
            f"boron search did not reach k_eff = {target_k} in "
            f"{max_iterations} iterations",
            iterations=max_iterations,
            residual=abs(f1),
        )

    def sweep(
        self,
        mutate: Callable[[Model, object], None],
        cases: Sequence,
        *,
        warm_start: bool = True,
        **overrides,
    ) -> list[Result]:
        """Solve a sequence of perturbed models in memory (FR-MODE-8).

        The geometry and the library are reused across cases, and by default
        each solve warm starts from the previous one.

        Parameters
        ----------
        mutate : callable
            ``mutate(model, case)``, applied before each solve.
        cases : sequence
            Opaque case descriptors handed to ``mutate``.
        """
        results = []
        for i, case in enumerate(cases):
            mutate(self, case)
            self.refresh()
            results.append(
                self.solve(warm_start=warm_start and i > 0, **overrides)
            )
        return results

    # --------------------------------------------------------- manipulation
    def refresh(self) -> None:
        """Re-read the geometry and library after mutating them in place.

        Call this after :meth:`swap_assemblies` or after changing cross
        sections, so the cached per-node data and coupling coefficients follow.
        The solver object survives, so buffers stay allocated.
        """
        self._solver.reset()

    def swap_assemblies(self, a: int, b: int) -> None:
        """Exchange the compositions of two radial positions (FR-OPT-7).

        Parameters
        ----------
        a, b : int
            Flat radial indices into the ``(ny, nx)`` lattice; the whole axial
            column moves with the assembly.
        """
        nz, ny, nx = self.geometry.shape
        mapping = self.geometry.lattice_to_node.reshape(nz, ny, nx)
        ja, ia = divmod(int(a), nx)
        jb, ib = divmod(int(b), nx)
        for k in range(nz):
            na, nb = mapping[k, ja, ia], mapping[k, jb, ib]
            if na < 0 or nb < 0:
                raise InputError(
                    "cannot swap a position that is inactive on some plane"
                )
            ca = self.geometry.compositions[na]
            cb = self.geometry.compositions[nb]
            self.geometry.set_composition(na, int(cb))
            self.geometry.set_composition(nb, int(ca))

    # ---------------------------------------------------------------- utils
    def _settings(self, settings, overrides) -> Settings:
        base = settings if settings is not None else self.settings
        if not overrides:
            return base
        merged = Settings()
        for name in dir(base._s):
            if name.startswith("_"):
                continue
            try:
                setattr(merged._s, name, getattr(base._s, name))
            except AttributeError:
                continue
        for key, value in overrides.items():
            setattr(merged, key, value)
        return merged

    def __repr__(self) -> str:
        return (
            f"<Model {self.geometry!r} {self.library!r} "
            f"kernel={self.settings.kernel!r}>"
        )
