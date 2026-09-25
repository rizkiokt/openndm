"""The top level model object and the calculation modes (FR-MODE, FR-OPT)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from . import _core
from .channel import absolute_power
from .exceptions import ConvergenceError, InputError
from .geometry import Geometry
from .settings import Settings
from .thermal import PicardCoupling
from .xslib import XSLibrary

__all__ = ["BoronSearchResult", "CoupledResult", "Model", "Result"]

_SECANT_SEED_STEP_PPM = 100.0
"""Offset from the initial guess to the second point of the secant, in ppm."""


class Result:
    """Results of one solve.

    Arrays are zero-copy views onto the C++ buffers where possible
    (FR-OPT-5); they stay valid as long as this object does.
    """

    def __init__(self, raw, geometry: Geometry, library: XSLibrary):
        self._raw = raw
        self._geom = geometry
        self._lib = library

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


class CoupledResult:
    """Outcome of a coupled steady state (FR-MODE-6).

    Attributes
    ----------
    result : Result
        The final neutronics solve, the one whose power the reported
        temperatures are in equilibrium with.
    power : ndarray, shape (n_nodes,)
        Node power that solve produced, in W. Sums to the thermal power
        asked for.
    temperatures : dict of str to ndarray
        Per node, in K, as the thermal solver last reported them.
    densities : dict of str to ndarray
        Per node, in kg/m^3.
    iterations : int
        Picard iterations performed.
    power_change : float
        Relative power change at the last iteration, below the tolerance.
    history : list of CouplingStep
        One entry per iteration, carrying the eigenvalue of each.
    """

    def __init__(
        self, result, power, temperatures, densities, iterations, power_change,
        history,
    ):
        self.result = result
        self.power = power
        self.temperatures = temperatures
        self.densities = densities
        self.iterations = iterations
        self.power_change = power_change
        self.history = history

    @property
    def k_eff(self) -> float:
        """Eigenvalue of the converged state."""
        return self.result.k_eff

    def __repr__(self) -> str:
        return (
            f"<CoupledResult k_eff={self.k_eff:.6f} "
            f"iterations={self.iterations} "
            f"power_change={self.power_change:.2e}>"
        )


@dataclass(frozen=True)
class TransientStep:
    """One completed time step (FR-KIN-6)."""

    time: float
    dt: float
    total_power: float
    peak_power: float
    iterations: int
    inner_iterations: int
    converged: bool

    def __repr__(self) -> str:
        return (
            f"<TransientStep t={self.time:.6g}s power={self.total_power:.6g} "
            f"iterations={self.iterations}>"
        )


class Transient:
    """A time-dependent solve in progress (FR-KIN-1, FR-KIN-2).

    Created by :meth:`Model.start_transient`. The model is advanced in place,
    so its flux is the transient flux from the first step onward.

    Cross sections, rod positions and geometry may be changed between steps;
    a change made before a step belongs to that step's interval. Re-finalize
    the library after writing a composition, as for any other mutation.
    """

    def __init__(self, model: Model, settings: Settings):
        self._model = model
        self._settings_ = settings

    @property
    def time(self) -> float:
        """Seconds since the transient started."""
        return self._model._solver.transient_time

    @property
    def precursors(self) -> np.ndarray:
        """Precursor concentrations, shape ``(n_nodes, n_precursors)``."""
        return self._model._solver.precursors

    @property
    def flux(self) -> np.ndarray:
        """Current flux, shape ``(n_nodes, n_groups)``, matching `Result.flux`."""
        flux = np.asarray(self._model._solver.flux)
        return flux.reshape(self._model.geometry.n_nodes, self._model.library.n_groups)

    def step(self, dt: float, settings: Settings | None = None, **overrides):
        """Advance by ``dt`` seconds.

        Returns
        -------
        TransientStep
        """
        self._model._require_finalized()
        resolved = (
            self._settings_
            if settings is None and not overrides
            else self._model._settings(settings, overrides)
        )
        record = self._model._solver.step(float(dt), resolved._s)
        return TransientStep(
            time=record.time,
            dt=record.dt,
            total_power=record.total_power,
            peak_power=record.peak_power,
            iterations=record.iterations,
            inner_iterations=record.inner_iterations,
            converged=record.converged,
        )

    def __repr__(self) -> str:
        return f"<Transient t={self.time:.6g}s>"


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
        self._last_result: Result | None = None

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
        self._require_finalized()
        self._last_result = Result(
            self._solver.solve(self._settings(settings, overrides)._s),
            self.geometry,
            self.library,
        )
        return self._last_result

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
            raise InputError(f"source must have shape {expected}, got {arr.shape}")
        self._require_finalized()
        settings = self._settings(None, overrides)
        return Result(
            self._solver.solve_fixed_source(arr.ravel(order="C"), settings._s),
            self.geometry,
            self.library,
        )

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
            If ``max_iterations`` is reached without meeting ``tolerance``, or
            if two evaluations return the same eigenvalue.
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

        k_is_above_target = f0 > 0
        toward_target = (
            _SECANT_SEED_STEP_PPM if k_is_above_target else -_SECANT_SEED_STEP_PPM
        )
        x0, x1 = guess, min(max(guess + toward_target, lo), hi)
        f1, r1 = evaluate(x1)

        for iteration in range(2, max_iterations + 1):
            if abs(f1) < tolerance:
                return BoronSearchResult(x1, r1, iteration, history)
            if f1 == f0:
                raise _stalled_search_error(
                    history, x0, x1, target_k, bracket, iteration, abs(f1)
                )
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            if not lo <= x2 <= hi or not np.isfinite(x2):
                x2 = 0.5 * (lo + hi)
            x0, f0 = x1, f1
            x1 = x2
            f1, r1 = evaluate(x1)
            lo, hi = _tightened_bracket(lo, hi, x1, f0, f1)

        raise ConvergenceError(
            f"boron search did not reach k_eff = {target_k} in "
            f"{max_iterations} iterations",
            iterations=max_iterations,
            residual=abs(f1),
        )

    def solve_coupled(
        self,
        thermal,
        apply_state: Callable[[dict, dict], None],
        *,
        total_power: float,
        percent: float = 100.0,
        relaxation: float = 1.0,
        tolerance: float = 1.0e-5,
        max_iterations: int = 50,
        settings: Settings | None = None,
        **overrides,
    ) -> CoupledResult:
        """Steady state with thermal-hydraulic feedback (FR-MODE-6).

        Solve, hand the power to the thermal model, push what comes back into
        the cross sections, solve again, until the power stops moving. The
        feedback model stays in the caller's hands, the way
        :meth:`search_boron` keeps the boron model there: nothing here reads
        a temperature or knows what a branch axis is.

        Parameters
        ----------
        thermal : ThermalSolver
            Anything satisfying the protocol, built-in or external.
        apply_state : callable
            ``apply_state(temperatures, densities)``, given what the thermal
            solver produced, per node. It must write the new state into
            ``self.library`` **in place** and re-finalize, because the solver
            holds a reference to that library object rather than a copy. See
            :meth:`~openndm.XSLibrary.interpolate_by_composition` and its
            ``out`` argument, or
            :meth:`~openndm.CompositionMapping.average` for reducing a node
            field onto compositions.
        total_power : float
            Thermal power of the modelled geometry at full power, W. A
            quarter core carries a quarter of the core's power.
        percent : float, optional
            Percent of full power.
        relaxation : float, optional
            Under-relaxation on the power, in ``(0, 1]``.
        tolerance : float, optional
            Relative power change below which the loop stops.
        max_iterations : int, optional
            Picard iteration cap.
        settings : Settings, optional
            Overrides the model's settings for every solve in the loop.
        **overrides
            Individual settings overridden the same way.

        Returns
        -------
        CoupledResult

        Raises
        ------
        ConvergenceError
            If the power is still moving after ``max_iterations``, or if any
            neutronics solve fails to converge.
        InputError
            If ``apply_state`` leaves the library unfinalized, or replaces it
            rather than writing into it.

        Notes
        -----
        Every iteration calls :meth:`refresh`, which discards the flux, so
        each solve starts cold. That is the same cost :meth:`search_boron`
        pays and the reason issue #82 matters here.
        """
        latest: dict[str, Result] = {}
        library = self.library

        def power_source() -> np.ndarray:
            if self.library is not library:
                raise InputError(
                    "apply_state replaced the model's library; write into it "
                    "in place instead, so the solver keeps its reference"
                )
            self.refresh()
            result = self.solve(settings, **overrides)
            latest["result"] = result
            return absolute_power(
                result.power, self.geometry.volumes, total_power, percent
            )

        coupling = PicardCoupling(
            thermal,
            apply_state,
            relaxation=relaxation,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        outcome = coupling.solve(
            power_source, k_eff_source=lambda: latest["result"].k_eff
        )
        result = latest["result"]
        return CoupledResult(
            result=result,
            power=absolute_power(
                result.power, self.geometry.volumes, total_power, percent
            ),
            temperatures=outcome.temperatures,
            densities=outcome.densities,
            iterations=outcome.iterations,
            power_change=outcome.history[-1].power_change,
            history=outcome.history,
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
            results.append(self.solve(warm_start=warm_start and i > 0, **overrides))
        return results

    def surface_currents(self) -> np.ndarray:
        """Net current on every surface, shape ``(n_surfaces, n_groups)``.

        Positive along the surface normal, which runs from the surface's
        low-side node to its high-side node. To form the leakage out of a
        node, add ``+J * area`` for every surface where the node is on the low
        side and ``-J * area`` where it is on the high side; that works
        unchanged for a boundary surface, where only one side exists.

        Requires a completed solve.
        """
        return self._solver.surface_currents()

    def neutron_balance(self) -> np.ndarray:
        r"""Residual of the node balance, shape ``(n_nodes, n_groups)``.

        For a converged solution every entry is zero to within the iteration
        tolerance:

        .. math::

            \sum_s \pm J_s A_s + \Sigma_{r,g} V \phi_g
                - \sum_{g' \neq g} \Sigma_{s,g' \to g} V \phi_{g'}
                - \frac{\chi_g}{k} \sum_{g'} \nu\Sigma_{f,g'} V \phi_{g'}
                = 0

        Residuals are normalised by the node's total reaction rate, so they
        are dimensionless and comparable between nodes. This is the standard
        internal consistency check for a nodal code: it fails on a wrong
        coupling coefficient, a mis-assembled scattering term or a boundary
        condition applied to the wrong face, none of which need move k_eff
        very far.

        The residual is bounded by whichever convergence criterion is looser,
        the inner linear solve or the outer iteration. At the library defaults
        expect around ``1e-5``, set by ``Settings.inner_tolerance``; tightening
        both that and ``fission_source_tolerance`` brings it to about
        ``1e-10``. Compare against those rather than against a fixed number.
        """
        result = self._last_result
        if result is None:
            raise InputError("no solution available; call solve() first")

        geometry = self.geometry
        library = self.library
        n_groups = library.n_groups
        flux = np.asarray(result.flux).reshape(geometry.n_nodes, n_groups)
        volume = geometry.volumes
        compositions = geometry.compositions

        removal = library.array("removal")[compositions]
        nu_fission = library.array("nu_fission")[compositions]
        chi = library.array("chi")[compositions]
        scatter = library.array("scatter")[compositions]

        residual = removal * flux * volume[:, None]
        scattered_into_group = np.einsum("nij,ni->nj", scatter, flux)
        stayed_within_group = np.einsum("nii,ni->ni", scatter, flux)
        in_scatter = scattered_into_group - stayed_within_group
        residual -= in_scatter * volume[:, None]
        fission = np.einsum("ng,ng->n", nu_fission, flux)
        residual -= chi * (fission / result.k_eff)[:, None] * volume[:, None]

        currents = self.surface_currents()
        for index, surface in enumerate(geometry._g.surfaces):
            flow = currents[index] * surface.area
            if surface.lo >= 0:
                residual[surface.lo] += flow
            if surface.hi >= 0:
                residual[surface.hi] -= flow

        scale = removal * flux * volume[:, None]
        scale = np.maximum(np.abs(scale), np.abs(scale).max() * 1.0e-12)
        return residual / scale

    def start_transient(self, settings: Settings | None = None, **overrides):
        """Begin a time-dependent solve from the converged static solution.

        The static eigenvalue is generally not one, so the fission source is
        divided by it for the whole transient. That criticality normalisation
        is what makes a null transient null: without it a core at k = 1.03
        would ramp from the first step, and the ramp would look like physics.

        Precursors start at the equilibrium of the initial fission source.

        Returns
        -------
        Transient
            Driver with a :meth:`Transient.step` method.
        """
        self._require_finalized()
        resolved = self._settings(settings, overrides)
        self._solver.start_transient(resolved._s)
        return Transient(self, resolved)

    def refresh(self) -> None:
        """Re-read the geometry and library after mutating them in place.

        Call this after :meth:`swap_assemblies` or after changing cross
        sections, so the cached per-node data and coupling coefficients follow.
        The solver object survives, so buffers stay allocated.

        Writing a composition marks the library unfinalized, because its
        removal cross sections are derived at finalize time. Re-finalize
        before calling this, or it raises rather than letting a solve run on
        stale data.

        Raises
        ------
        InputError
            If a transient is in progress. The retained flux is that
            transient's state rather than a cache, so discarding it would
            leave the next step with nothing to advance. A step re-reads the
            cross sections, the compositions and the coupling on its own, so
            there is nothing to refresh between steps.
        """
        self._require_finalized()
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

    def _require_finalized(self) -> None:
        """Refuse to solve a library that has been mutated since finalizing.

        ``removal`` is derived from absorption and the scattering matrix at
        finalize time, and the kernels read it rather than recomputing it.
        Writing a composition afterwards marks the library unfinalized but
        leaves the cached removal in place, so solving anyway silently uses
        the *old* absorption while every accessor reports the new one.
        """
        if not self.library.finalized:
            raise InputError(
                "the library has been modified since it was finalized, so its "
                "removal cross sections are stale; call library.finalize() "
                "before solving"
            )

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


def _tightened_bracket(lo, hi, x, f_previous, f_current):
    """Narrow the search bracket onto the side the root is now known to be on.

    The secant method does not maintain a bracket of its own, so the bisection
    fallback it takes on a step outside the bracket is only valid while the
    bracket still straddles the root.
    """
    if f_current > 0:
        return (max(lo, x) if f_previous < 0 else lo), hi
    return lo, (min(hi, x) if f_previous > 0 else hi)


def _stalled_search_error(history, x0, x1, target_k, bracket, iteration, residual):
    """Explain a secant step that two equal eigenvalues left undefined.

    The usual cause is not an insensitive model but an unreachable target: the
    search walks to a bracket edge, gets clamped, and evaluates the same
    concentration twice. Which of the three it is follows from the spread of
    the eigenvalues seen so far, and saying so is far more useful to the
    caller than reporting insensitivity.
    """
    observed = [k for _, k in history]
    low, high = min(observed), max(observed)
    if high - low < 1.0e-12:
        return ConvergenceError(
            f"k_eff did not respond to boron at all: it stayed at "
            f"{high:.6f} across {len(history)} concentrations "
            f"between {history[0][0]:.1f} and {history[-1][0]:.1f} "
            f"ppm. Check that apply_boron mutates the library it "
            f"is handed and re-finalizes it.",
            iterations=iteration,
            residual=residual,
        )
    if not low <= target_k <= high:
        return ConvergenceError(
            f"k_eff = {target_k} is not reachable within the "
            f"bracket {bracket} ppm: over {len(history)} "
            f"evaluations k_eff stayed between {low:.6f} and "
            f"{high:.6f}. Widen the bracket, or check that boron "
            f"moves k_eff in the direction you expect.",
            iterations=iteration,
            residual=residual,
        )
    return ConvergenceError(
        f"boron search stalled: k_eff did not change between "
        f"{x0:.1f} and {x1:.1f} ppm. Check that apply_boron "
        f"mutates the library it is handed and re-finalizes it.",
        iterations=iteration,
        residual=residual,
    )
