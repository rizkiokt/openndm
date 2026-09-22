"""Thermal-hydraulic coupling: the interface, the driver and the mapping.

There is no physics here, deliberately. FR-TH-6 forbids a solver kernel from
calling a thermal solver directly, and FR-TH-7 requires the Picard iteration,
the field mapping and the convergence test to be separable from whichever
thermal model sits behind them, so that an external solver can be attached
without touching the neutronics.

Field transfer is in memory throughout, as FR-TH-7 requires; nothing here
touches the filesystem.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from .exceptions import ConvergenceError, InputError

__all__ = [
    "AxialMapping",
    "CouplingResult",
    "CouplingStep",
    "PicardCoupling",
    "ThermalSolver",
]

ApplyState = Callable[[Mapping[str, np.ndarray], Mapping[str, np.ndarray]], None]


@runtime_checkable
class ThermalSolver(Protocol):
    """What a thermal solver must provide to be coupled (FR-TH-6).

    Temperatures and densities are keyed by name rather than fixed, because
    which fields exist is a property of the model. The names are the ones a
    branch-parameterised :class:`~openndm.XSLibrary` uses for its axes, so
    the result feeds :meth:`~openndm.XSLibrary.interpolate` directly.
    """

    def set_heat_source(self, q: np.ndarray) -> None:
        """Set the power deposited in each node, in W."""

    def solve(self) -> None:
        """Bring the temperatures and densities into equilibrium with it."""

    def get_temperatures(self) -> Mapping[str, np.ndarray]:
        """Temperature per node, in K, keyed by field name."""

    def get_densities(self) -> Mapping[str, np.ndarray]:
        """Density per node, in kg/m^3, keyed by field name."""


@dataclass(frozen=True)
class CouplingStep:
    """One Picard iteration.

    Attributes
    ----------
    iteration : int
        One-based.
    power_change : float
        Relative change in the node power against the previous iteration.
        The convergence measure.
    k_eff : float or None
        Whatever the neutronics step reported, carried through untouched.
    """

    iteration: int
    power_change: float
    k_eff: float | None = None


@dataclass(frozen=True)
class CouplingResult:
    """The state a Picard coupling converged to.

    Attributes
    ----------
    power : ndarray
        Node power the last neutronics solve produced, in W. A copy.
    temperatures : dict of str to ndarray
        Temperature per node, in K, as the thermal solver last reported it.
    densities : dict of str to ndarray
        Density per node, in kg/m^3, as the thermal solver last reported it.
    iterations : int
        Iterations performed.
    converged : bool
    history : list of CouplingStep
        One entry per iteration, in order.
    """

    power: np.ndarray
    temperatures: dict[str, np.ndarray]
    densities: dict[str, np.ndarray]
    iterations: int
    converged: bool
    history: list[CouplingStep] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"<CouplingResult iterations={self.iterations} "
            f"converged={self.converged}>"
        )


class PicardCoupling:
    """Iterate a neutronics solve against a thermal solve to a fixed point.

    The driver owns the loop, the under-relaxation and the convergence test,
    and reaches the thermal model only through :class:`ThermalSolver`
    (FR-TH-7). The neutronics step is a callable the caller supplies, which
    keeps the feedback model in the caller's hands the way
    :meth:`~openndm.Model.search_boron` keeps the boron model there.

    Parameters
    ----------
    thermal : ThermalSolver
        Any object satisfying the protocol.
    apply_state : callable
        ``apply_state(temperatures, densities)``, called with what the
        thermal solver produced. It must push that state into the cross
        sections and leave the model ready to solve. Nothing here interprets
        the fields, so a branch library, a
        :class:`~openndm.DopplerFeedback` or a hand-written correlation all
        work unchanged.
    relaxation : float, optional
        Under-relaxation on the power handed to the thermal solver, in
        ``(0, 1]``. One passes the new power straight through; lower values
        trade iterations for stability on a strongly coupled problem.
    tolerance : float, optional
        Relative power change below which the loop stops.
    max_iterations : int, optional
        Iteration cap.

    Raises
    ------
    InputError
        On an invalid relaxation, tolerance or iteration cap.

    Examples
    --------
    >>> coupling = PicardCoupling(thermal, apply_state)  # doctest: +SKIP
    >>> result = coupling.solve(node_power)  # doctest: +SKIP
    """

    def __init__(
        self,
        thermal: ThermalSolver,
        apply_state: ApplyState,
        *,
        relaxation: float = 1.0,
        tolerance: float = 1.0e-5,
        max_iterations: int = 50,
    ):
        if not 0.0 < relaxation <= 1.0:
            raise InputError(f"relaxation must lie in (0, 1], got {relaxation}")
        if not tolerance > 0.0:
            raise InputError(f"tolerance must be positive, got {tolerance}")
        if max_iterations < 1:
            raise InputError(
                f"max_iterations must be at least 1, got {max_iterations}"
            )
        self._thermal = thermal
        self._apply_state = apply_state
        self.relaxation = float(relaxation)
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)

    def solve(
        self,
        power_source: Callable[[], np.ndarray],
        *,
        k_eff_source: Callable[[], float] | None = None,
    ) -> CouplingResult:
        """Run the coupled iteration to convergence.

        Parameters
        ----------
        power_source : callable
            ``power_source()`` returning the node power in W. The caller
            runs whatever neutronics solve it wants inside this; the driver
            only asks for the answer.
        k_eff_source : callable, optional
            ``k_eff_source()`` returning the eigenvalue of the last solve,
            recorded in the history. For reporting only.

        Returns
        -------
        CouplingResult
            Holding copies, not views onto anything the thermal solver owns.

        Raises
        ------
        ConvergenceError
            If the power is still moving by more than ``tolerance`` after
            ``max_iterations``. Returning an unconverged state silently is
            the failure this raises instead of.
        InputError
            If the node power changes shape between iterations.
        """
        power = np.array(power_source(), dtype=float)
        history: list[CouplingStep] = []
        temperatures: dict[str, np.ndarray] = {}
        densities: dict[str, np.ndarray] = {}

        for iteration in range(1, self.max_iterations + 1):
            self._thermal.set_heat_source(power)
            self._thermal.solve()
            temperatures = _copy_fields(self._thermal.get_temperatures())
            densities = _copy_fields(self._thermal.get_densities())
            self._apply_state(temperatures, densities)

            updated = np.array(power_source(), dtype=float)
            if updated.shape != power.shape:
                raise InputError(
                    f"node power changed shape between iterations, "
                    f"{power.shape} to {updated.shape}"
                )
            change = _relative_change(updated, power)
            power = power + self.relaxation * (updated - power)
            history.append(
                CouplingStep(
                    iteration=iteration,
                    power_change=change,
                    k_eff=None if k_eff_source is None else float(k_eff_source()),
                )
            )
            if change < self.tolerance:
                return CouplingResult(
                    power=power,
                    temperatures=temperatures,
                    densities=densities,
                    iterations=iteration,
                    converged=True,
                    history=history,
                )

        raise ConvergenceError(
            f"thermal-hydraulic coupling did not converge in "
            f"{self.max_iterations} iterations",
            self.max_iterations,
            history[-1].power_change,
        )

    def __repr__(self) -> str:
        return (
            f"<PicardCoupling relaxation={self.relaxation:g} "
            f"tolerance={self.tolerance:g}>"
        )


def _copy_fields(fields: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Detach a field set from whatever buffer the thermal solver reuses."""
    return {
        str(name): np.array(values, dtype=float) for name, values in fields.items()
    }


def _relative_change(new: np.ndarray, old: np.ndarray) -> float:
    """Largest node-wise change, relative to the mean of the old field.

    Normalising by the mean rather than node by node keeps a node carrying
    almost no power from dominating the criterion, for the same reason the
    outer iteration normalises its fission source to unit mean.
    """
    scale = float(np.mean(np.abs(old)))
    largest = float(np.max(np.abs(new - old)))
    return largest if scale == 0.0 else largest / scale


class AxialMapping:
    """Volume-conservative transfer between two axial meshes (FR-TH-7).

    An external thermal solver rarely uses the neutronics mesh. For a channel
    of constant cross section, volume overlap is length overlap, so each
    target cell takes from every source cell it meets, weighted by the length
    they share.

    Parameters
    ----------
    source_edges : array_like, shape (n_source + 1,)
        Cell boundaries, strictly ascending, in cm.
    target_edges : array_like, shape (n_target + 1,)
        The same, for the mesh being mapped onto. Both meshes must span the
        same interval; a mismatch means they describe different channels.

    Attributes
    ----------
    source_edges : ndarray
        The boundaries given, in cm.
    target_edges : ndarray
        The boundaries given, in cm.
    overlap : ndarray, shape (n_target, n_source)
        Length shared by each target and source cell, in cm.

    Raises
    ------
    InputError
        If either mesh is not strictly ascending, or the two do not span the
        same interval.

    Notes
    -----
    The two transfers are not the same operation. An *extensive* field such
    as power must keep its total, so it is split by overlap fraction of the
    **source** cell. An *intensive* field such as temperature must keep its
    mean, so it is averaged by overlap fraction of the **target** cell. Using
    one where the other belongs conserves nothing and looks plausible, which
    is why they are separate methods rather than a flag.

    Examples
    --------
    >>> mapping = AxialMapping([0.0, 10.0, 20.0], [0.0, 5.0, 10.0, 15.0, 20.0])
    >>> mapping.distribute([100.0, 0.0]).tolist()
    [50.0, 50.0, 0.0, 0.0]
    >>> mapping.average([400.0, 600.0]).tolist()
    [400.0, 400.0, 600.0, 600.0]
    """

    def __init__(self, source_edges: Sequence[float], target_edges: Sequence[float]):
        source = np.asarray(source_edges, dtype=float)
        target = np.asarray(target_edges, dtype=float)
        for name, edges in (("source", source), ("target", target)):
            if edges.ndim != 1 or edges.size < 2:
                raise InputError(
                    f"{name}_edges must be one-dimensional with at least two "
                    f"boundaries, got shape {edges.shape}"
                )
            if np.any(np.diff(edges) <= 0.0):
                raise InputError(f"{name}_edges must increase strictly")
        span = max(source[-1] - source[0], target[-1] - target[0])
        ends_apart = max(abs(source[0] - target[0]), abs(source[-1] - target[-1]))
        if ends_apart > 1.0e-9 * span:
            raise InputError(
                f"the two meshes must span the same interval, got "
                f"[{source[0]}, {source[-1]}] and [{target[0]}, {target[-1]}]"
            )

        self.source_edges = source
        self.target_edges = target
        lower = np.maximum(target[:-1, np.newaxis], source[np.newaxis, :-1])
        upper = np.minimum(target[1:, np.newaxis], source[np.newaxis, 1:])
        self.overlap = np.clip(upper - lower, 0.0, None)

    @property
    def n_source(self) -> int:
        """Number of source cells."""
        return self.source_edges.size - 1

    @property
    def n_target(self) -> int:
        """Number of target cells."""
        return self.target_edges.size - 1

    def distribute(self, values) -> np.ndarray:
        """Map an extensive field onto the target mesh, keeping its total.

        Parameters
        ----------
        values : array_like, shape (..., n_source)
            Power per source cell, or any quantity whose sum is physical.
            Leading axes are free, so a whole core of channels maps in one
            call.

        Returns
        -------
        ndarray, shape (..., n_target)
            A new array.
        """
        per_length = self._checked(values) / np.diff(self.source_edges)
        return per_length @ self.overlap.T

    def average(self, values) -> np.ndarray:
        """Map an intensive field onto the target mesh, keeping its mean.

        Parameters
        ----------
        values : array_like, shape (..., n_source)
            Temperature or density per source cell. Leading axes are free.

        Returns
        -------
        ndarray, shape (..., n_target)
            A new array.
        """
        weighted = self._checked(values) @ self.overlap.T
        return weighted / np.diff(self.target_edges)

    def reverse(self) -> AxialMapping:
        """The mapping the other way, target mesh to source mesh."""
        return AxialMapping(self.target_edges, self.source_edges)

    def _checked(self, values) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim < 1 or arr.shape[-1] != self.n_source:
            raise InputError(
                f"expected a last axis of {self.n_source} source cells, got "
                f"shape {arr.shape}"
            )
        return arr

    def __repr__(self) -> str:
        return f"<AxialMapping {self.n_source} -> {self.n_target} cells>"
