"""Radial fuel pin conduction and the Doppler temperature (FR-TH-2, FR-TH-4).

One-dimensional conduction from the pellet centreline out to the coolant,
across four resistances in series: the pellet, the pellet-to-clad gap, the
cladding, and the film. The pellet is meshed radially; the other three are
single resistances, having no internal generation. :meth:`PinConduction.solve`
is the steady state and :meth:`PinConduction.step` one implicit time step,
storing heat in the pellet rings and the cladding (FR-MODE-7).

No conductivity correlation is *invented* here. Conductivities and
coefficients are injected the way :class:`~openndm.WaterProperties` backends
are, because the published correlations are temperature-dependent and one
written from memory is the failure mode this project has already paid for
once. Every check the model is verified against holds at constant
conductivity.

The ``neacrp_`` functions are the exception that proves the rule: they are
the four relations NEACRP-L-335 Section 2.7 tells participants to use,
transcribed from the specification and cited to it, and they are ordinary
callables a caller opts into rather than a default.

Lengths are metres, temperatures K, linear heat rates W/m, conductivities
W/(m K), the gap and film coefficients W/(m^2 K), densities kg/m^3 and heat
capacities J/(kg K).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .exceptions import ConvergenceError, InputError

if TYPE_CHECKING:
    from collections.abc import Callable

    from .channel import PinGeometry

__all__ = [
    "NEACRP_FUEL_CONDUCTIVITY_POLE",
    "PinConduction",
    "PinState",
    "neacrp_clad_conductivity",
    "neacrp_clad_heat_capacity",
    "neacrp_fuel_conductivity",
    "neacrp_fuel_heat_capacity",
]

NEACRP_FUEL_CONDUCTIVITY_POLE = 73.15
"""Temperature at which the NEACRP fuel conductivity diverges, K."""

_STEP_TEMPERATURE_TOLERANCE = 1.0e-9
"""Largest change between property iterations inside one step, K."""

_MAX_STEP_ITERATIONS = 50
"""Property iterations allowed inside one step before it is refused."""


class PinState:
    """Temperatures through one pin at every node (FR-TH-2).

    Parameters
    ----------
    profile : ndarray, shape (n_nodes, n_rings + 1)
        Pellet temperature at each ring boundary, centreline first, K.
    clad_inner : ndarray, shape (n_nodes,)
        Cladding inner surface temperature, K.
    clad_outer : ndarray, shape (n_nodes,)
        Cladding outer surface temperature, K.
    average : ndarray, shape (n_nodes,)
        Volume-average pellet temperature, K.
    doppler : ndarray, shape (n_nodes,)
        Effective Doppler temperature, K.
    surface_heat_rate : ndarray, shape (n_nodes,)
        Heat leaving the cladding into the coolant per unit pin length, W/m.
        In steady state it is the linear heat rate.

    Attributes
    ----------
    profile, clad_inner, clad_outer, average, doppler, surface_heat_rate
        As above. A float array passed in is stored as given, not copied;
        the arrays :meth:`PinConduction.solve` and :meth:`PinConduction.step`
        build are fresh.
    """

    def __init__(
        self, profile, clad_inner, clad_outer, average, doppler, surface_heat_rate
    ):
        self.profile = np.asarray(profile, dtype=float)
        self.clad_inner = np.asarray(clad_inner, dtype=float)
        self.clad_outer = np.asarray(clad_outer, dtype=float)
        self.average = np.asarray(average, dtype=float)
        self.doppler = np.asarray(doppler, dtype=float)
        self.surface_heat_rate = np.asarray(surface_heat_rate, dtype=float)

    @property
    def centre(self) -> np.ndarray:
        """Pellet centreline temperature per node, K. A view on the profile."""
        return self.profile[:, 0]

    @property
    def surface(self) -> np.ndarray:
        """Pellet surface temperature per node, K. A view on the profile."""
        return self.profile[:, -1]

    def __repr__(self) -> str:
        if self.profile.size == 0:
            return "<PinState empty>"
        return (
            f"<PinState peak centre={self.centre.max():.1f} K "
            f"peak Doppler={self.doppler.max():.1f} K>"
        )


class PinConduction:
    """Radial conduction through pellet, gap, cladding and film.

    The pellet is split into ``n_rings`` rings of equal thickness and the
    conduction equation is integrated exactly across each one, so a constant
    conductivity reproduces the analytic parabola at every ring boundary for
    any ring count. A conductivity given as a callable is evaluated at the
    outer boundary of each ring, which converges with the ring count rather
    than being exact.

    Parameters
    ----------
    pins : PinGeometry
        Pin dimensions. Only the radii are used; the counts belong to the
        channel.
    fuel_conductivity : float or callable
        Pellet conductivity in W/(m K), or ``k(temperature)`` returning it.
    clad_conductivity : float or callable
        Cladding conductivity in W/(m K), or ``k(temperature)``.
    gap_conductance : float
        Pellet-to-cladding conductance in W/(m^2 K), referred to the pellet
        outer surface.
    film_coefficient : float
        Cladding-to-coolant heat transfer coefficient in W/(m^2 K), referred
        to the cladding outer surface.
    n_rings : int, optional
        Rings across the pellet.
    doppler_weight : float, optional
        Weight on the pellet surface in the effective Doppler temperature,
        the rest falling on the centreline (FR-TH-4). The default is the
        0.7 surface, 0.3 centre convention; 0.5 is the volume average of the
        parabola.
    fuel_density, clad_density : float, optional
        Pellet and cladding density, kg/m^3. Needed only by :meth:`step`.
    fuel_heat_capacity, clad_heat_capacity : float or callable, optional
        Specific heat capacity in J/(kg K), or ``c(temperature)`` returning
        it. Needed only by :meth:`step`.

    Raises
    ------
    InputError
        On a non-positive conductivity, conductance, coefficient, density or
        heat capacity, fewer than one ring, or a weight outside ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from openndm import PinGeometry
    >>> pins = PinGeometry(5.0e-3, 1.0e-4, 6.0e-4, 1.3e-2, 264, 25)
    >>> conduction = PinConduction(
    ...     pins, fuel_conductivity=3.0, clad_conductivity=15.0,
    ...     gap_conductance=1.0e4, film_coefficient=3.0e4,
    ... )
    >>> state = conduction.solve(np.array([4.0 * np.pi * 3.0]), np.array([0.0]))
    >>> round(float(state.centre[0] - state.surface[0]), 12)
    1.0
    """

    def __init__(
        self,
        pins: PinGeometry,
        *,
        fuel_conductivity,
        clad_conductivity,
        gap_conductance: float,
        film_coefficient: float,
        n_rings: int = 10,
        doppler_weight: float = 0.7,
        fuel_density: float | None = None,
        clad_density: float | None = None,
        fuel_heat_capacity=None,
        clad_heat_capacity=None,
    ):
        if n_rings < 1:
            raise InputError(f"the pellet needs at least one ring, got {n_rings}")
        if not 0.0 <= doppler_weight <= 1.0:
            raise InputError(
                f"doppler_weight is a weight on the surface and must lie in "
                f"[0, 1], got {doppler_weight}"
            )
        for name, value in (
            ("gap_conductance", gap_conductance),
            ("film_coefficient", film_coefficient),
            ("fuel_density", fuel_density),
            ("clad_density", clad_density),
        ):
            if value is not None and not value > 0.0:
                raise InputError(f"{name} must be positive, got {value}")

        self.pins = pins
        self._fuel_conductivity = _checked_property(
            "fuel_conductivity", fuel_conductivity
        )
        self._clad_conductivity = _checked_property(
            "clad_conductivity", clad_conductivity
        )
        self._fuel_heat_capacity = (
            None
            if fuel_heat_capacity is None
            else _checked_property("fuel_heat_capacity", fuel_heat_capacity)
        )
        self._clad_heat_capacity = (
            None
            if clad_heat_capacity is None
            else _checked_property("clad_heat_capacity", clad_heat_capacity)
        )
        self.fuel_density = None if fuel_density is None else float(fuel_density)
        self.clad_density = None if clad_density is None else float(clad_density)
        self.gap_conductance = float(gap_conductance)
        self.film_coefficient = float(film_coefficient)
        self.doppler_weight = float(doppler_weight)
        self.radii = np.linspace(0.0, pins.fuel_radius, int(n_rings) + 1)

    @property
    def n_rings(self) -> int:
        """Rings across the pellet."""
        return self.radii.size - 1

    @property
    def is_transient(self) -> bool:
        """Whether both densities and both heat capacities are set."""
        return None not in (
            self.fuel_density,
            self.clad_density,
            self._fuel_heat_capacity,
            self._clad_heat_capacity,
        )

    def solve(self, linear_heat_rate, coolant_temperature) -> PinState:
        """Compute the steady temperatures through the pin at every node.

        Parameters
        ----------
        linear_heat_rate : array_like, shape (n_nodes,)
            Power raised per unit pin length, W/m. What the fuel produces,
            so the directly deposited fraction is already excluded.
        coolant_temperature : array_like, shape (n_nodes,)
            Bulk coolant temperature the film rejects heat into, K.

        Returns
        -------
        PinState
            Pellet, cladding, average and Doppler temperatures in K. Every
            array is new; nothing is retained between solves.

        Raises
        ------
        InputError
            If the two arrays are not one-dimensional and the same length, or
            if the linear heat rate is negative or not finite.
        """
        rate, bulk = _checked_rates(linear_heat_rate, coolant_temperature)

        clad_outer = bulk + self._film_rise(rate)
        clad_inner = clad_outer + self._clad_rise(rate, clad_outer)
        surface = clad_inner + self._gap_rise(rate)
        profile = self._pellet_profile(rate, surface)
        return self._state(profile, clad_inner, clad_outer, rate.copy())

    def step(
        self, linear_heat_rate, coolant_temperature, previous: PinState, dt: float
    ) -> PinState:
        """Advance the pin temperatures by one backward Euler step.

        Heat is stored in each pellet ring and in the cladding; the gap and
        the film store none. Within a ring the effective source is uniform,
        so the conduction across it is integrated exactly, and an infinite
        step reproduces :meth:`solve`. Temperature-dependent properties are
        iterated to convergence inside the step. See ``docs/theory.md``.

        Parameters
        ----------
        linear_heat_rate : array_like, shape (n_nodes,)
            Power raised per unit pin length at the end of the step, W/m.
        coolant_temperature : array_like, shape (n_nodes,)
            Bulk coolant temperature at the end of the step, K.
        previous : PinState
            State at the start of the step, from :meth:`solve` or
            :meth:`step` on the same nodes.
        dt : float
            Step length, s.

        Returns
        -------
        PinState
            Temperatures at the end of the step, with the heat leaving the
            cladding as ``surface_heat_rate``. Every array is new.

        Raises
        ------
        InputError
            Without densities and heat capacities, on a non-positive step,
            on a ``previous`` state of another shape, or on the conditions
            :meth:`solve` refuses.
        ConvergenceError
            If the temperature-dependent properties do not settle.
        """
        if not self.is_transient:
            raise InputError(
                "a transient step needs fuel_density, clad_density, "
                "fuel_heat_capacity and clad_heat_capacity"
            )
        if not dt > 0.0:
            raise InputError(f"the time step must be positive, got {dt}")
        rate, bulk = _checked_rates(linear_heat_rate, coolant_temperature)
        if previous.profile.shape != (rate.size, self.radii.size):
            raise InputError(
                f"the previous state has shape {previous.profile.shape}, "
                f"expected {(rate.size, self.radii.size)}"
            )

        unknowns = np.column_stack(
            [previous.profile, previous.clad_inner, previous.clad_outer]
        )
        for _ in range(_MAX_STEP_ITERATIONS):
            residual = self._step_residual(unknowns, rate, bulk, previous, dt)
            updated = unknowns + self._affine_correction(residual, unknowns)
            change = float(np.max(np.abs(updated - unknowns), initial=0.0))
            unknowns = updated
            if change < _STEP_TEMPERATURE_TOLERANCE:
                break
        else:
            raise ConvergenceError(
                "pin temperatures did not settle within the step",
                _MAX_STEP_ITERATIONS,
                change,
            )

        profile = unknowns[:, : self.radii.size].copy()
        clad_inner = unknowns[:, -2].copy()
        clad_outer = unknowns[:, -1].copy()
        clad_capacity = self._clad_capacity(unknowns)
        outflow = self._pellet_outflow(profile, rate, previous, dt, unknowns)
        surface_heat_rate = outflow - clad_capacity * (
            0.5 * (clad_inner + clad_outer)
            - 0.5 * (previous.clad_inner + previous.clad_outer)
        ) / dt
        return self._state(profile, clad_inner, clad_outer, surface_heat_rate)

    def stored_energy(self, state: PinState) -> np.ndarray:
        """Return the heat held in the pin per unit length, J/m.

        Measured from 0 K at the heat capacities of ``state`` itself, on the
        same ring averages :meth:`step` stores heat in, so a step at constant
        properties conserves it exactly.

        Parameters
        ----------
        state : PinState
            Pin temperatures, shape ``(n_nodes, n_rings + 1)`` profile.

        Returns
        -------
        ndarray, shape (n_nodes,)
            Stored energy per unit pin length, J/m. A new array.

        Raises
        ------
        InputError
            Without densities and heat capacities.
        """
        if not self.is_transient:
            raise InputError("stored energy needs densities and heat capacities")
        unknowns = np.column_stack([state.profile, state.clad_inner, state.clad_outer])
        rings = 0.5 * (state.profile[:, :-1] + state.profile[:, 1:])
        fuel = (self._ring_capacity(unknowns) * rings).sum(axis=1)
        clad = self._clad_capacity(unknowns) * 0.5 * (
            state.clad_inner + state.clad_outer
        )
        return fuel + clad

    def _state(self, profile, clad_inner, clad_outer, surface_heat_rate) -> PinState:
        average = self._volume_average(profile)
        doppler = (
            self.doppler_weight * profile[:, -1]
            + (1.0 - self.doppler_weight) * profile[:, 0]
        )
        return PinState(
            profile, clad_inner, clad_outer, average, doppler, surface_heat_rate
        )

    def _ring_capacity(self, unknowns: np.ndarray) -> np.ndarray:
        """Heat capacity of each pellet ring per unit length, J/(m K)."""
        profile = unknowns[:, : self.radii.size]
        rings = 0.5 * (profile[:, :-1] + profile[:, 1:])
        areas = np.pi * np.diff(self.radii**2)
        return self.fuel_density * self._fuel_heat_capacity(rings) * areas

    def _clad_capacity(self, unknowns: np.ndarray) -> np.ndarray:
        """Heat capacity of the cladding per unit length, J/(m K)."""
        average = 0.5 * (unknowns[:, -2] + unknowns[:, -1])
        area = np.pi * (
            self.pins.clad_outer_radius**2 - self.pins.clad_inner_radius**2
        )
        return self.clad_density * self._clad_heat_capacity(average) * area

    def _pellet_outflow(self, profile, rate, previous, dt, unknowns) -> np.ndarray:
        """Heat crossing the pellet surface per unit length, W/m."""
        rings = 0.5 * (profile[:, :-1] + profile[:, 1:])
        old = 0.5 * (previous.profile[:, :-1] + previous.profile[:, 1:])
        stored = (self._ring_capacity(unknowns) * (rings - old)).sum(axis=1) / dt
        return rate - stored

    def _step_residual(self, frozen, rate, bulk, previous, dt):
        """Build the affine step residual with properties frozen at ``frozen``.

        Returns a callable mapping candidate temperatures of shape
        ``(n_nodes, n_rings + 3)`` to the residual of the same shape.
        """
        n_boundaries = self.radii.size
        ring_capacity = self._ring_capacity(frozen)
        clad_capacity = self._clad_capacity(frozen)
        fuel_k = self._fuel_conductivity(frozen[:, 1:n_boundaries])
        clad_k = self._clad_conductivity(frozen[:, -1])
        areas = np.pi * np.diff(self.radii**2)
        inner_areas = np.pi * self.radii[:-1] ** 2
        with np.errstate(divide="ignore"):
            log_ratio = np.log(self.radii[1:] / self.radii[:-1])
        log_ratio[0] = 0.0
        generation = rate / (np.pi * self.pins.fuel_radius**2)
        old_rings = 0.5 * (previous.profile[:, :-1] + previous.profile[:, 1:])
        old_clad = 0.5 * (previous.clad_inner + previous.clad_outer)
        clad_log = np.log(self.pins.clad_outer_radius / self.pins.clad_inner_radius)
        gap_area = 2.0 * np.pi * self.pins.fuel_radius
        film_area = 2.0 * np.pi * self.pins.clad_outer_radius

        def residual(candidate):
            profile = candidate[:, :n_boundaries]
            clad_inner = candidate[:, -2]
            clad_outer = candidate[:, -1]
            rings = 0.5 * (profile[:, :-1] + profile[:, 1:])
            source = (
                generation[:, np.newaxis]
                - ring_capacity * (rings - old_rings) / (areas * dt)
            )
            ring_heat = source * areas
            inflow = np.cumsum(ring_heat, axis=1) - ring_heat
            drop = source * areas / (4.0 * np.pi * fuel_k) + (
                inflow - source * inner_areas
            ) * log_ratio / (2.0 * np.pi * fuel_k)
            pellet_out = ring_heat.sum(axis=1)
            clad_out = pellet_out - clad_capacity * (
                0.5 * (clad_inner + clad_outer) - old_clad
            ) / dt
            out = np.empty_like(candidate)
            out[:, : n_boundaries - 1] = profile[:, :-1] - profile[:, 1:] - drop
            gap_drop = pellet_out / (gap_area * self.gap_conductance)
            out[:, n_boundaries - 1] = profile[:, -1] - clad_inner - gap_drop
            out[:, -2] = clad_inner - clad_outer - 0.5 * (
                pellet_out + clad_out
            ) * clad_log / (2.0 * np.pi * clad_k)
            out[:, -1] = clad_outer - bulk - clad_out / (
                film_area * self.film_coefficient
            )
            return out

        return residual

    @staticmethod
    def _affine_correction(residual, around: np.ndarray) -> np.ndarray:
        """Return the change from ``around`` that zeroes an affine residual.

        The matrix is read off column by column from unit perturbations,
        which costs one residual evaluation per unknown and keeps the
        equations written once, in :meth:`_step_residual`. Solving for the
        correction rather than the temperatures keeps round-off proportional
        to the residual, which vanishes as the properties settle.
        """
        n_nodes, n_unknowns = around.shape
        offset = residual(around)
        matrix = np.empty((n_nodes, n_unknowns, n_unknowns))
        for column in range(n_unknowns):
            perturbed = around.copy()
            perturbed[:, column] += 1.0
            matrix[:, :, column] = residual(perturbed) - offset
        return np.linalg.solve(matrix, -offset[:, :, np.newaxis])[:, :, 0]

    def _film_rise(self, rate: np.ndarray) -> np.ndarray:
        """Cladding surface above the bulk coolant, K."""
        area = 2.0 * np.pi * self.pins.clad_outer_radius
        return rate / (area * self.film_coefficient)

    def _gap_rise(self, rate: np.ndarray) -> np.ndarray:
        """Jump across the pellet-to-cladding gap, K."""
        area = 2.0 * np.pi * self.pins.fuel_radius
        return rate / (area * self.gap_conductance)

    def _clad_rise(self, rate: np.ndarray, outer: np.ndarray) -> np.ndarray:
        """Drop across the cladding wall, K."""
        ratio = np.log(self.pins.clad_outer_radius / self.pins.clad_inner_radius)
        return rate * ratio / (2.0 * np.pi * self._clad_conductivity(outer))

    def _pellet_profile(self, rate: np.ndarray, surface: np.ndarray) -> np.ndarray:
        """March inwards from the pellet surface to the centreline."""
        generation = rate / (np.pi * self.pins.fuel_radius**2)
        shells = np.diff(self.radii**2)

        profile = np.empty((rate.size, self.radii.size), dtype=float)
        profile[:, -1] = surface
        for ring in range(self.n_rings - 1, -1, -1):
            outer = profile[:, ring + 1]
            step = generation * shells[ring] / (4.0 * self._fuel_conductivity(outer))
            profile[:, ring] = outer + step
        return profile

    def _volume_average(self, profile: np.ndarray) -> np.ndarray:
        """Volume-average pellet temperature, K.

        The pellet volume element is proportional to ``d(r^2)``, and a
        constant-conductivity profile is linear in ``r^2``, so integrating on
        that variable is exact rather than approximate.
        """
        squared = self.radii**2
        shells = np.diff(squared)
        midpoints = 0.5 * (profile[:, :-1] + profile[:, 1:])
        return midpoints @ shells / squared[-1]

    def __repr__(self) -> str:
        return (
            f"<PinConduction {self.n_rings} rings "
            f"doppler_weight={self.doppler_weight:g}>"
        )


def _checked_property(name: str, value) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a constant or a correlation into one callable, validating it."""
    if callable(value):
        return value
    constant = float(value)
    if not constant > 0.0:
        raise InputError(f"{name} must be positive, got {constant}")
    return lambda temperature: np.full_like(
        np.asarray(temperature, dtype=float), constant
    )


def _checked_rates(linear_heat_rate, coolant_temperature):
    """Validate a heat rate and coolant temperature pair, returning arrays."""
    rate = np.asarray(linear_heat_rate, dtype=float)
    bulk = np.asarray(coolant_temperature, dtype=float)
    if rate.ndim != 1 or rate.shape != bulk.shape:
        raise InputError(
            f"linear heat rate and coolant temperature must be matching "
            f"1D arrays, got {rate.shape} and {bulk.shape}"
        )
    if not np.all(np.isfinite(rate)):
        raise InputError("the linear heat rate holds a non-finite entry")
    if np.any(rate < 0.0):
        raise InputError(
            f"the linear heat rate cannot be negative, got {float(rate.min())} W/m"
        )
    return rate, bulk


def neacrp_fuel_conductivity(temperature) -> np.ndarray:
    r"""UO2 heat conductivity in W/(m K), NEACRP-L-335 Section 2.7.

    .. math:: \lambda_{UO_2} = 1.05 + \frac{2150}{T - 73.15}

    Falls from about 10.5 W/(m K) at 300 K to 3.0 at 1200 K. The
    specification states no validity range; the pole at 73.15 K is refused
    because below it the expression turns negative.

    Parameters
    ----------
    temperature : array_like, shape (...)
        Fuel temperature, K.

    Returns
    -------
    ndarray, shape (...)
        Pellet conductivity in W/(m K), shaped like ``temperature``. A new
        array.

    Raises
    ------
    InputError
        At or below :data:`NEACRP_FUEL_CONDUCTIVITY_POLE`.

    Examples
    --------
    >>> round(float(neacrp_fuel_conductivity(1200.0)), 3)
    2.958
    """
    t = np.asarray(temperature, dtype=float)
    if np.any(t <= NEACRP_FUEL_CONDUCTIVITY_POLE):
        raise InputError(
            f"the NEACRP fuel conductivity diverges at "
            f"{NEACRP_FUEL_CONDUCTIVITY_POLE} K; got {float(np.min(t))} K"
        )
    return 1.05 + 2150.0 / (t - NEACRP_FUEL_CONDUCTIVITY_POLE)


def neacrp_clad_conductivity(temperature) -> np.ndarray:
    r"""Zircaloy-4 heat conductivity in W/(m K), NEACRP-L-335 Section 2.7.

    .. math:: \lambda_{Zr} = 7.51 + 2.09\times10^{-2} T
              - 1.45\times10^{-5} T^2 + 7.67\times10^{-9} T^3

    Rises from about 12.7 W/(m K) at 300 K to 21.6 at 1000 K. The cubic
    climbs steeply beyond that and the specification states no upper bound,
    so treat it with suspicion above the cladding's melting point.

    Parameters
    ----------
    temperature : array_like, shape (...)
        Cladding temperature, K.

    Returns
    -------
    ndarray, shape (...)
        Cladding conductivity in W/(m K), shaped like ``temperature``. A new
        array.

    Examples
    --------
    >>> round(float(neacrp_clad_conductivity(600.0)), 3)
    16.487
    """
    t = np.asarray(temperature, dtype=float)
    return 7.51 + 2.09e-2 * t - 1.45e-5 * t**2 + 7.67e-9 * t**3


def neacrp_fuel_heat_capacity(temperature) -> np.ndarray:
    r"""UO2 specific heat capacity in J/(kg K), NEACRP-L-335 Section 2.7.

    .. math:: c_{p,UO_2} = 162.3 + 0.3038 T - 2.391\times10^{-4} T^2
              + 6.404\times10^{-8} T^3

    Pass it as ``fuel_heat_capacity`` for :meth:`PinConduction.step`.

    Parameters
    ----------
    temperature : array_like, shape (...)
        Fuel temperature, K.

    Returns
    -------
    ndarray, shape (...)
        Pellet specific heat capacity in J/(kg K), shaped like
        ``temperature``. A new array.

    Examples
    --------
    >>> round(float(neacrp_fuel_heat_capacity(800.0)), 1)
    285.1
    """
    t = np.asarray(temperature, dtype=float)
    return 162.3 + 0.3038 * t - 2.391e-4 * t**2 + 6.404e-8 * t**3


def neacrp_clad_heat_capacity(temperature) -> np.ndarray:
    """Zircaloy-4 specific heat capacity in J/(kg K), NEACRP-L-335 Section 2.7.

    .. math:: c_{p,Zr} = 252.54 + 0.11474 T

    Pass it as ``clad_heat_capacity`` for :meth:`PinConduction.step`.

    Parameters
    ----------
    temperature : array_like, shape (...)
        Cladding temperature, K.

    Returns
    -------
    ndarray, shape (...)
        Cladding specific heat capacity in J/(kg K), shaped like
        ``temperature``. A new array.

    Examples
    --------
    >>> round(float(neacrp_clad_heat_capacity(600.0)), 2)
    321.38
    """
    t = np.asarray(temperature, dtype=float)
    return 252.54 + 0.11474 * t
