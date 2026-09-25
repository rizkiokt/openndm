"""Radial fuel pin conduction and the Doppler temperature (FR-TH-2, FR-TH-4).

Steady one-dimensional conduction from the pellet centreline out to the
coolant, across four resistances in series: the pellet, the pellet-to-clad
gap, the cladding, and the film. The pellet is meshed radially; the other
three are single resistances, having no internal generation.

No conductivity correlation is written here. Conductivities and coefficients
are injected the way :class:`~openndm.WaterProperties` backends are, because
the published correlations are temperature-dependent and a correlation
written from memory is the failure mode this project has already paid for
once. Every check the model is verified against holds at constant
conductivity.

Lengths are metres, temperatures K, linear heat rates W/m, conductivities
W/(m K) and the gap and film coefficients W/(m^2 K).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .exceptions import InputError

if TYPE_CHECKING:
    from collections.abc import Callable

    from .channel import PinGeometry

__all__ = ["PinConduction", "PinState"]


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

    Attributes
    ----------
    profile, clad_inner, clad_outer, average, doppler
        As above. Every array is a copy the caller owns.
    """

    def __init__(self, profile, clad_inner, clad_outer, average, doppler):
        self.profile = np.asarray(profile, dtype=float)
        self.clad_inner = np.asarray(clad_inner, dtype=float)
        self.clad_outer = np.asarray(clad_outer, dtype=float)
        self.average = np.asarray(average, dtype=float)
        self.doppler = np.asarray(doppler, dtype=float)

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

    Raises
    ------
    InputError
        On a non-positive conductivity, conductance or coefficient, fewer
        than one ring, or a weight outside ``[0, 1]``.

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
        ):
            if not value > 0.0:
                raise InputError(f"{name} must be positive, got {value}")

        self.pins = pins
        self._fuel_conductivity = _checked_conductivity(
            "fuel_conductivity", fuel_conductivity
        )
        self._clad_conductivity = _checked_conductivity(
            "clad_conductivity", clad_conductivity
        )
        self.gap_conductance = float(gap_conductance)
        self.film_coefficient = float(film_coefficient)
        self.doppler_weight = float(doppler_weight)
        self.radii = np.linspace(0.0, pins.fuel_radius, int(n_rings) + 1)

    @property
    def n_rings(self) -> int:
        """Rings across the pellet."""
        return self.radii.size - 1

    def solve(self, linear_heat_rate, coolant_temperature) -> PinState:
        """Temperatures through the pin at every node.

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

        Raises
        ------
        InputError
            If the two arrays are not one-dimensional and the same length, or
            if the linear heat rate is negative or not finite.
        """
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

        clad_outer = bulk + self._film_rise(rate)
        clad_inner = clad_outer + self._clad_rise(rate, clad_outer)
        surface = clad_inner + self._gap_rise(rate)
        profile = self._pellet_profile(rate, surface)

        average = self._volume_average(profile)
        doppler = (
            self.doppler_weight * profile[:, -1]
            + (1.0 - self.doppler_weight) * profile[:, 0]
        )
        return PinState(profile, clad_inner, clad_outer, average, doppler)

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


def _checked_conductivity(name: str, value) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a constant or a correlation into one callable, validating it."""
    if callable(value):
        return value
    conductivity = float(value)
    if not conductivity > 0.0:
        raise InputError(f"{name} must be positive, got {conductivity}")
    return lambda temperature: np.full_like(
        np.asarray(temperature, dtype=float), conductivity
    )
