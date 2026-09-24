"""Water and steam properties behind a pluggable backend (FR-TH-3, FR-TH-8).

Three backends, all satisfying :class:`WaterProperties`:

:class:`ConstantWater`
    Fixed density and a constant specific heat. No physics, which is the
    point: a channel model run against it has a closed-form answer, so the
    channel model can be verified before the properties are trusted.
:class:`IF97Water`
    The IAPWS Industrial Formulation 1997, the default. Covers the PWR and
    BWR ranges FR-TH-3 asks for.
:func:`external_backend`
    Adapts the ``iapws`` package or CoolProp when one is installed, so
    property consistency with an external thermal-hydraulics code can be
    enforced (FR-TH-8). Neither is a dependency of this package.

Pressures are in Pa, temperatures in K, enthalpies in J/kg and densities in
kg/m^3 throughout. Every backend accepts arrays and broadcasts.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .exceptions import ConvergenceError, InputError

__all__ = [
    "CRITICAL_PRESSURE",
    "CRITICAL_TEMPERATURE",
    "GAS_CONSTANT",
    "ConstantWater",
    "IF97Water",
    "WaterProperties",
    "external_backend",
]

GAS_CONSTANT = 461.526
"""Specific gas constant of ordinary water, J/(kg K). IF97 Eq. (1)."""

_REGION1_PRESSURE_STAR = 16.53e6
_REGION1_TEMPERATURE_STAR = 1386.0

_REGION1_I = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
     3, 3, 3, 4, 4, 4, 5, 8, 8, 21, 23, 29, 30, 31, 32],
    dtype=float,
)
_REGION1_J = np.array(
    [-2, -1, 0, 1, 2, 3, 4, 5, -9, -7, -1, 0, 1, 3, -3, 0, 1, 3, 17,
     -4, 0, 6, -5, -2, 10, -8, -11, -6, -29, -31, -38, -39, -40, -41],
    dtype=float,
)
_REGION1_N = np.array(
    [
        0.14632971213167,
        -0.84548187169114,
        -0.37563603672040e1,
        0.33855169168385e1,
        -0.95791963387872,
        0.15772038513228,
        -0.16616417199501e-1,
        0.81214629983568e-3,
        0.28319080123804e-3,
        -0.60706301565874e-3,
        -0.18990068218419e-1,
        -0.32529748770505e-1,
        -0.21841717175414e-1,
        -0.52838357969930e-4,
        -0.47184321073267e-3,
        -0.30001780793026e-3,
        0.47661393906987e-4,
        -0.44141845330846e-5,
        -0.72694996297594e-15,
        -0.31679644845054e-4,
        -0.28270797985312e-5,
        -0.85205128120103e-9,
        -0.22425281908000e-5,
        -0.65171222895601e-6,
        -0.14341729937924e-12,
        -0.40516996860117e-6,
        -0.12734301741641e-8,
        -0.17424871230634e-9,
        -0.68762131295531e-18,
        0.14478307828521e-19,
        0.26335781662795e-22,
        -0.11947622640071e-22,
        0.18228094581404e-23,
        -0.93537087292458e-25,
    ]
)

_REGION4_N = np.array(
    [
        0.11670521452767e4,
        -0.72421316703206e6,
        -0.17073846940092e2,
        0.12020824702470e5,
        -0.32325550323333e7,
        0.14915108613530e2,
        -0.48232657361591e4,
        0.40511340542057e6,
        -0.23855557567849,
        0.65017534844798e3,
    ]
)

REGION1_TEMPERATURE_RANGE = (273.15, 623.15)
"""Temperatures over which IF97 region 1 holds, K. IF97 Eq. (7)."""

REGION1_PRESSURE_LIMIT = 100.0e6
"""Highest pressure at which IF97 region 1 holds, Pa; the lowest is the
saturation line rather than a constant. IF97 Eq. (7)."""

CRITICAL_TEMPERATURE = 647.096
"""Critical temperature of water, K. IF97 Eq. (2)."""

CRITICAL_PRESSURE = 22.064e6
"""Critical pressure of water, Pa; above it there is no saturation line to
cross. IF97 Eq. (3)."""

SATURATION_TEMPERATURE_RANGE = (273.15, CRITICAL_TEMPERATURE)
"""Temperatures over which the saturation line is defined, K. IF97 Section 8."""

SATURATION_PRESSURE_RANGE = (611.213, CRITICAL_PRESSURE)
"""Pressures over which the saturation line is defined, Pa. The lower one is
what IF97 Eq. (31) gives when extrapolated to 273.15 K. IF97 Section 8."""


@runtime_checkable
class WaterProperties(Protocol):
    """What a property backend must provide (FR-TH-8).

    The channel solve advances enthalpy and needs the temperature and density
    that go with it, so the inverse :meth:`temperature` is part of the
    interface rather than something a caller is left to invert.
    """

    def density(self, pressure, temperature) -> np.ndarray:
        """Density in kg/m^3 at a pressure in Pa and temperature in K."""

    def enthalpy(self, pressure, temperature) -> np.ndarray:
        """Specific enthalpy in J/kg at a pressure in Pa, temperature in K."""

    def temperature(self, pressure, enthalpy) -> np.ndarray:
        """Temperature in K at a pressure in Pa and enthalpy in J/kg."""

    def saturation_temperature(self, pressure) -> np.ndarray:
        """Saturation temperature in K at a pressure in Pa."""


class ConstantWater:
    """Incompressible water with a constant specific heat.

    Nothing here is a property of water; the numbers are whatever the caller
    gives. It exists so a channel model has an analytically verifiable
    reference: with constant density and specific heat the axial enthalpy
    rise of a closed channel is exactly the integral of the heat input, so a
    coupled solve has a closed-form answer to be checked against.

    Parameters
    ----------
    density : float, optional
        kg/m^3. The default is roughly water at PWR conditions.
    specific_heat : float, optional
        J/(kg K), constant.
    reference_temperature : float, optional
        K, the temperature at which the enthalpy is taken to be
        ``reference_enthalpy``.
    reference_enthalpy : float, optional
        J/kg.
    saturation_temperature : float, optional
        K, reported by :meth:`saturation_temperature` at any pressure.

    Raises
    ------
    InputError
        On a non-positive density or specific heat.

    Examples
    --------
    >>> water = ConstantWater(density=750.0, specific_heat=5000.0)
    >>> float(water.temperature(15.5e6, water.enthalpy(15.5e6, 600.0)))
    600.0
    """

    def __init__(
        self,
        *,
        density: float = 750.0,
        specific_heat: float = 5000.0,
        reference_temperature: float = 560.0,
        reference_enthalpy: float = 1.3e6,
        saturation_temperature: float = 618.0,
    ):
        if not density > 0.0:
            raise InputError(f"density must be positive, got {density}")
        if not specific_heat > 0.0:
            raise InputError(f"specific heat must be positive, got {specific_heat}")
        self._density = float(density)
        self.specific_heat = float(specific_heat)
        self.reference_temperature = float(reference_temperature)
        self.reference_enthalpy = float(reference_enthalpy)
        self._saturation = float(saturation_temperature)

    def density(self, pressure, temperature) -> np.ndarray:
        """Density in kg/m^3, the constant, broadcast to the inputs."""
        return np.broadcast_arrays(
            np.asarray(pressure, dtype=float),
            np.asarray(temperature, dtype=float),
            np.asarray(self._density),
        )[2].copy()

    def enthalpy(self, pressure, temperature) -> np.ndarray:
        """Specific enthalpy in J/kg, linear in temperature."""
        t = np.asarray(temperature, dtype=float)
        return self.reference_enthalpy + self.specific_heat * (
            t - self.reference_temperature
        )

    def temperature(self, pressure, enthalpy) -> np.ndarray:
        """Temperature in K, the exact inverse of :meth:`enthalpy`."""
        h = np.asarray(enthalpy, dtype=float)
        return self.reference_temperature + (
            h - self.reference_enthalpy
        ) / self.specific_heat

    def saturation_temperature(self, pressure) -> np.ndarray:
        """Saturation temperature in K, the constant, broadcast."""
        return np.broadcast_to(
            np.asarray(self._saturation), np.shape(pressure)
        ).astype(float, copy=True)

    def __repr__(self) -> str:
        return (
            f"<ConstantWater rho={self._density:g} kg/m^3 "
            f"cp={self.specific_heat:g} J/(kg K)>"
        )


class IF97Water:
    """IAPWS-IF97 water properties: region 1 and the saturation line.

    Region 1 is the compressed liquid, 273.15 K to 623.15 K at pressures from
    the saturation line to 100 MPa, which covers the coolant of a PWR and the
    subcooled part of a BWR. Region 2 (vapour) is not implemented, so a
    two-phase model needs a backend that has it; see
    :func:`external_backend`.

    Enthalpy and density come from the basic equation, Eq. (7), rather than
    from the backward equations. :meth:`temperature` inverts Eq. (7) by
    Newton iteration instead of using the backward equation of Table 6: the
    standard permits those two to disagree by up to 25 mK, and an inverse
    that is exact against the equation it inverts has no such gap and no
    second coefficient table to keep in step.

    Parameters
    ----------
    tolerance : float, optional
        Convergence tolerance of the enthalpy inversion, in K.
    max_iterations : int, optional
        Iteration cap for the inversion.

    Raises
    ------
    InputError
        On a non-positive tolerance or iteration cap.

    Notes
    -----
    Coefficients are Tables 2 and 34 of IAPWS R7-97(2012), "Revised Release
    on the IAPWS Industrial Formulation 1997 for the Thermodynamic Properties
    of Water and Steam", International Association for the Properties of
    Water and Steam, Lucerne, August 2007. They are transcribed from the
    release, and the release's own computer-program verification values
    (Tables 5, 35 and 36) are asserted in the test suite, which is what makes
    a transcription error visible.

    Examples
    --------
    >>> water = IF97Water()
    >>> round(float(water.enthalpy(3.0e6, 300.0)), 3)
    115331.273
    >>> round(float(water.saturation_temperature(10.0e6)), 6)
    584.149488
    """

    def __init__(self, *, tolerance: float = 1.0e-9, max_iterations: int = 30):
        if not tolerance > 0.0:
            raise InputError(f"tolerance must be positive, got {tolerance}")
        if max_iterations < 1:
            raise InputError(
                f"max_iterations must be at least 1, got {max_iterations}"
            )
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)

    def specific_volume(self, pressure, temperature) -> np.ndarray:
        """Specific volume in m^3/kg, from IF97 Eq. (7) and Table 3."""
        p, t = self._checked(pressure, temperature)
        pi = p / _REGION1_PRESSURE_STAR
        tau = _REGION1_TEMPERATURE_STAR / t
        return GAS_CONSTANT * t / p * pi * self._gamma_pi(pi, tau)

    def density(self, pressure, temperature) -> np.ndarray:
        """Density in kg/m^3 at a pressure in Pa and temperature in K."""
        return 1.0 / self.specific_volume(pressure, temperature)

    def enthalpy(self, pressure, temperature) -> np.ndarray:
        """Specific enthalpy in J/kg at a pressure in Pa, temperature in K."""
        p, t = self._checked(pressure, temperature)
        pi = p / _REGION1_PRESSURE_STAR
        tau = _REGION1_TEMPERATURE_STAR / t
        return GAS_CONSTANT * t * tau * self._gamma_tau(pi, tau)

    def specific_heat(self, pressure, temperature) -> np.ndarray:
        """Isobaric specific heat in J/(kg K), the derivative of enthalpy."""
        p, t = self._checked(pressure, temperature)
        pi = p / _REGION1_PRESSURE_STAR
        tau = _REGION1_TEMPERATURE_STAR / t
        return -GAS_CONSTANT * tau * tau * self._gamma_tau_tau(pi, tau)

    def temperature(self, pressure, enthalpy) -> np.ndarray:
        """Temperature in K at a pressure in Pa and enthalpy in J/kg.

        Newton iteration on :meth:`enthalpy`, whose derivative is the
        specific heat. Enthalpy rises monotonically with temperature
        throughout region 1, so the iteration has one root and reaches it in
        a handful of steps from any start in range.

        Raises
        ------
        ConvergenceError
            If the iteration does not settle within ``max_iterations``.
        """
        p = np.asarray(pressure, dtype=float)
        h = np.asarray(enthalpy, dtype=float)
        p, h = np.broadcast_arrays(p, h)

        low = REGION1_TEMPERATURE_RANGE[0]
        ceiling = self._region1_ceiling(p)
        t = 0.5 * (low + ceiling)
        for _ in range(self.max_iterations):
            step = (h - self.enthalpy(p, t)) / self.specific_heat(p, t)
            t = np.clip(t + step, low, ceiling)
            if np.all(np.abs(step) < self.tolerance):
                return t
        raise ConvergenceError(
            "IF97 enthalpy inversion did not converge",
            self.max_iterations,
            float(np.max(np.abs(step))),
        )

    def saturation_pressure(self, temperature) -> np.ndarray:
        """Saturation pressure in Pa, from IF97 Eq. (30).

        Valid from 273.15 K to the critical temperature, 647.096 K.
        """
        t = np.asarray(temperature, dtype=float)
        if np.any(t < SATURATION_TEMPERATURE_RANGE[0]) or np.any(
            t > SATURATION_TEMPERATURE_RANGE[1]
        ):
            low, high = SATURATION_TEMPERATURE_RANGE
            raise InputError(
                f"the saturation line runs from {low} K to the critical "
                f"temperature {high} K; got {float(np.min(t))} K to "
                f"{float(np.max(t))} K"
            )
        n = _REGION4_N
        theta = t + n[8] / (t - n[9])
        a = theta * theta + n[0] * theta + n[1]
        b = n[2] * theta * theta + n[3] * theta + n[4]
        c = n[5] * theta * theta + n[6] * theta + n[7]
        root = 2.0 * c / (-b + np.sqrt(b * b - 4.0 * a * c))
        return 1.0e6 * root**4

    def saturation_temperature(self, pressure) -> np.ndarray:
        """Saturation temperature in K, from IF97 Eq. (31).

        Valid from 611.213 Pa to the critical pressure, 22.064 MPa.
        """
        p = np.asarray(pressure, dtype=float)
        if np.any(p < SATURATION_PRESSURE_RANGE[0]) or np.any(
            p > SATURATION_PRESSURE_RANGE[1]
        ):
            low, high = SATURATION_PRESSURE_RANGE
            raise InputError(
                f"the saturation line runs from {low} Pa to the critical "
                f"pressure {high} Pa; above it there is no phase boundary. "
                f"Got {float(np.min(p))} Pa to {float(np.max(p))} Pa"
            )
        n = _REGION4_N
        beta = (p / 1.0e6) ** 0.25
        e = beta * beta + n[2] * beta + n[5]
        f = n[0] * beta * beta + n[3] * beta + n[6]
        g = n[1] * beta * beta + n[4] * beta + n[7]
        d = 2.0 * g / (-f - np.sqrt(f * f - 4.0 * e * g))
        return 0.5 * (
            n[9] + d - np.sqrt((n[9] + d) ** 2 - 4.0 * (n[8] + n[9] * d))
        )

    def _region1_ceiling(self, pressure) -> np.ndarray:
        """Highest temperature at which region 1 holds, per pressure, in K.

        Water stops being liquid at the saturation line, so the ceiling is
        the saturation temperature below the critical pressure and region 1's
        own upper bound above it, where there is no saturation line to cross.
        """
        p = np.asarray(pressure, dtype=float)
        high = REGION1_TEMPERATURE_RANGE[1]
        subcritical = p < CRITICAL_PRESSURE
        saturated = self.saturation_temperature(np.minimum(p, CRITICAL_PRESSURE))
        return np.minimum(high, np.where(subcritical, saturated, high))

    def _checked(self, pressure, temperature):
        p = np.asarray(pressure, dtype=float)
        t = np.asarray(temperature, dtype=float)
        low, high = REGION1_TEMPERATURE_RANGE
        if np.any(t < low) or np.any(t > high):
            raise InputError(
                f"temperature outside IF97 region 1, {low} K to {high} K; "
                f"got {float(np.min(t))} K to {float(np.max(t))} K"
            )
        if np.any(p <= 0.0) or np.any(p > REGION1_PRESSURE_LIMIT):
            raise InputError(
                f"pressure outside IF97 region 1, up to "
                f"{REGION1_PRESSURE_LIMIT} Pa; got {float(np.min(p))} Pa to "
                f"{float(np.max(p))} Pa"
            )
        if np.any(p < self.saturation_pressure(t)):
            raise InputError(
                "pressure is below the saturation pressure, so the water is "
                "not liquid; region 1 does not cover it"
            )
        return np.broadcast_arrays(p, t)

    @staticmethod
    def _gamma_pi(pi, tau) -> np.ndarray:
        base = 7.1 - pi[..., np.newaxis]
        return -np.sum(
            _REGION1_N
            * _REGION1_I
            * base ** (_REGION1_I - 1.0)
            * (tau[..., np.newaxis] - 1.222) ** _REGION1_J,
            axis=-1,
        )

    @staticmethod
    def _gamma_tau(pi, tau) -> np.ndarray:
        shifted = tau[..., np.newaxis] - 1.222
        return np.sum(
            _REGION1_N
            * (7.1 - pi[..., np.newaxis]) ** _REGION1_I
            * _REGION1_J
            * shifted ** (_REGION1_J - 1.0),
            axis=-1,
        )

    @staticmethod
    def _gamma_tau_tau(pi, tau) -> np.ndarray:
        shifted = tau[..., np.newaxis] - 1.222
        return np.sum(
            _REGION1_N
            * (7.1 - pi[..., np.newaxis]) ** _REGION1_I
            * _REGION1_J
            * (_REGION1_J - 1.0)
            * shifted ** (_REGION1_J - 2.0),
            axis=-1,
        )

    def __repr__(self) -> str:
        return "<IF97Water region 1 and saturation line>"


class _ExternalBackend:
    """Adapts a scalar property package to the array protocol.

    Subclasses supply the four scalar lookups. Everything vectorised here,
    because neither package takes arrays and a channel solve asks for a whole
    axial mesh at once.
    """

    def density(self, pressure, temperature) -> np.ndarray:
        return self._map2(self._density, pressure, temperature)

    def enthalpy(self, pressure, temperature) -> np.ndarray:
        return self._map2(self._enthalpy, pressure, temperature)

    def temperature(self, pressure, enthalpy) -> np.ndarray:
        return self._map2(self._temperature, pressure, enthalpy)

    def saturation_temperature(self, pressure) -> np.ndarray:
        p = np.asarray(pressure, dtype=float)
        flat = [self._saturation_temperature(float(v)) for v in p.ravel()]
        return np.asarray(flat, dtype=float).reshape(p.shape)

    @staticmethod
    def _map2(scalar, first, second) -> np.ndarray:
        a, b = np.broadcast_arrays(
            np.asarray(first, dtype=float), np.asarray(second, dtype=float)
        )
        pairs = zip(a.ravel(), b.ravel(), strict=True)
        flat = [scalar(float(x), float(y)) for x, y in pairs]
        return np.asarray(flat, dtype=float).reshape(a.shape)


class _IapwsBackend(_ExternalBackend):
    """The ``iapws`` package, which takes pressure in MPa and enthalpy in kJ/kg."""

    def __init__(self):
        from iapws import IAPWS97

        self._state = IAPWS97

    def _density(self, pressure, temperature):
        return self._state(P=pressure * 1.0e-6, T=temperature).rho

    def _enthalpy(self, pressure, temperature):
        return self._state(P=pressure * 1.0e-6, T=temperature).h * 1.0e3

    def _temperature(self, pressure, enthalpy):
        return self._state(P=pressure * 1.0e-6, h=enthalpy * 1.0e-3).T

    def _saturation_temperature(self, pressure):
        return self._state(P=pressure * 1.0e-6, x=0.0).T

    def __repr__(self) -> str:
        return "<external water properties: iapws>"


class _CoolPropBackend(_ExternalBackend):
    """CoolProp, which is SI throughout and needs no unit conversion."""

    def __init__(self):
        from CoolProp.CoolProp import PropsSI

        self._props = PropsSI

    def _density(self, pressure, temperature):
        return self._props("D", "P", pressure, "T", temperature, "Water")

    def _enthalpy(self, pressure, temperature):
        return self._props("H", "P", pressure, "T", temperature, "Water")

    def _temperature(self, pressure, enthalpy):
        return self._props("T", "P", pressure, "H", enthalpy, "Water")

    def _saturation_temperature(self, pressure):
        return self._props("T", "P", pressure, "Q", 0.0, "Water")


_BACKENDS = {"iapws": _IapwsBackend, "coolprop": _CoolPropBackend}


def external_backend(name: str = "auto") -> WaterProperties:
    """Adapt an installed property package to :class:`WaterProperties`.

    FR-TH-8 asks for this so that property consistency with an external
    thermal-hydraulics code can be enforced: run both codes off the same
    package and the properties cannot disagree.

    Neither package is a dependency of OpenNDM, and neither is imported until
    this is called. ``iapws`` is GPL-licensed, so installing it is the
    caller's decision rather than something this package makes for them.

    Parameters
    ----------
    name : {'auto', 'iapws', 'coolprop'}, optional
        Which package to adapt. ``'auto'`` takes the first one installed.

    Returns
    -------
    WaterProperties

    Raises
    ------
    InputError
        If the name is not one of the above, or the named package is not
        installed, or ``'auto'`` finds none.
    """
    if name == "auto":
        for candidate in _BACKENDS:
            try:
                return _BACKENDS[candidate]()
            except ImportError:
                continue
        raise InputError(
            f"no external property package installed; tried "
            f"{list(_BACKENDS)}. Use IF97Water for the built-in formulation."
        )
    if name not in _BACKENDS:
        raise InputError(
            f"unknown backend {name!r}; choose from {list(_BACKENDS)} or 'auto'"
        )
    try:
        return _BACKENDS[name]()
    except ImportError as error:
        raise InputError(f"the {name} package is not installed") from error
