"""Single-phase closed-channel coolant model (FR-TH-1).

One channel per radial column of the core map, carrying a fixed mass flow
with no momentum equation, so the only conservation law left is energy. The
model is a :class:`~openndm.ThermalSolver`, so it couples through
:class:`~openndm.PicardCoupling` like any external solver would.

Lengths in the pin geometry are metres, pressures Pa, temperatures K,
enthalpies J/kg and powers W. The node mesh is in cm, as everywhere else in
the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .exceptions import InputError
from .water import IF97Water, SaturationProperties

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .geometry import Geometry
    from .pin import PinConduction, PinState
    from .water import WaterProperties

__all__ = ["ChannelModel", "PinGeometry", "absolute_power"]

CM_PER_M = 100.0


@dataclass(frozen=True)
class PinGeometry:
    """Pin and assembly dimensions of one coolant channel.

    Guide tubes displace coolant and carry no heat. They are taken to have
    the cladding's outer radius, because the input set carries no separate
    dimension for them.

    Attributes
    ----------
    fuel_radius : float
        Fuel meat radius, m.
    gap_thickness : float
        Pellet-to-cladding gap, m.
    clad_thickness : float
        Cladding thickness, m.
    pin_pitch : float
        Centre-to-centre pin spacing, m.
    n_pins : int
        Heated fuel pins in the channel.
    n_guide_tubes : int
        Unheated guide tubes in the channel.

    Raises
    ------
    InputError
        On a non-positive dimension or a channel with no fuel pins.

    Examples
    --------
    >>> pins = PinGeometry(4.1195e-3, 6.8e-5, 5.71e-4, 1.2655e-2, 264, 25)
    >>> pins.n_rods
    289
    """

    fuel_radius: float
    gap_thickness: float
    clad_thickness: float
    pin_pitch: float
    n_pins: int
    n_guide_tubes: int

    def __post_init__(self):
        for name in ("fuel_radius", "gap_thickness", "clad_thickness", "pin_pitch"):
            value = float(getattr(self, name))
            if not value > 0.0:
                raise InputError(f"{name} must be positive, got {value}")
        if self.n_pins < 1:
            raise InputError(f"a channel needs at least one pin, got {self.n_pins}")
        if self.n_guide_tubes < 0:
            raise InputError(
                f"n_guide_tubes cannot be negative, got {self.n_guide_tubes}"
            )
        if self.clad_outer_radius >= 0.5 * self.pin_pitch:
            raise InputError(
                f"a pin of outer radius {self.clad_outer_radius} m does not fit "
                f"in a lattice cell of pitch {self.pin_pitch} m"
            )

    @property
    def clad_inner_radius(self) -> float:
        """Cladding inner radius, m: across the pellet and the gap."""
        return self.fuel_radius + self.gap_thickness

    @property
    def clad_outer_radius(self) -> float:
        """Cladding outer radius, m."""
        return self.clad_inner_radius + self.clad_thickness

    @property
    def n_rods(self) -> int:
        """Lattice positions the channel's coolant flows past."""
        return self.n_pins + self.n_guide_tubes

    @property
    def flow_area(self) -> float:
        """Free coolant area of the channel, m^2.

        The lattice cells the rods occupy, less the rods themselves.
        """
        cell = self.pin_pitch**2 - np.pi * self.clad_outer_radius**2
        return float(self.n_rods * cell)

    @property
    def heated_perimeter(self) -> float:
        """Perimeter the power crosses, m. Fuel pins only."""
        return float(self.n_pins * 2.0 * np.pi * self.clad_outer_radius)

    @property
    def wetted_perimeter(self) -> float:
        """Perimeter in contact with coolant, m. Pins and guide tubes."""
        return float(self.n_rods * 2.0 * np.pi * self.clad_outer_radius)

    @property
    def hydraulic_diameter(self) -> float:
        """Four times the flow area over the wetted perimeter, m."""
        return 4.0 * self.flow_area / self.wetted_perimeter


class ChannelModel:
    """Closed-channel coolant energy balance, one channel per column.

    Steady energy conservation at constant mass flow makes the enthalpy at
    any height the inlet enthalpy plus the power raised below it, divided by
    the flow. Temperature and density follow from the enthalpy through the
    property backend, so the model is exact to the properties it is given.

    In steady state every watt reaches the coolant whatever route it takes,
    so ``direct_heating`` does not enter the enthalpy rise. It splits the
    power between the coolant and the pin, and so sets
    :attr:`linear_heat_rate`, which is what a conduction model consumes.

    With ``two_phase`` the enthalpy integration is unchanged and only its
    inversion differs: above the saturated liquid enthalpy the temperature
    stops at the boiling point and the surplus becomes quality and void.

    Parameters
    ----------
    geometry : Geometry
        Supplies the channel map and the axial mesh. Each radial column of
        the lattice holding at least one active node becomes one channel.
    pins : PinGeometry
        Dimensions of one channel. A subdivided assembly is several channels,
        so the pin counts are per channel, not per assembly.
    mass_flow : float or array_like
        Coolant mass flow per channel, kg/s. A scalar applies to every
        channel; an array gives ``n_channels`` values in channel order.
    inlet_temperature : float
        Coolant temperature entering every channel, K.
    pressure : float, optional
        System pressure, Pa. Constant, there being no momentum equation.
    direct_heating : float, optional
        Fraction of the node power deposited straight in the coolant rather
        than raised in the fuel, in ``[0, 1]``.
    water : WaterProperties, optional
        Property backend. :class:`~openndm.IF97Water` by default.
    conduction : PinConduction, optional
        Radial pin conduction (FR-TH-2). Without it the model reports the
        coolant alone; with it, also the fuel and Doppler temperatures. It
        must carry the same ``pins``.
    two_phase : bool, optional
        Allow the coolant to boil (FR-TH-5). The backend must then satisfy
        :class:`~openndm.SaturationProperties`. A channel that stays
        subcooled gives bit-identical answers either way, so this extends the
        range of validity rather than changing the model.
    slip_ratio : float, optional
        Ratio of vapour to liquid velocity in the void fraction. One is the
        homogeneous equilibrium model and the default; anything else is a
        correlation the caller is choosing.

    Attributes
    ----------
    pressure : float
        System pressure, Pa.
    inlet_temperature : float
        Channel inlet temperature, K.
    direct_heating : float
        Fraction deposited directly in the coolant.

    Raises
    ------
    InputError
        On a non-positive mass flow or inlet temperature, a direct heating
        fraction outside ``[0, 1]``, a mass flow array of the wrong length,
        or a geometry with no active node.

    Examples
    --------
    >>> import numpy as np
    >>> from openndm import ConstantWater, Geometry
    >>> geom = Geometry.from_lattice(np.zeros((4, 1, 1), int), pitch=(20, 20, 25))
    >>> pins = PinGeometry(4.1195e-3, 6.8e-5, 5.71e-4, 1.2655e-2, 264, 25)
    >>> channel = ChannelModel(
    ...     geom, pins, mass_flow=80.0, inlet_temperature=560.0,
    ...     water=ConstantWater(specific_heat=5000.0),
    ... )
    >>> channel.set_heat_source(np.full(4, 1.0e6))
    >>> channel.solve()
    >>> float(channel.outlet_temperature[0] - 560.0)
    10.0
    """

    def __init__(
        self,
        geometry: Geometry,
        pins: PinGeometry,
        *,
        mass_flow,
        inlet_temperature: float,
        pressure: float = 15.5e6,
        direct_heating: float = 0.0,
        water: WaterProperties | None = None,
        conduction: PinConduction | None = None,
        two_phase: bool = False,
        slip_ratio: float = 1.0,
    ):
        if not inlet_temperature > 0.0:
            raise InputError(
                f"inlet temperature must be above absolute zero, got "
                f"{inlet_temperature}"
            )
        if not pressure > 0.0:
            raise InputError(f"pressure must be positive, got {pressure}")
        if not 0.0 <= direct_heating <= 1.0:
            raise InputError(
                f"direct_heating is a fraction and must lie in [0, 1], got "
                f"{direct_heating}"
            )

        if conduction is not None and conduction.pins != pins:
            raise InputError(
                "the conduction model was built on different pin dimensions "
                "from the channel"
            )
        if not slip_ratio > 0.0:
            raise InputError(f"slip_ratio must be positive, got {slip_ratio}")

        self._geometry = geometry
        self.pins = pins
        self.pressure = float(pressure)
        self.inlet_temperature = float(inlet_temperature)
        self.direct_heating = float(direct_heating)
        self._water = IF97Water() if water is None else water
        if two_phase and not isinstance(self._water, SaturationProperties):
            raise InputError(
                "a two-phase channel needs a backend giving both sides of the "
                "saturation line; ConstantWater does not, IF97Water does"
            )
        self._conduction = conduction
        self._pin_state = None
        self.two_phase = bool(two_phase)
        self.slip_ratio = float(slip_ratio)

        self._nodes = _channel_columns(geometry)
        self._active = self._nodes >= 0
        self._gather_index = np.where(self._active, self._nodes, 0)
        self._heights = np.asarray(geometry.dz, dtype=float) / CM_PER_M

        self._mass_flow = self._checked_mass_flow(mass_flow)
        self._power = np.zeros(geometry.n_nodes, dtype=float)
        self.solve()

    @property
    def n_channels(self) -> int:
        """Number of channels, one per radial column carrying a node."""
        return self._nodes.shape[0]

    @property
    def mass_flow(self) -> np.ndarray:
        """Mass flow per channel in kg/s, shape ``(n_channels,)``. A copy."""
        return self._mass_flow.copy()

    @property
    def enthalpy(self) -> np.ndarray:
        """Node-average specific enthalpy in J/kg, shape ``(n_nodes,)``.

        A copy.
        """
        return self._enthalpy.copy()

    @property
    def outlet_enthalpy(self) -> np.ndarray:
        """Specific enthalpy leaving each channel, J/kg. A copy."""
        return self._outlet_enthalpy.copy()

    @property
    def outlet_temperature(self) -> np.ndarray:
        """Temperature leaving each channel, K, shape ``(n_channels,)``."""
        return np.asarray(
            self._water.temperature(self.pressure, self._outlet_enthalpy),
            dtype=float,
        )

    @property
    def mass_flux(self) -> np.ndarray:
        """Coolant mass flux per channel, kg/(m^2 s)."""
        return self._mass_flow / self.pins.flow_area

    @property
    def quality(self) -> np.ndarray:
        """Equilibrium steam quality per node, shape ``(n_nodes,)``.

        Zero everywhere in a single-phase channel, and zero at and below the
        saturated liquid enthalpy in a two-phase one. A copy.
        """
        return self._quality.copy()

    @property
    def void_fraction(self) -> np.ndarray:
        """Void fraction per node, shape ``(n_nodes,)``. A copy."""
        return self._void_fraction.copy()

    @property
    def pin_state(self) -> PinState | None:
        """Pin temperatures from the last solve, or None without conduction."""
        return self._pin_state

    @property
    def linear_heat_rate(self) -> np.ndarray:
        """Power per unit pin length in each node, W/m, shape ``(n_nodes,)``.

        What the fuel raises, so the directly deposited fraction is excluded.
        """
        pin_length = self.pins.n_pins * self._node_heights()
        return (1.0 - self.direct_heating) * self._power / pin_length

    def set_heat_source(self, q: np.ndarray) -> None:
        """Set the power raised in each node, in W.

        Parameters
        ----------
        q : array_like, shape (n_nodes,)
            Total power per node, fission energy included whatever fraction
            of it is deposited in the coolant.

        Raises
        ------
        InputError
            If the shape is wrong, or any entry is negative or not finite.
        """
        power = np.asarray(q, dtype=float)
        if power.shape != (self._geometry.n_nodes,):
            raise InputError(
                f"expected a heat source of {self._geometry.n_nodes} nodes, got "
                f"shape {power.shape}"
            )
        if not np.all(np.isfinite(power)):
            raise InputError("the heat source holds a non-finite entry")
        if np.any(power < 0.0):
            raise InputError(
                f"node power cannot be negative, got {float(power.min())} W"
            )
        self._power = power.copy()

    def solve(self) -> None:
        """Integrate the enthalpy up every channel and invert for the state."""
        per_channel = np.where(self._active, self._power[self._gather_index], 0.0)
        rise = np.cumsum(per_channel, axis=1) / self._mass_flow[:, np.newaxis]
        inlet = self._inlet_enthalpy()

        average = inlet + rise - 0.5 * per_channel / self._mass_flow[:, np.newaxis]
        self._outlet_enthalpy = inlet + rise[:, -1]
        self._enthalpy = self._scatter(average, fill=inlet)
        (
            self._temperature,
            self._density,
            self._quality,
            self._void_fraction,
        ) = self._invert(self._enthalpy)
        if self._conduction is not None:
            self._pin_state = self._conduction.solve(
                self.linear_heat_rate, self._temperature
            )

    def _invert(self, enthalpy: np.ndarray):
        """Temperature, density, quality and void fraction from enthalpy.

        Single phase is the whole story unless ``two_phase`` is set. When it
        is, the subcooled nodes take exactly the same path they would have
        taken without it, so a channel that never reaches saturation gives
        bit-identical answers either way.
        """
        if not self.two_phase:
            temperature = np.asarray(
                self._water.temperature(self.pressure, enthalpy), dtype=float
            )
            density = np.asarray(
                self._water.density(self.pressure, temperature), dtype=float
            )
            zero = np.zeros_like(temperature)
            return temperature, density, zero, zero.copy()

        liquid_enthalpy = float(
            self._water.saturated_liquid_enthalpy(self.pressure)
        )
        vapour_enthalpy = float(
            self._water.saturated_vapour_enthalpy(self.pressure)
        )
        if np.any(enthalpy > vapour_enthalpy):
            hottest = float(enthalpy.max())
            raise InputError(
                f"the channel boils dry: node enthalpy reaches {hottest} J/kg "
                f"against a saturated vapour enthalpy of {vapour_enthalpy} "
                f"J/kg. Superheated steam is not modelled"
            )

        boiling = enthalpy > liquid_enthalpy
        temperature = np.empty_like(enthalpy)
        density = np.empty_like(enthalpy)
        quality = np.zeros_like(enthalpy)

        subcooled = ~boiling
        if subcooled.any():
            temperature[subcooled] = self._water.temperature(
                self.pressure, enthalpy[subcooled]
            )
            density[subcooled] = self._water.density(
                self.pressure, temperature[subcooled]
            )
        if boiling.any():
            temperature[boiling] = self._water.saturation_temperature(
                self.pressure
            )
            quality[boiling] = (enthalpy[boiling] - liquid_enthalpy) / (
                vapour_enthalpy - liquid_enthalpy
            )

        void_fraction = self._void_from_quality(quality)
        liquid_density = float(
            self._water.saturated_liquid_density(self.pressure)
        )
        vapour_density = float(
            self._water.saturated_vapour_density(self.pressure)
        )
        density[boiling] = (
            void_fraction[boiling] * vapour_density
            + (1.0 - void_fraction[boiling]) * liquid_density
        )
        return temperature, density, quality, void_fraction

    def _void_from_quality(self, quality: np.ndarray) -> np.ndarray:
        """Void fraction of a homogeneous mixture at a given quality.

        At ``slip_ratio`` one this is the homogeneous equilibrium model, where
        the mixture density it implies is exactly the inverse of the
        mass-weighted specific volume. A larger slip ratio is the caller's
        correlation, not this model's.
        """
        density_ratio = float(
            self._water.saturated_vapour_density(self.pressure)
        ) / float(self._water.saturated_liquid_density(self.pressure))
        void = np.zeros_like(quality)
        flowing = quality > 0.0
        if flowing.any():
            x = quality[flowing]
            void[flowing] = 1.0 / (
                1.0 + (1.0 - x) / x * density_ratio * self.slip_ratio
            )
        return void

    def get_temperatures(self) -> Mapping[str, np.ndarray]:
        """Temperature per node in K, keyed by field.

        Always ``'moderator_temperature'``. With a conduction model attached,
        also ``'fuel_temperature'``, the volume-average pellet temperature,
        and ``'doppler_temperature'``, the weighted one of FR-TH-4.

        Returns
        -------
        dict of str to ndarray
            Copies, not views onto the model's own buffers.
        """
        fields = {"moderator_temperature": self._temperature.copy()}
        if self._pin_state is not None:
            fields["fuel_temperature"] = self._pin_state.average.copy()
            fields["doppler_temperature"] = self._pin_state.doppler.copy()
        return fields

    def get_densities(self) -> Mapping[str, np.ndarray]:
        """Coolant density per node in kg/m^3, keyed ``'moderator_density'``.

        Returns
        -------
        dict of str to ndarray
            Copies, not views onto the model's own buffers.
        """
        return {"moderator_density": self._density.copy()}

    def _inlet_enthalpy(self) -> float:
        return float(self._water.enthalpy(self.pressure, self.inlet_temperature))

    def _node_heights(self) -> np.ndarray:
        """Height of each node in m, gathered back onto the node index."""
        stacked = np.broadcast_to(self._heights, self._nodes.shape)
        return self._scatter(stacked, fill=1.0)

    def _scatter(self, per_channel: np.ndarray, fill: float) -> np.ndarray:
        out = np.full(self._geometry.n_nodes, fill, dtype=float)
        out[self._nodes[self._active]] = per_channel[self._active]
        return out

    def _checked_mass_flow(self, mass_flow) -> np.ndarray:
        flow = np.asarray(mass_flow, dtype=float)
        if flow.ndim == 0:
            flow = np.full(self.n_channels, float(flow))
        if flow.shape != (self.n_channels,):
            raise InputError(
                f"expected one mass flow, or one per each of "
                f"{self.n_channels} channels, got shape {flow.shape}"
            )
        if np.any(flow <= 0.0):
            raise InputError(
                f"mass flow must be positive, got {float(flow.min())} kg/s"
            )
        return flow

    def __repr__(self) -> str:
        return (
            f"<ChannelModel {self.n_channels} channels "
            f"inlet={self.inlet_temperature:g} K "
            f"p={self.pressure / 1.0e6:g} MPa>"
        )


def _channel_columns(geometry: Geometry) -> np.ndarray:
    """Node index per channel and axial plane, ``(n_channels, nz)``, -1 empty.

    Channels run bottom to top, so a cumulative sum along the second axis is
    an integral up the channel.
    """
    mapping = np.asarray(geometry.lattice_to_node).reshape(geometry.shape)
    carries_nodes = (mapping >= 0).any(axis=0)
    if not carries_nodes.any():
        raise InputError("the geometry has no active node to build a channel from")
    rows, columns = np.nonzero(carries_nodes)
    return mapping[:, rows, columns].T


def absolute_power(
    relative_power: np.ndarray,
    volumes: np.ndarray,
    total_power: float,
    percent: float = 100.0,
) -> np.ndarray:
    """Turn a relative node power into watts per node.

    :attr:`~openndm.Result.power` is normalised to a mean of one over the
    powered nodes, so a node's share of the total is its relative power
    weighted by its volume.

    Parameters
    ----------
    relative_power : array_like, shape (n_nodes,)
        Node power as the solver reports it.
    volumes : array_like, shape (n_nodes,)
        Node volumes in cm^3.
    total_power : float
        Thermal power of the modelled geometry at full power, W. A quarter
        core carries a quarter of the core's power.
    percent : float, optional
        Percent of full power, so a zero-power deck costs no special case.

    Returns
    -------
    ndarray, shape (n_nodes,)
        Power per node in W, summing to ``total_power * percent / 100``.

    Raises
    ------
    InputError
        On a negative total power, a percent outside ``[0, 100]``, mismatched
        shapes, or a distribution carrying no power at all.

    Examples
    --------
    >>> import numpy as np
    >>> absolute_power(np.array([1.0, 1.0]), np.array([1.0, 3.0]), 8.0).tolist()
    [2.0, 6.0]
    """
    power = np.asarray(relative_power, dtype=float)
    volume = np.asarray(volumes, dtype=float)
    if power.shape != volume.shape:
        raise InputError(
            f"power and volumes must match, got {power.shape} and {volume.shape}"
        )
    if total_power < 0.0:
        raise InputError(f"total power cannot be negative, got {total_power}")
    if not 0.0 <= percent <= 100.0:
        raise InputError(f"percent power must lie in [0, 100], got {percent}")

    weighted = power * volume
    total = float(weighted.sum())
    if total <= 0.0:
        raise InputError("the power distribution carries no power")
    return weighted * (total_power * percent / 100.0) / total
