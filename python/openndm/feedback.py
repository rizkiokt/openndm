"""Analytic cross section feedback models (FR-XS-7).

A branch library generated from OpenMC carries the real temperature
dependence of every cross section and is strictly better than anything here.
This exists for the case where there is no such library: the LRA BWR
transient, for one, *defines* its Doppler feedback as a square-root law and
cannot be run as specified without it.

Note this is not parity work. KOMODO's ``%FTEM`` card is strictly linear per
Kelvin -- a reference value plus a per-Kelvin derivative -- with no
square-root form anywhere in its documentation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .exceptions import InputError

__all__ = ["DopplerFeedback"]

#: Cross section fields the model may scale.
_FIELDS = ("absorption", "nu_fission", "kappa_fission")


class DopplerFeedback:
    r"""Square-root temperature feedback on a set of compositions.

    .. math:: \Sigma(T) = \Sigma_0 \left[1 + \gamma\left(\sqrt{T} -
              \sqrt{T_0}\right)\right]

    The base cross sections are snapshotted at construction, so temperatures
    are absolute rather than incremental: applying twice gives the same
    library as applying once, and returning to the reference temperature
    restores exactly what was there before.

    Because cross sections live per composition rather than per node, a
    temperature distribution needs one composition per region that can have
    its own temperature. Give each such region its own index in the core map.

    Parameters
    ----------
    library : XSLibrary
        Must be finalized and collapsed. Written in place by :meth:`apply`,
        which re-finalizes.
    compositions : sequence of int
        The compositions that feel the feedback.
    gamma : float
        Feedback coefficient, in :math:`K^{-1/2}`. Positive gamma raises
        absorption with temperature, which is the physical sign for Doppler
        broadening of a capture resonance.
    reference_temperature : float
        :math:`T_0`, in Kelvin. Above absolute zero.
    field : str
        Which cross section the law scales. ``"absorption"`` is the Doppler
        case and the default.
    groups : sequence of int, optional
        Groups that feel it. Every group by default. The LRA specification
        applies it to the thermal group alone.
    """

    def __init__(
        self,
        library,
        compositions: Sequence[int],
        *,
        gamma: float,
        reference_temperature: float,
        field: str = "absorption",
        groups: Sequence[int] | None = None,
    ):
        if field not in _FIELDS:
            raise InputError(
                f"cannot apply feedback to {field!r}; choose from {list(_FIELDS)}"
            )
        if not (reference_temperature > 0.0):
            raise InputError(
                f"reference temperature must be above absolute zero, got "
                f"{reference_temperature}"
            )
        if library.n_states > 1:
            raise InputError(
                "a branch-parameterised library must be collapsed first; it "
                "already carries a temperature dependence of its own"
            )
        indices = [int(c) for c in compositions]
        if not indices:
            raise InputError("no compositions given")
        for index in indices:
            if not 0 <= index < library.n_compositions:
                raise InputError(
                    f"composition {index} is outside the library's "
                    f"{library.n_compositions}"
                )

        self._library = library
        self._compositions = tuple(indices)
        self.gamma = float(gamma)
        self.reference_temperature = float(reference_temperature)
        self.field = field

        n_groups = library.n_groups
        if groups is None:
            self._groups = tuple(range(n_groups))
        else:
            self._groups = tuple(int(g) for g in groups)
            for g in self._groups:
                if not 0 <= g < n_groups:
                    raise InputError(f"group {g} is outside the library's {n_groups}")

        # Snapshot every field, not just the scaled one: set_composition
        # writes a whole composition, so the others have to be handed back
        # unchanged or they revert to zero.
        self._base = {index: self._snapshot(index) for index in self._compositions}

    # ------------------------------------------------------------ properties
    @property
    def compositions(self) -> tuple[int, ...]:
        return self._compositions

    @property
    def groups(self) -> tuple[int, ...]:
        return self._groups

    # --------------------------------------------------------------- methods
    def factor(self, temperature: float) -> float:
        """The multiplier at one temperature."""
        if not (temperature >= 0.0):
            raise InputError(f"temperature must not be negative, got {temperature}")
        return 1.0 + self.gamma * (
            math.sqrt(temperature) - math.sqrt(self.reference_temperature)
        )

    def apply(self, temperatures) -> None:
        """Set the temperature of each composition and rewrite the library.

        Accepts one temperature for all of them, or one per composition in
        the order given to the constructor.

        The library is re-finalized here, so a model holding it only needs
        :meth:`~openndm.Model.refresh`.
        """
        values = np.atleast_1d(np.asarray(temperatures, dtype=float))
        if values.size == 1:
            values = np.repeat(values, len(self._compositions))
        if values.size != len(self._compositions):
            raise InputError(
                f"expected 1 or {len(self._compositions)} temperatures, got "
                f"{values.size}"
            )

        for index, temperature in zip(self._compositions, values, strict=True):
            factor = self.factor(float(temperature))
            fields = {
                name: np.array(value, dtype=float, copy=True)
                for name, value in self._base[index].items()
            }
            scaled = fields[self.field]
            for g in self._groups:
                scaled[g] *= factor
            if np.any(scaled < 0.0):
                raise InputError(
                    f"composition {index} at {temperature} K gives a negative "
                    f"{self.field}; gamma {self.gamma} is too large a "
                    "reduction for this temperature"
                )
            self._library.set_composition(
                index,
                D=fields["D"],
                absorption=fields["absorption"],
                nu_fission=fields["nu_fission"],
                kappa_fission=fields["kappa_fission"],
                chi=fields["chi"],
                scatter=fields["scatter"].reshape(
                    self._library.n_groups, self._library.n_groups
                ),
                inv_velocity=fields["inv_velocity"],
            )
        self._library.finalize(warn=False)

    def _snapshot(self, index: int) -> dict[str, np.ndarray]:
        composition = self._library.composition(index)
        return {
            name: np.asarray(getattr(composition, name), dtype=float).copy()
            for name in (
                "D",
                "absorption",
                "nu_fission",
                "kappa_fission",
                "chi",
                "scatter",
                "inv_velocity",
            )
        }

    def __repr__(self) -> str:
        return (
            f"<DopplerFeedback gamma={self.gamma:g} "
            f"T0={self.reference_temperature:g}K "
            f"compositions={list(self._compositions)}>"
        )
