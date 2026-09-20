"""Control rod banks (FR-MODE-5).

A bank is a set of radial lattice columns that share one axial position. The
position is given in *steps*, following the convention KOMODO's ``%CROD`` card
uses, so an existing deck translates without arithmetic:

``zero_position``
    Height of the rod tip above the bottom of the mesh at step 0, in cm.
``step_size``
    Centimetres of travel per step.

Step 0 is therefore the most inserted position and increasing steps withdraw
the bank upward. Rods enter from the top, so a node is rodded when it sits
above the tip.

Mutating the geometry does not by itself update a solver that is already
holding it, exactly as for :meth:`~openndm.Model.swap_assemblies`. Call
:meth:`~openndm.Model.refresh` after moving a bank.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .exceptions import InputError

__all__ = ["ControlRodBank", "ControlRods"]


@dataclass(frozen=True)
class ControlRodBank:
    """One bank of control rods.

    Parameters
    ----------
    name : str
        Identifies the bank when setting positions.
    columns : array_like of bool, shape (ny, nx)
        Radial lattice positions this bank occupies. This is KOMODO's
        ``BMAP`` reduced to a single bank.
    rodded : mapping of int to int
        Composition substitution applied to a node the bank covers, keyed by
        the node's *unrodded* composition. Every composition the bank can
        reach must appear, including reflector compositions if the rods
        travel through an axial reflector; a missing key is an error rather
        than a silent no-op, because an unsubstituted node looks exactly like
        a correctly withdrawn one.
    step_size : float
        Centimetres of travel per step.
    zero_position : float
        Height of the tip above the bottom of the mesh at step 0, in cm.
        Note this is the bottom of the *mesh*, not of the active core: with a
        bottom axial reflector present, add its height.
    max_steps : int, optional
        Upper bound on the position, enforced when a position is set.
    """

    name: str
    columns: np.ndarray
    rodded: Mapping[int, int]
    step_size: float = 1.0
    zero_position: float = 0.0
    max_steps: int | None = None

    def __post_init__(self) -> None:
        columns = np.asarray(self.columns, dtype=bool)
        if columns.ndim != 2:
            raise InputError(
                f"bank {self.name!r}: columns must be a 2D (ny, nx) mask, "
                f"got {columns.ndim}D"
            )
        if not columns.any():
            raise InputError(f"bank {self.name!r}: no columns selected")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "rodded", dict(self.rodded))
        if self.step_size <= 0.0:
            raise InputError(
                f"bank {self.name!r}: step_size must be positive, got {self.step_size}"
            )
        if self.max_steps is not None and self.max_steps < 0:
            raise InputError(f"bank {self.name!r}: max_steps must not be negative")

    def tip_height(self, steps: float) -> float:
        """Height of the rod tip above the bottom of the mesh, in cm."""
        return self.zero_position + float(steps) * self.step_size


class ControlRods:
    """Control rod banks bound to a geometry.

    The unrodded composition of every node the banks can reach is snapshotted
    at construction, so positions are absolute rather than incremental: moving
    a bank twice gives the same core as setting its final position once, and
    withdrawing it restores exactly what was there before.

    Parameters
    ----------
    geometry : Geometry
        Built with the banks **withdrawn** -- every node carrying its unrodded
        composition.
    banks : sequence of ControlRodBank
    """

    def __init__(self, geometry, banks: Sequence[ControlRodBank]):
        self._geometry = geometry
        self._banks: dict[str, ControlRodBank] = {}
        for bank in banks:
            if bank.name in self._banks:
                raise InputError(f"duplicate bank name {bank.name!r}")
            self._banks[bank.name] = bank

        nz, ny, nx = geometry.shape
        edges = np.concatenate([[0.0], np.cumsum(geometry.dz)])
        centres = 0.5 * (edges[:-1] + edges[1:])
        mapping = geometry.lattice_to_node.reshape(nz, ny, nx)
        compositions = geometry.compositions

        self._nodes: dict[str, np.ndarray] = {}
        self._centres: dict[str, np.ndarray] = {}
        self._base: dict[str, np.ndarray] = {}
        for name, bank in self._banks.items():
            if bank.columns.shape != (ny, nx):
                raise InputError(
                    f"bank {name!r}: columns has shape {bank.columns.shape}, "
                    f"but the lattice is ({ny}, {nx})"
                )
            covered = np.broadcast_to(bank.columns, (nz, ny, nx))
            nodes = mapping[covered & (mapping >= 0)]
            self._nodes[name] = nodes
            self._centres[name] = np.repeat(centres, ny * nx).reshape(nz, ny, nx)[
                covered & (mapping >= 0)
            ]
            self._base[name] = compositions[nodes].copy()

        self._positions: dict[str, float | None] = dict.fromkeys(self._banks)

    # ------------------------------------------------------------ properties
    @property
    def banks(self) -> tuple[str, ...]:
        return tuple(self._banks)

    @property
    def positions(self) -> dict[str, float | None]:
        """Position of each bank in steps; ``None`` means fully withdrawn."""
        return dict(self._positions)

    def tip_height(self, name: str) -> float:
        """Tip height of one bank above the bottom of the mesh, in cm.

        A fully withdrawn bank returns the top of the mesh.
        """
        bank = self._bank(name)
        steps = self._positions[name]
        if steps is None:
            return float(np.sum(self._geometry.dz))
        return bank.tip_height(steps)

    # --------------------------------------------------------------- methods
    def insert(self, positions: Mapping[str, float | None] | None = None, **kwargs):
        """Set bank positions in steps and rewrite the node compositions.

        Accepts a mapping, keyword arguments, or both. A bank left unnamed
        keeps its current position; ``None`` withdraws one fully.

        Call :meth:`~openndm.Model.refresh` afterwards if a solver already
        holds this geometry.
        """
        requested: dict[str, float | None] = {}
        requested.update(positions or {})
        requested.update(kwargs)

        for name, steps in requested.items():
            bank = self._bank(name)
            if steps is not None:
                steps = float(steps)
                if steps < 0.0:
                    raise InputError(
                        f"bank {name!r}: position must not be negative, got {steps}"
                    )
                if bank.max_steps is not None and steps > bank.max_steps:
                    raise InputError(
                        f"bank {name!r}: position {steps} exceeds max_steps "
                        f"{bank.max_steps}"
                    )
            self._positions[name] = steps

        self._apply()
        return self

    def withdraw(self, *names: str):
        """Fully withdraw the named banks, or all of them if none are named."""
        targets = names or tuple(self._banks)
        return self.insert(dict.fromkeys(targets, None))

    # ----------------------------------------------------------------- inner
    def _bank(self, name: str) -> ControlRodBank:
        try:
            return self._banks[name]
        except KeyError:
            raise InputError(
                f"unknown bank {name!r}; known banks are {sorted(self._banks)}"
            ) from None

    def _apply(self) -> None:
        geometry = self._geometry
        for name, bank in self._banks.items():
            nodes = self._nodes[name]
            base = self._base[name]
            steps = self._positions[name]
            if steps is None:
                rodded = np.zeros(nodes.shape, dtype=bool)
            else:
                # Rods enter from the top: a node is rodded when it sits above
                # the tip. Comparing centres makes the assignment exact
                # whenever the tip lands on a plane boundary, and a step
                # function in between -- see the cusping note in the docs.
                rodded = self._centres[name] > bank.tip_height(steps)

            for node, was, is_rodded in zip(nodes, base, rodded, strict=True):
                if not is_rodded:
                    geometry.set_composition(int(node), int(was))
                    continue
                try:
                    geometry.set_composition(int(node), int(bank.rodded[int(was)]))
                except KeyError:
                    raise InputError(
                        f"bank {name!r} covers a node of composition {int(was)}, "
                        f"which has no rodded counterpart; rodded maps "
                        f"{sorted(bank.rodded)}"
                    ) from None
